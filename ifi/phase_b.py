from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .functions import ComposedFunction, FunctionSystem
from .models import IFIPolicy, StaticHypernetwork
from .operators import FunctionalOperator, ProjectionResult, project_update
from .tasks import Episode, SyntheticFunctionBenchmark
from .teachers import RKHSFunctionalGradientTeacher


@dataclass
class RolloutLoss:
    total: Tensor
    action_metric: Tensor
    action_mse: Tensor
    functional_mse: Tensor
    differential_mse: Tensor
    used_steps: int
    stop_reasons: dict[str, int]


class PhaseBTrainer:
    """On-policy, locally supervised policy distillation with a permanently frozen operator."""

    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: SyntheticFunctionBenchmark,
        ifi_policy: IFIPolicy,
        ar_policy: IFIPolicy,
        static_policy: StaticHypernetwork,
    ):
        if not operator.frozen:
            raise ValueError("Phase B requires a frozen Phase-A operator")
        self.config = config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        self.ifi_policy = ifi_policy
        self.ar_policy = ar_policy
        self.static_policy = static_policy
        parameters = list(ifi_policy.parameters()) + list(ar_policy.parameters()) + list(static_policy.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=config.phase_b.learning_rate)

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    def _supervision(
        self, current: ComposedFunction, episode: Episode, probes: Tensor
    ) -> tuple[ProjectionResult, Tensor, dict[str, object]]:
        update = self.teacher.propose_update(
            current, (episode.support_x, episode.support_y), probes
        )
        projection = project_update(
            self.operator, current, update, probes, self.config.operator.projection_ridge
        )
        return projection, update.teacher(probes).detach(), update.metadata

    def _action_metric(self, difference: Tensor, gram: Tensor) -> Tensor:
        if self.config.phase_b.action_metric == "mse":
            return difference.square().mean()
        if self.config.phase_b.action_metric != "functional":
            raise ValueError(f"Unknown action metric: {self.config.phase_b.action_metric}")
        scale = torch.diagonal(gram).mean().detach().clamp_min(1e-8)
        return (difference @ gram.detach() @ difference) / (difference.numel() * scale)

    def _rollout_recurrent(
        self, policy: IFIPolicy, episode: Episode, horizon: int, differential: bool = False
    ) -> RolloutLoss:
        z_data = policy.encode_data(episode.support_x, episode.support_y)
        previous_action, state = policy.initial(z_data)
        current = self.initial_function()
        action_metrics, action_mses, function_losses, differential_losses = [], [], [], []
        stops = {"small_update": 0, "negligible_improvement": 0, "outside_span": 0}
        for _ in range(horizon):
            probes = self.benchmark.sample_inputs(self.config.phase_b.probe_size)
            projection, teacher_values, metadata = self._supervision(
                current, episode, probes
            )
            phi_star = projection.phi.detach()
            if projection.relative_residual.item() > self.config.phase_a.span_threshold:
                stops["outside_span"] += 1
                break
            if projection.target_energy.sqrt().item() < self.config.teacher.update_epsilon:
                stops["small_update"] += 1
                break
            if float(metadata["predicted_improvement"]) < self.config.teacher.improvement_epsilon:
                stops["negligible_improvement"] += 1
                break
            prediction = current(episode.support_x)
            z_residual = policy.encode_residual(episode.support_x, episode.support_y, prediction)
            phi_hat, next_state = policy.controller(z_data, z_residual, previous_action, state)
            difference = phi_hat - phi_star
            action_metrics.append(self._action_metric(difference, projection.gram))
            action_mses.append(difference.square().mean())
            candidate = current.append(phi_hat)
            function_losses.append((candidate(probes) - teacher_values).square().mean())

            if differential and self.config.phase_b.lambda_differential > 0:
                alternate_x = self.benchmark.sample_inputs(episode.support_x.shape[0])
                alternate_y = episode.target(alternate_x)
                alternate_episode = Episode(
                    episode.target, alternate_x, alternate_y, episode.split, episode.episode_id
                )
                alternate_projection, _, _ = self._supervision(current, alternate_episode, probes)
                phi_alt = alternate_projection.phi.detach()
                z_alt_data = policy.encode_data(alternate_x, alternate_y)
                z_alt_residual = policy.encode_residual(alternate_x, alternate_y, current(alternate_x))
                phi_alt_hat, _ = policy.controller(
                    z_alt_data, z_alt_residual, previous_action, state
                )
                differential_losses.append(
                    ((phi_hat - phi_alt_hat) - (phi_star - phi_alt)).square().mean()
                )

            # The next teacher sees IFI's actual state. Detachment makes supervision local.
            current = candidate.detached()
            previous_action = phi_hat.detach()
            state = next_state.detach()
        zero = z_data.sum() * 0.0
        action_metric = torch.stack(action_metrics).mean() if action_metrics else zero
        action_mse = torch.stack(action_mses).mean() if action_mses else zero
        functional = torch.stack(function_losses).mean() if function_losses else zero
        differential_loss = torch.stack(differential_losses).mean() if differential_losses else zero
        total = (
            action_metric
            + self.config.phase_b.lambda_functional * functional
            + self.config.phase_b.lambda_differential * differential_loss
        )
        return RolloutLoss(
            total, action_metric, action_mse, functional, differential_loss, len(action_mses), stops
        )

    def _rollout_static(self, episode: Episode, horizon: int) -> RolloutLoss:
        predicted_actions = self.static_policy(episode.support_x, episode.support_y)
        current = self.initial_function()
        action_metrics, action_mses, function_losses = [], [], []
        stops = {"small_update": 0, "negligible_improvement": 0, "outside_span": 0}
        for index in range(horizon):
            probes = self.benchmark.sample_inputs(self.config.phase_b.probe_size)
            projection, teacher_values, _ = self._supervision(current, episode, probes)
            phi_star = projection.phi.detach()
            if projection.relative_residual.item() > self.config.phase_a.span_threshold:
                stops["outside_span"] += 1
                break
            phi_hat = predicted_actions[index]
            difference = phi_hat - phi_star
            action_metrics.append(self._action_metric(difference, projection.gram))
            action_mses.append(difference.square().mean())
            candidate = current.append(phi_hat)
            function_losses.append((candidate(probes) - teacher_values).square().mean())
            current = candidate.detached()
        zero = predicted_actions.sum() * 0.0
        action_metric = torch.stack(action_metrics).mean() if action_metrics else zero
        action_mse = torch.stack(action_mses).mean() if action_mses else zero
        functional = torch.stack(function_losses).mean() if function_losses else zero
        return RolloutLoss(
            action_metric + self.config.phase_b.lambda_functional * functional,
            action_metric, action_mse, functional, zero, len(action_mses), stops
        )

    def train_step(self) -> dict[str, float]:
        cfg = self.config.phase_b
        self.optimizer.zero_grad(set_to_none=True)
        all_losses: list[RolloutLoss] = []
        for _ in range(cfg.batch_size):
            episode = self.benchmark.sample_episode(self.benchmark.sample_support_size(), "train")
            horizon = random.randint(1, cfg.rollout_max) if cfg.variable_rollout else cfg.rollout_max
            all_losses.append(self._rollout_recurrent(self.ifi_policy, episode, horizon, True))
            all_losses.append(self._rollout_recurrent(self.ar_policy, episode, horizon))
            all_losses.append(self._rollout_static(episode, horizon))
        total = torch.stack([loss.total for loss in all_losses]).mean()
        total.backward()
        trainable = (
            list(self.ifi_policy.parameters())
            + list(self.ar_policy.parameters())
            + list(self.static_policy.parameters())
        )
        nn.utils.clip_grad_norm_(trainable, cfg.gradient_clip)
        self.optimizer.step()
        if any(parameter.grad is not None for parameter in self.operator.parameters()):
            raise RuntimeError("Frozen operator received gradients during Phase B")
        return {
            "loss": float(total.detach()),
            "action_metric": sum(float(x.action_metric.detach()) for x in all_losses) / len(all_losses),
            "action_mse": sum(float(x.action_mse.detach()) for x in all_losses) / len(all_losses),
            "functional_mse": sum(float(x.functional_mse.detach()) for x in all_losses) / len(all_losses),
            "mean_used_steps": sum(x.used_steps for x in all_losses) / len(all_losses),
            "outside_span": float(sum(x.stop_reasons["outside_span"] for x in all_losses)),
        }

    def run(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for step in range(1, self.config.phase_b.steps + 1):
            metrics = self.train_step()
            if step == 1 or step % self.config.phase_b.log_every == 0:
                metrics["step"] = float(step)
                history.append(metrics)
                print(json.dumps({"phase": "B", **metrics}))
        return history
