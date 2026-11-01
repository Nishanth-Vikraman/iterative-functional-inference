from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, ProjectionResult, project_update
from .tasks import Episode, SyntheticFunctionBenchmark
from .teachers import RKHSFunctionalGradientTeacher
from .v21_models import DirectIFIPolicy


@dataclass
class DirectRolloutLoss:
    total: Tensor
    action_metric: Tensor
    action_mse: Tensor
    functional_mse: Tensor
    used_steps: int
    stop_reasons: dict[str, int]


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    squares = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


class DirectPhaseBTrainer:
    """v2.1 on-policy/local training for the matched direct-evidence A/B/C variants."""

    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: SyntheticFunctionBenchmark,
        policies: dict[str, DirectIFIPolicy],
    ):
        if not operator.frozen:
            raise ValueError("v2.1 requires the frozen canonical Phase-A operator")
        if set(policies) != {"direct_current", "direct_history", "direct_recurrent"}:
            raise ValueError("Exactly the three preregistered v2.1 variants are required")
        self.config = config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.policies = policies
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        parameters = [parameter for policy in policies.values() for parameter in policy.parameters()]
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
        scale = torch.diagonal(gram).mean().detach().clamp_min(1e-8)
        return (difference @ gram.detach() @ difference) / (difference.numel() * scale)

    def _rollout(
        self, policy: DirectIFIPolicy, episode: Episode, horizon: int
    ) -> DirectRolloutLoss:
        z_data = policy.encode_data(episode.support_x, episode.support_y)
        state = policy.initial_state(z_data)
        history: tuple[Tensor, ...] = ()
        current = self.initial_function()
        action_metrics: list[Tensor] = []
        action_mses: list[Tensor] = []
        function_losses: list[Tensor] = []
        stops = {"small_update": 0, "negligible_improvement": 0, "outside_span": 0}
        for _ in range(horizon):
            # T sees only current behavioral evidence before the independent teacher is called.
            support_prediction = current(episode.support_x)
            raw_tokens = policy.raw_residual_tokens(
                episode.support_x, episode.support_y, support_prediction
            )
            predicted = policy.step(raw_tokens, z_data, history, state)

            probes = self.benchmark.sample_inputs(self.config.phase_b.probe_size)
            projection, teacher_values, metadata = self._supervision(current, episode, probes)
            if projection.relative_residual.item() > self.config.phase_a.span_threshold:
                stops["outside_span"] += 1
                break
            if projection.target_energy.sqrt().item() < self.config.teacher.update_epsilon:
                stops["small_update"] += 1
                break
            if float(metadata["predicted_improvement"]) < self.config.teacher.improvement_epsilon:
                stops["negligible_improvement"] += 1
                break

            phi_star = projection.phi.detach()
            difference = predicted.action - phi_star
            action_metrics.append(self._action_metric(difference, projection.gram))
            action_mses.append(difference.square().mean())
            candidate = current.append(predicted.action)
            function_losses.append((candidate(probes) - teacher_values).square().mean())

            # Own action, on-policy state, and exact local-gradient truncation.
            current = candidate.detached()
            history = history + (predicted.action.detach(),)
            state = None if predicted.next_state is None else predicted.next_state.detach()

        zero = z_data.sum() * 0.0
        action_metric = torch.stack(action_metrics).mean() if action_metrics else zero
        action_mse = torch.stack(action_mses).mean() if action_mses else zero
        functional = torch.stack(function_losses).mean() if function_losses else zero
        total = action_metric + self.config.phase_b.lambda_functional * functional
        return DirectRolloutLoss(
            total, action_metric, action_mse, functional, len(action_mses), stops
        )

    def train_step(self) -> dict[str, float]:
        cfg = self.config.phase_b
        self.optimizer.zero_grad(set_to_none=True)
        losses: dict[str, list[DirectRolloutLoss]] = {name: [] for name in self.policies}
        for _ in range(cfg.batch_size):
            episode = self.benchmark.sample_episode(self.benchmark.sample_support_size(), "train")
            horizon = random.randint(1, cfg.rollout_max) if cfg.variable_rollout else cfg.rollout_max
            for name, policy in self.policies.items():
                losses[name].append(self._rollout(policy, episode, horizon))
        flat = [loss for variant_losses in losses.values() for loss in variant_losses]
        total = torch.stack([loss.total for loss in flat]).mean()
        total.backward()
        parameters_by_variant = {
            name: list(policy.parameters()) for name, policy in self.policies.items()
        }
        gradient_norms = {
            name: _gradient_norm(parameters) for name, parameters in parameters_by_variant.items()
        }
        all_parameters = [parameter for values in parameters_by_variant.values() for parameter in values]
        global_norm = float(nn.utils.clip_grad_norm_(all_parameters, cfg.gradient_clip))
        self.optimizer.step()
        if any(parameter.grad is not None for parameter in self.operator.parameters()):
            raise RuntimeError("Frozen operator received gradients during v2.1 Phase B")

        metrics: dict[str, float] = {
            "loss": float(total.detach()),
            "gradient_norm": global_norm,
            "gradient_clipped": float(global_norm > cfg.gradient_clip),
            "gradient_clip_ratio": min(1.0, cfg.gradient_clip / max(global_norm, 1e-12)),
        }
        for name, variant_losses in losses.items():
            count = len(variant_losses)
            metrics[f"{name}/loss"] = sum(float(item.total.detach()) for item in variant_losses) / count
            metrics[f"{name}/action_metric"] = (
                sum(float(item.action_metric.detach()) for item in variant_losses) / count
            )
            metrics[f"{name}/action_mse"] = (
                sum(float(item.action_mse.detach()) for item in variant_losses) / count
            )
            metrics[f"{name}/functional_mse"] = (
                sum(float(item.functional_mse.detach()) for item in variant_losses) / count
            )
            metrics[f"{name}/mean_used_steps"] = sum(item.used_steps for item in variant_losses) / count
            metrics[f"{name}/gradient_norm"] = gradient_norms[name]
            metrics[f"{name}/outside_span"] = float(
                sum(item.stop_reasons["outside_span"] for item in variant_losses)
            )
        return metrics

    def run(
        self,
        on_log: Callable[[int, list[dict[str, float]]], None] | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for step in range(1, self.config.phase_b.steps + 1):
            metrics = self.train_step()
            if step == 1 or step % self.config.phase_b.log_every == 0:
                metrics["step"] = float(step)
                history.append(metrics)
                print(json.dumps({"phase": "B_v2_1", **metrics}), flush=True)
                if on_log is not None:
                    on_log(step, history)
        return history


def rng_state_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    return payload
