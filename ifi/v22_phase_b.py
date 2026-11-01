from __future__ import annotations

import json
import random
from typing import Callable

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, project_update
from .tasks import SyntheticFunctionBenchmark
from .teachers import RKHSFunctionalGradientTeacher
from .v22_models import FixedResidualContrast, PredictiveRecurrentPolicy


def _norm_from_grads(grads: tuple[Tensor | None, ...]) -> float:
    values = [gradient.detach().square().sum() for gradient in grads if gradient is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def _parameter_gradient_norm(parameters: list[nn.Parameter]) -> float:
    values = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


class PredictiveStateTrainer:
    """Disjoint local objectives: teacher action supervision for T, InfoNCE only for U."""

    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: SyntheticFunctionBenchmark,
        policy: PredictiveRecurrentPolicy,
    ):
        if not operator.frozen:
            raise ValueError("v2.2 requires the frozen Phase-A operator")
        self.config = config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.policy = policy
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        self.contrast = FixedResidualContrast(
            policy.model_dimension,
            config.predictive_state.projection_seed,
            config.predictive_state.temperature,
        ).to(next(policy.parameters()).device)
        if any(True for _ in self.contrast.parameters()):
            raise RuntimeError("Fixed residual contrast unexpectedly has trainable parameters")
        self.optimizer = torch.optim.Adam(
            policy.t_parameters() + policy.u_parameters(),
            lr=config.phase_b.learning_rate,
        )

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    def _action_metric(self, difference: Tensor, gram: Tensor) -> Tensor:
        if self.config.phase_b.action_metric == "mse":
            return difference.square().mean()
        scale = torch.diagonal(gram).mean().detach().clamp_min(1e-8)
        return (difference @ gram.detach() @ difference) / (difference.numel() * scale)

    def train_step(self, diagnose_routing: bool = False) -> dict[str, float]:
        cfg = self.config.phase_b
        episodes = [
            self.benchmark.sample_episode(self.benchmark.sample_support_size(), "train")
            for _ in range(cfg.batch_size)
        ]
        horizons = [
            random.randint(1, cfg.rollout_max) if cfg.variable_rollout else cfg.rollout_max
            for _ in episodes
        ]
        z_data = [
            self.policy.encode_data(episode.support_x, episode.support_y)
            for episode in episodes
        ]
        states = [self.policy.initial_state(value) for value in z_data]
        histories: list[tuple[Tensor, ...]] = [() for _ in episodes]
        functions = [self.initial_function() for _ in episodes]
        alive = [True for _ in episodes]
        action_metrics: list[Tensor] = []
        action_mses: list[Tensor] = []
        functional_losses: list[Tensor] = []
        infonce_losses: list[Tensor] = []
        infonce_accuracies: list[Tensor] = []
        contrastive_batch_sizes: list[int] = []
        stops = {"small_update": 0, "negligible_improvement": 0, "outside_span": 0}

        for iteration in range(cfg.rollout_max):
            next_states: list[Tensor] = []
            next_residuals: list[Tensor] = []
            for index, episode in enumerate(episodes):
                if not alive[index] or iteration >= horizons[index]:
                    continue
                current = functions[index]
                support_prediction = current(episode.support_x)
                raw_tokens = self.policy.raw_residual_tokens(
                    episode.support_x, episode.support_y, support_prediction
                )
                action = self.policy.predict_action(
                    raw_tokens, z_data[index], states[index], histories[index]
                )

                probes = self.benchmark.sample_inputs(cfg.probe_size)
                update = self.teacher.propose_update(
                    current, (episode.support_x, episode.support_y), probes
                )
                projection = project_update(
                    self.operator, current, update, probes,
                    self.config.operator.projection_ridge,
                )
                if projection.relative_residual.item() > self.config.phase_a.span_threshold:
                    stops["outside_span"] += 1
                    alive[index] = False
                    continue
                if projection.target_energy.sqrt().item() < self.config.teacher.update_epsilon:
                    stops["small_update"] += 1
                    alive[index] = False
                    continue
                if float(update.metadata["predicted_improvement"]) < self.config.teacher.improvement_epsilon:
                    stops["negligible_improvement"] += 1
                    alive[index] = False
                    continue

                difference = action - projection.phi.detach()
                action_metrics.append(self._action_metric(difference, projection.gram))
                action_mses.append(difference.square().mean())
                candidate = current.append(action)
                functional_losses.append(
                    (candidate(probes) - update.teacher(probes).detach()).square().mean()
                )

                # U sees only detached trajectory variables. Its target is the actual next residual.
                next_state = self.policy.update_state(states[index], raw_tokens, action)
                detached_candidate = candidate.detached()
                future_tokens = self.policy.raw_residual_tokens(
                    episode.support_x,
                    episode.support_y,
                    detached_candidate(episode.support_x),
                ).detach()
                next_states.append(next_state)
                next_residuals.append(future_tokens)

                functions[index] = detached_candidate
                histories[index] = histories[index] + (action.detach(),)
                states[index] = next_state.detach()

            if len(next_states) >= 2:
                info_loss, info_accuracy = self.contrast(
                    torch.stack(next_states), next_residuals
                )
                infonce_losses.append(info_loss)
                infonce_accuracies.append(info_accuracy)
                contrastive_batch_sizes.append(len(next_states))

        zero = sum(value.sum() for value in z_data) * 0.0
        action_metric = torch.stack(action_metrics).mean() if action_metrics else zero
        action_mse = torch.stack(action_mses).mean() if action_mses else zero
        functional = torch.stack(functional_losses).mean() if functional_losses else zero
        teacher_loss = action_metric + cfg.lambda_functional * functional
        information_loss = torch.stack(infonce_losses).mean() if infonce_losses else zero
        information_accuracy = (
            torch.stack(infonce_accuracies).mean() if infonce_accuracies else zero
        )
        total = teacher_loss + self.config.predictive_state.information_weight * information_loss

        t_parameters = self.policy.t_parameters()
        u_parameters = self.policy.u_parameters()
        routing: dict[str, float] = {}
        if diagnose_routing:
            routing["teacher_to_T_grad_norm"] = _norm_from_grads(torch.autograd.grad(
                teacher_loss, t_parameters, retain_graph=True, allow_unused=True
            ))
            routing["teacher_to_U_grad_norm"] = _norm_from_grads(torch.autograd.grad(
                teacher_loss, u_parameters, retain_graph=True, allow_unused=True
            ))
            routing["info_to_T_grad_norm"] = _norm_from_grads(torch.autograd.grad(
                information_loss, t_parameters, retain_graph=True, allow_unused=True
            ))
            routing["info_to_U_grad_norm"] = _norm_from_grads(torch.autograd.grad(
                information_loss, u_parameters, retain_graph=True, allow_unused=True
            ))

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        preclip_t = _parameter_gradient_norm(t_parameters)
        preclip_u = _parameter_gradient_norm(u_parameters)
        all_parameters = t_parameters + u_parameters
        global_norm = float(nn.utils.clip_grad_norm_(all_parameters, cfg.gradient_clip))
        self.optimizer.step()
        if any(parameter.grad is not None for parameter in self.operator.parameters()):
            raise RuntimeError("Frozen operator received gradients in v2.2")

        metrics = {
            "loss": float(total.detach()),
            "teacher_loss": float(teacher_loss.detach()),
            "action_metric": float(action_metric.detach()),
            "action_mse": float(action_mse.detach()),
            "functional_mse": float(functional.detach()),
            "infonce_loss": float(information_loss.detach()),
            "infonce_accuracy": float(information_accuracy.detach()),
            "mean_contrastive_batch_size": (
                sum(contrastive_batch_sizes) / len(contrastive_batch_sizes)
                if contrastive_batch_sizes else 0.0
            ),
            "T_preclip_grad_norm": preclip_t,
            "U_preclip_grad_norm": preclip_u,
            "gradient_norm": global_norm,
            "gradient_clipped": float(global_norm > cfg.gradient_clip),
            "outside_span": float(stops["outside_span"]),
        }
        metrics.update(routing)
        return metrics

    def run(
        self,
        on_log: Callable[[int, list[dict[str, float]]], None] | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for step in range(1, self.config.phase_b.steps + 1):
            should_log = step == 1 or step % self.config.phase_b.log_every == 0
            metrics = self.train_step(diagnose_routing=should_log)
            if should_log:
                metrics["step"] = float(step)
                history.append(metrics)
                print(json.dumps({"phase": "B_v2_2", **metrics}), flush=True)
                if on_log is not None:
                    on_log(step, history)
        return history
