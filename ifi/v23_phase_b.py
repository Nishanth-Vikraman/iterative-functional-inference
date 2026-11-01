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
from .v23_models import ForwardTrainedPolicy


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    values = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


class ForwardCreditTrainer:
    """Coordinate-descent H/Phi blocks without BPTT through functional transitions."""

    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: SyntheticFunctionBenchmark,
        policy: ForwardTrainedPolicy,
    ):
        if not operator.frozen:
            raise ValueError("v2.3 requires the frozen Phase-A operator")
        self.config = config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.policy = policy
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.phase_b.learning_rate)

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    def _action_metric(self, difference: Tensor, gram: Tensor) -> Tensor:
        if self.config.phase_b.action_metric == "mse":
            return difference.square().mean()
        scale = torch.diagonal(gram).mean().detach().clamp_min(1e-8)
        return (difference @ gram.detach() @ difference) / (difference.numel() * scale)

    def _sample_episode_horizons(self):
        cfg = self.config.phase_b
        episodes = [
            self.benchmark.sample_episode(self.benchmark.sample_support_size(), "train")
            for _ in range(cfg.batch_size)
        ]
        horizons = [
            random.randint(1, cfg.rollout_max) if cfg.variable_rollout else cfg.rollout_max
            for _ in episodes
        ]
        return episodes, horizons

    def _h_step(self) -> dict[str, float]:
        self.policy.set_training_block("h")
        episodes, horizons = self._sample_episode_horizons()
        losses: list[Tensor] = []
        action_mses: list[Tensor] = []
        current_actions_require_grad = False
        for episode, horizon in zip(episodes, horizons):
            z_data = self.policy.encode_data(episode.support_x, episode.support_y)
            state = self.policy.initial_state(z_data)
            history: tuple[Tensor, ...] = ()
            current = self.initial_function()
            # A horizon of L actions has L-1 within-rollout next-decision targets.
            for _ in range(max(0, horizon - 1)):
                raw = self.policy.raw_residual_tokens(
                    episode.support_x, episode.support_y, current(episode.support_x)
                )
                generated = self.policy.step(raw, z_data, state, history)
                current_actions_require_grad |= generated.action.requires_grad

                # The functional transition and action history are hard stop-gradient boundaries.
                action = generated.action.detach()
                next_function = current.append(action, detach=True).detached()
                next_raw = self.policy.raw_residual_tokens(
                    episode.support_x,
                    episode.support_y,
                    next_function(episode.support_x),
                ).detach()
                next_history = history + (action,)

                probes = self.benchmark.sample_inputs(self.config.phase_b.probe_size)
                update = self.teacher.propose_update(
                    next_function, (episode.support_x, episode.support_y), probes
                )
                projection = project_update(
                    self.operator, next_function, update, probes,
                    self.config.operator.projection_ridge,
                )
                if projection.relative_residual.item() > self.config.phase_a.span_threshold:
                    break

                # Frozen T at i+1 remains differentiable with respect to its h input only.
                next_action = self.policy.predict_action(
                    next_raw,
                    z_data,
                    generated.next_state,
                    next_history,
                )
                difference = next_action - projection.phi.detach()
                losses.append(self._action_metric(difference, projection.gram))
                action_mses.append(difference.square().mean())

                current = next_function
                history = next_history
                state = generated.next_state.detach()

        zero = self.policy.controller.h_delta.weight.sum() * 0.0
        loss = torch.stack(losses).mean() if losses else zero
        action_mse = torch.stack(action_mses).mean() if action_mses else zero
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        h_parameters = self.policy.h_parameters()
        action_parameters = self.policy.action_parameters()
        h_norm = _gradient_norm(h_parameters)
        frozen_norm = _gradient_norm(action_parameters)
        global_norm = float(nn.utils.clip_grad_norm_(h_parameters, self.config.phase_b.gradient_clip))
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "next_action_mse": float(action_mse.detach()),
            "H_h_grad_norm": h_norm,
            "H_frozen_action_grad_norm": frozen_norm,
            "H_phi_i_grad_norm": float(current_actions_require_grad),
            "H_environment_grad_path": 0.0,
            "Phi_action_grad_norm": 0.0,
            "Phi_h_grad_norm": 0.0,
            "Phi_state_inputs_require_grad": 0.0,
            "gradient_norm": global_norm,
            "gradient_clipped": float(global_norm > self.config.phase_b.gradient_clip),
        }

    def _phi_step(self) -> dict[str, float]:
        self.policy.set_training_block("phi")
        episodes, horizons = self._sample_episode_horizons()
        action_losses: list[Tensor] = []
        action_mses: list[Tensor] = []
        functional_losses: list[Tensor] = []
        state_inputs_require_grad = False
        for episode, horizon in zip(episodes, horizons):
            z_data = self.policy.encode_data(episode.support_x, episode.support_y)
            state = self.policy.initial_state(z_data).detach()
            history: tuple[Tensor, ...] = ()
            current = self.initial_function()
            for _ in range(horizon):
                state_inputs_require_grad |= state.requires_grad
                raw = self.policy.raw_residual_tokens(
                    episode.support_x, episode.support_y, current(episode.support_x)
                )
                generated = self.policy.step(raw, z_data, state, history)
                probes = self.benchmark.sample_inputs(self.config.phase_b.probe_size)
                update = self.teacher.propose_update(
                    current, (episode.support_x, episode.support_y), probes
                )
                projection = project_update(
                    self.operator, current, update, probes,
                    self.config.operator.projection_ridge,
                )
                if projection.relative_residual.item() > self.config.phase_a.span_threshold:
                    break
                difference = generated.action - projection.phi.detach()
                action_losses.append(self._action_metric(difference, projection.gram))
                action_mses.append(difference.square().mean())
                candidate = current.append(generated.action)
                functional_losses.append(
                    (candidate(probes) - update.teacher(probes).detach()).square().mean()
                )
                current = candidate.detached()
                history = history + (generated.action.detach(),)
                state = generated.next_state.detach()

        zero = self.policy.controller.phi_head[-1].weight.sum() * 0.0
        action_loss = torch.stack(action_losses).mean() if action_losses else zero
        action_mse = torch.stack(action_mses).mean() if action_mses else zero
        functional = torch.stack(functional_losses).mean() if functional_losses else zero
        loss = action_loss + self.config.phase_b.lambda_functional * functional
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        action_parameters = self.policy.action_parameters()
        h_parameters = self.policy.h_parameters()
        action_norm = _gradient_norm(action_parameters)
        h_norm = _gradient_norm(h_parameters)
        global_norm = float(nn.utils.clip_grad_norm_(
            action_parameters, self.config.phase_b.gradient_clip
        ))
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "action_metric": float(action_loss.detach()),
            "action_mse": float(action_mse.detach()),
            "functional_mse": float(functional.detach()),
            "H_h_grad_norm": 0.0,
            "H_frozen_action_grad_norm": 0.0,
            "H_phi_i_grad_norm": 0.0,
            "H_environment_grad_path": 0.0,
            "Phi_action_grad_norm": action_norm,
            "Phi_h_grad_norm": h_norm,
            "Phi_state_inputs_require_grad": float(state_inputs_require_grad),
            "gradient_norm": global_norm,
            "gradient_clipped": float(global_norm > self.config.phase_b.gradient_clip),
        }

    def block_for_step(self, step: int) -> str:
        length = self.config.forward_state.block_length
        position = (step - 1) % (2 * length)
        return "h" if position < length else "phi"

    def train_step(self, step: int) -> dict[str, float]:
        block = self.block_for_step(step)
        metrics = self._h_step() if block == "h" else self._phi_step()
        metrics["block_h"] = float(block == "h")
        metrics["block_phi"] = float(block == "phi")
        if any(parameter.grad is not None for parameter in self.operator.parameters()):
            raise RuntimeError("Frozen operator received gradients in v2.3")
        return metrics

    def run(
        self,
        on_log: Callable[[int, list[dict[str, float]]], None] | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for step in range(1, self.config.phase_b.steps + 1):
            metrics = self.train_step(step)
            should_log = step == 1 or step % self.config.phase_b.log_every == 0
            if should_log:
                metrics["step"] = float(step)
                history.append(metrics)
                print(json.dumps({"phase": "B_v2_3", **metrics}), flush=True)
                if on_log is not None:
                    on_log(step, history)
        return history
