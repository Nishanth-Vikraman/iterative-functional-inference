from __future__ import annotations

import json
import random
from typing import Callable

import torch
from torch import Tensor, nn

from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, project_update
from .teachers import BayesianLatentSystemTeacher
from .v30_config import V30Config
from .v30_models import HistoryItem, V30Policy
from .v30_tasks import BayesianLatentBenchmark


VARIANTS = ("direct_current", "full_history", "query_history", "ar_no_feedback")


def _gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


class V30PhaseBTrainer:
    def __init__(
        self,
        config: V30Config,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: BayesianLatentBenchmark,
        policies: dict[str, V30Policy],
    ):
        if not operator.frozen:
            raise ValueError("v3.0 Phase B requires a frozen Phase-A operator")
        self.config = config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.policies = policies
        self.teacher = BayesianLatentSystemTeacher(
            benchmark.library, config.tasks.likelihood_noise
        )
        parameters = [
            parameter for policy in policies.values() for parameter in policy.parameters()
        ]
        self.optimizer = torch.optim.Adam(parameters, lr=config.phase_b.learning_rate)

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    def _action_metric(self, difference: Tensor, gram: Tensor) -> Tensor:
        if self.config.phase_b.action_metric == "mse":
            return difference.square().mean()
        scale = torch.diagonal(gram).mean().detach().clamp_min(1e-8)
        return (difference @ gram.detach() @ difference) / (difference.numel() * scale)

    def train_step(self) -> dict[str, float]:
        cfg = self.config.phase_b
        episodes = [
            self.benchmark.sample_episode(
                random.choice(tuple(self.config.tasks.support_sizes)),
                "train_latents",
                self.config.tasks.train_sequence_length,
            )
            for _ in range(cfg.batch_size)
        ]
        horizons = [
            random.randint(1, cfg.rollout_max) if cfg.variable_rollout else cfg.rollout_max
            for _ in episodes
        ]
        action_losses = {name: [] for name in VARIANTS}
        action_mses = {name: [] for name in VARIANTS}
        functional_losses = {name: [] for name in VARIANTS}
        projection_residuals = {name: [] for name in VARIANTS}
        for episode, horizon in zip(episodes, horizons):
            current = {name: self.initial_function() for name in VARIANTS}
            history: dict[str, tuple[HistoryItem, ...]] = {name: () for name in VARIANTS}
            static_z = {}
            for iteration in range(horizon):
                batch = episode.batches[iteration]
                cumulative = episode.cumulative(iteration)
                probes = self.benchmark.sample_query(cfg.probe_size)
                for name, policy in self.policies.items():
                    prediction = current[name](batch.x)
                    raw = policy.raw_residual_tokens(batch.x, batch.y, prediction)
                    current_z = policy.encode_data(batch.x, batch.y)
                    if iteration == 0:
                        static_z[name] = current_z
                    z_data = static_z[name] if name == "ar_no_feedback" else current_z
                    action = policy.predict(raw, z_data, history[name])
                    update = self.teacher.propose_update(current[name], cumulative, probes)
                    projection = project_update(
                        self.operator,
                        current[name],
                        update,
                        probes,
                        self.config.operator.projection_ridge,
                    )
                    difference = action - projection.phi.detach()
                    action_losses[name].append(
                        self._action_metric(difference, projection.gram)
                    )
                    action_mses[name].append(difference.square().mean())
                    candidate = current[name].append(action)
                    functional_losses[name].append(
                        (candidate(probes) - update.teacher(probes).detach()).square().mean()
                    )
                    projection_residuals[name].append(projection.relative_residual.detach())
                    current[name] = candidate.detached()
                    history[name] = history[name] + (
                        HistoryItem(raw.detach(), action.detach()),
                    )

        total_losses = []
        metrics = {}
        for name in VARIANTS:
            action_loss = torch.stack(action_losses[name]).mean()
            action_mse = torch.stack(action_mses[name]).mean()
            functional = torch.stack(functional_losses[name]).mean()
            loss = action_loss + cfg.lambda_functional * functional
            total_losses.append(loss)
            metrics[f"{name}_loss"] = float(loss.detach())
            metrics[f"{name}_action_mse"] = float(action_mse.detach())
            metrics[f"{name}_functional_mse"] = float(functional.detach())
            metrics[f"{name}_projection_residual"] = float(
                torch.stack(projection_residuals[name]).mean()
            )
        total = torch.stack(total_losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        for name, policy in self.policies.items():
            metrics[f"{name}_gradient_norm"] = _gradient_norm(list(policy.parameters()))
        parameters = [
            parameter for policy in self.policies.values() for parameter in policy.parameters()
        ]
        global_norm = float(nn.utils.clip_grad_norm_(parameters, cfg.gradient_clip))
        self.optimizer.step()
        metrics["loss"] = float(total.detach())
        metrics["gradient_norm"] = global_norm
        metrics["gradient_clipped"] = float(global_norm > cfg.gradient_clip)
        metrics["operator_gradient"] = float(any(
            parameter.grad is not None for parameter in self.operator.parameters()
        ))
        if metrics["operator_gradient"]:
            raise RuntimeError("Frozen v3.0 operator received a Phase-B gradient")
        return metrics

    def run(
        self,
        on_log: Callable[[int, list[dict[str, float]]], None] | None = None,
    ) -> list[dict[str, float]]:
        history = []
        for step in range(1, self.config.phase_b.steps + 1):
            metrics = self.train_step()
            if step == 1 or step % self.config.phase_b.log_every == 0:
                metrics["step"] = float(step)
                history.append(metrics)
                print(json.dumps({"phase": "B_v3_0", **metrics}), flush=True)
                if on_log is not None:
                    on_log(step, history)
        return history
