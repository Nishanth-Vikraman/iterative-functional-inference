from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, action_design, project_update
from .teachers import BayesianLatentSystemTeacher
from .v30_config import V30Config
from .v30_tasks import BayesianLatentBenchmark, LatentSplit


@dataclass
class V30PhaseAMetrics:
    split: str
    reconstruction_mse: float
    relative_projection_residual_mean: float
    relative_projection_residual_median: float
    relative_projection_residual_q90: float
    condition_number_median: float
    phi_mean: float
    phi_std: float
    phi_abs_max: float
    outside_span_fraction: float


def _conditioning_loss(design: Tensor) -> Tensor:
    gram = design.transpose(0, 1) @ design / max(1, design.shape[0])
    diagonal = torch.diagonal(gram).clamp_min(1e-6)
    normalized = gram / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    identity = torch.eye(gram.shape[0], device=gram.device)
    return (normalized - identity).square().mean() + 0.05 * torch.log(diagonal).square().mean()


class V30PhaseATrainer:
    def __init__(
        self,
        config: V30Config,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: BayesianLatentBenchmark,
    ):
        self.config = config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.teacher = BayesianLatentSystemTeacher(
            benchmark.library, config.tasks.likelihood_noise
        )
        self.optimizer = torch.optim.Adam(
            operator.parameters(), lr=config.phase_a.learning_rate
        )
        self.device = next(operator.parameters()).device

    def _sample_transition(self, split: LatentSplit):
        support_size = random.choice(tuple(self.config.tasks.support_sizes))
        episode = self.benchmark.sample_episode(
            support_size, split, self.config.tasks.train_sequence_length
        )
        iteration = random.randrange(self.config.tasks.train_sequence_length)
        cumulative = episode.cumulative(iteration)
        if iteration > 0 and random.random() < 0.7:
            prior_support = episode.cumulative(iteration - 1)
            weights = self.teacher.posterior(prior_support).detach()
        else:
            weights = torch.softmax(
                1.5 * torch.randn(self.benchmark.library.size, device=self.device), dim=0
            )
        scale = 0.65 + 0.7 * random.random()

        def initial(x: Tensor, posterior=weights, factor=scale) -> Tensor:
            return factor * self.teacher.predictive(posterior, x)

        current = ComposedFunction(self.system, self.operator, initial_function=initial)
        return current, cumulative

    def train_step(self) -> dict[str, float]:
        cfg = self.config.phase_a
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        reconstructions = []
        conditioning = []
        for _ in range(cfg.batch_size):
            current, cumulative = self._sample_transition("train_latents")
            probes = self.benchmark.sample_query(cfg.probe_size)
            update = self.teacher.propose_update(current, cumulative, probes)
            with torch.no_grad():
                phi = project_update(
                    self.operator,
                    current,
                    update,
                    probes,
                    self.config.operator.projection_ridge,
                ).phi
            design = action_design(self.operator, current, probes).reshape(
                -1, self.operator.action_dimension
            )
            target = update.displacement(probes).reshape(-1).detach()
            reconstruction = (design @ phi - target).square().mean()
            condition = _conditioning_loss(design)
            losses.append(
                reconstruction
                + cfg.conditioning_weight * condition
                + cfg.phi_weight * phi.square().mean()
            )
            reconstructions.append(reconstruction.detach())
            conditioning.append(condition.detach())
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            self.operator.parameters(), 5.0
        ))
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "reconstruction": float(torch.stack(reconstructions).mean()),
            "conditioning_loss": float(torch.stack(conditioning).mean()),
            "gradient_norm": gradient_norm,
        }

    @torch.no_grad()
    def evaluate(self, split: LatentSplit) -> V30PhaseAMetrics:
        cfg = self.config.phase_a
        reconstruction = []
        relative = []
        conditions = []
        phis = []
        outside = 0
        for _ in range(cfg.validation_episodes):
            current, cumulative = self._sample_transition(split)
            probes = self.benchmark.sample_query(cfg.probe_size)
            update = self.teacher.propose_update(current, cumulative, probes)
            result = project_update(
                self.operator,
                current,
                update,
                probes,
                self.config.operator.projection_ridge,
            )
            reconstruction.append(result.reconstruction_mse)
            relative.append(result.relative_residual)
            conditions.append(result.condition_number)
            phis.append(result.phi)
            outside += int(result.relative_residual > cfg.span_threshold)
        relative_tensor = torch.stack(relative)
        phi = torch.stack(phis)
        return V30PhaseAMetrics(
            split=split,
            reconstruction_mse=float(torch.stack(reconstruction).mean()),
            relative_projection_residual_mean=float(relative_tensor.mean()),
            relative_projection_residual_median=float(relative_tensor.median()),
            relative_projection_residual_q90=float(torch.quantile(relative_tensor, 0.9)),
            condition_number_median=float(torch.stack(conditions).median()),
            phi_mean=float(phi.mean()),
            phi_std=float(phi.std()),
            phi_abs_max=float(phi.abs().max()),
            outside_span_fraction=outside / cfg.validation_episodes,
        )

    def run(self, on_log=None) -> list[dict[str, float]]:
        history = []
        for step in range(1, self.config.phase_a.steps + 1):
            metrics = self.train_step()
            if step == 1 or step % self.config.phase_a.log_every == 0:
                metrics["step"] = float(step)
                history.append(metrics)
                print(json.dumps({"phase": "A_v3_0", **metrics}), flush=True)
                if on_log is not None:
                    on_log(step, history)
        return history


def save_v30_phase_a(
    path: str | Path,
    config: V30Config,
    system: FunctionSystem,
    operator: FunctionalOperator,
    metrics: dict[str, V30PhaseAMetrics],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.to_dict(),
        "system": system.state_dict(),
        "operator": operator.state_dict(),
        "metrics": {key: asdict(value) for key, value in metrics.items()},
    }, path)
