from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import ExperimentConfig
from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, action_design, project_update
from .tasks import SyntheticFunctionBenchmark
from .teachers import RKHSFunctionalGradientTeacher


@dataclass
class PhaseAMetrics:
    reconstruction_mse: float
    relative_projection_residual: float
    condition_number: float
    phi_mean: float
    phi_std: float
    phi_abs_max: float
    outside_span_fraction: float


def _conditioning_loss(design: Tensor) -> Tensor:
    gram = design.transpose(0, 1) @ design / max(1, design.shape[0])
    diagonal = torch.diagonal(gram).clamp_min(1e-6)
    normalized = gram / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    identity = torch.eye(gram.shape[0], device=gram.device)
    scale_loss = torch.log(diagonal).square().mean()
    return (normalized - identity).square().mean() + 0.05 * scale_loss


class PhaseATrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: SyntheticFunctionBenchmark,
    ):
        self.config = config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        self.optimizer = torch.optim.Adam(operator.parameters(), lr=config.phase_a.learning_rate)
        self.device = next(operator.parameters()).device

    def _sample_current(self) -> ComposedFunction:
        # A separate randomly sampled hypothesis prevents Phase A from seeing only F0.
        candidate = self.benchmark.sample_target("train")
        scale = 0.15 + 0.85 * torch.rand((), device="cpu").item()
        return ComposedFunction(
            self.system,
            self.operator,
            initial_function=lambda x, fn=candidate, s=scale: s * fn(x),
        )

    def train_step(self) -> dict[str, float]:
        cfg = self.config.phase_a
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        reconstruction: list[Tensor] = []
        conditioning: list[Tensor] = []
        for _ in range(cfg.batch_size):
            episode = self.benchmark.sample_episode(self.benchmark.sample_support_size(), "train")
            current = self._sample_current()
            probes = self.benchmark.sample_inputs(cfg.probe_size)
            update = self.teacher.propose_update(
                current, (episode.support_x, episode.support_y), probes
            )
            # Alternating dictionary learning: infer coordinates, then update shared bases.
            with torch.no_grad():
                phi = project_update(
                    self.operator, current, update, probes, self.config.operator.projection_ridge
                ).phi
            design = action_design(self.operator, current, probes).reshape(
                -1, self.operator.action_dimension
            )
            target = update.displacement(probes).reshape(-1).detach()
            prediction = design @ phi
            recon = (prediction - target).square().mean()
            cond = _conditioning_loss(design)
            loss = recon + cfg.conditioning_weight * cond + cfg.phi_weight * phi.square().mean()
            losses.append(loss)
            reconstruction.append(recon.detach())
            conditioning.append(cond.detach())
        total = torch.stack(losses).mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.operator.parameters(), 5.0)
        self.optimizer.step()
        return {
            "loss": float(total.detach()),
            "reconstruction": float(torch.stack(reconstruction).mean()),
            "conditioning_loss": float(torch.stack(conditioning).mean()),
        }

    @torch.no_grad()
    def evaluate(self, episodes: int | None = None, split: str = "seen_parameters") -> PhaseAMetrics:
        cfg = self.config.phase_a
        count = episodes or cfg.validation_episodes
        recon, relative, conditions, phis = [], [], [], []
        outside = 0
        for _ in range(count):
            episode = self.benchmark.sample_episode(self.benchmark.sample_support_size(), split)  # type: ignore[arg-type]
            current = self._sample_current()
            probes = self.benchmark.sample_inputs(cfg.probe_size)
            update = self.teacher.propose_update(current, (episode.support_x, episode.support_y), probes)
            result = project_update(
                self.operator, current, update, probes, self.config.operator.projection_ridge
            )
            recon.append(result.reconstruction_mse)
            relative.append(result.relative_residual)
            conditions.append(result.condition_number)
            phis.append(result.phi)
            outside += int(result.relative_residual.item() > cfg.span_threshold)
        phi = torch.stack(phis)
        return PhaseAMetrics(
            reconstruction_mse=float(torch.stack(recon).mean()),
            relative_projection_residual=float(torch.stack(relative).mean()),
            condition_number=float(torch.stack(conditions).median()),
            phi_mean=float(phi.mean()),
            phi_std=float(phi.std()),
            phi_abs_max=float(phi.abs().max()),
            outside_span_fraction=outside / count,
        )

    def run(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for step in range(1, self.config.phase_a.steps + 1):
            metrics = self.train_step()
            if step == 1 or step % self.config.phase_a.log_every == 0:
                metrics["step"] = float(step)
                history.append(metrics)
                print(json.dumps({"phase": "A", **metrics}))
        return history


def save_phase_a(
    path: str | Path,
    config: ExperimentConfig,
    system: FunctionSystem,
    operator: FunctionalOperator,
    metrics: PhaseAMetrics,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.to_dict(),
        "system": system.state_dict(),
        "operator": operator.state_dict(),
        "metrics": asdict(metrics),
    }, path)


def write_phase_a_report(path: str | Path, metrics: PhaseAMetrics) -> None:
    Path(path).write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
