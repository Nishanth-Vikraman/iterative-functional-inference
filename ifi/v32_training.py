from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import numpy as np
from torch import Tensor, nn

from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, project_update
from .teachers import BayesianLatentSystemTeacher
from .v30_config import V30Config
from .v30_models import HistoryItem, V30Policy
from .v30_tasks import BayesianLatentBenchmark
from .v32_config import ARMS, V32Config
from .v32_losses import (
    antithetic_logit_noise,
    functional_update,
    normalized_functional_loss,
    normalized_mahalanobis_loss,
    posterior_function_perturbations,
    teacher_geometry,
)


@dataclass(frozen=True)
class LossCalibration:
    normalized_functional_epsilon: float
    normalized_mahalanobis_epsilon: float
    covariance_regularization: float
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_functional_epsilon": self.normalized_functional_epsilon,
            "normalized_mahalanobis_epsilon": self.normalized_mahalanobis_epsilon,
            "covariance_regularization": self.covariance_regularization,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LossCalibration":
        return cls(
            float(payload["normalized_functional_epsilon"]),
            float(payload["normalized_mahalanobis_epsilon"]),
            float(payload["covariance_regularization"]),
            dict(payload["diagnostics"]),
        )


def _quantile(values: list[float], probability: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, probability))


@torch.no_grad()
def calibrate_losses(
    config: V32Config,
    train_config: V30Config,
    benchmark: BayesianLatentBenchmark,
    grid: Tensor,
    seed: int,
) -> LossCalibration:
    """Lock epsilon/lambda using train-latent teacher trajectories only."""
    rng = random.Random(seed)
    noise_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    hypothesis_values = benchmark.library.evaluate(grid).squeeze(-1)
    energies: list[float] = []
    state_payloads: list[tuple[Tensor, Tensor]] = []
    eigenvalues: list[float] = []
    induced_rms_ratios: list[float] = []
    ranks: list[int] = []
    leading_spectra: list[list[float]] = []
    for _ in range(config.calibration_episodes):
        support_size = rng.choice(tuple(train_config.tasks.support_sizes))
        episode = benchmark.sample_episode(
            support_size, "train_latents", train_config.phase_b.rollout_max
        )
        previous = torch.zeros_like(grid)
        for iteration in range(train_config.phase_b.rollout_max):
            posterior = benchmark.posterior(episode.cumulative(iteration))
            target = benchmark.posterior_predictive(posterior, grid)
            desired = target - previous
            energy = float(desired.square().mean())
            energies.append(energy)
            noise = antithetic_logit_noise(
                config.covariance_samples,
                benchmark.library.size,
                config.posterior_perturbation_scale,
                noise_generator,
                grid.device,
            )
            perturbations = posterior_function_perturbations(
                posterior, hypothesis_values, noise
            )
            provisional = teacher_geometry(
                perturbations,
                1.0,
                config.covariance_rank_tolerance,
            )
            eigenvalues.extend(provisional.eigenvalues.cpu().tolist())
            spectrum = provisional.eigenvalues.cpu().tolist()
            leading_spectra.append(spectrum + [0.0] * (7 - len(spectrum)))
            ranks.append(provisional.raw_rank)
            induced = float(perturbations.square().mean().sqrt())
            desired_rms = float(desired.square().mean().sqrt())
            induced_rms_ratios.append(induced / max(desired_rms, 1e-12))
            state_payloads.append((perturbations.cpu(), desired.flatten().cpu()))
            previous = target
    positive_energies = [value for value in energies if value > 0]
    positive_eigenvalues = [value for value in eigenvalues if value > 0]
    regularization = max(
        config.minimum_epsilon,
        config.covariance_lambda_multiplier
        * _quantile(positive_eigenvalues, config.covariance_lambda_quantile),
    )
    functional_epsilon = max(
        config.minimum_epsilon,
        config.epsilon_multiplier * _quantile(positive_energies, config.epsilon_quantile),
    )
    mahalanobis_denominators = []
    effective_ranks = []
    condition_after = []
    for perturbations_cpu, desired_cpu in state_payloads:
        geometry = teacher_geometry(
            perturbations_cpu.to(grid.device),
            regularization,
            config.covariance_rank_tolerance,
        )
        desired = desired_cpu.to(grid.device)
        mahalanobis_denominators.append(
            float(geometry.quadratic(desired) / desired.numel())
        )
        effective_ranks.append(geometry.effective_rank)
        condition_after.append(geometry.condition_after)
    positive_maha = [value for value in mahalanobis_denominators if value > 0]
    mahalanobis_epsilon = max(
        config.minimum_epsilon,
        config.epsilon_multiplier * _quantile(positive_maha, config.epsilon_quantile),
    )
    diagnostics = {
        "source_split": "train_latents",
        "states": len(energies),
        "posterior_state": "32 finite-hypothesis log-posterior coordinates",
        "perturbation": "antithetic iid Gaussian logit noise followed by softmax",
        "posterior_perturbation_scale": config.posterior_perturbation_scale,
        "covariance_samples": config.covariance_samples,
        "teacher_update_energy": {
            "q01": _quantile(energies, 0.01),
            "median": _quantile(energies, 0.5),
            "q99": _quantile(energies, 0.99),
        },
        "positive_covariance_eigenvalue": {
            "q10": _quantile(positive_eigenvalues, 0.1),
            "median": _quantile(positive_eigenvalues, 0.5),
            "q90": _quantile(positive_eigenvalues, 0.9),
        },
        "rank": {"median": _quantile(ranks, 0.5), "max": max(ranks)},
        "effective_rank_median": _quantile(effective_ranks, 0.5),
        "regularized_condition_median": _quantile(condition_after, 0.5),
        "leading_eigenvalue_median": [
            _quantile([spectrum[index] for spectrum in leading_spectra], 0.5)
            for index in range(7)
        ],
        "induced_perturbation_rms_over_update_rms": {
            "median": _quantile(induced_rms_ratios, 0.5),
            "q90": _quantile(induced_rms_ratios, 0.9),
        },
        "selection_rules": {
            "lambda": (
                f"{config.covariance_lambda_multiplier:g} * q"
                f"{100 * config.covariance_lambda_quantile:g} positive training "
                "covariance eigenvalue"
            ),
            "epsilon_nf": (
                f"{config.epsilon_multiplier:g} * q"
                f"{100 * config.epsilon_quantile:g} positive training "
                "teacher-update energy"
            ),
            "epsilon_nm": (
                f"{config.epsilon_multiplier:g} * q"
                f"{100 * config.epsilon_quantile:g} positive training "
                "normalized-Mahalanobis target energy"
            ),
            "ood_used": False,
        },
    }
    return LossCalibration(
        functional_epsilon,
        mahalanobis_epsilon,
        regularization,
        diagnostics,
    )


def _gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


class V32Trainer:
    def __init__(
        self,
        config: V32Config,
        train_config: V30Config,
        system: FunctionSystem,
        operator: FunctionalOperator,
        benchmark: BayesianLatentBenchmark,
        policies: dict[str, V30Policy],
        calibration: LossCalibration,
        seed: int,
    ):
        if not operator.frozen:
            raise ValueError("v3.2 requires frozen Phase-A O_phi")
        self.config = config
        self.train_config = train_config
        self.system = system
        self.operator = operator
        self.benchmark = benchmark
        self.policies = policies
        self.calibration = calibration
        self.teacher = BayesianLatentSystemTeacher(
            benchmark.library, train_config.tasks.likelihood_noise
        )
        self.optimizers = {
            arm: torch.optim.Adam(policy.parameters(), lr=train_config.phase_b.learning_rate)
            for arm, policy in policies.items()
        }
        self.grid = torch.linspace(
            train_config.tasks.x_min,
            train_config.tasks.x_max,
            config.functional_grid_size,
            device=next(operator.parameters()).device,
        ).unsqueeze(-1)
        self.hypothesis_values = benchmark.library.evaluate(self.grid).squeeze(-1)
        self.geometry_generator = torch.Generator(device="cpu").manual_seed(seed + 900)
        self.episode_rng = random.Random(seed + 901)

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    def _geometry(self, posterior: Tensor):
        noise = antithetic_logit_noise(
            self.config.covariance_samples,
            self.benchmark.library.size,
            self.config.posterior_perturbation_scale,
            self.geometry_generator,
            self.grid.device,
        )
        perturbations = posterior_function_perturbations(
            posterior, self.hypothesis_values, noise
        )
        return teacher_geometry(
            perturbations,
            self.calibration.covariance_regularization,
            self.config.covariance_rank_tolerance,
        )

    def train_step(self) -> dict[str, float]:
        phase = self.train_config.phase_b
        episodes = [
            self.benchmark.sample_episode(
                self.episode_rng.choice(tuple(self.train_config.tasks.support_sizes)),
                "train_latents",
                self.train_config.tasks.train_sequence_length,
            )
            for _ in range(phase.batch_size)
        ]
        horizons = [
            self.episode_rng.randint(1, phase.rollout_max)
            if phase.variable_rollout else phase.rollout_max
            for _ in episodes
        ]
        losses = {arm: [] for arm in ARMS}
        numerators = {arm: [] for arm in ARMS}
        denominators = {arm: [] for arm in ARMS}
        phi_errors = {arm: [] for arm in ARMS}
        functional_errors = {arm: [] for arm in ARMS}
        action_norms = {arm: [] for arm in ARMS}
        geometry_ranks = []
        geometry_effective_ranks = []
        geometry_conditions = []
        for episode, horizon in zip(episodes, horizons):
            current = {arm: self.initial_function() for arm in ARMS}
            histories: dict[str, tuple[HistoryItem, ...]] = {arm: () for arm in ARMS}
            for iteration in range(horizon):
                batch = episode.batches[iteration]
                cumulative = episode.cumulative(iteration)
                posterior = self.teacher.posterior(cumulative).detach()
                target_values = self.teacher.predictive(posterior, self.grid).detach()
                geometry = self._geometry(posterior)
                geometry_ranks.append(float(geometry.raw_rank))
                geometry_effective_ranks.append(geometry.effective_rank)
                geometry_conditions.append(geometry.condition_after)
                for arm, policy in self.policies.items():
                    prediction = current[arm](batch.x)
                    raw = policy.raw_residual_tokens(batch.x, batch.y, prediction)
                    z_data = policy.encode_data(batch.x, batch.y)
                    action = policy.predict(raw, z_data, histories[arm])
                    current_values = current[arm](self.grid).detach()
                    candidate = current[arm].append(action)
                    predicted_update = functional_update(
                        candidate(self.grid), current_values
                    )
                    target_update = target_values - current_values
                    update = self.teacher.propose_update(
                        current[arm], cumulative, self.grid
                    )
                    projection = project_update(
                        self.operator,
                        current[arm],
                        update,
                        self.grid,
                        self.train_config.operator.projection_ridge,
                    )
                    phi_error = (action - projection.phi.detach()).square().mean()
                    functional_error = (
                        predicted_update - target_update
                    ).square().mean()
                    if arm == "phi_mse":
                        loss = phi_error
                        numerator = phi_error
                        denominator = torch.ones_like(phi_error)
                    elif arm == "normalized_functional":
                        loss, numerator, denominator = normalized_functional_loss(
                            predicted_update,
                            target_update,
                            self.calibration.normalized_functional_epsilon,
                        )
                    else:
                        loss, numerator, denominator = normalized_mahalanobis_loss(
                            predicted_update,
                            target_update,
                            geometry,
                            self.calibration.normalized_mahalanobis_epsilon,
                        )
                    losses[arm].append(loss)
                    numerators[arm].append(numerator.detach())
                    denominators[arm].append(denominator.detach())
                    phi_errors[arm].append(phi_error.detach())
                    functional_errors[arm].append(functional_error.detach())
                    action_norms[arm].append(action.detach().norm())
                    current[arm] = candidate.detached()
                    histories[arm] = histories[arm] + (
                        HistoryItem(raw.detach(), action.detach()),
                    )
        metrics: dict[str, float] = {}
        for arm in ARMS:
            parameters = list(self.policies[arm].parameters())
            loss = torch.stack(losses[arm]).mean()
            self.optimizers[arm].zero_grad(set_to_none=True)
            loss.backward()
            metrics[f"{arm}_gradient_norm"] = _gradient_norm(parameters)
            global_norm = float(nn.utils.clip_grad_norm_(parameters, phase.gradient_clip))
            metrics[f"{arm}_gradient_clipped"] = float(global_norm > phase.gradient_clip)
            self.optimizers[arm].step()
            metrics[f"{arm}_loss"] = float(loss.detach())
            metrics[f"{arm}_numerator"] = float(torch.stack(numerators[arm]).mean())
            metrics[f"{arm}_denominator"] = float(torch.stack(denominators[arm]).mean())
            metrics[f"{arm}_phi_mse"] = float(torch.stack(phi_errors[arm]).mean())
            metrics[f"{arm}_functional_mse"] = float(
                torch.stack(functional_errors[arm]).mean()
            )
            metrics[f"{arm}_action_norm"] = float(torch.stack(action_norms[arm]).mean())
        metrics["geometry_rank"] = sum(geometry_ranks) / len(geometry_ranks)
        metrics["geometry_effective_rank"] = (
            sum(geometry_effective_ranks) / len(geometry_effective_ranks)
        )
        metrics["geometry_condition_after"] = (
            sum(geometry_conditions) / len(geometry_conditions)
        )
        metrics["operator_gradient"] = float(any(
            parameter.grad is not None for parameter in self.operator.parameters()
        ))
        if metrics["operator_gradient"]:
            raise RuntimeError("Frozen v3.2 operator received a gradient")
        return metrics

    def run(
        self,
        steps: int,
        on_log: Callable[[int, list[dict[str, float]]], None] | None = None,
        start_step: int = 0,
        history: list[dict[str, float]] | None = None,
    ) -> list[dict[str, float]]:
        history = [] if history is None else list(history)
        elapsed_offset = 0.0 if not history else float(history[-1]["elapsed_seconds"])
        start = time.time()
        for step in range(start_step + 1, steps + 1):
            metrics = self.train_step()
            if step == 1 or step % self.train_config.phase_b.log_every == 0:
                metrics["step"] = float(step)
                metrics["elapsed_seconds"] = elapsed_offset + time.time() - start
                history.append(metrics)
                print(json.dumps({"phase": "training_v3_2", **metrics}), flush=True)
                if on_log is not None:
                    on_log(step, history)
        return history


def atomic_training_checkpoint(
    path: Path,
    step: int,
    policies: dict[str, V30Policy],
    trainer: V32Trainer,
    history: list[dict[str, float]],
    seed: int,
    complete: bool,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": step,
        "seed": seed,
        "complete": complete,
        "arms": {arm: policy.state_dict() for arm, policy in policies.items()},
        "optimizers": {
            arm: optimizer.state_dict() for arm, optimizer in trainer.optimizers.items()
        },
        "history": history,
        "calibration": trainer.calibration.to_dict(),
        "rng": {
            "episode_rng": trainer.episode_rng.getstate(),
            "geometry_generator": trainer.geometry_generator.get_state(),
            "benchmark_generator": trainer.benchmark.generator.get_state(),
            "benchmark_next_episode": trainer.benchmark._next_episode,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }, temporary)
    temporary.replace(path)
