from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, project_update
from .v30_config import V30Config
from .v30_models import HistoryItem, V30Policy
from .v31_targets import (
    V31OracleTeacher,
    V31Target,
    episode_seed,
    registry_groups,
    sample_episode,
)
from .v32_config import ARMS, V32Config
from .v32_losses import (
    antithetic_logit_noise,
    normalized_functional_loss,
    posterior_function_perturbations,
    teacher_geometry,
)
from .v32_training import LossCalibration


RECORD_FIELDS = (
    "bucket", "severity", "target_id", "seed", "support_size", "episode", "arm",
    "iteration", "query_mse", "action_norm", "phi_prediction_mse",
    "normalized_functional_error", "functional_error_numerator", "target_update_energy",
    "next_mse_change", "stable",
)
PROJECTION_FIELDS = (
    "bucket", "severity", "target_id", "seed", "support_size", "episode", "arm",
    "iteration", "relative_projection_residual", "condition_number", "phi_star_norm",
    "phi_hat_norm", "phi_prediction_mse",
)
GEOMETRY_FIELDS = (
    "distribution", "severity", "seed", "support_size", "target_id", "episode",
    "iteration", "arm", "rank", "effective_rank", "condition_before",
    "condition_after", "top1_update_energy", "top3_update_energy", "top5_update_energy",
    "eigenvalue_1", "eigenvalue_2", "eigenvalue_3", "eigenvalue_4",
    "eigenvalue_5", "eigenvalue_6", "eigenvalue_7",
)


def _append(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _mse(prediction, target) -> float:
    return float((prediction - target).square().mean())


def _energy_fraction(vector, eigenvectors, count: int) -> float:
    flat = vector.flatten()
    total = float(flat.square().sum())
    if total <= 1e-20 or eigenvectors.shape[1] == 0:
        return 0.0
    selected = eigenvectors[:, : min(count, eigenvectors.shape[1])]
    captured = (selected.transpose(0, 1) @ flat).square().sum()
    return float(captured / total)


class V32Evaluator:
    def __init__(
        self,
        config: V32Config,
        train_config: V30Config,
        system: FunctionSystem,
        operator: FunctionalOperator,
        targets: tuple[V31Target, ...],
        calibration: LossCalibration,
        output_dir: str | Path,
    ):
        self.config = config
        self.train_config = train_config
        self.system = system
        self.operator = operator
        self.targets = targets
        self.groups = registry_groups(targets)
        self.calibration = calibration
        self.output = Path(output_dir)
        self.device = next(operator.parameters()).device
        self.query_x = torch.linspace(
            train_config.tasks.x_min,
            train_config.tasks.x_max,
            256,
            device=self.device,
        ).unsqueeze(-1)
        self.grid = torch.linspace(
            train_config.tasks.x_min,
            train_config.tasks.x_max,
            config.functional_grid_size,
            device=self.device,
        ).unsqueeze(-1)

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    def _geometry(
        self,
        posterior,
        hypotheses: tuple[V31Target, ...],
        generator: torch.Generator,
    ):
        values = torch.stack([target.evaluate(self.grid).squeeze(-1) for target in hypotheses])
        noise = antithetic_logit_noise(
            self.config.covariance_samples,
            len(hypotheses),
            self.config.posterior_perturbation_scale,
            generator,
            self.device,
        )
        perturbations = posterior_function_perturbations(posterior, values, noise)
        return teacher_geometry(
            perturbations,
            self.calibration.covariance_regularization,
            self.config.covariance_rank_tolerance,
        )

    @torch.inference_mode()
    def _rollout(
        self,
        target: V31Target,
        episode,
        oracle: V31OracleTeacher,
        policies: dict[str, V30Policy],
        seed: int,
        support_size: int,
        episode_index: int,
        geometry_budget: bool,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        target_query = target.evaluate(self.query_x)
        current = {arm: self.initial_function() for arm in ARMS}
        histories: dict[str, tuple[HistoryItem, ...]] = {arm: () for arm in ARMS}
        records = []
        projections = []
        geometry_rows = []
        for arm in ARMS:
            records.append({
                "bucket": target.bucket,
                "severity": target.severity,
                "target_id": target.target_id,
                "seed": seed,
                "support_size": support_size,
                "episode": episode_index,
                "arm": arm,
                "iteration": 0,
                "query_mse": _mse(current[arm](self.query_x), target_query),
                "action_norm": "",
                "phi_prediction_mse": "",
                "normalized_functional_error": "",
                "functional_error_numerator": "",
                "target_update_energy": "",
                "next_mse_change": "",
                "stable": True,
            })
        geometry_generator = torch.Generator(device="cpu").manual_seed(
            episode_seed(seed + 3200, support_size, target.target_id, episode_index)
        )
        for iteration in range(self.config.evaluation_rollout):
            batch = episode.batches[iteration]
            cumulative = episode.cumulative(iteration)
            posterior = oracle.posterior(cumulative)
            geometry = self._geometry(
                posterior,
                self.groups[(target.bucket, target.severity)],
                geometry_generator,
            )
            for arm, policy in policies.items():
                before_query = current[arm](self.query_x)
                before_mse = _mse(before_query, target_query)
                prediction = current[arm](batch.x)
                raw = policy.raw_residual_tokens(batch.x, batch.y, prediction)
                z_data = policy.encode_data(batch.x, batch.y)
                visible_history = histories[arm][-self.config.history_capacity :]
                action = policy.predict(raw, z_data, visible_history).detach()
                update = oracle.propose_update(current[arm], cumulative, self.grid)
                projection = project_update(
                    self.operator,
                    current[arm],
                    update,
                    self.grid,
                    self.train_config.operator.projection_ridge,
                )
                current_values = current[arm](self.grid)
                candidate = current[arm].append(action, detach=True)
                predicted_update = candidate(self.grid) - current_values
                target_update = update.teacher(self.grid) - current_values
                _, functional_numerator, target_energy = normalized_functional_loss(
                    predicted_update,
                    target_update,
                    self.calibration.normalized_functional_epsilon,
                )
                normalized_error = functional_numerator / (
                    target_energy + self.calibration.normalized_functional_epsilon
                )
                candidate_query = candidate(self.query_x)
                after_mse = _mse(candidate_query, target_query)
                stable = bool(
                    torch.isfinite(action).all() and torch.isfinite(candidate_query).all()
                )
                phi_mse = float((action - projection.phi).square().mean())
                common = {
                    "bucket": target.bucket,
                    "severity": target.severity,
                    "target_id": target.target_id,
                    "seed": seed,
                    "support_size": support_size,
                    "episode": episode_index,
                    "arm": arm,
                }
                records.append({
                    **common,
                    "iteration": iteration + 1,
                    "query_mse": after_mse if stable else math.nan,
                    "action_norm": float(action.norm()),
                    "phi_prediction_mse": phi_mse,
                    "normalized_functional_error": float(normalized_error),
                    "functional_error_numerator": float(functional_numerator),
                    "target_update_energy": float(target_energy),
                    "next_mse_change": after_mse - before_mse,
                    "stable": stable,
                })
                projections.append({
                    **common,
                    "iteration": iteration,
                    "relative_projection_residual": float(projection.relative_residual),
                    "condition_number": float(projection.condition_number),
                    "phi_star_norm": float(projection.phi.norm()),
                    "phi_hat_norm": float(action.norm()),
                    "phi_prediction_mse": phi_mse,
                })
                if geometry_budget:
                    eigenvalues = geometry.eigenvalues.detach().cpu().tolist()
                    eigenvalues += [0.0] * (7 - len(eigenvalues))
                    geometry_rows.append({
                        "distribution": target.bucket,
                        "severity": target.severity,
                        "seed": seed,
                        "support_size": support_size,
                        "target_id": target.target_id,
                        "episode": episode_index,
                        "iteration": iteration,
                        "arm": arm,
                        "rank": geometry.raw_rank,
                        "effective_rank": geometry.effective_rank,
                        "condition_before": geometry.condition_before,
                        "condition_after": geometry.condition_after,
                        "top1_update_energy": _energy_fraction(
                            target_update, geometry.eigenvectors, 1
                        ),
                        "top3_update_energy": _energy_fraction(
                            target_update, geometry.eigenvectors, 3
                        ),
                        "top5_update_energy": _energy_fraction(
                            target_update, geometry.eigenvectors, 5
                        ),
                        **{
                            f"eigenvalue_{index + 1}": eigenvalues[index]
                            for index in range(7)
                        },
                    })
                current[arm] = candidate
                histories[arm] = histories[arm] + (
                    HistoryItem(raw.detach(), action.detach()),
                )
        return records, projections, geometry_rows

    def run(self, policy_loader, max_cells: int | None = None) -> dict[str, object]:
        records_path = self.output / "evaluation_records.csv"
        projection_path = self.output / "projection_records.csv"
        geometry_path = self.output / "mahalanobis_geometry_records.csv"
        for path in (records_path, projection_path, geometry_path):
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite v3.2 evaluation result: {path}")
        cells = [
            (seed, support_size, target)
            for seed in self.config.training_seeds
            for support_size in self.train_config.tasks.support_sizes
            for target in self.targets
        ]
        if max_cells is not None:
            selected = []
            seen = set()
            for cell in cells:
                group = (cell[2].bucket, cell[2].severity)
                if group not in seen:
                    selected.append(cell)
                    seen.add(group)
                if len(selected) >= max_cells:
                    break
            cells = selected
        start = time.time()
        loaded_seed = None
        policies = None
        geometry_counts = defaultdict(int)
        for completed, (seed, support_size, target) in enumerate(cells, start=1):
            if seed != loaded_seed:
                policies = policy_loader(seed)
                loaded_seed = seed
            assert policies is not None
            oracle = V31OracleTeacher(
                self.groups[(target.bucket, target.severity)],
                self.train_config.tasks.likelihood_noise,
            )
            cell_records = []
            cell_projections = []
            cell_geometry = []
            for episode_index in range(4):
                seed_value = episode_seed(seed, support_size, target.target_id, episode_index)
                episode = sample_episode(
                    target,
                    self.train_config,
                    support_size,
                    self.config.evaluation_rollout,
                    seed_value,
                    self.device,
                )
                group = (target.bucket, target.severity)
                budget = geometry_counts[group] < self.config.diagnostic_geometry_states_per_group
                records, projections, geometry = self._rollout(
                    target,
                    episode,
                    oracle,
                    policies,
                    seed,
                    support_size,
                    episode_index,
                    budget,
                )
                cell_records.extend(records)
                cell_projections.extend(projections)
                cell_geometry.extend(geometry)
                if budget:
                    geometry_counts[group] += self.config.evaluation_rollout
            _append(records_path, RECORD_FIELDS, cell_records)
            _append(projection_path, PROJECTION_FIELDS, cell_projections)
            _append(geometry_path, GEOMETRY_FIELDS, cell_geometry)
            elapsed = time.time() - start
            rate = completed / max(elapsed, 1e-9)
            progress = {
                "phase": "evaluation",
                "completed_cells": completed,
                "total_cells": len(cells),
                "fraction": completed / len(cells),
                "seed": seed,
                "support_size": support_size,
                "target_id": target.target_id,
                "elapsed_seconds": elapsed,
                "eta_seconds": (len(cells) - completed) / rate,
            }
            temporary = self.output / "progress.json.tmp"
            temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            temporary.replace(self.output / "progress.json")
            print(json.dumps({"phase": "evaluation_v3_2", **progress}), flush=True)
        return {"cells": len(cells), "elapsed_seconds": time.time() - start}
