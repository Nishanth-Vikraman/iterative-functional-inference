from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, project_update
from .v30_config import V30Config
from .v30_models import HistoryItem, V30Policy
from .v31_config import V31Config
from .v31_targets import (
    V31Episode,
    V31OracleTeacher,
    V31Target,
    episode_seed,
    registry_groups,
    sample_episode,
)


METHODS = ("query_history", "full_history", "direct_current", "ar_no_feedback")

RECORD_FIELDS = (
    "bucket", "severity", "ood_distance", "factor_count", "target_id", "seed",
    "support_size", "episode", "method", "iteration", "query_mse", "action_mse",
    "functional_teacher_imitation_mse", "numerically_stable",
)
PROJECTION_FIELDS = (
    "bucket", "severity", "ood_distance", "target_id", "seed", "support_size",
    "episode", "method", "iteration", "relative_projection_residual",
    "reconstruction_mse", "target_energy", "condition_number", "phi_norm", "phi_abs_max",
) + tuple(f"phi_{index}" for index in range(12))
EVENT_FIELDS = (
    "bucket", "severity", "target_id", "seed", "support_size", "episode", "method",
    "iteration", "kind", "detail",
)


def _mse(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).square().mean())


def _append_rows(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class V31Evaluator:
    def __init__(
        self,
        config: V31Config,
        train_config: V30Config,
        system: FunctionSystem,
        operator: FunctionalOperator,
        policies: dict[str, V30Policy],
        targets: tuple[V31Target, ...],
        output_dir: str | Path,
    ):
        self.config = config
        self.train_config = train_config
        self.system = system
        self.operator = operator
        self.policies = policies
        self.targets = targets
        self.groups = registry_groups(targets)
        self.output = Path(output_dir)
        self.device = next(operator.parameters()).device
        self.query_x = torch.linspace(
            train_config.tasks.x_min,
            train_config.tasks.x_max,
            config.query_size,
            device=self.device,
        ).unsqueeze(-1)
        self.probes = torch.linspace(
            train_config.tasks.x_min,
            train_config.tasks.x_max,
            config.projection_size,
            device=self.device,
        ).unsqueeze(-1)

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    def _common(self, target, seed, support_size, episode_index, method, iteration):
        return {
            "bucket": target.bucket,
            "severity": target.severity,
            "ood_distance": target.ood_distance,
            "factor_count": target.factor_count,
            "target_id": target.target_id,
            "seed": seed,
            "support_size": support_size,
            "episode": episode_index,
            "method": method,
            "iteration": iteration,
        }

    @torch.inference_mode()
    def _oracle_records(
        self,
        episode: V31Episode,
        oracle: V31OracleTeacher,
        seed: int,
        support_size: int,
        episode_index: int,
    ) -> list[dict]:
        target_y = episode.target.evaluate(self.query_x)
        initial = self.initial_function()
        rows = [{
            **self._common(episode.target, seed, support_size, episode_index, "bayesian_oracle", 0),
            "query_mse": _mse(initial(self.query_x), target_y),
            "action_mse": "",
            "functional_teacher_imitation_mse": "",
            "numerically_stable": True,
        }]
        for iteration in range(self.config.rollout_steps):
            posterior = oracle.posterior(episode.cumulative(iteration))
            prediction = oracle.predictive(posterior, self.query_x)
            rows.append({
                **self._common(
                    episode.target, seed, support_size, episode_index,
                    "bayesian_oracle", iteration + 1,
                ),
                "query_mse": _mse(prediction, target_y),
                "action_mse": "",
                "functional_teacher_imitation_mse": "",
                "numerically_stable": bool(torch.isfinite(prediction).all()),
            })
        return rows

    @torch.inference_mode()
    def _policy_records(
        self,
        episode: V31Episode,
        oracle: V31OracleTeacher,
        method: str,
        policy: V30Policy,
        seed: int,
        support_size: int,
        episode_index: int,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        target_y = episode.target.evaluate(self.query_x)
        current = self.initial_function()
        history: tuple[HistoryItem, ...] = ()
        static_z = None
        records = [{
            **self._common(episode.target, seed, support_size, episode_index, method, 0),
            "query_mse": _mse(current(self.query_x), target_y),
            "action_mse": "",
            "functional_teacher_imitation_mse": "",
            "numerically_stable": True,
        }]
        projections: list[dict] = []
        events: list[dict] = []
        for iteration in range(self.config.rollout_steps):
            batch = episode.batches[iteration]
            prediction = current(batch.x)
            raw = policy.raw_residual_tokens(batch.x, batch.y, prediction)
            current_z = policy.encode_data(batch.x, batch.y)
            if static_z is None:
                static_z = current_z
            z_data = static_z if method == "ar_no_feedback" else current_z
            visible_history = history[-self.config.history_capacity :]
            try:
                action = policy.predict(raw, z_data, visible_history).detach()
                update = oracle.propose_update(current, episode.cumulative(iteration), self.probes)
                projection = project_update(
                    self.operator,
                    current,
                    update,
                    self.probes,
                    self.train_config.operator.projection_ridge,
                )
                candidate = current.append(action, detach=True)
                candidate_query = candidate(self.query_x)
                stable = bool(
                    torch.isfinite(action).all()
                    and torch.isfinite(candidate_query).all()
                    and torch.isfinite(projection.relative_residual)
                )
                action_mse = float((action - projection.phi).square().mean())
                functional_mse = _mse(candidate(self.probes), update.teacher(self.probes))
                query_mse = _mse(candidate_query, target_y) if stable else math.nan
                phi_values = projection.phi.detach().cpu().tolist()
                projections.append({
                    **{key: value for key, value in self._common(
                        episode.target, seed, support_size, episode_index, method, iteration,
                    ).items() if key != "factor_count"},
                    "relative_projection_residual": float(projection.relative_residual),
                    "reconstruction_mse": float(projection.reconstruction_mse),
                    "target_energy": float(projection.target_energy),
                    "condition_number": float(projection.condition_number),
                    "phi_norm": float(projection.phi.norm()),
                    "phi_abs_max": float(projection.phi.abs().max()),
                    **{f"phi_{index}": value for index, value in enumerate(phi_values)},
                })
                records.append({
                    **self._common(
                        episode.target, seed, support_size, episode_index, method, iteration + 1,
                    ),
                    "query_mse": query_mse,
                    "action_mse": action_mse,
                    "functional_teacher_imitation_mse": functional_mse,
                    "numerically_stable": stable,
                })
                if not stable:
                    events.append({
                        **{key: value for key, value in self._common(
                            episode.target, seed, support_size, episode_index, method, iteration,
                        ).items() if key not in {"ood_distance", "factor_count"}},
                        "kind": "nonfinite",
                        "detail": "action, candidate function, or projection became nonfinite",
                    })
                    break
                current = candidate
                history = history + (HistoryItem(raw.detach(), action.detach()),)
            except (RuntimeError, ValueError) as error:
                events.append({
                    **{key: value for key, value in self._common(
                        episode.target, seed, support_size, episode_index, method, iteration,
                    ).items() if key not in {"ood_distance", "factor_count"}},
                    "kind": "runtime_error",
                    "detail": str(error).replace("\n", " ")[:500],
                })
                break
        return records, projections, events

    def run(self, max_cells: int | None = None) -> dict[str, object]:
        records_path = self.output / "evaluation_records.csv"
        projection_path = self.output / "projection_records.csv"
        events_path = self.output / "numerical_events.csv"
        for path in (records_path, projection_path, events_path):
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite existing v3.1 result: {path}")
        cells = [
            (seed, support_size, target)
            for seed in self.config.seeds
            for support_size in self.config.support_sizes
            for target in self.targets
        ]
        if max_cells is not None:
            # Engineering smoke mode only; canonical config never supplies max_cells.
            selected = []
            seen = set()
            for cell in cells:
                key = (cell[2].bucket, cell[2].severity)
                if key not in seen:
                    selected.append(cell)
                    seen.add(key)
                if len(selected) >= max_cells:
                    break
            cells = selected
        start = time.time()
        for completed, (seed, support_size, target) in enumerate(cells, start=1):
            oracle = V31OracleTeacher(
                self.groups[(target.bucket, target.severity)],
                self.train_config.tasks.likelihood_noise,
            )
            cell_records: list[dict] = []
            cell_projections: list[dict] = []
            cell_events: list[dict] = []
            for episode_index in range(self.config.episodes_per_target):
                sample_seed = episode_seed(seed, support_size, target.target_id, episode_index)
                episode = sample_episode(
                    target,
                    self.train_config,
                    support_size,
                    self.config.rollout_steps,
                    sample_seed,
                    self.device,
                )
                cell_records.extend(self._oracle_records(
                    episode, oracle, seed, support_size, episode_index,
                ))
                for method in METHODS:
                    records, projections, events = self._policy_records(
                        episode,
                        oracle,
                        method,
                        self.policies[method],
                        seed,
                        support_size,
                        episode_index,
                    )
                    cell_records.extend(records)
                    cell_projections.extend(projections)
                    cell_events.extend(events)
            _append_rows(records_path, RECORD_FIELDS, cell_records)
            _append_rows(projection_path, PROJECTION_FIELDS, cell_projections)
            _append_rows(events_path, EVENT_FIELDS, cell_events)
            elapsed = time.time() - start
            rate = completed / max(elapsed, 1e-9)
            progress = {
                "completed_cells": completed,
                "total_cells": len(cells),
                "fraction": completed / len(cells),
                "seed": seed,
                "support_size": support_size,
                "target_id": target.target_id,
                "bucket": target.bucket,
                "severity": target.severity,
                "elapsed_seconds": elapsed,
                "cells_per_second": rate,
                "eta_seconds": (len(cells) - completed) / rate,
                "canonical": max_cells is None,
            }
            temporary = self.output / "evaluation_progress.json.tmp"
            temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            temporary.replace(self.output / "evaluation_progress.json")
            print(json.dumps({"phase": "evaluation_v3_1", **progress}), flush=True)
        return {
            "cells": len(cells),
            "elapsed_seconds": time.time() - start,
            "records_path": str(records_path),
            "projection_path": str(projection_path),
            "events_path": str(events_path),
        }
