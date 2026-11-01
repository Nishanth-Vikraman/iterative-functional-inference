from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, project_update
from .teachers import BayesianLatentSystemTeacher
from .v30_config import V30Config
from .v30_models import HistoryItem, V30Policy
from .v30_tasks import BayesianEpisode, BayesianLatentBenchmark


@dataclass(frozen=True)
class V30Record:
    seed: int
    split: str
    support_size: int
    method: str
    iteration: int
    query_mse: float
    action_mse: float | None
    functional_teacher_imitation_mse: float | None
    projection_residual: float | None


@dataclass(frozen=True)
class V30Trace:
    functions: tuple[ComposedFunction, ...]
    actions: tuple[Tensor, ...]
    residuals: tuple[Tensor, ...]
    histories: tuple[tuple[HistoryItem, ...], ...]
    z_data: tuple[Tensor, ...]
    action_mse: tuple[float, ...]
    functional_mse: tuple[float, ...]
    projection_residual: tuple[float, ...]


def _mse_values(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).square().mean())


def _shuffle_residual_correspondence(tokens: Tensor) -> Tensor:
    permutation = torch.arange(tokens.shape[0] - 1, -1, -1, device=tokens.device)
    result = tokens.clone()
    result[:, 1:] = result[permutation, 1:]
    return result


class V30Evaluator:
    def __init__(
        self,
        config: V30Config,
        system: FunctionSystem,
        operator: FunctionalOperator,
        policies: dict[str, V30Policy],
        output_dir: str | Path,
    ):
        self.config = config
        self.system = system
        self.operator = operator
        self.policies = policies
        self.output_dir = Path(output_dir)
        self.device = next(operator.parameters()).device

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    @torch.no_grad()
    def _rollout(
        self,
        policy: V30Policy,
        episode: BayesianEpisode,
        benchmark: BayesianLatentBenchmark,
        teacher: BayesianLatentSystemTeacher,
        steps: int,
    ) -> V30Trace:
        current = self.initial_function()
        history: tuple[HistoryItem, ...] = ()
        functions = [current]
        actions = []
        residuals = []
        histories = []
        z_values = []
        action_mses = []
        function_mses = []
        projection_residuals = []
        static_z = None
        for iteration in range(steps):
            batch = episode.batches[iteration]
            raw = policy.raw_residual_tokens(
                batch.x, batch.y, current(batch.x)
            )
            current_z = policy.encode_data(batch.x, batch.y)
            if static_z is None:
                static_z = current_z
            z_data = static_z if policy.variant == "ar_no_feedback" else current_z
            action = policy.predict(raw, z_data, history).detach()
            probes = benchmark.sample_query(self.config.phase_b.probe_size)
            update = teacher.propose_update(
                current, episode.cumulative(iteration), probes
            )
            projection = project_update(
                self.operator,
                current,
                update,
                probes,
                self.config.operator.projection_ridge,
            )
            next_function = current.append(action, detach=True)
            action_mses.append(float((action - projection.phi).square().mean()))
            function_mses.append(float(
                (next_function(probes) - update.teacher(probes)).square().mean()
            ))
            projection_residuals.append(float(projection.relative_residual))
            histories.append(history)
            residuals.append(raw.detach())
            z_values.append(z_data.detach())
            actions.append(action)
            history = history + (HistoryItem(raw.detach(), action),)
            current = next_function
            functions.append(current)
        return V30Trace(
            tuple(functions),
            tuple(actions),
            tuple(residuals),
            tuple(histories),
            tuple(z_values),
            tuple(action_mses),
            tuple(function_mses),
            tuple(projection_residuals),
        )

    @staticmethod
    def _history_controls(
        history: tuple[HistoryItem, ...],
        wrong_history: tuple[HistoryItem, ...],
    ) -> dict[str, tuple[HistoryItem, ...]]:
        if not history:
            return {"correct": history}
        reversed_history = tuple(reversed(history))
        temporal_shuffle = tuple(history[1::2] + history[::2])
        actions = [item.action for item in history]
        rotated_actions = actions[1:] + actions[:1]
        broken = tuple(
            HistoryItem(item.residual, action)
            for item, action in zip(history, rotated_actions)
        )
        return {
            "correct": history,
            "remove_all": (),
            "wrong_task": wrong_history,
            "reversed": reversed_history,
            "shuffled_temporal": temporal_shuffle,
            "break_correspondence": broken,
            "residuals_only": tuple(
                HistoryItem(item.residual, None) for item in history
            ),
            "actions_only": tuple(
                HistoryItem(None, item.action) for item in history
            ),
            "recent_only": (history[-1],),
            "earliest_only": (history[0],),
        }

    @torch.no_grad()
    def _interventions(
        self,
        name: str,
        policy: V30Policy,
        episode: BayesianEpisode,
        wrong_episode: BayesianEpisode,
        trace: V30Trace,
        wrong_trace: V30Trace,
        query_x: Tensor,
        steps: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        target = episode_target(episode, query_x, self._benchmark.library)
        residual_rows = []
        history_rows = []
        for iteration in range(steps):
            current = trace.functions[iteration]
            correct_action = trace.actions[iteration]
            correct_next = trace.functions[iteration + 1]
            correct_error = _mse_values(correct_next(query_x), target)
            raw = trace.residuals[iteration]
            z_data = trace.z_data[iteration]
            history = trace.histories[iteration]
            wrong_batch = wrong_episode.batches[iteration]
            wrong_raw = policy.raw_residual_tokens(
                wrong_batch.x, wrong_batch.y, current(wrong_batch.x)
            )
            controls = {
                "correct": raw,
                "zero": torch.zeros_like(raw),
                "wrong_task": wrong_raw,
                "shuffled": _shuffle_residual_correspondence(raw),
            }
            if iteration > 0:
                controls["stale"] = trace.residuals[iteration - 1]
            for intervention, branch_raw in controls.items():
                action = policy.predict(branch_raw, z_data, history).detach()
                function = current.append(action, detach=True)
                error = _mse_values(function(query_x), target)
                residual_rows.append({
                    "method": name,
                    "iteration": iteration,
                    "intervention": intervention,
                    "phi_change_from_correct": float((action - correct_action).norm()),
                    "function_change_from_correct": float(
                        (function(query_x) - correct_next(query_x)).square().mean()
                    ),
                    "query_mse": error,
                    "query_mse_delta_from_correct": error - correct_error,
                })
            if name not in {"full_history", "query_history"} or iteration == 0:
                continue
            wrong_history = wrong_trace.histories[iteration]
            for intervention, branch_history in self._history_controls(
                history, wrong_history
            ).items():
                action = policy.predict(raw, z_data, branch_history).detach()
                function = current.append(action, detach=True)
                error = _mse_values(function(query_x), target)
                history_rows.append({
                    "method": name,
                    "iteration": iteration,
                    "intervention": intervention,
                    "phi_change_from_correct": float((action - correct_action).norm()),
                    "function_change_from_correct": float(
                        (function(query_x) - correct_next(query_x)).square().mean()
                    ),
                    "query_mse": error,
                    "query_mse_delta_from_correct": error - correct_error,
                })
        return residual_rows, history_rows

    @torch.no_grad()
    def run(self):
        cfg = self.config.evaluation
        records = []
        residual_interventions = []
        history_interventions = []
        total_cells = (
            len(cfg.seeds) * 2 * len(self.config.tasks.support_sizes)
        )
        completed = 0
        for seed in cfg.seeds:
            benchmark = BayesianLatentBenchmark(self.config.tasks, seed, self.device)
            self._benchmark = benchmark
            teacher = BayesianLatentSystemTeacher(
                benchmark.library, self.config.tasks.likelihood_noise
            )
            for split in ("train_latents", "heldout_latents"):
                for support_size in self.config.tasks.support_sizes:
                    for episode_index in range(cfg.episodes_per_cell):
                        episode = benchmark.sample_episode(
                            support_size,
                            split,
                            cfg.extended_rollout_steps,
                        )
                        wrong_episode = benchmark.sample_episode(
                            support_size,
                            split,
                            cfg.extended_rollout_steps,
                        )
                        query_x = benchmark.sample_query(cfg.query_size)
                        target = episode_target(episode, query_x, benchmark.library)
                        initial = self.initial_function()
                        records.append(V30Record(
                            seed, split, support_size, "f0", 0,
                            _mse_values(initial(query_x), target), None, None, None,
                        ))
                        for iteration in range(cfg.extended_rollout_steps):
                            posterior = teacher.posterior(episode.cumulative(iteration))
                            prediction = teacher.predictive(posterior, query_x)
                            records.append(V30Record(
                                seed, split, support_size, "bayesian_teacher", iteration + 1,
                                _mse_values(prediction, target), None, None, None,
                            ))
                        for name, policy in self.policies.items():
                            trace = self._rollout(
                                policy, episode, benchmark, teacher,
                                cfg.extended_rollout_steps,
                            )
                            records.append(V30Record(
                                seed, split, support_size, name, 0,
                                _mse_values(trace.functions[0](query_x), target),
                                None, None, None,
                            ))
                            for iteration in range(1, cfg.extended_rollout_steps + 1):
                                index = iteration - 1
                                records.append(V30Record(
                                    seed, split, support_size, name, iteration,
                                    _mse_values(trace.functions[iteration](query_x), target),
                                    trace.action_mse[index], trace.functional_mse[index],
                                    trace.projection_residual[index],
                                ))
                            if episode_index < max(1, cfg.episodes_per_cell // 8):
                                wrong_trace = self._rollout(
                                    policy, wrong_episode, benchmark, teacher,
                                    min(cfg.rollout_steps, self.config.phase_b.rollout_max),
                                )
                                residual_rows, history_rows = self._interventions(
                                    name,
                                    policy,
                                    episode,
                                    wrong_episode,
                                    trace,
                                    wrong_trace,
                                    query_x,
                                    min(cfg.rollout_steps, self.config.phase_b.rollout_max),
                                )
                                shared = {
                                    "seed": seed,
                                    "split": split,
                                    "support_size": support_size,
                                }
                                residual_interventions.extend([
                                    {**shared, **row} for row in residual_rows
                                ])
                                history_interventions.extend([
                                    {**shared, **row} for row in history_rows
                                ])
                    completed += 1
                    progress = {
                        "completed_cells": completed,
                        "total_cells": total_cells,
                        "seed": seed,
                        "split": split,
                        "support_size": support_size,
                    }
                    (self.output_dir / "evaluation_progress.json").write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
                    print(json.dumps({"phase": "evaluation_v3_0", **progress}), flush=True)
        return records, residual_interventions, history_interventions


def episode_target(episode: BayesianEpisode, x: Tensor, library) -> Tensor:
    return library.function(episode.latent_index, x)


def _write_csv(path: Path, rows) -> None:
    dictionaries = [asdict(row) if hasattr(row, "__dataclass_fields__") else row for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def write_v30_outputs(output_dir, records, residual, history):
    output = Path(output_dir)
    _write_csv(output / "evaluation_records.csv", records)
    _write_csv(output / "residual_interventions.csv", residual)
    _write_csv(output / "history_interventions.csv", history)
    grouped = defaultdict(list)
    for row in records:
        grouped[(row.split, row.support_size, row.method, row.iteration)].append(row)
    summary = []
    for (split, support, method, iteration), values in grouped.items():
        errors = [row.query_mse for row in values]
        actions = [row.action_mse for row in values if row.action_mse is not None]
        summary.append({
            "split": split,
            "support_size": support,
            "method": method,
            "iteration": iteration,
            "query_mse_mean": mean(errors),
            "query_mse_std": pstdev(errors),
            "action_mse_mean": mean(actions) if actions else None,
            "n": len(values),
        })
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def make_v30_plots(output_dir, config: V30Config):
    output = Path(output_dir)
    rows = list(csv.DictReader((output / "evaluation_records.csv").open(encoding="utf-8")))
    methods = [
        "direct_current", "full_history", "query_history",
        "ar_no_feedback", "bayesian_teacher",
    ]
    plt.figure(figsize=(8, 5))
    for method in methods:
        values = []
        for iteration in range(config.evaluation.extended_rollout_steps + 1):
            selected = [
                float(row["query_mse"]) for row in rows
                if row["split"] == "heldout_latents"
                and row["method"] == method
                and int(row["iteration"]) == iteration
            ]
            values.append(mean(selected) if selected else float("nan"))
        plt.plot(range(len(values)), values, marker="o", label=method)
    plt.xlabel("IFI iteration")
    plt.ylabel("held-out latent query MSE")
    plt.legend()
    plt.tight_layout()
    path = output / "error_vs_iteration.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return [path]
