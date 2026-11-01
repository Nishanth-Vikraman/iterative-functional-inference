from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor

from .config import ExperimentConfig
from .functions import ComposedFunction, Functional, FunctionSystem
from .inference import adapt_without_gradients
from .models import IFIPolicy, StaticHypernetwork, residual_features
from .operators import FunctionalOperator, project_update
from .tasks import Episode, Split, SyntheticFunctionBenchmark
from .teachers import RKHSFunctionalGradientTeacher


@dataclass(frozen=True)
class EvaluationRecord:
    seed: int
    split: str
    support_size: int
    method: str
    iteration: int
    query_mse: float
    action_mse: float | None = None
    functional_teacher_imitation_mse: float | None = None


def _mse(function: Functional, target: Functional, x: Tensor) -> float:
    return float((function(x) - target(x)).square().mean())


class Evaluator:
    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        ifi_policy: IFIPolicy,
        ar_policy: IFIPolicy,
        static_policy: StaticHypernetwork,
    ):
        self.config = config
        self.system = system
        self.operator = operator
        self.ifi = ifi_policy.eval()
        self.ar = ar_policy.eval()
        self.static = static_policy.eval()
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        self.device = next(ifi_policy.parameters()).device
        self.trajectory_rows: list[dict[str, float | str | int]] = []

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    @torch.no_grad()
    def _teacher_trace(self, episode: Episode, steps: int, probes: Tensor) -> list[Functional]:
        trace: list[Functional] = [self.initial_function()]
        current: Functional = trace[0]
        for _ in range(steps):
            update = self.teacher.propose_update(
                current, (episode.support_x, episode.support_y), probes
            )
            current = update.updated_function
            trace.append(current)
        return trace

    @torch.no_grad()
    def _static_trace(self, episode: Episode, steps: int) -> list[ComposedFunction]:
        actions = self.static(episode.support_x, episode.support_y)
        current = self.initial_function()
        trace = [current]
        for index in range(steps):
            if index < actions.shape[0]:
                current = current.append(actions[index], detach=True)
            trace.append(current)
        return trace

    @torch.no_grad()
    def _action_diagnostics(
        self, episode: Episode, functions: tuple[ComposedFunction, ...], actions: tuple[Tensor, ...],
        benchmark: SyntheticFunctionBenchmark,
    ) -> tuple[list[float], list[float], list[Tensor]]:
        errors, functional_errors, targets = [], [], []
        for current, predicted in zip(functions[:-1], actions):
            probes = benchmark.sample_inputs(self.config.phase_b.probe_size)
            update = self.teacher.propose_update(
                current, (episode.support_x, episode.support_y), probes
            )
            target = project_update(
                self.operator, current, update, probes, self.config.operator.projection_ridge
            ).phi
            errors.append(float((predicted - target).square().mean()))
            functional_errors.append(float(
                (current.append(predicted)(probes) - update.teacher(probes)).square().mean()
            ))
            targets.append(target)
        return errors, functional_errors, targets

    @torch.no_grad()
    def feedback_interventions(
        self,
        episode: Episode,
        wrong_episode: Episode,
        query_x: Tensor,
        step: int,
    ) -> list[dict[str, float | str | int]]:
        """Branch only z_R while holding F_i, z_D, prior actions, and history fixed."""
        z_data = self.ifi.encode_data(episode.support_x, episode.support_y)
        previous, state = self.ifi.initial(z_data)
        current = self.initial_function()
        stale: Tensor | None = None
        for _ in range(step):
            prediction = current(episode.support_x)
            features = residual_features(episode.support_x, episode.support_y, prediction)
            z_residual = self.ifi.residual_encoder(features)
            action, state = self.ifi.controller(z_data, z_residual, previous, state)
            current = current.append(action, detach=True)
            previous = action
            stale = features

        prediction = current(episode.support_x)
        correct_features = residual_features(episode.support_x, episode.support_y, prediction)
        correct_z = self.ifi.residual_encoder(correct_features)
        correct_action, _ = self.ifi.controller(z_data, correct_z, previous, state)
        correct_next = current.append(correct_action, detach=True)
        correct_error = _mse(correct_next, episode.target, query_x)
        rows: list[dict[str, float | str | int]] = []
        for intervention in ("correct", "zero", "wrong_task", "shuffled", "stale"):
            if intervention == "correct":
                z_residual = correct_z
            elif intervention == "zero":
                z_residual = torch.zeros_like(correct_z)
            elif intervention == "wrong_task":
                wrong_prediction = current(wrong_episode.support_x)
                z_residual = self.ifi.residual_encoder(residual_features(
                    wrong_episode.support_x, wrong_episode.support_y, wrong_prediction
                ))
            elif intervention == "shuffled":
                permutation = torch.arange(
                    episode.support_x.shape[0] - 1, -1, -1, device=self.device
                )
                corrupted = correct_features.clone()
                corrupted[:, 3:] = corrupted[permutation, 3:]
                z_residual = self.ifi.residual_encoder(corrupted)
            else:
                z_residual = correct_z if stale is None else self.ifi.residual_encoder(stale)
            action, _ = self.ifi.controller(z_data, z_residual, previous, state)
            next_function = current.append(action, detach=True)
            rows.append({
                "intervention": intervention,
                "iteration": step,
                "phi_change_from_correct": float((action - correct_action).norm()),
                "function_change_from_correct": float(
                    (next_function(query_x) - correct_next(query_x)).square().mean()
                ),
                "query_mse": _mse(next_function, episode.target, query_x),
                "query_mse_delta_from_correct": _mse(next_function, episode.target, query_x)
                - correct_error,
            })
        return rows

    @torch.no_grad()
    def run(self) -> tuple[list[EvaluationRecord], list[dict[str, float | str | int]]]:
        cfg = self.config.evaluation
        records: list[EvaluationRecord] = []
        interventions: list[dict[str, float | str | int]] = []
        eval_splits: tuple[Split, ...] = (
            "seen_parameters", "heldout_combinations", "heldout_compositions"
        )
        for seed in cfg.seeds:
            benchmark = SyntheticFunctionBenchmark(self.config.tasks, seed, self.device)
            for split in eval_splits:
                for support_size in self.config.tasks.support_sizes:
                    for episode_index in range(cfg.episodes_per_cell):
                        episode = benchmark.sample_episode(support_size, split)
                        wrong = benchmark.sample_episode(support_size, split)
                        query_x = benchmark.sample_inputs(cfg.query_size)
                        initial = self.initial_function()
                        ifi_trace = adapt_without_gradients(
                            self.ifi, initial, episode.support_x, episode.support_y, cfg.rollout_steps
                        )
                        ar_trace = adapt_without_gradients(
                            self.ar, initial, episode.support_x, episode.support_y, cfg.rollout_steps
                        )
                        static_trace = self._static_trace(episode, cfg.rollout_steps)
                        teacher_trace = self._teacher_trace(episode, cfg.rollout_steps, query_x)
                        ifi_action_errors, ifi_function_errors, ifi_targets = self._action_diagnostics(
                            episode, ifi_trace.functions, ifi_trace.actions, benchmark
                        )
                        ar_action_errors, ar_function_errors, _ = self._action_diagnostics(
                            episode, ar_trace.functions, ar_trace.actions, benchmark
                        )
                        if episode_index == 0:
                            for iteration, (predicted, target) in enumerate(
                                zip(ifi_trace.actions, ifi_targets)
                            ):
                                for coordinate in range(predicted.numel()):
                                    self.trajectory_rows.append({
                                        "seed": seed, "split": split,
                                        "support_size": support_size, "iteration": iteration,
                                        "coordinate": coordinate,
                                        "ifi_phi": float(predicted[coordinate]),
                                        "teacher_phi": float(target[coordinate]),
                                    })
                        traces = {
                            "ifi": ifi_trace.functions,
                            "ar_no_feedback": ar_trace.functions,
                            "static": static_trace,
                            "rkhs_teacher": teacher_trace,
                        }
                        for method, trace in traces.items():
                            for iteration, function in enumerate(trace):
                                action_error = None
                                functional_error = None
                                if iteration > 0 and method == "ifi":
                                    action_error = ifi_action_errors[iteration - 1]
                                    functional_error = ifi_function_errors[iteration - 1]
                                elif iteration > 0 and method == "ar_no_feedback":
                                    action_error = ar_action_errors[iteration - 1]
                                    functional_error = ar_function_errors[iteration - 1]
                                records.append(EvaluationRecord(
                                    seed, split, support_size, method, iteration,
                                    _mse(function, episode.target, query_x), action_error,
                                    functional_error,
                                ))
                        records.append(EvaluationRecord(
                            seed, split, support_size, "f0", 0,
                            _mse(initial, episode.target, query_x), None
                        ))
                        if episode_index < max(1, cfg.episodes_per_cell // 8):
                            for step in range(min(cfg.rollout_steps, self.config.phase_b.rollout_max)):
                                row = self.feedback_interventions(episode, wrong, query_x, step)
                                for item in row:
                                    item.update({
                                        "seed": seed, "split": split, "support_size": support_size
                                    })
                                interventions.extend(row)
        return records, interventions


def aggregate_records(records: list[EvaluationRecord]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str, int], list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[(record.split, record.support_size, record.method, record.iteration)].append(record)
    summary = []
    for key, rows in sorted(groups.items()):
        values = [row.query_mse for row in rows]
        action_values = [row.action_mse for row in rows if row.action_mse is not None]
        summary.append({
            "split": key[0], "support_size": key[1], "method": key[2], "iteration": key[3],
            "query_mse_mean": mean(values), "query_mse_std": pstdev(values), "n": len(values),
            "action_mse_mean": mean(action_values) if action_values else None,
        })
    return summary


def write_evaluation(
    output_dir: str | Path,
    records: list[EvaluationRecord],
    interventions: list[dict[str, float | str | int]],
    trajectories: list[dict[str, float | str | int]] | None = None,
) -> list[dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    if interventions:
        with (output / "feedback_interventions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(interventions[0].keys()))
            writer.writeheader()
            writer.writerows(interventions)
    if trajectories:
        with (output / "phi_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectories[0].keys()))
            writer.writeheader()
            writer.writerows(trajectories)
    summary = aggregate_records(records)
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
