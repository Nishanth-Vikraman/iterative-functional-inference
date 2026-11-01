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

from .config import ExperimentConfig
from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator, project_update
from .tasks import Episode, Split, SyntheticFunctionBenchmark
from .teachers import RKHSFunctionalGradientTeacher
from .v21_models import DirectIFIPolicy


@dataclass(frozen=True)
class DirectTrace:
    functions: tuple[ComposedFunction, ...]
    actions: tuple[Tensor, ...]
    residual_tokens: tuple[Tensor, ...]
    states_before: tuple[Tensor | None, ...]
    states_after: tuple[Tensor | None, ...]


@dataclass(frozen=True)
class V21EvaluationRecord:
    seed: int
    split: str
    support_size: int
    method: str
    iteration: int
    query_mse: float
    action_mse: float | None
    functional_teacher_imitation_mse: float | None
    residual_distance_from_previous: float | None
    teacher_action_change_from_previous: float | None
    recurrent_state_norm: float | None
    recurrent_state_change: float | None


@torch.no_grad()
def direct_adapt(
    policy: DirectIFIPolicy,
    initial_function: ComposedFunction,
    support_x: Tensor,
    support_y: Tensor,
    steps: int,
) -> DirectTrace:
    policy.eval()
    z_data = policy.encode_data(support_x, support_y)
    state = policy.initial_state(z_data)
    history: tuple[Tensor, ...] = ()
    current = initial_function
    functions = [current]
    actions: list[Tensor] = []
    tokens: list[Tensor] = []
    states_before: list[Tensor | None] = []
    states_after: list[Tensor | None] = []
    for _ in range(steps):
        raw = policy.raw_residual_tokens(support_x, support_y, current(support_x))
        result = policy.step(raw, z_data, history, state)
        action = result.action.detach()
        next_state = None if result.next_state is None else result.next_state.detach()
        tokens.append(raw.detach())
        states_before.append(None if state is None else state.detach())
        states_after.append(next_state)
        actions.append(action)
        current = current.append(action, detach=True)
        functions.append(current)
        history = history + (action,)
        state = next_state
    return DirectTrace(
        tuple(functions), tuple(actions), tuple(tokens), tuple(states_before), tuple(states_after)
    )


def _query_mse(function: ComposedFunction, episode: Episode, query_x: Tensor) -> float:
    return float((function(query_x) - episode.target(query_x)).square().mean())


def _shuffle_correspondence(tokens: Tensor) -> Tensor:
    permutation = torch.arange(tokens.shape[0] - 1, -1, -1, device=tokens.device)
    shuffled = tokens.clone()
    # Keep x fixed and reassign y, F(x), residual, |residual|, and residual^2 together.
    shuffled[:, 1:] = shuffled[permutation, 1:]
    return shuffled


class V21Evaluator:
    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        policies: dict[str, DirectIFIPolicy],
        output_dir: str | Path,
    ):
        self.config = config
        self.system = system
        self.operator = operator
        self.policies = {name: policy.eval() for name, policy in policies.items()}
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        self.device = next(iter(policies.values())).controller.position_embedding.device
        self.output_dir = Path(output_dir)

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    @torch.no_grad()
    def _diagnostics(
        self,
        episode: Episode,
        trace: DirectTrace,
        benchmark: SyntheticFunctionBenchmark,
    ) -> tuple[list[float], list[float], list[float | None], list[Tensor]]:
        action_errors: list[float] = []
        function_errors: list[float] = []
        teacher_changes: list[float | None] = []
        targets: list[Tensor] = []
        previous_target: Tensor | None = None
        for current, predicted in zip(trace.functions[:-1], trace.actions):
            probes = benchmark.sample_inputs(self.config.phase_b.probe_size)
            update = self.teacher.propose_update(
                current, (episode.support_x, episode.support_y), probes
            )
            projection = project_update(
                self.operator, current, update, probes, self.config.operator.projection_ridge
            )
            target = projection.phi.detach()
            action_errors.append(float((predicted - target).square().mean()))
            function_errors.append(float(
                (current.append(predicted)(probes) - update.teacher(probes)).square().mean()
            ))
            teacher_changes.append(
                None if previous_target is None else float((target - previous_target).norm())
            )
            targets.append(target)
            previous_target = target
        return action_errors, function_errors, teacher_changes, targets

    @torch.no_grad()
    def _interventions(
        self,
        policy: DirectIFIPolicy,
        episode: Episode,
        wrong_episode: Episode,
        query_x: Tensor,
        steps: int,
    ) -> list[dict[str, float | str | int | None]]:
        z_data = policy.encode_data(episode.support_x, episode.support_y)
        state = policy.initial_state(z_data)
        history: tuple[Tensor, ...] = ()
        current = self.initial_function()
        stale_tokens: Tensor | None = None
        previous_target: Tensor | None = None
        rows: list[dict[str, float | str | int | None]] = []
        for iteration in range(steps):
            correct_tokens = policy.raw_residual_tokens(
                episode.support_x, episode.support_y, current(episode.support_x)
            )
            correct = policy.step(correct_tokens, z_data, history, state)
            correct_next = current.append(correct.action, detach=True)
            correct_error = _query_mse(correct_next, episode, query_x)

            probes = query_x[: self.config.phase_b.probe_size]
            update = self.teacher.propose_update(
                current, (episode.support_x, episode.support_y), probes
            )
            target = project_update(
                self.operator, current, update, probes, self.config.operator.projection_ridge
            ).phi
            teacher_change = (
                None if previous_target is None else float((target - previous_target).norm())
            )
            residual_distance = (
                None
                if stale_tokens is None
                else float((correct_tokens - stale_tokens).square().mean().sqrt())
            )
            wrong_tokens = policy.raw_residual_tokens(
                wrong_episode.support_x,
                wrong_episode.support_y,
                current(wrong_episode.support_x),
            )
            corruptions: dict[str, Tensor] = {
                "correct": correct_tokens,
                "zero": torch.zeros_like(correct_tokens),
                "wrong_task": wrong_tokens,
                "shuffled": _shuffle_correspondence(correct_tokens),
            }
            if stale_tokens is not None:
                corruptions["stale"] = stale_tokens
            for intervention, tokens in corruptions.items():
                branch = policy.step(tokens, z_data, history, state)
                next_function = current.append(branch.action, detach=True)
                query_error = _query_mse(next_function, episode, query_x)
                rows.append({
                    "variant": policy.variant,
                    "iteration": iteration,
                    "intervention": intervention,
                    "phi_change_from_correct": float((branch.action - correct.action).norm()),
                    "function_change_from_correct": float(
                        (next_function(query_x) - correct_next(query_x)).square().mean()
                    ),
                    "query_mse": query_error,
                    "query_mse_delta_from_correct": query_error - correct_error,
                    "residual_distance_from_stale": residual_distance,
                    "teacher_action_change_from_previous": teacher_change,
                    "fixed_state_norm": None if state is None else float(state.norm()),
                })
            stale_tokens = correct_tokens.detach()
            previous_target = target.detach()
            current = correct_next
            history = history + (correct.action.detach(),)
            state = None if correct.next_state is None else correct.next_state.detach()
        return rows

    @torch.no_grad()
    def run(
        self,
    ) -> tuple[
        list[V21EvaluationRecord],
        list[dict[str, float | str | int | None]],
        list[dict[str, float | str | int]],
    ]:
        cfg = self.config.evaluation
        records: list[V21EvaluationRecord] = []
        interventions: list[dict[str, float | str | int | None]] = []
        trajectories: list[dict[str, float | str | int]] = []
        splits: tuple[Split, ...] = (
            "seen_parameters", "heldout_combinations", "heldout_compositions"
        )
        total_cells = len(cfg.seeds) * len(splits) * len(self.config.tasks.support_sizes)
        completed_cells = 0
        for seed in cfg.seeds:
            benchmark = SyntheticFunctionBenchmark(self.config.tasks, seed, self.device)
            for split in splits:
                for support_size in self.config.tasks.support_sizes:
                    for episode_index in range(cfg.episodes_per_cell):
                        episode = benchmark.sample_episode(support_size, split)
                        wrong = benchmark.sample_episode(support_size, split)
                        query_x = benchmark.sample_inputs(cfg.query_size)
                        for name, policy in self.policies.items():
                            trace = direct_adapt(
                                policy,
                                self.initial_function(),
                                episode.support_x,
                                episode.support_y,
                                cfg.rollout_steps,
                            )
                            action_errors, function_errors, teacher_changes, targets = (
                                self._diagnostics(episode, trace, benchmark)
                            )
                            for iteration, function in enumerate(trace.functions):
                                if iteration == 0:
                                    records.append(V21EvaluationRecord(
                                        seed, split, support_size, name, iteration,
                                        _query_mse(function, episode, query_x),
                                        None, None, None, None, 0.0 if policy.uses_recurrent_state else None,
                                        None,
                                    ))
                                    continue
                                index = iteration - 1
                                residual_distance = None
                                if index > 0:
                                    residual_distance = float(
                                        (trace.residual_tokens[index] - trace.residual_tokens[index - 1])
                                        .square().mean().sqrt()
                                    )
                                before = trace.states_before[index]
                                after = trace.states_after[index]
                                state_norm = None if after is None else float(after.norm())
                                state_change = None
                                if after is not None and before is not None:
                                    state_change = float((after - before).norm())
                                records.append(V21EvaluationRecord(
                                    seed, split, support_size, name, iteration,
                                    _query_mse(function, episode, query_x),
                                    action_errors[index], function_errors[index], residual_distance,
                                    teacher_changes[index], state_norm, state_change,
                                ))
                            if episode_index == 0:
                                for iteration, (action, target) in enumerate(
                                    zip(trace.actions, targets)
                                ):
                                    for coordinate in range(action.numel()):
                                        trajectories.append({
                                            "seed": seed, "split": split,
                                            "support_size": support_size, "variant": name,
                                            "iteration": iteration, "coordinate": coordinate,
                                            "predicted_phi": float(action[coordinate]),
                                            "teacher_phi": float(target[coordinate]),
                                        })
                            if episode_index < max(1, cfg.episodes_per_cell // 8):
                                rows = self._interventions(
                                    policy, episode, wrong, query_x,
                                    min(cfg.rollout_steps, self.config.phase_b.rollout_max),
                                )
                                for row in rows:
                                    row.update({
                                        "seed": seed, "split": split,
                                        "support_size": support_size,
                                    })
                                interventions.extend(rows)
                    completed_cells += 1
                    progress = {
                        "completed_cells": completed_cells,
                        "total_cells": total_cells,
                        "seed": seed,
                        "split": split,
                        "support_size": support_size,
                    }
                    (self.output_dir / "evaluation_progress.json").write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
                    print(json.dumps({"phase": "evaluation_v2_1", **progress}), flush=True)
        return records, interventions, trajectories


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_v21(records: list[V21EvaluationRecord]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str, int], list[V21EvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[(record.split, record.support_size, record.method, record.iteration)].append(record)
    summary: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        query = [row.query_mse for row in rows]
        action = [row.action_mse for row in rows if row.action_mse is not None]
        summary.append({
            "split": key[0], "support_size": key[1], "method": key[2], "iteration": key[3],
            "query_mse_mean": mean(query), "query_mse_std": pstdev(query), "n": len(query),
            "action_mse_mean": mean(action) if action else None,
        })
    return summary


def write_v21_outputs(
    output_dir: str | Path,
    records: list[V21EvaluationRecord],
    interventions: list[dict[str, object]],
    trajectories: list[dict[str, object]],
) -> list[dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "evaluation_records.csv", [asdict(record) for record in records])
    _write_csv(output / "feedback_interventions.csv", interventions)
    _write_csv(output / "phi_trajectories.csv", trajectories)
    summary = aggregate_v21(records)
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_v21_plots(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    records = _load_csv(output / "evaluation_records.csv")
    interventions = _load_csv(output / "feedback_interventions.csv")
    paths: list[Path] = []
    heldout = [row for row in records if row["split"] == "heldout_compositions"]

    plt.figure(figsize=(7.2, 4.6))
    for method in ("direct_current", "direct_history", "direct_recurrent"):
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in heldout:
            if row["method"] == method:
                grouped[int(row["iteration"])].append(float(row["query_mse"]))
        iterations = sorted(grouped)
        plt.plot(iterations, [mean(grouped[index]) for index in iterations], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("IFI iteration")
    plt.ylabel("held-out composition query MSE")
    plt.legend()
    plt.tight_layout()
    path = output / "direct_variants_vs_iteration.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    plt.figure(figsize=(7.2, 4.6))
    for variant in ("direct_current", "direct_history", "direct_recurrent"):
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in interventions:
            if row["variant"] == variant and row["intervention"] == "stale":
                grouped[int(row["iteration"])].append(float(row["query_mse_delta_from_correct"]))
        iterations = sorted(grouped)
        plt.plot(iterations, [mean(grouped[index]) for index in iterations], marker="o", label=variant)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("IFI iteration")
    plt.ylabel("stale minus fresh query MSE")
    plt.legend()
    plt.tight_layout()
    path = output / "fresh_vs_stale_by_iteration.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    recurrent = [
        row for row in heldout
        if row["method"] == "direct_recurrent" and row["recurrent_state_norm"]
    ]
    grouped_state: dict[int, list[float]] = defaultdict(list)
    for row in recurrent:
        grouped_state[int(row["iteration"])].append(float(row["recurrent_state_norm"]))
    plt.figure(figsize=(6.4, 4.2))
    iterations = sorted(grouped_state)
    plt.plot(iterations, [mean(grouped_state[index]) for index in iterations], marker="o")
    plt.xlabel("IFI iteration")
    plt.ylabel("recurrent learning-state norm")
    plt.tight_layout()
    path = output / "recurrent_state_norm.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths
