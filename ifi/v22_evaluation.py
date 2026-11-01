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
from .v22_models import PredictiveRecurrentPolicy


@dataclass(frozen=True)
class PredictiveTrace:
    functions: tuple[ComposedFunction, ...]
    actions: tuple[Tensor, ...]
    residual_tokens: tuple[Tensor, ...]
    states: tuple[Tensor, ...]


@dataclass(frozen=True)
class V22Record:
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
    recurrent_state_norm: float
    recurrent_state_change: float | None


@torch.no_grad()
def predictive_adapt(
    policy: PredictiveRecurrentPolicy,
    initial_function: ComposedFunction,
    support_x: Tensor,
    support_y: Tensor,
    steps: int,
) -> PredictiveTrace:
    policy.eval()
    z_data = policy.encode_data(support_x, support_y)
    state = policy.initial_state(z_data)
    history: tuple[Tensor, ...] = ()
    current = initial_function
    functions = [current]
    actions: list[Tensor] = []
    tokens: list[Tensor] = []
    states = [state]
    for _ in range(steps):
        raw = policy.raw_residual_tokens(support_x, support_y, current(support_x))
        action = policy.predict_action(raw, z_data, state, history).detach()
        next_state = policy.update_state(state, raw, action).detach()
        current = current.append(action, detach=True)
        functions.append(current)
        actions.append(action)
        tokens.append(raw.detach())
        states.append(next_state)
        history = history + (action,)
        state = next_state
    return PredictiveTrace(tuple(functions), tuple(actions), tuple(tokens), tuple(states))


def _mse(function: ComposedFunction, episode: Episode, query_x: Tensor) -> float:
    return float((function(query_x) - episode.target(query_x)).square().mean())


def _shuffle(tokens: Tensor) -> Tensor:
    permutation = torch.arange(tokens.shape[0] - 1, -1, -1, device=tokens.device)
    result = tokens.clone()
    result[:, 1:] = result[permutation, 1:]
    return result


class V22Evaluator:
    def __init__(
        self,
        config: ExperimentConfig,
        system: FunctionSystem,
        operator: FunctionalOperator,
        policy: PredictiveRecurrentPolicy,
        output_dir: str | Path,
    ):
        self.config = config
        self.system = system
        self.operator = operator
        self.policy = policy.eval()
        self.output_dir = Path(output_dir)
        self.teacher = RKHSFunctionalGradientTeacher(config.teacher)
        self.device = next(policy.parameters()).device

    def initial_function(self) -> ComposedFunction:
        return ComposedFunction(self.system, self.operator)

    @torch.no_grad()
    def _diagnostics(
        self,
        episode: Episode,
        trace: PredictiveTrace,
        benchmark: SyntheticFunctionBenchmark,
    ) -> tuple[list[float], list[float], list[float | None], list[Tensor]]:
        action_errors: list[float] = []
        function_errors: list[float] = []
        teacher_changes: list[float | None] = []
        targets: list[Tensor] = []
        previous: Tensor | None = None
        for current, action in zip(trace.functions[:-1], trace.actions):
            probes = benchmark.sample_inputs(self.config.phase_b.probe_size)
            update = self.teacher.propose_update(
                current, (episode.support_x, episode.support_y), probes
            )
            target = project_update(
                self.operator, current, update, probes,
                self.config.operator.projection_ridge,
            ).phi.detach()
            action_errors.append(float((action - target).square().mean()))
            function_errors.append(float(
                (current.append(action)(probes) - update.teacher(probes)).square().mean()
            ))
            teacher_changes.append(None if previous is None else float((target - previous).norm()))
            targets.append(target)
            previous = target
        return action_errors, function_errors, teacher_changes, targets

    @torch.no_grad()
    def _interventions(
        self,
        episode: Episode,
        wrong_episode: Episode,
        query_x: Tensor,
        steps: int,
    ) -> tuple[
        list[dict[str, float | str | int | None]],
        list[dict[str, float | str | int | None]],
    ]:
        z_data = self.policy.encode_data(episode.support_x, episode.support_y)
        state = self.policy.initial_state(z_data)
        history: tuple[Tensor, ...] = ()
        current = self.initial_function()
        stale_tokens: Tensor | None = None
        stale_state: Tensor | None = None

        wrong_z_data = self.policy.encode_data(
            wrong_episode.support_x, wrong_episode.support_y
        )
        wrong_state = self.policy.initial_state(wrong_z_data)
        wrong_history: tuple[Tensor, ...] = ()
        wrong_current = self.initial_function()

        residual_rows: list[dict[str, float | str | int | None]] = []
        state_rows: list[dict[str, float | str | int | None]] = []
        for iteration in range(steps):
            raw = self.policy.raw_residual_tokens(
                episode.support_x, episode.support_y, current(episode.support_x)
            )
            correct_action = self.policy.predict_action(raw, z_data, state, history)
            correct_next = current.append(correct_action, detach=True)
            correct_error = _mse(correct_next, episode, query_x)

            wrong_raw_trajectory = self.policy.raw_residual_tokens(
                wrong_episode.support_x,
                wrong_episode.support_y,
                wrong_current(wrong_episode.support_x),
            )
            wrong_action = self.policy.predict_action(
                wrong_raw_trajectory, wrong_z_data, wrong_state, wrong_history
            )
            wrong_next_state = self.policy.update_state(
                wrong_state, wrong_raw_trajectory, wrong_action
            )

            wrong_raw_current = self.policy.raw_residual_tokens(
                wrong_episode.support_x,
                wrong_episode.support_y,
                current(wrong_episode.support_x),
            )
            residual_controls = {
                "correct": raw,
                "zero": torch.zeros_like(raw),
                "wrong_task": wrong_raw_current,
                "shuffled": _shuffle(raw),
            }
            if stale_tokens is not None:
                residual_controls["stale"] = stale_tokens
            for intervention, branch_raw in residual_controls.items():
                branch_action = self.policy.predict_action(
                    branch_raw, z_data, state, history
                )
                branch_function = current.append(branch_action, detach=True)
                branch_error = _mse(branch_function, episode, query_x)
                residual_rows.append({
                    "iteration": iteration,
                    "intervention": intervention,
                    "phi_change_from_correct": float((branch_action - correct_action).norm()),
                    "function_change_from_correct": float(
                        (branch_function(query_x) - correct_next(query_x)).square().mean()
                    ),
                    "query_mse": branch_error,
                    "query_mse_delta_from_correct": branch_error - correct_error,
                })

            state_controls = {
                "correct": state,
                "zero": torch.zeros_like(state),
                "wrong_task": wrong_state,
            }
            if stale_state is not None:
                state_controls["stale"] = stale_state
            for intervention, branch_state in state_controls.items():
                branch_action = self.policy.predict_action(
                    raw, z_data, branch_state, history
                )
                branch_function = current.append(branch_action, detach=True)
                branch_error = _mse(branch_function, episode, query_x)
                state_rows.append({
                    "iteration": iteration,
                    "intervention": intervention,
                    "state_distance_from_correct": float((branch_state - state).norm()),
                    "phi_change_from_correct": float((branch_action - correct_action).norm()),
                    "function_change_from_correct": float(
                        (branch_function(query_x) - correct_next(query_x)).square().mean()
                    ),
                    "query_mse": branch_error,
                    "query_mse_delta_from_correct": branch_error - correct_error,
                })

            next_state = self.policy.update_state(state, raw, correct_action).detach()
            stale_tokens = raw.detach()
            stale_state = state.detach()
            current = correct_next
            history = history + (correct_action.detach(),)
            state = next_state

            wrong_current = wrong_current.append(wrong_action, detach=True)
            wrong_history = wrong_history + (wrong_action.detach(),)
            wrong_state = wrong_next_state.detach()
        return residual_rows, state_rows

    @torch.no_grad()
    def run(self):
        cfg = self.config.evaluation
        records: list[V22Record] = []
        residual_interventions: list[dict[str, object]] = []
        state_interventions: list[dict[str, object]] = []
        trajectories: list[dict[str, object]] = []
        diversity: list[dict[str, object]] = []
        splits: tuple[Split, ...] = (
            "seen_parameters", "heldout_combinations", "heldout_compositions"
        )
        total_cells = len(cfg.seeds) * len(splits) * len(self.config.tasks.support_sizes)
        completed_cells = 0
        for seed in cfg.seeds:
            benchmark = SyntheticFunctionBenchmark(self.config.tasks, seed, self.device)
            for split in splits:
                for support_size in self.config.tasks.support_sizes:
                    cell_states: dict[int, list[Tensor]] = defaultdict(list)
                    for episode_index in range(cfg.episodes_per_cell):
                        episode = benchmark.sample_episode(support_size, split)
                        wrong = benchmark.sample_episode(support_size, split)
                        query_x = benchmark.sample_inputs(cfg.query_size)
                        trace = predictive_adapt(
                            self.policy, self.initial_function(),
                            episode.support_x, episode.support_y, cfg.rollout_steps,
                        )
                        action_errors, function_errors, teacher_changes, targets = (
                            self._diagnostics(episode, trace, benchmark)
                        )
                        for iteration, function in enumerate(trace.functions):
                            state = trace.states[iteration]
                            cell_states[iteration].append(state.detach().cpu())
                            if iteration == 0:
                                records.append(V22Record(
                                    seed, split, support_size, "predictive_recurrent", iteration,
                                    _mse(function, episode, query_x), None, None, None, None,
                                    float(state.norm()), None,
                                ))
                                continue
                            index = iteration - 1
                            residual_distance = None
                            if index > 0:
                                residual_distance = float(
                                    (trace.residual_tokens[index] - trace.residual_tokens[index - 1])
                                    .square().mean().sqrt()
                                )
                            records.append(V22Record(
                                seed, split, support_size, "predictive_recurrent", iteration,
                                _mse(function, episode, query_x), action_errors[index],
                                function_errors[index], residual_distance, teacher_changes[index],
                                float(state.norm()),
                                float((state - trace.states[iteration - 1]).norm()),
                            ))
                        if episode_index == 0:
                            for iteration, (action, target) in enumerate(zip(trace.actions, targets)):
                                for coordinate in range(action.numel()):
                                    trajectories.append({
                                        "seed": seed, "split": split,
                                        "support_size": support_size, "iteration": iteration,
                                        "coordinate": coordinate,
                                        "predicted_phi": float(action[coordinate]),
                                        "teacher_phi": float(target[coordinate]),
                                    })
                        if episode_index < max(1, cfg.episodes_per_cell // 8):
                            residual_rows, state_rows = self._interventions(
                                episode, wrong, query_x,
                                min(cfg.rollout_steps, self.config.phase_b.rollout_max),
                            )
                            shared = {"seed": seed, "split": split, "support_size": support_size}
                            for row in residual_rows:
                                row.update(shared)
                            for row in state_rows:
                                row.update(shared)
                            residual_interventions.extend(residual_rows)
                            state_interventions.extend(state_rows)

                    for iteration, values in sorted(cell_states.items()):
                        states = torch.stack(values)
                        pairwise = torch.pdist(states)
                        diversity.append({
                            "seed": seed, "split": split, "support_size": support_size,
                            "iteration": iteration,
                            "mean_pairwise_state_distance": (
                                float(pairwise.mean()) if pairwise.numel() else 0.0
                            ),
                            "mean_feature_std": float(states.std(dim=0, unbiased=False).mean()),
                            "mean_state_norm": float(states.norm(dim=-1).mean()),
                        })
                    completed_cells += 1
                    progress = {
                        "completed_cells": completed_cells, "total_cells": total_cells,
                        "seed": seed, "split": split, "support_size": support_size,
                    }
                    (self.output_dir / "evaluation_progress.json").write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
                    print(json.dumps({"phase": "evaluation_v2_2", **progress}), flush=True)
        return records, residual_interventions, state_interventions, trajectories, diversity


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_v22_outputs(output_dir: str | Path, records, residual, state, trajectories, diversity):
    output = Path(output_dir)
    _write_csv(output / "evaluation_records.csv", [asdict(record) for record in records])
    _write_csv(output / "residual_interventions.csv", residual)
    _write_csv(output / "state_interventions.csv", state)
    _write_csv(output / "phi_trajectories.csv", trajectories)
    _write_csv(output / "state_diversity.csv", diversity)
    groups: dict[tuple[str, int, int], list[V22Record]] = defaultdict(list)
    for record in records:
        groups[(record.split, record.support_size, record.iteration)].append(record)
    summary = [{
        "split": key[0], "support_size": key[1], "iteration": key[2],
        "query_mse_mean": mean(row.query_mse for row in rows),
        "query_mse_std": pstdev(row.query_mse for row in rows), "n": len(rows),
        "action_mse_mean": mean(
            row.action_mse for row in rows if row.action_mse is not None
        ) if key[2] > 0 else None,
    } for key, rows in sorted(groups.items())]
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_v22_plots(output_dir: str | Path, v21_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    current = _read(output / "evaluation_records.csv")
    v21 = _read(Path(v21_dir) / "evaluation_records.csv")
    paths: list[Path] = []
    plt.figure(figsize=(7.2, 4.6))
    series = {
        "v2.2 predictive": [row for row in current if row["split"] == "heldout_compositions"],
        "v2.1 recurrent": [row for row in v21 if row["split"] == "heldout_compositions" and row["method"] == "direct_recurrent"],
        "direct current": [row for row in v21 if row["split"] == "heldout_compositions" and row["method"] == "direct_current"],
        "direct history": [row for row in v21 if row["split"] == "heldout_compositions" and row["method"] == "direct_history"],
    }
    for label, rows in series.items():
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            grouped[int(row["iteration"])].append(float(row["query_mse"]))
        iterations = sorted(grouped)
        plt.plot(iterations, [mean(grouped[i]) for i in iterations], marker="o", label=label)
    plt.yscale("log")
    plt.xlabel("IFI iteration")
    plt.ylabel("held-out composition query MSE")
    plt.legend()
    plt.tight_layout()
    path = output / "deep_rollout_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    for filename, ylabel in (
        ("residual_interventions.csv", "residual corruption MSE delta"),
        ("state_interventions.csv", "state corruption MSE delta"),
    ):
        rows = _read(output / filename)
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["intervention"] != "correct":
                grouped[row["intervention"]].append(float(row["query_mse_delta_from_correct"]))
        plt.figure(figsize=(6.4, 4.2))
        labels = list(grouped)
        plt.bar(labels, [mean(grouped[label]) for label in labels])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel(ylabel)
        plt.tight_layout()
        path = output / filename.replace(".csv", ".png")
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)
    return paths
