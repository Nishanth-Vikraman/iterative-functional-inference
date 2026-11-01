from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import replace

import torch
from torch import Tensor

from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_models import PopulationController
from .v40_worlds import (
    analytic_action,
    sample_initial_residuals,
    sample_probe_actions,
    transition,
)
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v44_config import V44Config


ALL_HISTORY = HistoryPolicy("all_history", 0, True, True)
FIFO_16 = HistoryPolicy("fifo_16", 16, False)


def clone_state(state: RolloutState, start: int, stop: int) -> RolloutState:
    return RolloutState(**{name: value[start:stop].clone() for name, value in state.__dict__.items()})


def fresh_state_from_matrices(
    config: V40Config,
    matrices: Tensor,
    world_seeds: Tensor,
    capacity: int,
    seed: int,
    device: torch.device,
) -> RolloutState:
    matrices = matrices.cpu()
    count = matrices.shape[0]
    generator = torch.Generator().manual_seed(seed)
    residual = sample_initial_residuals(count, config.residual_dimension, generator)
    probes = sample_probe_actions(
        count,
        config.bootstrap_actions,
        config.action_dimension,
        config.bootstrap_action_norm,
        generator,
        config.bootstrap_orthogonal,
    )
    before = torch.zeros(count, capacity, config.residual_dimension)
    action = torch.zeros(count, capacity, config.action_dimension)
    after = torch.zeros(count, capacity, config.residual_dimension)
    for index in range(config.bootstrap_actions):
        before[:, index] = residual
        action[:, index] = probes[:, index]
        residual = transition(residual, matrices, probes[:, index])
        after[:, index] = residual
    return RolloutState(
        matrices=matrices.to(device),
        singular_values=torch.linalg.svdvals(matrices).to(device),
        world_seeds=world_seeds.to(device),
        current=residual.to(device),
        previous_action=probes[:, -1].to(device),
        history_before=before.to(device),
        history_action=action.to(device),
        history_after=after.to(device),
        history_lengths=torch.full(
            (count,), config.bootstrap_actions, dtype=torch.int64, device=device
        ),
    )


def stored_state(
    config: V40Config,
    population: dict[str, object],
    indices: Tensor,
    capacity: int,
    device: torch.device,
) -> RolloutState:
    count = indices.numel()
    before = torch.zeros(count, capacity, config.residual_dimension)
    action = torch.zeros(count, capacity, config.action_dimension)
    after = torch.zeros(count, capacity, config.residual_dimension)
    before[:, : config.history_capacity] = population["history_before"][indices]
    action[:, : config.history_capacity] = population["history_action"][indices]
    after[:, : config.history_capacity] = population["history_after"][indices]
    matrices = population["matrices"][indices]
    singular = (
        population["singular_values"][indices]
        if "singular_values" in population
        else torch.linalg.svdvals(matrices)
    )
    return RolloutState(
        matrices=matrices.to(device),
        singular_values=singular.to(device),
        world_seeds=population["world_seeds"][indices].to(device),
        current=population["current"][indices].to(device),
        previous_action=population["previous_action"][indices].to(device),
        history_before=before.to(device),
        history_action=action.to(device),
        history_after=after.to(device),
        history_lengths=population["history_lengths"][indices].to(device),
    )


@torch.inference_mode()
def rollout_summary(
    experiment: V44Config,
    config: V40Config,
    model: PopulationController | None,
    model_label: str,
    trained_seed: int | str,
    fixture: str,
    base: RolloutState,
    policy: HistoryPolicy,
) -> list[dict[str, object]]:
    accumulators: dict[int, dict[str, list[Tensor]]] = {
        iteration: {
            "norm": [],
            "q": [],
            "action": [],
            "ratio": [],
        }
        for iteration in range(experiment.evaluation_horizon + 1)
    }
    batch_size = config.evaluation_batch_size
    for start in range(0, base.current.shape[0], batch_size):
        state = clone_state(base, start, min(start + batch_size, base.current.shape[0]))
        previous = state.current.norm(dim=-1)
        accumulators[0]["norm"].append(previous.cpu())
        for iteration in range(1, experiment.evaluation_horizon + 1):
            before = state.current
            action = (
                analytic_action(state.matrices, before)
                if model is None
                else model(controller_batch(config, state))
            )
            after = transition(before, state.matrices, action)
            norm = after.norm(dim=-1)
            accumulators[iteration]["norm"].append(norm.cpu())
            accumulators[iteration]["q"].append((norm / previous.clamp_min(1e-12)).cpu())
            accumulators[iteration]["action"].append(action.norm(dim=-1).cpu())
            accumulators[iteration]["ratio"].append(
                (action.norm(dim=-1) / previous.clamp_min(1e-12)).cpu()
            )
            append_with_policy(
                state,
                before,
                action,
                after,
                policy,
                config.bootstrap_actions,
                base.history_before.shape[1],
            )
            previous = norm
    rows = []
    for iteration, values in accumulators.items():
        norm = torch.cat(values["norm"])
        q = torch.cat(values["q"]) if values["q"] else None
        action = torch.cat(values["action"]) if values["action"] else torch.zeros_like(norm)
        ratio = torch.cat(values["ratio"]) if values["ratio"] else torch.zeros_like(norm)
        rows.append({
            "trained_seed": trained_seed,
            "fixture": fixture,
            "model": model_label,
            "history_policy": policy.name,
            "iteration": iteration,
            "residual_norm_median": float(norm.median()),
            "residual_norm_mean": float(norm.mean()),
            "q_median": "" if q is None else float(q.median()),
            "q_fraction_below_one": "" if q is None else float((q < 1).float().mean()),
            "regression_fraction": "" if q is None else float((q > 1).float().mean()),
            "action_norm_median": float(action.median()),
            "action_residual_ratio_median": float(ratio.median()),
            "divergence_fraction": float((norm > config.divergence_threshold).float().mean()),
        })
    return rows


def intervention_batch(
    config: V40Config,
    state: RolloutState,
    wrong: RolloutState,
    condition: str,
):
    modified = state.clone()
    if condition == "no_history":
        modified.history_lengths.zero_()
    elif condition == "wrong_world":
        modified.history_before = wrong.history_before.roll(1, 0)
        modified.history_action = wrong.history_action.roll(1, 0)
        modified.history_after = wrong.history_after.roll(1, 0)
        modified.history_lengths = wrong.history_lengths.roll(1, 0)
    elif condition != "correct":
        raise ValueError(condition)
    return controller_batch(config, modified)


@torch.inference_mode()
def history_intervention_summary(
    experiment: V44Config,
    config: V40Config,
    model: PopulationController,
    model_label: str,
    trained_seed: int | str,
    device: torch.device,
) -> list[dict[str, object]]:
    capacity = config.bootstrap_actions + experiment.intervention_at + experiment.intervention_horizon
    storage = replace(config, history_capacity=capacity)
    records: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    for start in range(0, experiment.intervention_episodes, config.evaluation_batch_size):
        count = min(config.evaluation_batch_size, experiment.intervention_episodes - start)
        state = make_rollout_state(storage, count, config.seed + 700 + start, device)
        wrong = make_rollout_state(storage, count, config.seed + 1700 + start, device)
        for _ in range(experiment.intervention_at):
            for active in (state, wrong):
                before = active.current
                action = model(controller_batch(config, active))
                after = transition(before, active.matrices, action)
                append_with_policy(
                    active, before, action, after, ALL_HISTORY, config.bootstrap_actions, capacity
                )
        intervention_norm = state.current.norm(dim=-1)
        for condition in ("correct", "no_history", "wrong_world"):
            rollout = state.clone()
            before = rollout.current
            action = model(intervention_batch(config, rollout, wrong, condition))
            after = transition(before, rollout.matrices, action)
            append_with_policy(
                rollout, before, action, after, ALL_HISTORY, config.bootstrap_actions, capacity
            )
            records[(condition, 1)].append(
                (after.norm(dim=-1) / intervention_norm.clamp_min(1e-12)).cpu()
            )
            for iteration in range(2, experiment.intervention_horizon + 1):
                before = rollout.current
                action = model(controller_batch(config, rollout))
                after = transition(before, rollout.matrices, action)
                append_with_policy(
                    rollout,
                    before,
                    action,
                    after,
                    ALL_HISTORY,
                    config.bootstrap_actions,
                    capacity,
                )
                if iteration in (16, 64, 128):
                    records[(condition, iteration)].append(
                        (after.norm(dim=-1) / intervention_norm.clamp_min(1e-12)).cpu()
                    )
    return [
        {
            "trained_seed": trained_seed,
            "model": model_label,
            "condition": condition,
            "followup_iteration": iteration,
            "residual_ratio_median": float(torch.cat(values).median()),
            "residual_ratio_mean": float(torch.cat(values).mean()),
        }
        for (condition, iteration), values in sorted(records.items())
    ]


def capacity_rows(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        model = str(row["model"])
        if model.startswith("small_seed") or model.startswith("large_seed"):
            if row["fixture"] == "unseen_same_family" and row["history_policy"] == "all_history":
                arm = model.split("_", 1)[0]
                groups[(arm, int(row["trained_seed"]))].append(row)
    seed_rows = []
    for (arm, seed), rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["iteration"]))
        best = min(rows, key=lambda row: float(row["residual_norm_median"]))
        by_iteration = {int(row["iteration"]): row for row in rows}
        f0 = float(by_iteration[0]["residual_norm_median"])
        f8 = float(by_iteration[8]["residual_norm_median"])
        seed_rows.append({
            "row_type": "seed",
            "arm": arm,
            "seed": seed,
            "f8_over_f0": f8 / f0,
            "best_residual": float(best["residual_norm_median"]),
            "best_step": int(best["iteration"]),
            "f64_over_best": float(by_iteration[64]["residual_norm_median"]) / float(best["residual_norm_median"]),
            "f128_over_best": float(by_iteration[128]["residual_norm_median"]) / float(best["residual_norm_median"]),
            "q_f1_f16": statistics.median(float(row["q_median"]) for row in rows if 1 <= int(row["iteration"]) <= 16),
            "q_f17_f32": statistics.median(float(row["q_median"]) for row in rows if 17 <= int(row["iteration"]) <= 32),
            "q_f33_f64": statistics.median(float(row["q_median"]) for row in rows if 33 <= int(row["iteration"]) <= 64),
            "q_f65_f128": statistics.median(float(row["q_median"]) for row in rows if 65 <= int(row["iteration"]) <= 128),
        })
    output.extend(seed_rows)
    for arm in sorted({str(row["arm"]) for row in seed_rows}):
        selected = [row for row in seed_rows if row["arm"] == arm]
        row: dict[str, object] = {"row_type": "mean", "arm": arm, "seed": "mean"}
        for key in (
            "f8_over_f0",
            "best_residual",
            "best_step",
            "f64_over_best",
            "f128_over_best",
            "q_f1_f16",
            "q_f17_f32",
            "q_f33_f64",
            "q_f65_f128",
        ):
            row[key] = statistics.fmean(float(item[key]) for item in selected)
        output.append(row)
    return output
