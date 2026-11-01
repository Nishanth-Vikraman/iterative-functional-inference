from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .v40_config import V40Config
from .v40_models import ControllerBatch, PopulationController, transition_tokens
from .v40_worlds import (
    analytic_action,
    sample_initial_residuals,
    sample_probe_actions,
    sample_well_conditioned_worlds,
    transition,
)


@dataclass
class RolloutState:
    matrices: Tensor
    singular_values: Tensor
    world_seeds: Tensor
    current: Tensor
    previous_action: Tensor
    history_before: Tensor
    history_action: Tensor
    history_after: Tensor
    history_lengths: Tensor

    def clone(self) -> "RolloutState":
        return RolloutState(**{
            name: value.clone()
            for name, value in self.__dict__.items()
        })


def make_rollout_state(
    config: V40Config,
    count: int,
    seed: int,
    device: torch.device,
    harder: bool = False,
) -> RolloutState:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    condition_range = (
        (config.harder_condition_min, config.harder_condition_max) if harder else None
    )
    worlds = sample_well_conditioned_worlds(
        count,
        config.residual_dimension,
        config.harder_singular_min if harder else config.singular_min,
        config.harder_singular_max if harder else config.singular_max,
        generator,
        seed * 1_000_000,
        condition_range,
    )
    sampled_residual = sample_initial_residuals(
        count,
        config.residual_dimension,
        generator,
        config.evaluation_residual_norm_min,
        config.evaluation_residual_norm_max,
    )
    probes = sample_probe_actions(
        count,
        config.bootstrap_actions,
        config.action_dimension,
        config.bootstrap_action_norm,
        generator,
        config.bootstrap_orthogonal,
    )
    if config.evaluation_residual_is_post_bootstrap:
        total_probe_effect = torch.bmm(
            worlds.matrices,
            probes.sum(dim=1).unsqueeze(-1),
        ).squeeze(-1)
        residual = sampled_residual - total_probe_effect
    else:
        residual = sampled_residual
    before = torch.zeros(count, config.history_capacity, config.residual_dimension)
    actions = torch.zeros(count, config.history_capacity, config.action_dimension)
    after = torch.zeros(count, config.history_capacity, config.residual_dimension)
    for probe_index in range(config.bootstrap_actions):
        before[:, probe_index] = residual
        actions[:, probe_index] = probes[:, probe_index]
        residual = transition(residual, worlds.matrices, probes[:, probe_index])
        after[:, probe_index] = residual
    return RolloutState(
        worlds.matrices.to(device),
        worlds.singular_values.to(device),
        worlds.seeds.to(device),
        residual.to(device),
        probes[:, -1].to(device),
        before.to(device),
        actions.to(device),
        after.to(device),
        torch.full((count,), config.bootstrap_actions, dtype=torch.int64, device=device),
    )


def controller_batch(config: V40Config, state: RolloutState) -> ControllerBatch:
    return ControllerBatch(
        state.current,
        state.previous_action,
        transition_tokens(
            state.history_before,
            state.history_action,
            state.history_after,
            config.magnitude_scale,
            config.numerical_epsilon,
        ),
        state.history_lengths,
    )


def append_transition(
    state: RolloutState,
    before: Tensor,
    action: Tensor,
    after: Tensor,
    capacity: int,
) -> None:
    full = state.history_lengths >= capacity
    if full.any():
        state.history_before[full, :-1] = state.history_before[full, 1:].clone()
        state.history_action[full, :-1] = state.history_action[full, 1:].clone()
        state.history_after[full, :-1] = state.history_after[full, 1:].clone()
    positions = torch.minimum(
        state.history_lengths,
        torch.tensor(capacity - 1, device=state.current.device),
    )
    rows = torch.arange(state.current.shape[0], device=state.current.device)
    state.history_before[rows, positions] = before
    state.history_action[rows, positions] = action
    state.history_after[rows, positions] = after
    state.history_lengths = torch.minimum(
        state.history_lengths + 1,
        torch.tensor(capacity, device=state.current.device),
    )
    state.current = after
    state.previous_action = action


def model_action(
    name: str,
    models: dict[str, PopulationController],
    state: RolloutState,
    config: V40Config,
    fixed_eta: float,
) -> Tensor:
    if name in models:
        return models[name](controller_batch(config, state))
    if name == "fixed_linear":
        return -fixed_eta * state.current
    if name == "analytic_oracle":
        return analytic_action(state.matrices, state.current)
    raise ValueError(name)


@torch.inference_mode()
def calibrate_fixed_eta(config: V40Config, device: torch.device) -> dict[str, object]:
    state = make_rollout_state(
        config,
        config.fixed_eta_calibration_worlds,
        config.seed + 400,
        device,
    )
    scores = {}
    denominator = state.current.square().sum(-1).clamp_min(config.numerical_epsilon)
    for eta in config.fixed_eta_candidates:
        action = -eta * state.current
        next_residual = transition(state.current, state.matrices, action)
        scores[str(eta)] = float((next_residual.square().sum(-1) / denominator).mean())
    selected = min(config.fixed_eta_candidates, key=lambda value: scores[str(value)])
    return {
        "selected_eta": selected,
        "candidate_mean_contraction_squared": scores,
        "source": "preregistered train-family calibration worlds only",
        "worlds": config.fixed_eta_calibration_worlds,
        "test_data_used": False,
    }


@torch.inference_mode()
def quick_gate(
    model: PopulationController,
    config: V40Config,
    device: torch.device,
    step: int,
) -> dict[str, float | bool]:
    state = make_rollout_state(config, config.gate_episodes, config.seed + 500, device)
    initial = state.current.norm(dim=-1)
    q_values = []
    for _ in range(config.gate_rollout):
        before = state.current
        action = model(controller_batch(config, state))
        after = transition(before, state.matrices, action)
        q_values.append(after.norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12))
        append_transition(state, before, action, after, config.history_capacity)
    median_q = float(torch.cat(q_values).median())
    endpoint_ratio = float((state.current.norm(dim=-1) / initial).median())
    passed = (
        median_q < config.gate_median_q_max
        and endpoint_ratio < config.gate_endpoint_ratio_max
    )
    return {
        "step": step,
        "median_q": median_q,
        "median_endpoint_over_initial": endpoint_ratio,
        "median_q_threshold": config.gate_median_q_max,
        "endpoint_ratio_threshold": config.gate_endpoint_ratio_max,
        "passed": passed,
    }


def _world_rows(state: RolloutState, bucket: str) -> list[dict[str, object]]:
    rows = []
    seeds = state.world_seeds.cpu().tolist()
    singular_values = state.singular_values.cpu().tolist()
    matrices = state.matrices.flatten(1).cpu().tolist()
    for episode in range(state.current.shape[0]):
        row: dict[str, object] = {
            "arm": f"evaluation:{bucket}",
            "world_seed": int(seeds[episode]),
            "created_at_optimizer_step": -1,
        }
        for index, value in enumerate(singular_values[episode]):
            row[f"singular_{index}"] = value
        for index, value in enumerate(matrices[episode]):
            row[f"B_{index}"] = value
        rows.append(row)
    return rows


@torch.inference_mode()
def evaluate_rollouts(
    models: dict[str, PopulationController],
    config: V40Config,
    device: torch.device,
    fixed_eta: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_names = [*models, "fixed_linear", "analytic_oracle"]
    buckets = [("same_family", False)]
    if config.harder_bucket_enabled:
        buckets.append(("condition_4_6", True))
    records: list[dict[str, object]] = []
    world_rows: list[dict[str, object]] = []
    for bucket_index, (bucket, harder) in enumerate(buckets):
        base = make_rollout_state(
            config,
            config.evaluation_episodes,
            config.seed + 600 + bucket_index,
            device,
            harder,
        )
        world_rows.extend(_world_rows(base, bucket))
        for name in model_names:
            state = base.clone()
            initial = state.current.norm(dim=-1)
            previous_norm = initial
            initial_values = initial.cpu().tolist()
            seed_values = state.world_seeds.cpu().tolist()
            for episode in range(config.evaluation_episodes):
                records.append({
                    "bucket": bucket,
                    "model": name,
                    "episode": episode,
                    "world_seed": int(seed_values[episode]),
                    "iteration": 0,
                    "residual_norm": initial_values[episode],
                    "q": "",
                    "action_norm": 0.0,
                    "converged": initial_values[episode] < config.convergence_threshold,
                    "diverged": initial_values[episode] > config.divergence_threshold,
                })
            for iteration in range(1, config.evaluation_rollout + 1):
                before = state.current
                action = model_action(name, models, state, config, fixed_eta)
                after = transition(before, state.matrices, action)
                after_norm = after.norm(dim=-1)
                q = after_norm / previous_norm.clamp_min(1e-12)
                after_values = after_norm.cpu().tolist()
                q_values = q.cpu().tolist()
                action_values = action.norm(dim=-1).cpu().tolist()
                for episode in range(config.evaluation_episodes):
                    records.append({
                        "bucket": bucket,
                        "model": name,
                        "episode": episode,
                        "world_seed": int(seed_values[episode]),
                        "iteration": iteration,
                        "residual_norm": after_values[episode],
                        "q": q_values[episode],
                        "action_norm": action_values[episode],
                        "converged": after_values[episode] < config.convergence_threshold,
                        "diverged": after_values[episode] > config.divergence_threshold,
                    })
                append_transition(state, before, action, after, config.history_capacity)
                previous_norm = after_norm
    return records, world_rows


def _intervened_batch(
    config: V40Config,
    state: RolloutState,
    condition: str,
    wrong: RolloutState,
) -> ControllerBatch:
    modified = state.clone()
    if condition == "no_history":
        modified.history_lengths.zero_()
    elif condition == "wrong_world":
        modified.history_before = wrong.history_before.roll(1, 0)
        modified.history_action = wrong.history_action.roll(1, 0)
        modified.history_after = wrong.history_after.roll(1, 0)
        modified.history_lengths = wrong.history_lengths.roll(1, 0)
    elif condition == "actions_removed":
        modified.history_action.zero_()
    elif condition == "residual_transitions_removed":
        modified.history_before.zero_()
        modified.history_after.zero_()
    elif condition == "recent_subset_only":
        recent = 2
        rows = torch.arange(modified.current.shape[0], device=modified.current.device)
        start = (modified.history_lengths - recent).clamp_min(0)
        before = torch.zeros_like(modified.history_before)
        action = torch.zeros_like(modified.history_action)
        after = torch.zeros_like(modified.history_after)
        for offset in range(recent):
            source = (start + offset).clamp_max(config.history_capacity - 1)
            valid = offset < modified.history_lengths
            before[rows[valid], offset] = modified.history_before[rows[valid], source[valid]]
            action[rows[valid], offset] = modified.history_action[rows[valid], source[valid]]
            after[rows[valid], offset] = modified.history_after[rows[valid], source[valid]]
        modified.history_before = before
        modified.history_action = action
        modified.history_after = after
        modified.history_lengths = modified.history_lengths.clamp_max(recent)
    elif condition != "correct":
        raise ValueError(condition)
    return controller_batch(config, modified)


@torch.inference_mode()
def evaluate_history_interventions(
    model: PopulationController,
    config: V40Config,
    device: torch.device,
) -> list[dict[str, object]]:
    state = make_rollout_state(
        config, config.intervention_episodes, config.seed + 700, device
    )
    wrong = make_rollout_state(
        config, config.intervention_episodes, config.seed + 701, device
    )
    for _ in range(config.intervention_at):
        for active in (state, wrong):
            before = active.current
            action = model(controller_batch(config, active))
            after = transition(before, active.matrices, action)
            append_transition(active, before, action, after, config.history_capacity)
    conditions = (
        "correct",
        "no_history",
        "wrong_world",
        "actions_removed",
        "residual_transitions_removed",
        "recent_subset_only",
    )
    correct_action = model(controller_batch(config, state))
    initial_norm = state.current.norm(dim=-1)
    seed_values = state.world_seeds.cpu().tolist()
    records = []
    for condition in conditions:
        rollout = state.clone()
        action = model(_intervened_batch(config, rollout, condition, wrong))
        before = rollout.current
        after = transition(before, rollout.matrices, action)
        one_step_q = after.norm(dim=-1) / initial_norm.clamp_min(1e-12)
        append_transition(rollout, before, action, after, config.history_capacity)
        for _ in range(1, config.intervention_followup):
            before = rollout.current
            followup_action = model(controller_batch(config, rollout))
            after = transition(before, rollout.matrices, followup_action)
            append_transition(
                rollout, before, followup_action, after, config.history_capacity
            )
        endpoint_ratio = rollout.current.norm(dim=-1) / initial_norm.clamp_min(1e-12)
        delta_action = (action - correct_action).norm(dim=-1)
        one_step_values = one_step_q.cpu().tolist()
        endpoint_values = endpoint_ratio.cpu().tolist()
        delta_values = delta_action.cpu().tolist()
        for episode in range(config.intervention_episodes):
            records.append({
                "condition": condition,
                "episode": episode,
                "world_seed": int(seed_values[episode]),
                "intervention_iteration": config.intervention_at,
                "delta_phi": delta_values[episode],
                "one_step_q": one_step_values[episode],
                "endpoint_over_intervention_residual": endpoint_values[episode],
                "followup_steps": config.intervention_followup,
            })
    return records


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
