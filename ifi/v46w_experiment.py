from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor, nn

from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_models import ControllerBatch, PopulationController, transition_tokens
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v45_replay import RawReplayBatch, TrajectoryCollection, collect_current_policy
from .v45q_config import V45QConfig
from .v45q_model import ContractionCritic, CriticBatch
from .v46w_config import V46WConfig
from .v46w_model import ResidualWorldModel, WorldModelBatch


CAPPED_HISTORY = HistoryPolicy("all_available_capped_128", 128, False)
ACTION_KINDS = ("t_proposal", "scaled_t", "random", "orthogonal")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_t(
    experiment: V46WConfig, device: torch.device
) -> tuple[V40Config, PopulationController]:
    source = torch.load(
        experiment.t_source_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint = torch.load(experiment.t_checkpoint, map_location="cpu", weights_only=False)
    config = V40Config(**source["config"])
    model = PopulationController("query_history", config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return config, model


def collection_spec(
    experiment: V46WConfig, worlds: int, seed: int
) -> SimpleNamespace:
    return SimpleNamespace(
        collection_depths=experiment.collection_depths,
        collection_worlds=worlds,
        collection_batch_size=experiment.collection_batch_size,
        max_history=experiment.max_history,
        seed=seed,
    )


@torch.inference_mode()
def collect_datasets(
    experiment: V46WConfig,
    config: V40Config,
    controller: PopulationController,
    device: torch.device,
) -> dict[str, TrajectoryCollection]:
    definitions = {
        "train": (experiment.train_worlds, experiment.seed + 10_000, False),
        "validation": (experiment.validation_worlds, experiment.seed + 20_000, False),
        "heldout_same_family": (
            experiment.heldout_worlds,
            experiment.seed + 30_000,
            False,
        ),
        "condition_4_6": (experiment.ood_worlds, experiment.seed + 40_000, True),
    }
    collections = {}
    for index, (name, (worlds, seed, harder)) in enumerate(definitions.items()):
        collections[name] = collect_current_policy(
            collection_spec(experiment, worlds, seed),
            config,
            controller,
            policy_version=200 + index,
            device=device,
            harder=harder,
        )
        if collections[name].history_error() > 2e-6:
            raise RuntimeError(f"Non-causal history in {name}")
    seed_sets = {
        name: set(collection.world_seeds.tolist())
        for name, collection in collections.items()
    }
    for left in seed_sets:
        for right in seed_sets:
            if left < right and seed_sets[left] & seed_sets[right]:
                raise RuntimeError(f"World leakage between {left} and {right}")
    return collections


def transition_positions(collection: TrajectoryCollection) -> Tensor:
    valid = collection.flat_depth < collection.horizons[collection.flat_episode]
    return torch.where(valid)[0]


def stored_t_actions(collection: TrajectoryCollection, positions: Tensor) -> Tensor:
    episodes = collection.flat_episode[positions]
    depths = collection.flat_depth[positions]
    return collection.actions[episodes, depths]


def random_unit(count: int, dimension: int, generator: torch.Generator) -> Tensor:
    value = torch.randn(count, dimension, generator=generator)
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def random_norms(
    count: int, experiment: V46WConfig, generator: torch.Generator
) -> Tensor:
    low = math.log(experiment.random_action_norm_min)
    high = math.log(experiment.random_action_norm_max)
    return torch.exp(low + (high - low) * torch.rand(count, generator=generator))


def candidate_actions(
    t_action: Tensor,
    kinds: Tensor,
    experiment: V46WConfig,
    config: V40Config,
    generator: torch.Generator,
) -> Tensor:
    count, dimension = t_action.shape
    result = t_action.clone()
    scaled = kinds == 1
    if scaled.any():
        indices = torch.randint(
            len(experiment.scaled_factors),
            (int(scaled.sum()),),
            generator=generator,
        )
        factors = torch.tensor(experiment.scaled_factors)[indices]
        result[scaled] = t_action[scaled] * factors[:, None]
    random_mask = kinds == 2
    if random_mask.any():
        amount = int(random_mask.sum())
        result[random_mask] = random_unit(amount, dimension, generator) * random_norms(
            amount, experiment, generator
        )[:, None]
    orthogonal = kinds == 3
    if orthogonal.any():
        amount = int(orthogonal.sum())
        proposal = t_action[orthogonal]
        direction = random_unit(amount, dimension, generator)
        unit_proposal = proposal / proposal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        direction = direction - (direction * unit_proposal).sum(
            dim=-1, keepdim=True
        ) * unit_proposal
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result[orthogonal] = direction * random_norms(
            amount, experiment, generator
        )[:, None]
    return result.clamp(-config.action_max, config.action_max)


def mixed_action_kinds(count: int, experiment: V46WConfig) -> Tensor:
    fractions = (
        experiment.actual_action_fraction,
        experiment.scaled_action_fraction,
        experiment.random_action_fraction,
        experiment.orthogonal_action_fraction,
    )
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-8):
        raise ValueError("Candidate-action fractions must sum to one")
    boundaries = np.cumsum((0.0,) + fractions)
    kinds = torch.empty(count, dtype=torch.int64)
    for index in range(4):
        start = round(float(boundaries[index]) * count)
        stop = round(float(boundaries[index + 1]) * count)
        kinds[start:stop] = index
    return kinds


def world_batch(
    raw: RawReplayBatch,
    candidate: Tensor,
    config: V40Config,
    device: torch.device,
    history_mode: str = "full",
    recent_history: int = 16,
) -> WorldModelBatch:
    before = raw.history_before.clone()
    action = raw.history_action.clone()
    after = raw.history_after.clone()
    lengths = raw.history_lengths.clone()
    if history_mode == "no_history":
        before.zero_()
        action.zero_()
        after.zero_()
        lengths.zero_()
    elif history_mode == "wrong_world":
        before = before.roll(1, 0)
        action = action.roll(1, 0)
        after = after.roll(1, 0)
        lengths = lengths.roll(1, 0)
    elif history_mode == "bootstrap_only":
        lengths = lengths.clamp_max(config.bootstrap_actions)
    elif history_mode == "recent_only":
        new_before = torch.zeros_like(before)
        new_action = torch.zeros_like(action)
        new_after = torch.zeros_like(after)
        new_lengths = lengths.clamp_max(recent_history)
        for row, length in enumerate(lengths.tolist()):
            keep = min(int(length), recent_history)
            start = int(length) - keep
            new_before[row, :keep] = before[row, start : start + keep]
            new_action[row, :keep] = action[row, start : start + keep]
            new_after[row, :keep] = after[row, start : start + keep]
        before, action, after, lengths = (
            new_before,
            new_action,
            new_after,
            new_lengths,
        )
    elif history_mode != "full":
        raise ValueError(history_mode)
    state = ControllerBatch(
        current=raw.current.to(device),
        previous_action=raw.previous_action.to(device),
        history=transition_tokens(
            before.to(device),
            action.to(device),
            after.to(device),
            config.magnitude_scale,
            config.numerical_epsilon,
        ),
        history_lengths=lengths.to(device),
    )
    return WorldModelBatch(state, candidate.to(device))


def sample_training_batch(
    collection: TrajectoryCollection,
    positions: Tensor,
    experiment: V46WConfig,
    config: V40Config,
    state_generator: torch.Generator,
    action_generator: torch.Generator,
    device: torch.device,
) -> tuple[WorldModelBatch, Tensor, Tensor, Tensor]:
    selected = positions[
        torch.randint(positions.numel(), (experiment.batch_size,), generator=state_generator)
    ]
    raw = collection.batch_from_positions(selected, 0)
    kinds = mixed_action_kinds(experiment.batch_size, experiment)
    kinds = kinds[torch.randperm(kinds.numel(), generator=action_generator)]
    t_action = stored_t_actions(collection, selected)
    action = candidate_actions(t_action, kinds, experiment, config, action_generator)
    target = torch.bmm(raw.matrices, action.unsqueeze(-1)).squeeze(-1)
    return (
        world_batch(raw, action, config, device),
        target.to(device),
        raw.current.to(device),
        kinds,
    )


def normalized_delta_loss(
    predicted: Tensor, target: Tensor, floor: float
) -> Tensor:
    numerator = (predicted - target).square().sum(dim=-1)
    denominator = target.square().sum(dim=-1) + floor**2
    return numerator / denominator


def fit_global_response(collection: TrajectoryCollection) -> Tensor:
    action = collection.bootstrap_action.reshape(-1, collection.current_dimension)
    delta = (collection.bootstrap_after - collection.bootstrap_before).reshape(
        -1, collection.current_dimension
    )
    return torch.linalg.lstsq(action, delta).solution.T


def empirical_history_prediction(raw: RawReplayBatch, candidate: Tensor) -> Tensor:
    action = raw.history_action
    delta = raw.history_after - raw.history_before
    positions = torch.arange(action.shape[1])[None]
    valid = (positions < raw.history_lengths[:, None]).float()
    weighted_action = action * valid[:, :, None]
    xtx = torch.einsum("bhi,bhj->bij", weighted_action, action)
    xty = torch.einsum("bhi,bhk->bik", weighted_action, delta)
    identity = torch.eye(action.shape[-1])[None]
    response_transpose = torch.linalg.solve(xtx + 1e-6 * identity, xty)
    return torch.einsum("bi,bij->bj", candidate, response_transpose)


def gradient_norm(model: nn.Module) -> float:
    values = [
        parameter.grad.square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def train_to_step(
    experiment: V46WConfig,
    config: V40Config,
    model: ResidualWorldModel,
    collection: TrajectoryCollection,
    seed: int,
    start_step: int,
    end_step: int,
    device: torch.device,
    optimizer_state: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    positions = transition_positions(collection)
    state_generator = torch.Generator().manual_seed(seed + 60_000)
    action_generator = torch.Generator().manual_seed(seed + 70_000)
    # Reproduce the exact random stream when resuming the smoke seed.
    if start_step:
        for _ in range(start_step):
            torch.randint(
                positions.numel(), (experiment.batch_size,), generator=state_generator
            )
            kinds = mixed_action_kinds(experiment.batch_size, experiment)
            kinds = kinds[torch.randperm(kinds.numel(), generator=action_generator)]
            # Consume the action RNG with the same shape-dependent sequence.
            dummy = torch.ones(experiment.batch_size, config.action_dimension)
            candidate_actions(dummy, kinds, experiment, config, action_generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    model.train()
    rows = []
    started = time.time()
    for step in range(start_step + 1, end_step + 1):
        batch, target, current, kinds = sample_training_batch(
            collection,
            positions,
            experiment,
            config,
            state_generator,
            action_generator,
            device,
        )
        predicted = model(batch)
        per_example = normalized_delta_loss(predicted, target, experiment.delta_floor)
        loss = per_example.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite v4.6-W loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = gradient_norm(model)
        if not math.isfinite(gradients):
            raise RuntimeError(f"Non-finite v4.6-W gradient at step {step}")
        optimizer.step()
        if step == start_step + 1 or step % experiment.log_every == 0:
            with torch.no_grad():
                cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
                residual_loss = (predicted - target).square().sum(-1) / (
                    current.square().sum(-1) + config.numerical_epsilon
                )
                for index, name in enumerate(ACTION_KINDS):
                    selected = kinds.to(device) == index
                    rows.append({
                        "seed": seed,
                        "step": step,
                        "action_kind": name,
                        "loss_delta": float(per_example[selected].mean()),
                        "loss_residual_normalized": float(residual_loss[selected].mean()),
                        "cosine": float(cosine[selected].mean()),
                        "gradient_norm": gradients,
                        "steps_per_second": (step - start_step)
                        / max(time.time() - started, 1e-6),
                    })
    return rows, optimizer.state_dict()


def fixed_evaluation_positions(
    collection: TrajectoryCollection, count: int, seed: int
) -> Tensor:
    positions = transition_positions(collection)
    generator = torch.Generator().manual_seed(seed)
    return positions[torch.randperm(positions.numel(), generator=generator)[:count]]


def fixed_candidate(
    collection: TrajectoryCollection,
    positions: Tensor,
    action_kind: str,
    experiment: V46WConfig,
    config: V40Config,
    seed: int,
) -> Tensor:
    t_action = stored_t_actions(collection, positions)
    generator = torch.Generator().manual_seed(seed)
    kind = ACTION_KINDS.index(action_kind)
    kinds = torch.full((positions.numel(),), kind, dtype=torch.int64)
    if action_kind == "scaled_t":
        factors = torch.tensor(experiment.scaled_factors)
        result = t_action * factors[
            torch.arange(positions.numel()) % factors.numel()
        ][:, None]
        return result.clamp(-config.action_max, config.action_max)
    return candidate_actions(t_action, kinds, experiment, config, generator)


def roc_auc(scores: Tensor, labels: Tensor) -> float:
    labels = labels.float()
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    positive = sorted_labels.sum().clamp_min(1)
    negative = (1 - sorted_labels).sum().clamp_min(1)
    tpr = torch.cat([torch.zeros(1), sorted_labels.cumsum(0) / positive])
    fpr = torch.cat([torch.zeros(1), (1 - sorted_labels).cumsum(0) / negative])
    return float(torch.trapezoid(tpr, fpr))


def summarize_predictions(
    distribution: str,
    action_kind: str,
    model_name: str,
    predicted: Tensor,
    target: Tensor,
    current: Tensor,
    experiment: V46WConfig,
    config: V40Config,
) -> dict[str, object]:
    normalized = normalized_delta_loss(predicted, target, experiment.delta_floor)
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
    magnitude = predicted.norm(dim=-1) / target.norm(dim=-1).clamp_min(
        config.numerical_epsilon
    )
    true_next = current + target
    predicted_next = current + predicted
    denominator = current.norm(dim=-1).clamp_min(config.numerical_epsilon)
    q_true = true_next.norm(dim=-1) / denominator
    q_hat = predicted_next.norm(dim=-1) / denominator
    harmful = q_true >= 1
    return {
        "distribution": distribution,
        "action_kind": action_kind,
        "model": model_name,
        "samples": int(target.shape[0]),
        "normalized_delta_loss_mean": float(normalized.mean()),
        "normalized_delta_loss_median": float(normalized.median()),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(cosine.median()),
        "magnitude_ratio_mean": float(magnitude.mean()),
        "magnitude_ratio_median": float(magnitude.median()),
        "residual_normalized_loss_mean": float(
            ((predicted_next - true_next).square().sum(-1) / (
                current.square().sum(-1) + config.numerical_epsilon
            )).mean()
        ),
        "q_mae": float((q_hat - q_true).abs().mean()),
        "q_bias": float((q_hat - q_true).mean()),
        "harmful_auroc": roc_auc(q_hat, harmful),
        "false_safe_1.00": float(
            ((q_hat < 1.0) & harmful).sum() / harmful.sum().clamp_min(1)
        ),
        "false_safe_0.98": float(
            ((q_hat < 0.98) & harmful).sum() / harmful.sum().clamp_min(1)
        ),
        "false_safe_0.95": float(
            ((q_hat < 0.95) & harmful).sum() / harmful.sum().clamp_min(1)
        ),
        "harmful_fraction": float(harmful.float().mean()),
    }


@torch.inference_mode()
def evaluate_world_model(
    experiment: V46WConfig,
    config: V40Config,
    model: ResidualWorldModel,
    collections: dict[str, TrajectoryCollection],
    global_response: Tensor,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    rows = []
    for distribution, collection in collections.items():
        positions = fixed_evaluation_positions(
            collection, experiment.evaluation_states, experiment.seed + 80_000
        )
        for kind_index, action_kind in enumerate(ACTION_KINDS):
            action = fixed_candidate(
                collection,
                positions,
                action_kind,
                experiment,
                config,
                experiment.seed + 81_000 + kind_index,
            )
            predictions: dict[str, list[Tensor]] = {
                "world_model": [],
                "zero_transition": [],
                "global_average_response": [],
                "history_least_squares": [],
            }
            targets, currents = [], []
            for start in range(0, positions.numel(), 256):
                selected = positions[start : start + 256]
                selected_action = action[start : start + 256]
                raw = collection.batch_from_positions(selected, 0)
                target = torch.bmm(
                    raw.matrices, selected_action.unsqueeze(-1)
                ).squeeze(-1)
                batch = world_batch(raw, selected_action, config, device)
                predictions["world_model"].append(model(batch).cpu())
                predictions["zero_transition"].append(torch.zeros_like(target))
                predictions["global_average_response"].append(
                    selected_action @ global_response.T
                )
                predictions["history_least_squares"].append(
                    empirical_history_prediction(raw, selected_action)
                )
                targets.append(target)
                currents.append(raw.current)
            target = torch.cat(targets)
            current = torch.cat(currents)
            for model_name, values in predictions.items():
                row = summarize_predictions(
                    distribution,
                    action_kind,
                    model_name,
                    torch.cat(values),
                    target,
                    current,
                    experiment,
                    config,
                )
                row["seed"] = seed
                rows.append(row)
    return rows


@torch.inference_mode()
def evaluate_history_interventions(
    experiment: V46WConfig,
    config: V40Config,
    model: ResidualWorldModel,
    collection: TrajectoryCollection,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    positions = fixed_evaluation_positions(
        collection, experiment.intervention_states, experiment.seed + 82_000
    )
    action = fixed_candidate(
        collection,
        positions,
        "random",
        experiment,
        config,
        experiment.seed + 83_000,
    )
    accumulators: dict[str, dict[str, list[Tensor]]] = {
        mode: {"prediction": [], "target": [], "current": []}
        for mode in (
            "full",
            "no_history",
            "wrong_world",
            "bootstrap_only",
            "recent_only",
        )
    }
    for start in range(0, positions.numel(), 256):
        selected = positions[start : start + 256]
        selected_action = action[start : start + 256]
        raw = collection.batch_from_positions(selected, 0)
        target = torch.bmm(raw.matrices, selected_action.unsqueeze(-1)).squeeze(-1)
        for mode in accumulators:
            batch = world_batch(
                raw,
                selected_action,
                config,
                device,
                history_mode=mode,
                recent_history=experiment.recent_history,
            )
            accumulators[mode]["prediction"].append(model(batch).cpu())
            accumulators[mode]["target"].append(target)
            accumulators[mode]["current"].append(raw.current)
    rows = []
    for mode, values in accumulators.items():
        row = summarize_predictions(
            "heldout_same_family",
            "random",
            f"world_model_{mode}",
            torch.cat(values["prediction"]),
            torch.cat(values["target"]),
            torch.cat(values["current"]),
            experiment,
            config,
        )
        row["seed"] = seed
        row["history_mode"] = mode
        rows.append(row)
    return rows


def load_old_q(
    experiment: V46WConfig,
    config: V40Config,
    device: torch.device,
) -> ContractionCritic:
    old_config = V45QConfig.from_json(experiment.old_q_config)
    checkpoint = torch.load(
        experiment.old_q_checkpoint, map_location="cpu", weights_only=False
    )
    model = ContractionCritic(config, old_config).to(device)
    model.load_state_dict(checkpoint["critic"])
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def old_q_row(
    experiment: V46WConfig,
    config: V40Config,
    critic: ContractionCritic,
    collection: TrajectoryCollection,
    distribution: str,
    device: torch.device,
) -> dict[str, object]:
    positions = fixed_evaluation_positions(
        collection, experiment.evaluation_states, experiment.seed + 80_000
    )
    action = stored_t_actions(collection, positions)
    truth, prediction = [], []
    for start in range(0, positions.numel(), 256):
        selected = positions[start : start + 256]
        selected_action = action[start : start + 256]
        raw = collection.batch_from_positions(selected, 0)
        target = torch.bmm(raw.matrices, selected_action.unsqueeze(-1)).squeeze(-1)
        batch = world_batch(raw, selected_action, config, device)
        log_q, _ = critic(CriticBatch(batch.state, batch.candidate_action))
        q_true = (raw.current + target).norm(dim=-1) / raw.current.norm(
            dim=-1
        ).clamp_min(config.numerical_epsilon)
        truth.append(q_true)
        prediction.append(log_q.exp().cpu())
    q_true = torch.cat(truth)
    q_hat = torch.cat(prediction)
    harmful = q_true >= 1
    boundary = (q_true > 0.9) & (q_true < 1.1)
    return {
        "distribution": distribution,
        "model": "old_scalar_q",
        "action_kind": "t_proposal",
        "samples": int(q_true.numel()),
        "q_mae": float((q_hat - q_true).abs().mean()),
        "q_bias": float((q_hat - q_true).mean()),
        "harmful_auroc": roc_auc(q_hat, harmful),
        "boundary_auroc": roc_auc(q_hat[boundary], harmful[boundary]),
        "false_safe_1.00": float(
            ((q_hat < 1) & harmful).sum() / harmful.sum().clamp_min(1)
        ),
        "false_safe_0.98": float(
            ((q_hat < 0.98) & harmful).sum() / harmful.sum().clamp_min(1)
        ),
        "false_safe_0.95": float(
            ((q_hat < 0.95) & harmful).sum() / harmful.sum().clamp_min(1)
        ),
    }


def smoke_verdict(
    experiment: V46WConfig,
    rows: list[dict[str, object]],
    interventions: list[dict[str, object]],
    old_q: dict[str, object],
) -> dict[str, object]:
    def find(kind: str, model: str) -> dict[str, object]:
        return next(
            row
            for row in rows
            if row["distribution"] == "heldout_same_family"
            and row["action_kind"] == kind
            and row["model"] == model
        )

    world = find("t_proposal", "world_model")
    zero = find("t_proposal", "zero_transition")
    average = find("t_proposal", "global_average_response")
    random_world = find("random", "world_model")
    random_zero = find("random", "zero_transition")
    random_average = find("random", "global_average_response")
    history = {row["history_mode"]: row for row in interventions}
    checks = {
        "heldout_below_zero": float(world["normalized_delta_loss_mean"])
        <= experiment.smoke_relative_baseline_max
        * float(zero["normalized_delta_loss_mean"]),
        "heldout_below_average": float(world["normalized_delta_loss_mean"])
        <= experiment.smoke_relative_baseline_max
        * float(average["normalized_delta_loss_mean"]),
        "heldout_cosine": float(world["cosine_mean"]) >= experiment.smoke_cosine_min,
        "random_below_zero": float(random_world["normalized_delta_loss_mean"])
        <= experiment.smoke_relative_baseline_max
        * float(random_zero["normalized_delta_loss_mean"]),
        "random_below_average": float(random_world["normalized_delta_loss_mean"])
        <= experiment.smoke_relative_baseline_max
        * float(random_average["normalized_delta_loss_mean"]),
        "random_cosine": float(random_world["cosine_mean"])
        >= experiment.smoke_cosine_min,
        "no_history_degrades": float(
            history["no_history"]["normalized_delta_loss_mean"]
        )
        >= experiment.smoke_history_degradation_min
        * float(history["full"]["normalized_delta_loss_mean"]),
        "wrong_history_degrades": float(
            history["wrong_world"]["normalized_delta_loss_mean"]
        )
        >= experiment.smoke_history_degradation_min
        * float(history["full"]["normalized_delta_loss_mean"]),
        "derived_q_improves_old_false_safe": float(world["false_safe_0.98"])
        < float(old_q["false_safe_0.98"]),
    }
    return {
        "passed": all(checks.values()),
        "verdict": "SMOKE PASSED" if all(checks.values()) else "SMOKE FAILED",
        "checks": checks,
        "heldout_t_action_world_model": world,
        "heldout_random_action_world_model": random_world,
        "history_interventions": history,
        "old_scalar_q": old_q,
        "canonical_allowed": all(checks.values()),
    }


def clone_state(state: RolloutState) -> RolloutState:
    return state.clone()


@torch.inference_mode()
def gate_rollout(
    experiment: V46WConfig,
    config: V40Config,
    controller: PopulationController,
    world_model: ResidualWorldModel,
    base_state: RolloutState,
    distribution: str,
    threshold: float,
) -> list[dict[str, object]]:
    state = clone_state(base_state)
    initial = state.current.norm(dim=-1).cpu()
    stopped = torch.zeros(state.current.shape[0], dtype=torch.bool, device=state.current.device)
    accepted = torch.zeros_like(stopped, dtype=torch.float32)
    harmful_accepted = torch.zeros_like(accepted)
    values = [initial]
    for _ in range(experiment.gate_rollout_horizon):
        before = state.current
        batch = controller_batch(config, state)
        proposal = controller(batch)
        predicted_delta = world_model(WorldModelBatch(batch, proposal))
        predicted_q = (before + predicted_delta).norm(dim=-1) / before.norm(
            dim=-1
        ).clamp_min(config.numerical_epsilon)
        actual_after = transition(before, state.matrices, proposal)
        true_q = actual_after.norm(dim=-1) / before.norm(dim=-1).clamp_min(
            config.numerical_epsilon
        )
        accept = (~stopped) & (predicted_q < threshold)
        reject = (~stopped) & (~accept)
        stopped |= reject
        accepted += accept.float()
        harmful_accepted += (accept & (true_q >= 1)).float()
        action = torch.where(accept[:, None], proposal, torch.zeros_like(proposal))
        after = torch.where(accept[:, None], actual_after, before)
        append_with_policy(
            state,
            before,
            action,
            after,
            CAPPED_HISTORY,
            config.bootstrap_actions,
            experiment.max_history,
        )
        values.append(after.norm(dim=-1).cpu())
    stacked = torch.stack(values)
    best = stacked.min(dim=0).values
    return [
        {
            "distribution": distribution,
            "threshold": threshold,
            "iteration": step,
            "residual_median": float(residual.median()),
            "normalized_f_over_f0": float((residual / initial).median()),
            "final_over_best_median": float((values[-1] / best).median()),
            "accepted_actions_mean": float(accepted.mean()),
            "harmful_actions_accepted_mean": float(harmful_accepted.mean()),
        }
        for step, residual in enumerate(values)
    ]


def final_verdict(
    experiment: V46WConfig,
    metrics: list[dict[str, object]],
    interventions: list[dict[str, object]],
    gate_run: bool,
) -> dict[str, object]:
    heldout = [
        row
        for row in metrics
        if row["distribution"] == "heldout_same_family"
        and row["action_kind"] == "random"
        and row["model"] == "world_model"
    ]
    ood = [
        row
        for row in metrics
        if row["distribution"] == "condition_4_6"
        and row["action_kind"] == "random"
        and row["model"] == "world_model"
    ]
    heldout_loss = float(np.mean([row["normalized_delta_loss_mean"] for row in heldout]))
    heldout_cosine = float(np.mean([row["cosine_mean"] for row in heldout]))
    heldout_false_safe = float(np.mean([row["false_safe_0.98"] for row in heldout]))
    ood_false_safe = float(np.mean([row["false_safe_0.98"] for row in ood]))
    full = [
        row for row in interventions if row["history_mode"] == "full"
    ]
    wrong = [
        row for row in interventions if row["history_mode"] == "wrong_world"
    ]
    history_degradation = float(
        np.mean([row["normalized_delta_loss_mean"] for row in wrong])
        / np.mean([row["normalized_delta_loss_mean"] for row in full])
    )
    world_model_supported = (
        heldout_loss < 0.5
        and heldout_cosine >= experiment.smoke_cosine_min
        and history_degradation >= experiment.smoke_history_degradation_min
    )
    q_gate_ready = (
        heldout_false_safe <= experiment.final_heldout_false_safe_max
        and ood_false_safe <= experiment.final_ood_false_safe_max
    )
    return {
        "residual_world_model_supported": world_model_supported,
        "task_invariant_consequence_model_supported": world_model_supported
        and q_gate_ready,
        "world_model_helps_gating_evaluated": gate_run,
        "heldout_random_normalized_delta_loss": heldout_loss,
        "heldout_random_cosine": heldout_cosine,
        "heldout_false_safe_0.98": heldout_false_safe,
        "ood_false_safe_0.98": ood_false_safe,
        "wrong_history_loss_ratio": history_degradation,
        "verdict": (
            "TASK-INVARIANT CONSEQUENCE MODEL SUPPORTED"
            if world_model_supported and q_gate_ready
            else "RESIDUAL WORLD MODEL SUPPORTED; GATING NOT SUPPORTED"
            if world_model_supported
            else "RESIDUAL WORLD MODEL NOT SUPPORTED"
        ),
    }


def report(
    smoke: dict[str, object], final: dict[str, object] | None
) -> str:
    if final is None:
        return f"""# IFI v4.6-W — Task-Invariant Residual World Model

## Verdict

**{smoke['verdict']}**. The locked 2,000-step smoke gate did not pass every mechanistic criterion, so canonical training and runtime gating were not run.

The learned object was the vector transition `W(R,H,u) -> delta R`; neither B, q labels, optimal actions, nor world identity were training targets or inputs.
"""
    return f"""# IFI v4.6-W — Task-Invariant Residual World Model

## Verdict

**{final['verdict']}**

The model was trained only to predict the observed vector residual change. Hidden-system inference is assessed by correct-versus-wrong/no-history interventions; counterfactual generalization is assessed on scaled, random, and orthogonal actions at held-out states; q is derived from the predicted vector transition rather than trained directly.

- Held-out random-action normalized delta loss: `{final['heldout_random_normalized_delta_loss']:.6f}`
- Held-out random-action cosine: `{final['heldout_random_cosine']:.6f}`
- Held-out false-safe rate at 0.98: `{final['heldout_false_safe_0.98']:.6f}`
- OOD false-safe rate at 0.98: `{final['ood_false_safe_0.98']:.6f}`
- Wrong-history/full-history loss ratio: `{final['wrong_history_loss_ratio']:.6f}`

Runtime W-derived gating was evaluated only if both locked false-safe thresholds passed. No joint T/W training or multi-step planning was performed.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_6_residual_world_model.json")
    args = parser.parse_args()
    experiment = V46WConfig.from_json(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output = Path(experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "canonical.lock"
    if lock.exists():
        raise RuntimeError("v4.6-W has already been initiated")
    lock.write_text(str(time.time()), encoding="utf-8")
    atomic_json(output / "status.json", {"phase": "initializing"})
    seed_all(experiment.seed)
    config, controller = load_frozen_t(experiment, device)
    t_hash = sha256(experiment.t_checkpoint)
    collections = collect_datasets(experiment, config, controller, device)
    atomic_json(output / "status.json", {"phase": "datasets_complete"})
    global_response = fit_global_response(collections["train"])
    old_q = load_old_q(experiment, config, device)

    dataset_details = {
        name: {
            "worlds": int(collection.horizons.numel()),
            "transitions": int(transition_positions(collection).numel()),
            "history_error_max": collection.history_error(),
            "world_seed_min": int(collection.world_seeds.min()),
            "world_seed_max": int(collection.world_seeds.max()),
        }
        for name, collection in collections.items()
    }
    atomic_json(output / "config.json", {
        **experiment.to_dict(),
        "base_world_config": config.to_dict(),
        "t_checkpoint_sha256": t_hash,
        "t_frozen": all(not parameter.requires_grad for parameter in controller.parameters()),
        "world_split": dataset_details,
        "candidate_action_semantics": {
            "actual": "stored frozen-T proposal",
            "scaled": list(experiment.scaled_factors),
            "random_norm_range": [
                experiment.random_action_norm_min,
                experiment.random_action_norm_max,
            ],
            "orthogonal": "random action orthogonal to T proposal",
        },
    })

    train_rows: list[dict[str, object]] = []
    seed_all(experiment.seed)
    smoke_model = ResidualWorldModel(config, experiment).to(device)
    smoke_rows, smoke_optimizer = train_to_step(
        experiment,
        config,
        smoke_model,
        collections["train"],
        experiment.seed,
        0,
        experiment.smoke_steps,
        device,
    )
    train_rows.extend(smoke_rows)
    torch.save({
        "model": smoke_model.state_dict(),
        "optimizer": smoke_optimizer,
        "step": experiment.smoke_steps,
        "seed": experiment.seed,
        "t_checkpoint_sha256": t_hash,
    }, output / "smoke_checkpoint.pt")
    smoke_metrics = evaluate_world_model(
        experiment,
        config,
        smoke_model,
        collections,
        global_response,
        experiment.seed,
        device,
    )
    smoke_interventions = evaluate_history_interventions(
        experiment,
        config,
        smoke_model,
        collections["heldout_same_family"],
        experiment.seed,
        device,
    )
    old_heldout = old_q_row(
        experiment,
        config,
        old_q,
        collections["heldout_same_family"],
        "heldout_same_family",
        device,
    )
    smoke = smoke_verdict(
        experiment, smoke_metrics, smoke_interventions, old_heldout
    )
    atomic_json(output / "smoke_gate.json", smoke)
    write_csv(output / "train_metrics.csv", train_rows)
    if not smoke["passed"]:
        write_csv(output / "heldout_world_metrics.csv", smoke_metrics)
        write_csv(
            output / "counterfactual_action_metrics.csv",
            [row for row in smoke_metrics if row["action_kind"] != "t_proposal"],
        )
        write_csv(output / "history_interventions.csv", smoke_interventions)
        write_csv(
            output / "derived_q_metrics.csv",
            [row for row in smoke_metrics if row["model"] == "world_model"],
        )
        write_csv(output / "q_comparison.csv", [old_heldout])
        write_csv(
            output / "baseline_comparison.csv",
            [row for row in smoke_metrics if row["model"] != "world_model"],
        )
        write_csv(output / "optional_gate_rollouts.csv", [])
        atomic_json(output / "verdict.json", {
            "verdict": "SMOKE FAILED; CANONICAL FORBIDDEN",
            "smoke": smoke,
        })
        (output / "report.md").write_text(report(smoke, None), encoding="utf-8")
        atomic_json(output / "parameter_counts.json", {
            "proposal_T_frozen": sum(parameter.numel() for parameter in controller.parameters()),
            "world_model": sum(parameter.numel() for parameter in smoke_model.parameters()),
            "old_scalar_q_frozen": sum(parameter.numel() for parameter in old_q.parameters()),
        })
        atomic_json(output / "status.json", {"phase": "complete", "smoke_passed": False})
        print(json.dumps({"phase": "complete", "smoke": smoke}, indent=2), flush=True)
        return

    atomic_json(output / "status.json", {"phase": "canonical_training", "seed": 11})
    final_models = {}
    continuation, optimizer_state = train_to_step(
        experiment,
        config,
        smoke_model,
        collections["train"],
        experiment.seed,
        experiment.smoke_steps,
        experiment.canonical_steps,
        device,
        smoke_optimizer,
    )
    train_rows.extend(continuation)
    final_models[experiment.seed] = smoke_model
    torch.save({
        "model": smoke_model.state_dict(),
        "optimizer": optimizer_state,
        "step": experiment.canonical_steps,
        "seed": experiment.seed,
        "t_checkpoint_sha256": t_hash,
    }, output / f"checkpoint_seed_{experiment.seed}.pt")
    for seed in experiment.canonical_seeds:
        if seed == experiment.seed:
            continue
        atomic_json(output / "status.json", {"phase": "canonical_training", "seed": seed})
        seed_all(seed)
        model = ResidualWorldModel(config, experiment).to(device)
        rows, optimizer_state = train_to_step(
            experiment,
            config,
            model,
            collections["train"],
            seed,
            0,
            experiment.canonical_steps,
            device,
        )
        train_rows.extend(rows)
        final_models[seed] = model
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer_state,
            "step": experiment.canonical_steps,
            "seed": seed,
            "t_checkpoint_sha256": t_hash,
        }, output / f"checkpoint_seed_{seed}.pt")
    write_csv(output / "train_metrics.csv", train_rows)

    final_metrics, final_interventions, q_rows = [], [], []
    for seed, model in final_models.items():
        metrics = evaluate_world_model(
            experiment,
            config,
            model,
            collections,
            global_response,
            seed,
            device,
        )
        interventions = evaluate_history_interventions(
            experiment,
            config,
            model,
            collections["heldout_same_family"],
            seed,
            device,
        )
        final_metrics.extend(metrics)
        final_interventions.extend(interventions)
        q_rows.extend(
            row for row in metrics if row["model"] == "world_model"
        )
    old_q_rows = [
        old_q_row(
            experiment,
            config,
            old_q,
            collections[name],
            name,
            device,
        )
        for name in ("heldout_same_family", "condition_4_6")
    ]
    world_t_rows = [
        row
        for row in q_rows
        if row["action_kind"] == "t_proposal"
        and row["distribution"] in {"heldout_same_family", "condition_4_6"}
    ]
    q_comparison = old_q_rows + [{
        "distribution": row["distribution"],
        "model": "world_model_derived_q",
        "action_kind": row["action_kind"],
        "samples": row["samples"],
        "q_mae": row["q_mae"],
        "q_bias": row["q_bias"],
        "harmful_auroc": row["harmful_auroc"],
        "boundary_auroc": "",
        "false_safe_1.00": row["false_safe_1.00"],
        "false_safe_0.98": row["false_safe_0.98"],
        "false_safe_0.95": row["false_safe_0.95"],
    } for row in world_t_rows]

    provisional = final_verdict(
        experiment, final_metrics, final_interventions, gate_run=False
    )
    gate_rows = []
    if provisional["task_invariant_consequence_model_supported"]:
        storage = replace(config, history_capacity=experiment.max_history)
        fixtures = {
            "heldout_same_family": make_rollout_state(
                storage,
                experiment.gate_rollout_episodes,
                experiment.seed + 920,
                device,
            ),
            "condition_4_6": make_rollout_state(
                storage,
                experiment.gate_rollout_episodes,
                experiment.seed + 930,
                device,
                harder=True,
            ),
        }
        for seed, model in final_models.items():
            model.eval().requires_grad_(False)
            for distribution, fixture in fixtures.items():
                for threshold in experiment.gate_thresholds:
                    rows = gate_rollout(
                        experiment,
                        config,
                        controller,
                        model,
                        fixture,
                        distribution,
                        threshold,
                    )
                    for row in rows:
                        row["seed"] = seed
                    gate_rows.extend(rows)
    final = final_verdict(
        experiment, final_metrics, final_interventions, gate_run=bool(gate_rows)
    )
    write_csv(output / "heldout_world_metrics.csv", final_metrics)
    write_csv(
        output / "counterfactual_action_metrics.csv",
        [row for row in final_metrics if row["action_kind"] != "t_proposal"],
    )
    write_csv(output / "history_interventions.csv", final_interventions)
    write_csv(output / "derived_q_metrics.csv", q_rows)
    write_csv(output / "q_comparison.csv", q_comparison)
    write_csv(
        output / "baseline_comparison.csv",
        [row for row in final_metrics if row["model"] != "world_model"],
    )
    write_csv(output / "optional_gate_rollouts.csv", gate_rows)
    atomic_json(output / "parameter_counts.json", {
        "proposal_T_frozen": sum(parameter.numel() for parameter in controller.parameters()),
        "world_model": sum(
            parameter.numel() for parameter in next(iter(final_models.values())).parameters()
        ),
        "old_scalar_q_frozen": sum(parameter.numel() for parameter in old_q.parameters()),
    })
    atomic_json(output / "verdict.json", final)
    (output / "report.md").write_text(report(smoke, final), encoding="utf-8")
    atomic_json(output / "status.json", {"phase": "complete", **final})
    print(json.dumps({"phase": "complete", **final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
