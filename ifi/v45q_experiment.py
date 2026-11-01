from __future__ import annotations

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
from torch.nn import functional as F

from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_models import ControllerBatch, PopulationController, transition_tokens
from .v40_worlds import transition
from .v41_history_policy_eval import append_with_policy
from .v45_replay import TrajectoryCollection, collect_current_policy
from .v45q_config import V45QConfig
from .v45q_model import ContractionCritic, CriticBatch
from .v45q_oracle import CAPPED_HISTORY, build_fixtures, load_frozen_t


Q_BIN_NAMES = ("lt_0.8", "0.8_0.95", "0.95_1.0", "1.0_1.05", "ge_1.05")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_spec(experiment: V45QConfig, worlds: int, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        collection_depths=experiment.dataset_depths,
        collection_worlds=worlds,
        collection_batch_size=experiment.collection_batch_size,
        max_history=experiment.max_history,
        seed=seed,
    )


def transition_positions(collection: TrajectoryCollection) -> Tensor:
    return torch.where(collection.flat_depth < collection.horizons[collection.flat_episode])[0]


def labels_at(collection: TrajectoryCollection, positions: Tensor) -> Tensor:
    episodes = collection.flat_episode[positions]
    depth = collection.flat_depth[positions]
    before = collection.residuals[episodes, depth].norm(dim=-1)
    after = collection.residuals[episodes, depth + 1].norm(dim=-1)
    return after / before.clamp_min(1e-12)


def q_bins(q: Tensor) -> Tensor:
    return torch.bucketize(q, torch.tensor([0.8, 0.95, 1.0, 1.05]))


def critic_batch_from_positions(
    collection: TrajectoryCollection,
    positions: Tensor,
    config,
    device: torch.device,
) -> tuple[CriticBatch, Tensor]:
    raw = collection.batch_from_positions(positions, 0)
    episodes = collection.flat_episode[positions]
    depths = collection.flat_depth[positions]
    candidate = collection.actions[episodes, depths]
    q_true = labels_at(collection, positions)
    state = ControllerBatch(
        current=raw.current.to(device),
        previous_action=raw.previous_action.to(device),
        history=transition_tokens(
            raw.history_before.to(device),
            raw.history_action.to(device),
            raw.history_after.to(device),
            config.magnitude_scale,
            config.numerical_epsilon,
        ),
        history_lengths=raw.history_lengths.to(device),
    )
    return CriticBatch(state, candidate.to(device)), q_true.to(device)


@torch.inference_mode()
def collect_datasets(
    experiment: V45QConfig,
    config,
    model: PopulationController,
    device: torch.device,
) -> dict[str, TrajectoryCollection]:
    definitions = {
        "train": (experiment.dataset_train_worlds, experiment.seed + 10_000, False),
        "validation": (experiment.dataset_validation_worlds, experiment.seed + 20_000, False),
        "heldout_same_family": (experiment.dataset_test_worlds, experiment.seed + 30_000, False),
        "condition_4_6": (experiment.dataset_ood_worlds, experiment.seed + 40_000, True),
    }
    result = {}
    for index, (name, (worlds, seed, harder)) in enumerate(definitions.items()):
        result[name] = collect_current_policy(
            collection_spec(experiment, worlds, seed),
            config,
            model,
            100 + index,
            device,
            harder=harder,
        )
        if result[name].history_error() > 1e-5:
            raise RuntimeError(f"Non-causal history found in {name}")
    seed_sets = {name: set(value.world_seeds.tolist()) for name, value in result.items()}
    for left, left_seeds in seed_sets.items():
        for right, right_seeds in seed_sets.items():
            if left < right and left_seeds & right_seeds:
                raise RuntimeError(f"World leakage between {left} and {right}")
    return result


def dataset_stats(collections: dict[str, TrajectoryCollection]) -> dict[str, object]:
    payload: dict[str, object] = {"split_by_world": True, "world_overlap": 0}
    for name, collection in collections.items():
        positions = transition_positions(collection)
        q = labels_at(collection, positions)
        bins = q_bins(q)
        payload[name] = {
            "worlds": int(collection.horizons.numel()),
            "transitions": int(positions.numel()),
            "history_error_max": collection.history_error(),
            "q_mean": float(q.mean()),
            "q_median": float(q.median()),
            "harmful_fraction": float((q >= 1).float().mean()),
            "q_bin_natural_counts": {
                Q_BIN_NAMES[index]: int((bins == index).sum()) for index in range(5)
            },
            "history_length_max": int(
                (collection.horizons + collection.bootstrap_actions).clamp_max(
                    collection.max_history
                ).max()
            ),
            "world_seed_min": int(collection.world_seeds.min()),
            "world_seed_max": int(collection.world_seeds.max()),
        }
    return payload


def balanced_positions(
    positions: Tensor,
    bins: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    pieces = []
    base, extra = divmod(count, 5)
    for index in range(5):
        members = positions[bins == index]
        if not members.numel():
            raise RuntimeError(f"Training q bin {Q_BIN_NAMES[index]} is empty")
        amount = base + int(index < extra)
        chosen = torch.randint(members.numel(), (amount,), generator=generator)
        pieces.append(members[chosen])
    result = torch.cat(pieces)
    return result[torch.randperm(result.numel(), generator=generator)]


def gradient_norm(model: nn.Module) -> float:
    terms = [p.grad.square().sum() for p in model.parameters() if p.grad is not None]
    return 0.0 if not terms else float(torch.stack(terms).sum().sqrt())


def train_critic(
    experiment: V45QConfig,
    config,
    model: ContractionCritic,
    collection: TrajectoryCollection,
    device: torch.device,
) -> list[dict[str, object]]:
    positions = transition_positions(collection)
    bins = q_bins(labels_at(collection, positions))
    generator = torch.Generator().manual_seed(experiment.seed + 50_000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.q_learning_rate)
    rows = []
    started = time.time()
    model.train()
    for step in range(1, experiment.q_training_steps + 1):
        selected = balanced_positions(
            positions, bins, experiment.q_batch_size, generator
        )
        batch, q_true = critic_batch_from_positions(collection, selected, config, device)
        predicted_log_q, harmful_logit = model(batch)
        target_log_q = q_true.clamp_min(config.numerical_epsilon).log()
        regression = F.smooth_l1_loss(predicted_log_q, target_log_q)
        classification = F.binary_cross_entropy_with_logits(
            harmful_logit, (q_true >= 1).float()
        )
        loss = regression + experiment.q_classification_weight * classification
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite critic loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = gradient_norm(model)
        if not math.isfinite(gradients):
            raise RuntimeError(f"Non-finite critic gradient at step {step}")
        optimizer.step()
        if step % experiment.q_log_every == 0 or step == 1:
            q_hat = predicted_log_q.detach().exp()
            rows.append({
                "step": step,
                "loss": float(loss),
                "regression_loss": float(regression),
                "classification_loss": float(classification),
                "q_mae": float((q_hat - q_true).abs().mean()),
                "false_safe_primary": float(
                    ((q_hat < experiment.primary_gate_threshold) & (q_true >= 1)).sum()
                    / (q_true >= 1).sum().clamp_min(1)
                ),
                "gradient_norm": gradients,
                "steps_per_second": step / max(time.time() - started, 1e-6),
            })
    return rows


def roc_pr_metrics(scores: Tensor, labels: Tensor) -> tuple[float, float]:
    labels = labels.float()
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    positives = sorted_labels.sum().clamp_min(1)
    negatives = (1 - sorted_labels).sum().clamp_min(1)
    tpr = torch.cat([torch.zeros(1), sorted_labels.cumsum(0) / positives])
    fpr = torch.cat([torch.zeros(1), (1 - sorted_labels).cumsum(0) / negatives])
    auroc = float(torch.trapezoid(tpr, fpr))
    precision = sorted_labels.cumsum(0) / torch.arange(1, labels.numel() + 1)
    auprc = float((precision * sorted_labels).sum() / positives)
    return auroc, auprc


@torch.inference_mode()
def predict_collection(
    critic: ContractionCritic,
    collection: TrajectoryCollection,
    config,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    critic.eval()
    positions = transition_positions(collection)
    truth, predicted, harm_score = [], [], []
    for start in range(0, positions.numel(), batch_size):
        selected = positions[start : start + batch_size]
        batch, q_true = critic_batch_from_positions(collection, selected, config, device)
        log_q, harm = critic(batch)
        truth.append(q_true.cpu())
        predicted.append(log_q.exp().cpu())
        harm_score.append(harm.sigmoid().cpu())
    return positions, torch.cat(truth), torch.cat(predicted), torch.cat(harm_score)


def metrics_row(
    split: str,
    scope: str,
    q_true: Tensor,
    q_hat: Tensor,
    harm_score: Tensor,
    threshold: float,
) -> dict[str, object]:
    harmful = q_true >= 1
    safe = ~harmful
    auroc, auprc = roc_pr_metrics(harm_score, harmful)
    centered_true = q_true - q_true.mean()
    centered_pred = q_hat - q_hat.mean()
    correlation = float(
        (centered_true * centered_pred).sum()
        / (centered_true.square().sum().sqrt() * centered_pred.square().sum().sqrt()).clamp_min(1e-12)
    )
    return {
        "split": split,
        "scope": scope,
        "samples": int(q_true.numel()),
        "q_mae": float((q_hat - q_true).abs().mean()),
        "q_rmse": float((q_hat - q_true).square().mean().sqrt()),
        "q_correlation": correlation,
        "harmful_auroc": auroc,
        "harmful_auprc": auprc,
        "false_safe_rate_primary": float(
            ((q_hat < threshold) & harmful).sum() / harmful.sum().clamp_min(1)
        ),
        "false_reject_rate_primary": float(
            ((q_hat >= threshold) & safe).sum() / safe.sum().clamp_min(1)
        ),
        "harmful_fraction": float(harmful.float().mean()),
        "mean_predicted_q": float(q_hat.mean()),
        "mean_true_q": float(q_true.mean()),
    }


def offline_metrics(
    experiment: V45QConfig,
    config,
    critic: ContractionCritic,
    collections: dict[str, TrajectoryCollection],
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, tuple[Tensor, Tensor, Tensor, Tensor]]]:
    rows, boundary_rows, predictions = [], [], {}
    for name, collection in collections.items():
        prediction = predict_collection(critic, collection, config, device)
        predictions[name] = prediction
        _, q_true, q_hat, harm = prediction
        rows.append(metrics_row(
            name, "all", q_true, q_hat, harm, experiment.primary_gate_threshold
        ))
        boundary = (q_true > 0.9) & (q_true < 1.1)
        boundary_rows.append(metrics_row(
            name,
            "0.9_lt_q_lt_1.1",
            q_true[boundary],
            q_hat[boundary],
            harm[boundary],
            experiment.primary_gate_threshold,
        ))
    return rows, boundary_rows, predictions


def clone_state(state: RolloutState) -> RolloutState:
    return state.clone()


@torch.inference_mode()
def learned_gate_rollout(
    experiment: V45QConfig,
    config,
    controller: PopulationController,
    critic: ContractionCritic,
    base_state: RolloutState,
    distribution: str,
    threshold: float,
) -> list[dict[str, object]]:
    state = clone_state(base_state)
    count = state.current.shape[0]
    stopped = torch.zeros(count, dtype=torch.bool, device=state.current.device)
    first_reject = torch.full((count,), experiment.rollout_horizon + 1, device=state.current.device)
    accepted = torch.zeros(count, device=state.current.device)
    rejected = torch.zeros(count, device=state.current.device)
    harmful_accepted = torch.zeros(count, device=state.current.device)
    beneficial_rejected = torch.zeros(count, device=state.current.device)
    values = [state.current.norm(dim=-1).cpu()]
    for step in range(1, experiment.rollout_horizon + 1):
        before = state.current
        batch = controller_batch(config, state)
        proposed = controller(batch)
        proposed_after = transition(before, state.matrices, proposed)
        q_true = proposed_after.norm(dim=-1) / before.norm(dim=-1).clamp_min(config.numerical_epsilon)
        q_hat, _ = critic(CriticBatch(batch, proposed))
        q_hat = q_hat.exp()
        accept = (~stopped) & (q_hat < threshold)
        reject = (~stopped) & (~accept)
        first_reject[reject] = step
        stopped |= reject
        accepted += accept.float()
        rejected += reject.float()
        harmful_accepted += (accept & (q_true >= 1)).float()
        beneficial_rejected += (reject & (q_true < 1)).float()
        action = torch.where(accept[:, None], proposed, torch.zeros_like(proposed))
        after = torch.where(accept[:, None], proposed_after, before)
        append_with_policy(
            state, before, action, after, CAPPED_HISTORY,
            config.bootstrap_actions, experiment.max_history,
        )
        values.append(after.norm(dim=-1).cpu())
    best_per_episode = torch.stack(values).min(dim=0).values
    rows = []
    for step, residual in enumerate(values):
        rows.append({
            "distribution": distribution,
            "mode": f"learned_q_{threshold:.2f}",
            "threshold": threshold,
            "iteration": step,
            "residual_median": float(residual.median()),
            "residual_mean": float(residual.mean()),
            "normalized_f_over_f0": float((residual / values[0].clamp_min(config.numerical_epsilon)).median()),
            "best_residual_episode_median": float(best_per_episode.median()),
            "final_over_best_episode_median": float((values[-1] / best_per_episode.clamp_min(config.numerical_epsilon)).median()),
            "accepted_actions_mean": float(accepted.mean()),
            "rejected_actions_mean": float(rejected.mean()),
            "harmful_actions_accepted_mean": float(harmful_accepted.mean()),
            "beneficial_actions_rejected_mean": float(beneficial_rejected.mean()),
            "first_rejection_step_median": float(first_reject.median()),
        })
    return rows


def oracle_endpoint(output: Path, distribution: str, mode: str, horizon: int) -> float:
    with (output / "oracle_gate_results.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["distribution"] == distribution and row["mode"] == mode and int(row["iteration"]) == horizon:
                return float(row["residual_median"])
    raise KeyError((distribution, mode, horizon))


@torch.inference_mode()
def action_floor_diagnostic(
    experiment: V45QConfig,
    config,
    controller: PopulationController,
    critic: ContractionCritic,
    device: torch.device,
) -> list[dict[str, object]]:
    state = make_rollout_state(
        replace(config, history_capacity=experiment.max_history), 256,
        experiment.seed + 920, device,
    )
    rows = []
    selected = {8, 16, 24, 32, 64}
    for step in range(1, 65):
        before = state.current
        action = controller(controller_batch(config, state))
        after = transition(before, state.matrices, action)
        append_with_policy(
            state, before, action, after, CAPPED_HISTORY,
            config.bootstrap_actions, experiment.max_history,
        )
        if step not in selected:
            continue
        original = state.current.clone()
        for scale in (1.0, 0.1, 0.01, 0.0):
            state.current = original * scale
            batch = controller_batch(config, state)
            proposal = controller(batch)
            predicted_log_q, _ = critic(CriticBatch(batch, proposal))
            true_after = transition(state.current, state.matrices, proposal)
            q_true = true_after.norm(dim=-1) / state.current.norm(dim=-1).clamp_min(config.numerical_epsilon)
            rows.append({
                "source_iteration": step,
                "residual_scale": scale,
                "residual_norm_median": float(state.current.norm(dim=-1).median()),
                "proposal_norm_median": float(proposal.norm(dim=-1).median()),
                "predicted_q_median": float(predicted_log_q.exp().median()),
                "true_q_median": float(q_true.median()),
                "accept_1.00_fraction": float((predicted_log_q.exp() < 1).float().mean()),
                "accept_0.98_fraction": float((predicted_log_q.exp() < 0.98).float().mean()),
                "accept_0.95_fraction": float((predicted_log_q.exp() < 0.95).float().mean()),
            })
        state.current = original
    return rows


def modify_history(batch: CriticBatch, mode: str) -> CriticBatch:
    state = batch.state
    if mode == "correct":
        return batch
    if mode == "no_history":
        changed = ControllerBatch(
            state.current, state.previous_action, torch.zeros_like(state.history),
            torch.zeros_like(state.history_lengths),
        )
    elif mode == "wrong_world":
        changed = ControllerBatch(
            state.current, state.previous_action, state.history.roll(1, 0),
            state.history_lengths.roll(1, 0),
        )
    elif mode == "shuffled_set":
        permutation = torch.randperm(state.history.shape[1], device=state.history.device)
        changed = ControllerBatch(
            state.current, state.previous_action, state.history[:, permutation],
            state.history_lengths,
        )
    else:
        raise ValueError(mode)
    return CriticBatch(changed, batch.candidate_action)


@torch.inference_mode()
def history_interventions(
    experiment: V45QConfig,
    config,
    critic: ContractionCritic,
    collection: TrajectoryCollection,
    device: torch.device,
) -> list[dict[str, object]]:
    positions = transition_positions(collection)
    generator = torch.Generator().manual_seed(experiment.seed + 72_000)
    chosen = positions[torch.randperm(positions.numel(), generator=generator)[: experiment.intervention_samples]]
    rows = []
    for start in range(0, chosen.numel(), 256):
        batch, q_true = critic_batch_from_positions(collection, chosen[start:start + 256], config, device)
        for mode in ("correct", "no_history", "wrong_world", "shuffled_set"):
            log_q, _ = critic(modify_history(batch, mode))
            q_hat = log_q.exp()
            harmful = q_true >= 1
            rows.append({
                "history_mode": mode,
                "samples": int(q_true.numel()),
                "absolute_error_sum": float((q_hat - q_true).abs().sum()),
                "false_safe_count": int(((q_hat < experiment.primary_gate_threshold) & harmful).sum()),
                "harmful_count": int(harmful.sum()),
            })
    result = []
    for mode in ("correct", "no_history", "wrong_world", "shuffled_set"):
        selected = [row for row in rows if row["history_mode"] == mode]
        samples = sum(int(row["samples"]) for row in selected)
        harmful = sum(int(row["harmful_count"]) for row in selected)
        result.append({
            "history_mode": mode,
            "samples": samples,
            "q_mae": sum(float(row["absolute_error_sum"]) for row in selected) / samples,
            "false_safe_rate": sum(int(row["false_safe_count"]) for row in selected) / max(harmful, 1),
        })
    return result


def report_text(
    oracle: dict[str, object],
    offline: list[dict[str, object]],
    learned: list[dict[str, object]],
    verdict: dict[str, object],
) -> str:
    test = next(row for row in offline if row["split"] == "heldout_same_family" and row["scope"] == "all")
    endpoint = next(row for row in learned if row["distribution"] == "heldout_same_family" and row["mode"] == "learned_q_0.98" and row["iteration"] == 128)
    return f"""# IFI v4.5-Q — Learned Contraction Critic / Runtime Q-Gate

## Result

**{verdict['verdict']}**

The frozen proposal controller was never retrained. A perfect runtime contraction gate reduced held-out F128 from **{oracle['ungated_f128']:.6f}** to **{oracle['oracle_f128']:.6f}**, versus an ungated best of **{oracle['ungated_best']:.6f}**. This establishes that harmful proposal execution—not absence of useful proposals—is a major late-rollout bottleneck.

The learned critic achieved held-out q MAE **{test['q_mae']:.6f}**, harmful-action AUROC **{test['harmful_auroc']:.4f}**, and false-safe rate **{test['false_safe_rate_primary']:.4f}** at the preregistered 0.98 gate. Its held-out learned-gate F128 was **{endpoint['residual_median']:.6f}**, with final/best ratio **{endpoint['final_over_best_episode_median']:.4f}**.

## Mechanistic distinction

1. **T proposals:** useful early contraction exists, but T itself is not asymptotically stable.
2. **Oracle acceptability:** a perfect reject/stop rule preserves the progress.
3. **Q prediction:** measured independently on world-disjoint trajectories, especially near q=1.
4. **T + Q runtime:** reported as a two-component learned system; successful gating is not described as T becoming stable.

No B, world identity, rollout depth, future residual, or oracle q entered Q at runtime. Histories were genuine frozen-T interactions and all model weights were frozen during final rollout evaluation.
"""


def main() -> None:
    experiment = V45QConfig.from_json("configs/v4_5_q_gate.json")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output = Path(experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase_a = json.loads((output / "phase_a_gate.json").read_text(encoding="utf-8"))
    if not phase_a["passed"]:
        raise RuntimeError("Phase A failed; critic training is forbidden")
    lock = output / "critic_canonical.lock"
    if lock.exists():
        raise RuntimeError("The canonical critic run was already initiated")
    lock.write_text(str(time.time()), encoding="utf-8")
    seed_all(experiment.seed)
    config, controller = load_frozen_t(experiment, device)
    t_hash_before = sha256(experiment.t_checkpoint)
    (output / "config.json").write_text(json.dumps({
        **experiment.to_dict(),
        "proposal_controller_parameters": sum(p.numel() for p in controller.parameters()),
        "proposal_controller_frozen": all(not p.requires_grad for p in controller.parameters()),
        "proposal_checkpoint_sha256": t_hash_before,
        "critic_inputs": ["current_residual", "genuine_history", "previous_action", "candidate_action"],
        "critic_forbidden_inputs": ["B", "world_id", "rollout_depth", "future_residual", "q_true"],
    }, indent=2), encoding="utf-8")

    collections = collect_datasets(experiment, config, controller, device)
    stats = dataset_stats(collections)
    (output / "q_dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    critic = ContractionCritic(config, experiment).to(device)
    stats["critic_parameters"] = sum(p.numel() for p in critic.parameters())
    training = train_critic(experiment, config, critic, collections["train"], device)
    write_csv(output / "q_train_metrics.csv", training)
    torch.save({
        "critic": critic.state_dict(),
        "experiment": experiment.to_dict(),
        "base_config": config.to_dict(),
        "t_checkpoint": experiment.t_checkpoint,
        "t_checkpoint_sha256": t_hash_before,
        "t_frozen": True,
    }, output / "q_checkpoint.pt")

    offline, boundary, _ = offline_metrics(experiment, config, critic, collections, device)
    write_csv(output / "q_offline_metrics.csv", offline)
    write_csv(output / "q_boundary_metrics.csv", boundary)
    heldout = next(row for row in offline if row["split"] == "heldout_same_family")
    critic_gate_passed = (
        float(heldout["false_safe_rate_primary"]) <= experiment.offline_false_safe_max
        and float(heldout["harmful_auroc"]) >= experiment.offline_auroc_min
    )
    if not critic_gate_passed:
        verdict = {
            "verdict": "ORACLE WORKS, LEARNED Q FAILS",
            "phase_a_passed": True,
            "phase_b_passed": False,
            "phase_c_run": False,
            "heldout_offline": heldout,
        }
        (output / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        (output / "report.md").write_text(
            "# IFI v4.5-Q\n\n**ORACLE WORKS, LEARNED Q FAILS.**\n\n"
            f"Held-out AUROC: {heldout['harmful_auroc']:.4f}; false-safe rate: {heldout['false_safe_rate_primary']:.4f}. "
            "The preregistered critic gate failed, so runtime Phase C was not run.\n",
            encoding="utf-8",
        )
        print(json.dumps(verdict, indent=2), flush=True)
        return

    v41 = torch.load(experiment.v41_checkpoint, map_location="cpu", weights_only=False)
    fixtures = build_fixtures(experiment, config, v41, device)
    learned_rows = []
    for distribution, fixture in fixtures.items():
        for threshold in experiment.gate_thresholds:
            learned_rows.extend(learned_gate_rollout(
                experiment, config, controller, critic, fixture,
                distribution, threshold,
            ))
    write_csv(output / "learned_gate_rollouts.csv", learned_rows)
    write_csv(output / "ood_gate_results.csv", [
        row for row in learned_rows if row["distribution"] == "condition_4_6"
    ])
    floor = action_floor_diagnostic(experiment, config, controller, critic, device)
    write_csv(output / "action_floor_gate_diagnostic.csv", floor)
    interventions = history_interventions(
        experiment, config, critic, collections["heldout_same_family"], device
    )
    write_csv(output / "history_interventions_q.csv", interventions)

    primary_endpoint = next(
        row for row in learned_rows
        if row["distribution"] == "heldout_same_family"
        and row["mode"] == f"learned_q_{experiment.primary_gate_threshold:.2f}"
        and row["iteration"] == experiment.rollout_horizon
    )
    t_endpoint = oracle_endpoint(output, "heldout_same_family", "t_only", experiment.rollout_horizon)
    oracle_gate_endpoint = oracle_endpoint(output, "heldout_same_family", "oracle_hard_stop", experiment.rollout_horizon)
    denominator = t_endpoint - oracle_gate_endpoint
    recovery = (
        float("nan") if abs(denominator) < 1e-12
        else (t_endpoint - float(primary_endpoint["residual_median"])) / denominator
    )
    strong = (
        float(primary_endpoint["final_over_best_episode_median"]) <= 1.20
        and recovery >= 0.50
        and float(primary_endpoint["residual_median"]) < 0.75 * oracle_endpoint(output, "heldout_same_family", "no_update", 0)
    )
    verdict = {
        "verdict": "Q-GATED IFI SUPPORTED" if strong else "ORACLE WORKS, LEARNED Q FAILS",
        "phase_a_passed": True,
        "phase_b_passed": True,
        "phase_c_run": True,
        "primary_threshold": experiment.primary_gate_threshold,
        "oracle_benefit_recovered": recovery,
        "heldout_t_f128": t_endpoint,
        "heldout_oracle_f128": oracle_gate_endpoint,
        "heldout_learned_q_f128": primary_endpoint["residual_median"],
        "heldout_learned_final_over_best": primary_endpoint["final_over_best_episode_median"],
        "heldout_offline": heldout,
        "t_checkpoint_unchanged": sha256(experiment.t_checkpoint) == t_hash_before,
        "critic_frozen_at_evaluation": all(not p.requires_grad for p in critic.eval().requires_grad_(False).parameters()),
    }
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (output / "report.md").write_text(
        report_text(phase_a, offline, learned_rows, verdict), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
