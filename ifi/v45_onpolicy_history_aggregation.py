from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .v40_cli import hardware
from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_models import ControllerBatch, PopulationController
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v44_evaluation import ALL_HISTORY, FIFO_16, fresh_state_from_matrices, rollout_summary
from .v45_config import V45Config
from .v45_replay import (
    TrajectoryCollection,
    collect_current_policy,
    mixed_replay_batch,
)


CAPPED_ALL_128 = HistoryPolicy("all_available_capped_128", 128, False)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def gradient_norm(model: nn.Module) -> float:
    values = [
        parameter.grad.square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def collection_distribution(
    collection: TrajectoryCollection,
    cycle: int,
) -> list[dict[str, object]]:
    rows = []
    for horizon in sorted(collection.horizons.unique().tolist()):
        episodes = torch.where(collection.horizons == horizon)[0]
        for depth in (0, 8, 16, 24, 32, 48, 64):
            if depth > horizon:
                continue
            residual = collection.residuals[episodes, depth].norm(dim=-1)
            rows.append({
                "cycle": cycle,
                "policy_version": collection.policy_version,
                "collection_horizon": horizon,
                "state_depth": depth,
                "episode_count": int(episodes.numel()),
                "residual_median": float(residual.median()),
                "residual_mean": float(residual.mean()),
                "residual_q10": float(torch.quantile(residual, 0.10)),
                "residual_q90": float(torch.quantile(residual, 0.90)),
                "history_length": collection.bootstrap_actions + depth,
            })
    return rows


@torch.inference_mode()
def diagnostic_rollout(
    experiment: V45Config,
    config: V40Config,
    model: PopulationController,
    cycle: int,
    device: torch.device,
) -> list[dict[str, object]]:
    selected = {0, 8, 16, 24, 32, 64}
    values: dict[int, dict[str, list[Tensor]]] = {
        step: defaultdict(list) for step in selected
    }
    storage = replace(config, history_capacity=experiment.max_history)
    was_training = model.training
    model.eval()
    for start in range(0, experiment.diagnostic_episodes, config.evaluation_batch_size):
        count = min(
            config.evaluation_batch_size,
            experiment.diagnostic_episodes - start,
        )
        state = make_rollout_state(
            storage,
            count,
            experiment.seed + 5500 + start,
            device,
        )
        values[0]["residual"].append(state.current.norm(dim=-1).cpu())
        for step in range(1, experiment.diagnostic_horizon + 1):
            before = state.current
            batch = controller_batch(config, state)
            action = model(batch)
            after = transition(before, state.matrices, action)
            if step in selected:
                residual = after.norm(dim=-1)
                before_norm = before.norm(dim=-1)
                delta = after - before
                alignment = (-before * delta).sum(dim=-1) / (
                    before_norm * delta.norm(dim=-1)
                ).clamp_min(config.numerical_epsilon)
                zero_batch = ControllerBatch(
                    current=torch.zeros_like(batch.current),
                    previous_action=batch.previous_action,
                    history=batch.history,
                    history_lengths=batch.history_lengths,
                )
                values[step]["residual"].append(residual.cpu())
                values[step]["q"].append(
                    (residual / before_norm.clamp_min(config.numerical_epsilon)).cpu()
                )
                values[step]["action"].append(action.norm(dim=-1).cpu())
                values[step]["ratio"].append(
                    (
                        action.norm(dim=-1)
                        / before_norm.clamp_min(config.numerical_epsilon)
                    ).cpu()
                )
                values[step]["alignment"].append(alignment.cpu())
                values[step]["zero_action"].append(
                    model(zero_batch).norm(dim=-1).cpu()
                )
            append_with_policy(
                state,
                before,
                action,
                after,
                CAPPED_ALL_128,
                config.bootstrap_actions,
                experiment.max_history,
            )
    if was_training:
        model.train()
    rows = []
    for step in sorted(selected):
        residual = torch.cat(values[step]["residual"])
        q = torch.cat(values[step]["q"]) if step else None
        action = torch.cat(values[step]["action"]) if step else None
        alignment = torch.cat(values[step]["alignment"]) if step else None
        zero_action = torch.cat(values[step]["zero_action"]) if step else None
        rows.append({
            "cycle": cycle,
            "iteration": step,
            "residual_median": float(residual.median()),
            "residual_mean": float(residual.mean()),
            "q_median": "" if q is None else float(q.median()),
            "regression_fraction": "" if q is None else float((q > 1).float().mean()),
            "action_norm_median": "" if action is None else float(action.median()),
            "action_residual_ratio_median": (
                ""
                if step == 0
                else float(torch.cat(values[step]["ratio"]).median())
            ),
            "alignment_median": "" if alignment is None else float(alignment.median()),
            "zero_current_action_median": (
                "" if zero_action is None else float(zero_action.median())
            ),
        })
    return rows


def train_cycle(
    experiment: V45Config,
    config: V40Config,
    model: PopulationController,
    optimizer: torch.optim.Optimizer,
    latest: TrajectoryCollection,
    older: list[TrajectoryCollection],
    anchor_population: dict[str, object],
    cycle: int,
    generator: torch.Generator,
    action_generator: torch.Generator,
    device: torch.device,
) -> list[dict[str, object]]:
    model.train()
    rows = []
    for local_step in range(1, experiment.steps_per_cycle + 1):
        batch, matrices, metadata = mixed_replay_batch(
            experiment,
            config,
            latest,
            older,
            anchor_population,
            generator,
            device,
        )
        action = model(batch)
        exploration = config.exploration_std * torch.randn(
            action.shape,
            generator=action_generator,
            device=device,
        )
        executed = (action + exploration).clamp(-config.action_max, config.action_max)
        after = transition(batch.current, matrices, executed)
        before_norm = batch.current.norm(dim=-1)
        after_norm = after.norm(dim=-1)
        q = after_norm / before_norm.clamp_min(config.numerical_epsilon)
        progress = F.softplus(q - config.progress_tau)
        safety = F.relu(after.square().sum(-1) - batch.current.square().sum(-1))
        per_example = progress + experiment.lambda_safe * safety
        loss = per_example.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite v4.5 loss in cycle {cycle}, step {local_step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = gradient_norm(model)
        if not math.isfinite(gradients):
            raise RuntimeError("Non-finite v4.5 gradient")
        optimizer.step()
        if local_step % experiment.log_every == 0:
            for source_id, source_name in ((0, "latest"), (1, "older"), (2, "anchor")):
                mask = metadata["source"].to(device) == source_id
                rows.append({
                    "cycle": cycle,
                    "local_step": local_step,
                    "global_posttrain_step": (
                        (cycle - 1) * experiment.steps_per_cycle + local_step
                    ),
                    "source": source_name,
                    "loss": float(per_example.detach()[mask].mean()),
                    "q_median": float(q.detach()[mask].median()),
                    "regression_fraction": float((q.detach()[mask] > 1).float().mean()),
                    "residual_norm_median": float(before_norm.detach()[mask].median()),
                    "action_norm_median": float(
                        action.detach().norm(dim=-1)[mask].median()
                    ),
                    "history_length_median": float(
                        metadata["history_lengths"][mask.cpu()].float().median()
                    ),
                    "depth_median_logging_only": float(
                        metadata["depth"][mask.cpu()].float().median()
                    ),
                    "policy_version_median_logging_only": float(
                        metadata["policy_version"][mask.cpu()].float().median()
                    ),
                    "gradient_norm": gradients,
                })
    return rows


def clone_state(state: RolloutState) -> RolloutState:
    return RolloutState(**{name: value.clone() for name, value in state.__dict__.items()})


@torch.inference_mode()
def history_interventions(
    experiment: V45Config,
    config: V40Config,
    model: PopulationController,
    device: torch.device,
) -> list[dict[str, object]]:
    records: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    storage = replace(config, history_capacity=experiment.max_history)
    for start in range(0, experiment.intervention_episodes, config.evaluation_batch_size):
        count = min(
            config.evaluation_batch_size,
            experiment.intervention_episodes - start,
        )
        state = make_rollout_state(storage, count, experiment.seed + 7500 + start, device)
        wrong = make_rollout_state(storage, count, experiment.seed + 8500 + start, device)
        for _ in range(experiment.intervention_at):
            for active in (state, wrong):
                before = active.current
                action = model(controller_batch(config, active))
                after = transition(before, active.matrices, action)
                append_with_policy(
                    active,
                    before,
                    action,
                    after,
                    ALL_HISTORY,
                    config.bootstrap_actions,
                    experiment.max_history,
                )
        initial = state.current.norm(dim=-1)
        for condition in ("correct", "no_history", "wrong_world"):
            active = clone_state(state)
            for step in range(1, experiment.intervention_horizon + 1):
                before = active.current
                batch = controller_batch(config, active)
                if step == 1 and condition == "no_history":
                    batch = ControllerBatch(
                        current=batch.current,
                        previous_action=batch.previous_action,
                        history=batch.history,
                        history_lengths=torch.zeros_like(batch.history_lengths),
                    )
                elif step == 1 and condition == "wrong_world":
                    wrong_batch = controller_batch(config, wrong)
                    batch = ControllerBatch(
                        current=batch.current,
                        previous_action=batch.previous_action,
                        history=wrong_batch.history.roll(1, 0),
                        history_lengths=wrong_batch.history_lengths.roll(1, 0),
                    )
                action = model(batch)
                after = transition(before, active.matrices, action)
                append_with_policy(
                    active,
                    before,
                    action,
                    after,
                    ALL_HISTORY,
                    config.bootstrap_actions,
                    experiment.max_history,
                )
                if step in (1, 16, 32, 64):
                    records[(condition, step)].append(
                        (after.norm(dim=-1) / initial.clamp_min(config.numerical_epsilon)).cpu()
                    )
    return [
        {
            "condition": condition,
            "followup_iteration": step,
            "residual_ratio_median": float(torch.cat(values).median()),
            "residual_ratio_mean": float(torch.cat(values).mean()),
        }
        for (condition, step), values in sorted(records.items())
    ]


@torch.inference_mode()
def alignment_rollout(
    experiment: V45Config,
    config: V40Config,
    model: PopulationController,
    model_label: str,
    device: torch.device,
) -> list[dict[str, object]]:
    accumulators: dict[int, dict[str, list[Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    storage = replace(config, history_capacity=experiment.max_history)
    for start in range(0, experiment.evaluation_episodes, config.evaluation_batch_size):
        count = min(config.evaluation_batch_size, experiment.evaluation_episodes - start)
        state = make_rollout_state(storage, count, experiment.seed + 6500 + start, device)
        for step in range(1, experiment.evaluation_horizon + 1):
            before = state.current
            action = model(controller_batch(config, state))
            after = transition(before, state.matrices, action)
            delta = after - before
            alignment = (-before * delta).sum(dim=-1) / (
                before.norm(dim=-1) * delta.norm(dim=-1)
            ).clamp_min(config.numerical_epsilon)
            accumulators[step]["alignment"].append(alignment.cpu())
            accumulators[step]["q"].append(
                (
                    after.norm(dim=-1)
                    / before.norm(dim=-1).clamp_min(config.numerical_epsilon)
                ).cpu()
            )
            accumulators[step]["action"].append(action.norm(dim=-1).cpu())
            append_with_policy(
                state,
                before,
                action,
                after,
                CAPPED_ALL_128,
                config.bootstrap_actions,
                experiment.max_history,
            )
    return [
        {
            "model": model_label,
            "iteration": step,
            "alignment_median": float(torch.cat(values["alignment"]).median()),
            "q_median": float(torch.cat(values["q"]).median()),
            "regression_fraction": float((torch.cat(values["q"]) > 1).float().mean()),
            "action_norm_median": float(torch.cat(values["action"]).median()),
        }
        for step, values in sorted(accumulators.items())
    ]


def save_cycle_checkpoint(
    output: Path,
    cycle: int,
    model: PopulationController,
    optimizer: torch.optim.Optimizer,
    source: Path,
) -> None:
    payload = {
        "experiment": "IFI v4.5 on-policy history aggregation",
        "cycle": cycle,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "initialized_from": str(source.resolve()),
        "initialized_from_sha256": sha256(source),
    }
    path = output / f"checkpoint_cycle_{cycle}.pt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def window_q(rows: list[dict[str, object]], first: int, last: int) -> float:
    values = [
        float(row["q_median"])
        for row in rows
        if first <= int(row["iteration"]) <= last and row["q_median"] != ""
    ]
    return float(np.median(values))


def write_report(
    output: Path,
    cycle_rows: list[dict[str, object]],
    long_rows: list[dict[str, object]],
) -> dict[str, object]:
    baseline = {
        int(row["iteration"]): row for row in cycle_rows if int(row["cycle"]) == 0
    }
    final_cycle = max(int(row["cycle"]) for row in cycle_rows)
    final = {
        int(row["iteration"]): row
        for row in cycle_rows
        if int(row["cycle"]) == final_cycle
    }
    curve = [
        row
        for row in long_rows
        if row["model"] == "v45_onpolicy"
        and row["fixture"] == "unseen_same_family"
        and row["history_policy"] == "all_available_capped_128"
    ]
    by_step = {int(row["iteration"]): row for row in curve}
    best = min(curve, key=lambda row: float(row["residual_norm_median"]))
    early_preserved = float(final[8]["residual_median"]) / float(final[0]["residual_median"]) <= (
        1.20 * float(baseline[8]["residual_median"]) / float(baseline[0]["residual_median"])
    )
    late_improved = (
        float(final[32]["q_median"]) < float(baseline[32]["q_median"])
        and float(final[32]["alignment_median"]) > float(baseline[32]["alignment_median"])
    )
    q_33_64 = window_q(curve, 33, 64)
    q_65_128 = window_q(curve, 65, 128)
    endpoint_ratio = float(by_step[128]["residual_norm_median"]) / float(
        best["residual_norm_median"]
    )
    stationary = q_33_64 <= 1 and q_65_128 <= 1 and endpoint_ratio <= 1.25
    if stationary and early_preserved:
        verdict = "STATIONARY LEARNING LAW SUPPORTED"
    elif late_improved and int(best["iteration"]) > 24:
        verdict = "MOVING-FRONTIER FAILURE"
    elif late_improved and early_preserved:
        verdict = "ON-POLICY DISTRIBUTION SHIFT SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    result = {
        "verdict": verdict,
        "early_preserved": early_preserved,
        "late_q_alignment_improved": late_improved,
        "stationary": stationary,
        "best_iteration": int(best["iteration"]),
        "best_residual": float(best["residual_norm_median"]),
        "f64_over_best": float(by_step[64]["residual_norm_median"])
        / float(best["residual_norm_median"]),
        "f128_over_best": endpoint_ratio,
        "q_f33_f64": q_33_64,
        "q_f65_f128": q_65_128,
        "baseline_f32_q": float(baseline[32]["q_median"]),
        "final_f32_q": float(final[32]["q_median"]),
        "baseline_f32_alignment": float(baseline[32]["alignment_median"]),
        "final_f32_alignment": float(final[32]["alignment_median"]),
    }
    lines = [
        "# IFI v4.5 — On-Policy History Aggregation",
        "",
        f"## {verdict}",
        "",
        f"- Best residual: **{result['best_residual']:.6g}** at F{result['best_iteration']}.",
        f"- F64/best: **{result['f64_over_best']:.6g}**.",
        f"- F128/best: **{result['f128_over_best']:.6g}**.",
        f"- Median q F33–64: **{q_33_64:.6g}**.",
        f"- Median q F65–128: **{q_65_128:.6g}**.",
        f"- F32 q: **{result['baseline_f32_q']:.6g} → {result['final_f32_q']:.6g}**.",
        f"- F32 alignment: **{result['baseline_f32_alignment']:.6g} → {result['final_f32_alignment']:.6g}**.",
        "",
        "The architecture, worlds, action interface, and one-step objective were unchanged. "
        "Only the residual–action–history training distribution was changed.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(output / "verdict.json", result)
    return result


def validate(experiment: V45Config, base: V40Config) -> None:
    if experiment.latest_batch + experiment.older_batch + experiment.anchor_batch != experiment.batch_size:
        raise ValueError("v4.5 replay mixture must exactly fill the batch")
    if (experiment.latest_batch, experiment.older_batch, experiment.anchor_batch) != (45, 13, 6):
        raise ValueError("Locked v4.5 replay mixture is 45/13/6")
    if experiment.max_history < base.bootstrap_actions + max(experiment.collection_depths):
        raise ValueError("History capacity cannot hold the longest collection trajectory")
    if experiment.lambda_safe != 1.0:
        raise ValueError("v4.5 preserves the v4.4 safety weight")


def main() -> None:
    parser = argparse.ArgumentParser(description="IFI v4.5 on-policy history aggregation")
    parser.add_argument("--config", default="configs/v4_5_onpolicy_history_aggregation.json")
    args = parser.parse_args()
    experiment = V45Config.from_json(args.config)
    base = V40Config.from_json(experiment.base_config)
    validate(experiment, base)
    if not torch.cuda.is_available():
        raise RuntimeError("IFI v4.5 requires CUDA")
    output = Path(experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "canonical_run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    except FileExistsError as error:
        raise RuntimeError("Refusing a duplicate v4.5 canonical run") from error
    device = torch.device("cuda")
    seed_all(experiment.seed)
    source = Path(experiment.initial_checkpoint)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    config = V40Config(**checkpoint["config"])
    model = PopulationController("query_history", config).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = experiment.learning_rate
    v41 = torch.load(experiment.v41_checkpoint, map_location="cpu", weights_only=False)
    anchor_population = v41["populations"]["query_history"]
    atomic_json(output / "config.json", experiment.to_dict())
    atomic_json(output / "hardware.json", hardware(device))
    provenance = {
        "initial_checkpoint": str(source.resolve()),
        "initial_checkpoint_sha256": sha256(source),
        "source_cycle_or_step": int(checkpoint["step"]),
        "parameter_count": parameter_count(model),
        "architecture_unchanged": True,
        "optimizer_restored": True,
        "no_positional_parameters": not any(
            "position" in name.lower() for name, _ in model.named_parameters()
        ),
    }
    atomic_json(output / "provenance.json", provenance)
    generator = torch.Generator().manual_seed(experiment.seed + 4510)
    action_generator = torch.Generator(device=device).manual_seed(experiment.seed + 4511)
    cycle_rows = diagnostic_rollout(experiment, config, model, 0, device)
    write_csv(output / "rollout_diagnostics.csv", cycle_rows)
    baseline = {int(row["iteration"]): row for row in cycle_rows}
    older: list[TrajectoryCollection] = []
    replay_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    latest = None
    started = time.time()
    for cycle in range(1, experiment.cycles + 1):
        atomic_json(output / "status.json", {
            "phase": "collect",
            "cycle": cycle,
            "cycles": experiment.cycles,
        })
        latest = collect_current_policy(experiment, config, model, cycle - 1, device)
        replay_rows.extend(collection_distribution(latest, cycle))
        if latest.state_count < 5000 or latest.history_error() >= 2e-6:
            raise RuntimeError("v4.5 collection integrity gate failed")
        atomic_json(output / "status.json", {
            "phase": "train",
            "cycle": cycle,
            "cycles": experiment.cycles,
            "latest_states": latest.state_count,
        })
        train_rows.extend(
            train_cycle(
                experiment,
                config,
                model,
                optimizer,
                latest,
                older,
                anchor_population,
                cycle,
                generator,
                action_generator,
                device,
            )
        )
        diagnostics = diagnostic_rollout(experiment, config, model, cycle, device)
        cycle_rows.extend(diagnostics)
        write_csv(output / "cycle_metrics.csv", train_rows)
        write_csv(output / "replay_distribution.csv", replay_rows)
        write_csv(output / "rollout_diagnostics.csv", cycle_rows)
        save_cycle_checkpoint(output, cycle, model, optimizer, source)
        current = {int(row["iteration"]): row for row in diagnostics}
        smoke = {
            "cycle": cycle,
            "collection_states": latest.state_count,
            "history_error": latest.history_error(),
            "f8_ratio": float(current[8]["residual_median"])
            / float(current[0]["residual_median"]),
            "f8_retention_limit": experiment.smoke_early_retention_factor
            * float(baseline[8]["residual_median"])
            / float(baseline[0]["residual_median"]),
            "f32_q": float(current[32]["q_median"]),
        }
        smoke["passed"] = (
            smoke["history_error"] < 2e-6
            and smoke["f8_ratio"] <= smoke["f8_retention_limit"]
            and smoke["f32_q"] < experiment.smoke_q_upper_bound
        )
        if cycle == 1:
            atomic_json(output / "smoke_gate.json", smoke)
            if not smoke["passed"]:
                atomic_json(output / "status.json", {
                    "phase": "complete",
                    "stopped_by_smoke": True,
                    "smoke": smoke,
                })
                (output / "report.md").write_text(
                    "# IFI v4.5 — On-Policy History Aggregation\n\n"
                    "## NOT SUPPORTED — SMOKE/RETENTION GATE FAILED\n",
                    encoding="utf-8",
                )
                return
        older.append(latest)
        older = older[-experiment.older_versions_keep :]
        elapsed = time.time() - started
        atomic_json(output / "status.json", {
            "phase": "cycle_complete",
            "cycle": cycle,
            "cycles": experiment.cycles,
            "elapsed_seconds": elapsed,
            "eta_seconds": elapsed / cycle * (experiment.cycles - cycle),
            "diagnostic": current,
        })
        print(json.dumps({
            "phase": "v45_cycle_complete",
            "cycle": cycle,
            "f8_ratio": smoke["f8_ratio"],
            "f32_q": smoke["f32_q"],
            "f32_alignment": current[32]["alignment_median"],
        }), flush=True)
    assert latest is not None
    model.eval().requires_grad_(False)
    storage = replace(config, history_capacity=experiment.max_history)
    unseen = make_rollout_state(
        storage,
        experiment.evaluation_episodes,
        experiment.seed + 600,
        device,
    )
    familiar = fresh_state_from_matrices(
        config,
        anchor_population["matrices"][: experiment.evaluation_episodes],
        anchor_population["world_seeds"][: experiment.evaluation_episodes],
        experiment.max_history,
        experiment.seed + 610,
        device,
    )
    stored_long = latest.deepest_rollout_state(experiment.evaluation_episodes, device)
    harder = make_rollout_state(
        storage,
        experiment.evaluation_episodes,
        experiment.seed + 620,
        device,
        harder=True,
    )
    fixtures = {
        "training_B_fresh_residual": familiar,
        "unseen_same_family": unseen,
        "stored_long_policy_states": stored_long,
        "condition_4_6": harder,
    }
    long_rows = []
    for fixture, state in fixtures.items():
        for policy in (CAPPED_ALL_128, FIFO_16):
            long_rows.extend(
                rollout_summary(
                    experiment,
                    config,
                    model,
                    "v45_onpolicy",
                    experiment.seed,
                    fixture,
                    state,
                    policy,
                )
            )
    v44_model = PopulationController("query_history", config).to(device)
    v44_model.load_state_dict(checkpoint["model"])
    v44_model.eval().requires_grad_(False)
    v41_model = PopulationController("query_history", base).to(device)
    v41_model.load_state_dict(v41["models"]["query_history"])
    v41_model.eval().requires_grad_(False)
    for label, baseline_model, baseline_config in (
        ("v44_large_12k", v44_model, config),
        ("v41_original", v41_model, base),
    ):
        long_rows.extend(
            rollout_summary(
                experiment,
                baseline_config,
                baseline_model,
                label,
                "historical",
                "unseen_same_family",
                unseen,
                CAPPED_ALL_128,
            )
        )
    interventions = history_interventions(experiment, config, model, device)
    alignment = alignment_rollout(
        experiment, config, model, "v45_onpolicy", device
    )
    alignment.extend(
        alignment_rollout(experiment, config, v44_model, "v44_large_12k", device)
    )
    write_csv(output / "long_rollouts.csv", long_rows)
    write_csv(output / "history_interventions.csv", interventions)
    write_csv(output / "action_alignment.csv", alignment)
    result = write_report(output, cycle_rows, long_rows)
    manifest = {
        "complete": True,
        "cycles": experiment.cycles,
        "states_per_collection": latest.state_count,
        "verdict": result,
        "provenance": provenance,
    }
    atomic_json(output / "experiment_manifest.json", manifest)
    atomic_json(output / "status.json", {"phase": "complete", **manifest})
    print(json.dumps({"phase": "v45_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
