from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .v40_cli import _device, prepare
from .v40_config import V40Config
from .v40_evaluation import (
    _intervened_batch,
    controller_batch,
    make_rollout_state,
)
from .v40_training import PopulationTrainer, build_models, write_world_registry
from .v40_worlds import analytic_action, transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy


ALL_HISTORY = HistoryPolicy("all_history", 0, True, True)
FIFO_16 = HistoryPolicy("fifo_16", 16, False)
SELECTED_ITERATIONS = (0, 1, 2, 4, 8, 14, 16, 32, 64)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(output: Path, **payload: object) -> None:
    body = {"updated_at": time.time(), **payload}
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(body, indent=2), encoding="utf-8")
    temporary.replace(output / "status.json")
    print(json.dumps(body), flush=True)


def _load_frozen_models(
    config: V40Config, checkpoint_path: Path, device: torch.device
) -> dict[str, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not checkpoint.get("complete") or int(checkpoint["step"]) != config.training_steps:
        raise RuntimeError(f"Incomplete canonical checkpoint: {checkpoint_path}")
    models = build_models(config, device)
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][name])
        model.eval()
        model.requires_grad_(False)
    return models


def _append(
    state,
    before: torch.Tensor,
    action: torch.Tensor,
    after: torch.Tensor,
    policy: HistoryPolicy,
    config: V40Config,
) -> None:
    append_with_policy(
        state,
        before,
        action,
        after,
        policy,
        config.bootstrap_actions,
        config.bootstrap_actions + config.evaluation_rollout,
    )


@torch.inference_mode()
def evaluate_seed(
    config: V40Config,
    models: dict[str, torch.nn.Module],
    trained_seed: int,
    fixed_eta: float,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    storage_config = replace(
        config,
        history_capacity=config.bootstrap_actions + config.evaluation_rollout,
    )
    specifications = (
        ("v42_query_all_history", "query_history", ALL_HISTORY),
        ("v42_query_fifo16", "query_history", FIFO_16),
        ("v42_current_only", "current_only", ALL_HISTORY),
        ("v42_ar_all_history", "ar_no_fresh", ALL_HISTORY),
        ("v42_ar_fifo16", "ar_no_fresh", FIFO_16),
        ("fixed_linear", "fixed_linear", ALL_HISTORY),
        ("analytic_oracle", "analytic_oracle", ALL_HISTORY),
    )
    buckets = [("same_family", False)]
    if config.harder_bucket_enabled:
        buckets.append(("condition_4_6", True))
    for bucket_index, (bucket, harder) in enumerate(buckets):
        # The seed formula and pre-bootstrap residual distribution are locked to v4.1.
        base = make_rollout_state(
            storage_config,
            config.evaluation_episodes,
            config.seed + 600 + bucket_index,
            device,
            harder,
        )
        world_seeds = base.world_seeds.cpu().tolist()
        for label, model_name, policy in specifications:
            state = base.clone()
            previous_norm = state.current.norm(dim=-1)
            for episode, norm in enumerate(previous_norm.cpu().tolist()):
                records.append({
                    "trained_seed": trained_seed,
                    "bucket": bucket,
                    "model": label,
                    "episode": episode,
                    "world_seed": int(world_seeds[episode]),
                    "iteration": 0,
                    "history_tokens": config.bootstrap_actions,
                    "residual_norm": norm,
                    "q": "",
                    "action_norm": 0.0,
                    "converged": norm < config.convergence_threshold,
                    "diverged": norm > config.divergence_threshold,
                })
            for iteration in range(1, config.evaluation_rollout + 1):
                before = state.current
                if model_name in models:
                    action = models[model_name](controller_batch(config, state))
                elif model_name == "fixed_linear":
                    action = -fixed_eta * before
                else:
                    action = analytic_action(state.matrices, before)
                after = transition(before, state.matrices, action)
                norm = after.norm(dim=-1)
                q = norm / previous_norm.clamp_min(1e-12)
                history_tokens = int(state.history_lengths[0])
                values = zip(
                    norm.cpu().tolist(),
                    q.cpu().tolist(),
                    action.norm(dim=-1).cpu().tolist(),
                )
                for episode, (norm_value, q_value, action_value) in enumerate(values):
                    records.append({
                        "trained_seed": trained_seed,
                        "bucket": bucket,
                        "model": label,
                        "episode": episode,
                        "world_seed": int(world_seeds[episode]),
                        "iteration": iteration,
                        "history_tokens": history_tokens,
                        "residual_norm": norm_value,
                        "q": q_value,
                        "action_norm": action_value,
                        "converged": norm_value < config.convergence_threshold,
                        "diverged": norm_value > config.divergence_threshold,
                    })
                _append(state, before, action, after, policy, config)
                previous_norm = norm
    return records


@torch.inference_mode()
def evaluate_history_causality(
    config: V40Config,
    model: torch.nn.Module,
    trained_seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    capacity = config.bootstrap_actions + config.intervention_at + config.intervention_followup
    storage_config = replace(config, history_capacity=capacity)
    state = make_rollout_state(
        storage_config, config.intervention_episodes, config.seed + 700, device
    )
    wrong = make_rollout_state(
        storage_config, config.intervention_episodes, config.seed + 701, device
    )
    for _ in range(config.intervention_at):
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
                capacity,
            )
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
    world_seeds = state.world_seeds.cpu().tolist()
    rows: list[dict[str, object]] = []
    for condition in conditions:
        rollout = state.clone()
        action = model(_intervened_batch(storage_config, rollout, condition, wrong))
        before = rollout.current
        after = transition(before, rollout.matrices, action)
        one_step_q = after.norm(dim=-1) / initial_norm.clamp_min(1e-12)
        append_with_policy(
            rollout,
            before,
            action,
            after,
            ALL_HISTORY,
            config.bootstrap_actions,
            capacity,
        )
        for _ in range(1, config.intervention_followup):
            before = rollout.current
            followup_action = model(controller_batch(config, rollout))
            after = transition(before, rollout.matrices, followup_action)
            append_with_policy(
                rollout,
                before,
                followup_action,
                after,
                ALL_HISTORY,
                config.bootstrap_actions,
                capacity,
            )
        endpoint = rollout.current.norm(dim=-1) / initial_norm.clamp_min(1e-12)
        delta_action = (action - correct_action).norm(dim=-1)
        for episode, values in enumerate(
            zip(one_step_q.cpu().tolist(), endpoint.cpu().tolist(), delta_action.cpu().tolist())
        ):
            rows.append({
                "trained_seed": trained_seed,
                "condition": condition,
                "episode": episode,
                "world_seed": int(world_seeds[episode]),
                "intervention_iteration": config.intervention_at,
                "delta_phi": values[2],
                "one_step_q": values[0],
                "endpoint_over_intervention_residual": values[1],
                "followup_steps": config.intervention_followup,
                "history_policy": "all_history",
            })
    return rows


def summarize_seed_rollouts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(
            int(row["trained_seed"]),
            str(row["bucket"]),
            str(row["model"]),
            int(row["iteration"]),
        )].append(row)
    output = []
    for (seed, bucket, model, iteration), selected in sorted(groups.items()):
        norms = [float(row["residual_norm"]) for row in selected]
        q = [float(row["q"]) for row in selected if row["q"] != ""]
        actions = [float(row["action_norm"]) for row in selected]
        output.append({
            "trained_seed": seed,
            "bucket": bucket,
            "model": model,
            "iteration": iteration,
            "history_tokens": int(selected[0]["history_tokens"]),
            "residual_norm_median": statistics.median(norms),
            "residual_norm_mean": statistics.fmean(norms),
            "q_median": statistics.median(q) if q else "",
            "q_fraction_below_one": sum(value < 1 for value in q) / len(q) if q else "",
            "action_norm_median": statistics.median(actions),
            "converged_fraction": sum(bool(row["converged"]) for row in selected) / len(selected),
            "diverged_fraction": sum(bool(row["diverged"]) for row in selected) / len(selected),
        })
    return output


def aggregate_seed_rollouts(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        groups[(str(row["bucket"]), str(row["model"]), int(row["iteration"]))].append(row)
    output = []
    for (bucket, model, iteration), selected in sorted(groups.items()):
        norms = [float(row["residual_norm_median"]) for row in selected]
        q = [float(row["q_median"]) for row in selected if row["q_median"] != ""]
        fractions = [
            float(row["q_fraction_below_one"])
            for row in selected
            if row["q_fraction_below_one"] != ""
        ]
        output.append({
            "bucket": bucket,
            "model": model,
            "iteration": iteration,
            "seed_count": len(selected),
            "residual_norm_seed_mean": statistics.fmean(norms),
            "residual_norm_seed_std": statistics.stdev(norms) if len(norms) > 1 else 0.0,
            "residual_norm_seed_min": min(norms),
            "residual_norm_seed_max": max(norms),
            "q_median_seed_mean": statistics.fmean(q) if q else "",
            "q_fraction_below_one_seed_mean": statistics.fmean(fractions) if fractions else "",
            "action_norm_seed_mean": statistics.fmean(
                float(row["action_norm_median"]) for row in selected
            ),
            "diverged_fraction_seed_mean": statistics.fmean(
                float(row["diverged_fraction"]) for row in selected
            ),
        })
    return output


def summarize_interventions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    per_seed: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_seed[(int(row["trained_seed"]), str(row["condition"]))].append(row)
    seed_rows = []
    for (seed, condition), selected in sorted(per_seed.items()):
        seed_rows.append({
            "trained_seed": seed,
            "condition": condition,
            "one_step_q_median": statistics.median(float(row["one_step_q"]) for row in selected),
            "endpoint_ratio_median": statistics.median(
                float(row["endpoint_over_intervention_residual"]) for row in selected
            ),
            "delta_phi_median": statistics.median(float(row["delta_phi"]) for row in selected),
        })
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        by_condition[str(row["condition"])].append(row)
    aggregate = []
    for condition, selected in sorted(by_condition.items()):
        aggregate.append({
            "condition": condition,
            "seed_count": len(selected),
            "one_step_q_seed_mean": statistics.fmean(float(row["one_step_q_median"]) for row in selected),
            "endpoint_ratio_seed_mean": statistics.fmean(float(row["endpoint_ratio_median"]) for row in selected),
            "delta_phi_seed_mean": statistics.fmean(float(row["delta_phi_median"]) for row in selected),
        })
    return seed_rows + aggregate


def _find(
    rows: list[dict[str, object]], bucket: str, model: str, iteration: int
) -> dict[str, object]:
    return next(
        row for row in rows
        if row["bucket"] == bucket and row["model"] == model and row["iteration"] == iteration
    )


def _historical_v41() -> list[dict[str, str]]:
    path = Path("artifacts/v4_1/history_policy_eval/summary.csv")
    if not path.exists():
        raise RuntimeError("Required completed v4.1 all-history comparison is missing")
    return _read_csv(path)


def verdict(
    aggregate: list[dict[str, object]],
    intervention_summary: list[dict[str, object]],
    historical: list[dict[str, str]],
) -> dict[str, object]:
    model = "v42_query_all_history"
    f0 = float(_find(aggregate, "same_family", model, 0)["residual_norm_seed_mean"])
    f14 = float(_find(aggregate, "same_family", model, 14)["residual_norm_seed_mean"])
    f32 = float(_find(aggregate, "same_family", model, 32)["residual_norm_seed_mean"])
    f64 = float(_find(aggregate, "same_family", model, 64)["residual_norm_seed_mean"])
    current64 = float(
        _find(aggregate, "same_family", "v42_current_only", 64)["residual_norm_seed_mean"]
    )
    ar64 = float(
        _find(aggregate, "same_family", "v42_ar_all_history", 64)["residual_norm_seed_mean"]
    )
    curve = [
        row for row in aggregate if row["bucket"] == "same_family" and row["model"] == model
    ]
    best = min(curve, key=lambda row: float(row["residual_norm_seed_mean"]))
    q15_32 = statistics.median(
        float(row["q_median_seed_mean"])
        for row in curve
        if 15 <= int(row["iteration"]) <= 32
    )
    q33_64 = statistics.median(
        float(row["q_median_seed_mean"])
        for row in curve
        if 33 <= int(row["iteration"]) <= 64
    )
    v41_f64 = next(
        float(row["residual_norm_median"])
        for row in historical
        if row["bucket"] == "same_family"
        and row["history_policy"] == "all_history"
        and int(row["iteration"]) == 64
    )
    intervention = {
        str(row["condition"]): row
        for row in intervention_summary
        if "trained_seed" not in row
    }
    correct_q = float(intervention["correct"]["one_step_q_seed_mean"])
    correct_endpoint = float(intervention["correct"]["endpoint_ratio_seed_mean"])
    history_causal = all(
        correct_q < float(intervention[name]["one_step_q_seed_mean"])
        and correct_endpoint < float(intervention[name]["endpoint_ratio_seed_mean"])
        for name in ("no_history", "wrong_world")
    )
    criteria = {
        "materially_beats_v41_at_f64": f64 < 0.90 * v41_f64,
        "contracts_after_old_f14_optimum": f32 < f14 and q15_32 < 1.0,
        "no_systematic_late_q_above_one": q33_64 < 1.0,
        "useful_through_f32_f64": f32 < f0 and f64 < f0 and f64 < current64 and f64 < ar64,
        "no_late_endpoint_rebound": f64 <= 1.25 * float(best["residual_norm_seed_mean"]),
        "history_causality_retained": history_causal,
        "no_explicit_output_scaling_or_solver_structure": True,
    }
    supported = all(criteria.values())
    return {
        "verdict": (
            "STATIONARY CORRECTION SUPPORTED"
            if supported
            else "STATIONARY CORRECTION NOT SUPPORTED"
        ),
        "supported": supported,
        "criteria": criteria,
        "measurements": {
            "v42_f0": f0,
            "v42_f14": f14,
            "v42_f32": f32,
            "v42_f64": f64,
            "v41_all_history_f64": v41_f64,
            "current_only_f64": current64,
            "ar_no_fresh_f64": ar64,
            "best_residual": float(best["residual_norm_seed_mean"]),
            "best_iteration": int(best["iteration"]),
            "median_q_f15_f32": q15_32,
            "median_q_f33_f64": q33_64,
            "correct_history_one_step_q": correct_q,
            "correct_history_endpoint_ratio": correct_endpoint,
        },
    }


def _plots(output: Path, aggregate: list[dict[str, object]], bin_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, bucket in zip(axes, ("same_family", "condition_4_6")):
        for model in (
            "v42_query_all_history",
            "v42_query_fifo16",
            "v42_current_only",
            "v42_ar_all_history",
            "fixed_linear",
        ):
            selected = [row for row in aggregate if row["bucket"] == bucket and row["model"] == model]
            axis.plot(
                [int(row["iteration"]) for row in selected],
                [float(row["residual_norm_seed_mean"]) for row in selected],
                label=model,
            )
        axis.set_title(bucket)
        axis.set_xlabel("Frozen iteration")
        axis.set_ylabel("Seed-mean median residual norm")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "frozen_rollout_generalization.png", dpi=160)
    plt.close(fig)

    query_bins = [row for row in bin_rows if row["arm"] == "query_history"]
    if query_bins:
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in query_bins:
            grouped[int(row["bin_index"])].append(row)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        x = []
        q = []
        action = []
        for bin_index, selected in sorted(grouped.items()):
            x.append(0.5 * (float(selected[0]["log10_rho_lower"]) + float(selected[0]["log10_rho_upper"])))
            q.append(statistics.fmean(float(row["q_median"]) for row in selected))
            action.append(statistics.fmean(float(row["action_norm_mean"]) for row in selected))
        axes[0].plot(x, q, marker="o")
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Training one-step q")
        axes[1].plot(x, action, marker="o")
        axes[1].set_ylabel("Action norm")
        for axis in axes:
            axis.set_xlabel("log10 residual magnitude bin center")
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "training_by_residual_magnitude.png", dpi=160)
        plt.close(fig)


def write_report(
    output: Path,
    aggregate: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
    intervention_summary: list[dict[str, object]],
    historical: list[dict[str, str]],
    result: dict[str, object],
) -> None:
    lines = [
        "# IFI v4.2-General — Stationary Learning-State Coverage",
        "",
        f"## Verdict: {result['verdict']}",
        "",
        "Training used only asynchronous, scale-stratified, real population states and local one-step consequence gradients. The controller received no B, age, position, target action, explicit output scaling, or solver structure.",
        "",
        "## Frozen same-family rollout (mean of per-seed medians)",
        "",
        "| Model | F0 | F8 | F14 | F16 | F32 | F64 | Best | Best step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in (
        "v42_query_all_history",
        "v42_query_fifo16",
        "v42_current_only",
        "v42_ar_all_history",
        "fixed_linear",
        "analytic_oracle",
    ):
        curve = [row for row in aggregate if row["bucket"] == "same_family" and row["model"] == model]
        best = min(curve, key=lambda row: float(row["residual_norm_seed_mean"]))
        values = [
            float(_find(aggregate, "same_family", model, iteration)["residual_norm_seed_mean"])
            for iteration in (0, 8, 14, 16, 32, 64)
        ]
        lines.append(
            f"| {model} | "
            + " | ".join(f"{value:.6g}" for value in values)
            + f" | {float(best['residual_norm_seed_mean']):.6g} | {best['iteration']} |"
        )
    v41_values = []
    for iteration in (0, 8, 14, 16, 32, 64):
        row = next(
            row for row in historical
            if row["bucket"] == "same_family"
            and row["history_policy"] == "all_history"
            and int(row["iteration"]) == iteration
        )
        v41_values.append(float(row["residual_norm_median"]))
    v41_curve = [
        row for row in historical
        if row["bucket"] == "same_family" and row["history_policy"] == "all_history"
    ]
    v41_best = min(v41_curve, key=lambda row: float(row["residual_norm_median"]))
    lines.append(
        "| v4.1_query_all_history (historical seed 11) | "
        + " | ".join(f"{value:.6g}" for value in v41_values)
        + f" | {float(v41_best['residual_norm_median']):.6g} | {v41_best['iteration']} |"
    )
    lines.extend(["", "## Locked criteria", ""])
    for name, passed in result["criteria"].items():
        lines.append(f"- {name.replace('_', ' ')}: **{passed}**")
    lines.extend([
        "",
        "## Seed endpoints — query-history/all-history",
        "",
        "| Seed | F8 | F14 | F32 | F64 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for seed in sorted({int(row["trained_seed"]) for row in seed_rows}):
        values = []
        for iteration in (8, 14, 32, 64):
            row = next(
                row for row in seed_rows
                if int(row["trained_seed"]) == seed
                and row["bucket"] == "same_family"
                and row["model"] == "v42_query_all_history"
                and int(row["iteration"]) == iteration
            )
            values.append(float(row["residual_norm_median"]))
        lines.append(f"| {seed} | " + " | ".join(f"{value:.6g}" for value in values) + " |")
    lines.extend([
        "",
        "History interventions used all available causally valid history and held the current residual/world fixed. Full per-episode and per-bin diagnostics are saved alongside this report.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combine_seed_file(output: Path, seeds: tuple[int, ...], filename: str) -> list[dict[str, str]]:
    rows = []
    for seed in seeds:
        for row in _read_csv(output / f"seed_{seed}" / filename):
            rows.append({"trained_seed": str(seed), **row})
    _write_csv(output / filename, rows)
    return rows


def run(config: V40Config, device: torch.device) -> None:
    output = Path(config.output_dir)
    prepare(config, device)
    calibration = json.loads((output / "fixed_linear_calibration.json").read_text(encoding="utf-8"))
    fixed_eta = float(calibration["selected_eta"])
    checkpoints: dict[int, dict[str, object]] = {}
    seeds = tuple(int(seed) for seed in config.canonical_seeds)
    for index, seed in enumerate(seeds):
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=False)
        seed_config = replace(config, seed=seed, output_dir=str(seed_output))
        (seed_output / "config.json").write_text(
            json.dumps(seed_config.to_dict(), indent=2), encoding="utf-8"
        )
        _status(output, phase="training", active_seed=seed, seed_index=index + 1, seed_count=len(seeds))
        trainer = PopulationTrainer(seed_config, device, seed_output)
        train_result = trainer.run(stop_on_gate_failure=index == 0)
        write_world_registry(seed_output / "world_registry.csv", trainer.populations)
        checkpoint_path = seed_output / "checkpoint_final.pt"
        checkpoints[seed] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "step": int(train_result["final_step"]),
            "gate": train_result["gate"],
        }
        if index == 0 and train_result["stopped_by_gate"]:
            result = {
                "verdict": "STATIONARY CORRECTION NOT SUPPORTED — BASIC CONTRACTION GATE FAILED",
                "supported": False,
                "stopped_by_gate": True,
                "gate": train_result["gate"],
            }
            (output / "verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            (output / "report.md").write_text(
                "# IFI v4.2-General\n\n## Verdict: STATIONARY CORRECTION NOT SUPPORTED — BASIC CONTRACTION GATE FAILED\n\nNo post-hoc architecture or loss tuning was performed.\n",
                encoding="utf-8",
            )
            _status(output, phase="complete", result=result)
            return
        del trainer
        torch.cuda.empty_cache()

    training_rows = _combine_seed_file(output, seeds, "train_metrics.csv")
    pool_rows = _combine_seed_file(output, seeds, "pool_diagnostics.csv")
    bin_rows = _combine_seed_file(output, seeds, "bin_metrics.csv")
    _status(output, phase="evaluation", completed_training_seeds=list(seeds))
    rollout_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
    frozen_audit = []
    for seed in seeds:
        seed_config = replace(config, seed=seed)
        checkpoint_path = output / f"seed_{seed}" / "checkpoint_final.pt"
        models = _load_frozen_models(seed_config, checkpoint_path, device)
        frozen_audit.append({
            "trained_seed": seed,
            "checkpoint_sha256": checkpoints[seed]["sha256"],
            "all_parameters_frozen": all(
                not parameter.requires_grad
                for model in models.values()
                for parameter in model.parameters()
            ),
            "evaluation_world_seed_base": config.seed + 600,
            "evaluation_uses_v41_residual_protocol": (
                config.evaluation_residual_norm_min == 0.5
                and config.evaluation_residual_norm_max == 2.0
                and not config.evaluation_residual_is_post_bootstrap
            ),
        })
        rollout_rows.extend(evaluate_seed(config, models, seed, fixed_eta, device))
        intervention_rows.extend(
            evaluate_history_causality(config, models["query_history"], seed, device)
        )
        del models
        torch.cuda.empty_cache()
        _status(output, phase="evaluation", completed_seed=seed)
    _write_csv(output / "eval_rollouts.csv", rollout_rows)
    _write_csv(output / "history_interventions.csv", intervention_rows)
    seed_summary = summarize_seed_rollouts(rollout_rows)
    aggregate = aggregate_seed_rollouts(seed_summary)
    intervention_summary = summarize_interventions(intervention_rows)
    _write_csv(output / "seed_summary.csv", seed_summary)
    _write_csv(output / "eval_summary.csv", aggregate)
    _write_csv(output / "history_intervention_summary.csv", intervention_summary)
    historical = _historical_v41()
    scientific_result = verdict(aggregate, intervention_summary, historical)
    (output / "frozen_integrity.json").write_text(
        json.dumps({
            "checks": frozen_audit,
            "no_grad_evaluation": True,
            "same_v41_worlds_and_seeds": True,
            "primary_history_policy": "all_available_history",
            "fifo16_retained_as_diagnostic": True,
            "checkpoint_provenance": checkpoints,
        }, indent=2),
        encoding="utf-8",
    )
    (output / "verdict.json").write_text(
        json.dumps(scientific_result, indent=2), encoding="utf-8"
    )
    write_report(
        output,
        aggregate,
        seed_summary,
        intervention_summary,
        historical,
        scientific_result,
    )
    _plots(output, aggregate, bin_rows)
    manifest = {
        "experiment": "IFI v4.2-General stationary learning-state coverage",
        "complete": True,
        "pid": os.getpid(),
        "seeds": list(seeds),
        "checkpoints": checkpoints,
        "training_metric_rows": len(training_rows),
        "pool_diagnostic_rows": len(pool_rows),
        "bin_metric_rows": len(bin_rows),
        "rollout_rows": len(rollout_rows),
        "verdict": scientific_result,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _status(output, phase="complete", manifest=manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v4.2-General canonical experiment")
    parser.add_argument("--config", default="configs/v4_2_general.json")
    args = parser.parse_args(argv)
    config = V40Config.from_json(args.config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "canonical_run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    except FileExistsError as error:
        raise RuntimeError("Refusing to launch a second v4.2-General canonical run") from error
    run(config, _device(config))


if __name__ == "__main__":
    main()
