from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .v40_cli import hardware
from .v40_config import V40Config
from .v40_evaluation import make_rollout_state
from .v40_models import PopulationController
from .v40_training import build_models
from .v44_config import V44Config
from .v44_evaluation import (
    ALL_HISTORY,
    FIFO_16,
    capacity_rows,
    fresh_state_from_matrices,
    history_intervention_summary,
    rollout_summary,
    stored_state,
)
from .v44_training import (
    JointSeedTrainer,
    arm_config,
    fixed_convergence_metrics,
    parameter_count,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate(experiment: V44Config, base: V40Config) -> None:
    if tuple(experiment.seeds) != (11, 23, 47):
        raise ValueError("Canonical v4.4 seeds must remain 11, 23, 47")
    if round(experiment.batch_size * experiment.ordinary_fraction) != 38:
        raise ValueError("Locked v4.4 batch must be 38 ordinary / 26 convergence")
    if experiment.lambda_safe != 1.0:
        raise ValueError("v4.4 lambda_safe was preregistered at 1.0")
    if experiment.learning_rate_small != experiment.learning_rate_large:
        raise ValueError("Both capacity arms use the same locked learning rate")
    if experiment.smoke_step >= experiment.training_steps:
        raise ValueError("Smoke gate must precede canonical completion")
    if base.residual_dimension != base.action_dimension:
        raise ValueError("v4.4 preserves square linear worlds")
    if experiment.recovery_source == "genuine_archive" and (
        experiment.recovery_gate_every <= 0
        or experiment.recovery_refresh_every <= 0
    ):
        raise ValueError("Archive recovery requires positive refresh and gate intervals")


def fixture_states(
    experiment: V44Config,
    base: V40Config,
    reference_population: dict[str, object],
    device: torch.device,
):
    capacity = base.bootstrap_actions + experiment.evaluation_horizon
    storage = replace(base, history_capacity=capacity)
    unseen = make_rollout_state(
        storage, experiment.evaluation_episodes, base.seed + 600, device
    )
    matrices = reference_population["matrices"][: experiment.evaluation_episodes]
    world_seeds = reference_population["world_seeds"][: experiment.evaluation_episodes]
    familiar = fresh_state_from_matrices(
        base,
        matrices,
        world_seeds,
        capacity,
        base.seed + 610,
        device,
    )
    generator = torch.Generator().manual_seed(41_128)
    indices = torch.randperm(base.pool_size, generator=generator)[: experiment.evaluation_episodes]
    stored = stored_state(base, reference_population, indices, capacity, device)
    return {
        "unseen_same_family": unseen,
        "familiar_B_fresh_residual": familiar,
        "stored_training_states": stored,
    }


def load_trained_model(
    checkpoint_path: Path,
    base: V40Config,
    experiment: V44Config,
    arm: str,
    seed: int,
    device: torch.device,
) -> PopulationController:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not checkpoint["complete"] or int(checkpoint["step"]) != experiment.training_steps:
        raise RuntimeError(f"Incomplete v4.4 checkpoint: {checkpoint_path}")
    config = arm_config(base, experiment, arm, seed)
    model = PopulationController("query_history", config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model


def model_curve(
    rows: list[dict[str, object]], model: str, fixture: str = "unseen_same_family"
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["model"] == model
        and row["fixture"] == fixture
        and row["history_policy"] == "all_history"
    ]


def history_supported(
    interventions: list[dict[str, object]], arm: str
) -> bool:
    labels = sorted({str(row["model"]) for row in interventions if str(row["model"]).startswith(arm)})
    seed_results = []
    for label in labels:
        lookup = {
            (str(row["condition"]), int(row["followup_iteration"])): float(
                row["residual_ratio_median"]
            )
            for row in interventions
            if row["model"] == label
        }
        seed_results.append(all(
            lookup[("correct", step)] <= 0.95 * lookup[(condition, step)]
            for step in (1, 64, 128)
            for condition in ("no_history", "wrong_world")
        ))
    return bool(seed_results) and all(seed_results)


def verdicts(
    long_rows: list[dict[str, object]],
    capacity: list[dict[str, object]],
    interventions: list[dict[str, object]],
) -> dict[str, object]:
    original = model_curve(long_rows, "original_v41")
    original_best = min(original, key=lambda row: float(row["residual_norm_median"]))
    original_by_step = {int(row["iteration"]): row for row in original}
    original_f8_ratio = float(original_by_step[8]["residual_norm_median"]) / float(
        original_by_step[0]["residual_norm_median"]
    )
    original_f128_best = float(original_by_step[128]["residual_norm_median"]) / float(
        original_best["residual_norm_median"]
    )
    means = {
        str(row["arm"]): row
        for row in capacity
        if row["row_type"] == "mean"
    }
    arm_results = {}
    for arm, row in means.items():
        checks = {
            "v41_level_early_correction": float(row["f8_over_f0"]) <= 1.20 * original_f8_ratio,
            "late_median_q_not_expansive": float(row["q_f33_f64"]) <= 1 and float(row["q_f65_f128"]) <= 1,
            "f64_f128_drift_substantially_reduced": float(row["f64_over_best"]) <= 1.5
            and float(row["f128_over_best"]) <= 1.5
            and float(row["f128_over_best"]) <= 0.5 * original_f128_best,
            "history_causality_preserved": history_supported(interventions, arm),
        }
        arm_results[arm] = {
            "supported": all(checks.values()),
            "checks": checks,
            "metrics": row,
        }
    joint = any(result["supported"] for result in arm_results.values())
    if "small" in means and "large" in means:
        large_better_early = float(means["large"]["f8_over_f0"]) <= 0.95 * float(
            means["small"]["f8_over_f0"]
        )
        large_better_late = float(means["large"]["f128_over_best"]) <= 0.95 * float(
            means["small"]["f128_over_best"]
        )
        capacity_supported = large_better_early and large_better_late
    else:
        capacity_supported = False
    return {
        "joint_stationary_learning": (
            "JOINT STATIONARY LEARNING SUPPORTED"
            if joint
            else "JOINT STATIONARY LEARNING NOT SUPPORTED"
        ),
        "capacity_verdict": (
            "CAPACITY-LIMITED INTERFERENCE SUPPORTED"
            if capacity_supported
            else "CAPACITY INSUFFICIENT"
        ),
        "general_track_weakened": not arm_results.get("large", {}).get("supported", False),
        "arm_results": arm_results,
        "reference": {
            "v41_f8_over_f0": original_f8_ratio,
            "v41_f128_over_best": original_f128_best,
        },
    }


def write_report(
    output: Path,
    long_rows: list[dict[str, object]],
    capacity: list[dict[str, object]],
    result: dict[str, object],
) -> None:
    lines = [
        "# IFI v4.4 — Joint-from-Scratch Identification + Convergence",
        "",
        f"## {result['joint_stationary_learning']}",
        "",
        f"## {result['capacity_verdict']}",
        "",
        "| Arm | F8/F0 | Best | Best step | F64/best | F128/best | q F33–64 | q F65–128 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in capacity:
        if row["row_type"] == "mean":
            lines.append(
                f"| {row['arm']} | {float(row['f8_over_f0']):.6g}"
                f" | {float(row['best_residual']):.6g} | {float(row['best_step']):.3g}"
                f" | {float(row['f64_over_best']):.6g} | {float(row['f128_over_best']):.6g}"
                f" | {float(row['q_f33_f64']):.6g} | {float(row['q_f65_f128']):.6g} |"
            )
    lines.extend(["", "## Locked criteria", ""])
    for arm, arm_result in result["arm_results"].items():
        lines.append(f"### {arm}")
        lines.append("")
        for name, passed in arm_result["checks"].items():
            lines.append(f"- {name.replace('_', ' ')}: **{passed}**")
        lines.append("")
    lines.extend([
        "Both arms were initialized from scratch and trained with identical 60/40 ordinary/convergence curricula, one-step consequence gradients, and safety loss. No trajectory labels, B input, teacher target, positional signal, BPTT, or analytic scaling entered either controller.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axis = plt.subplots(figsize=(7, 5))
    for row in capacity:
        if row["row_type"] == "seed":
            axis.scatter(
                float(row["f8_over_f0"]),
                float(row["f128_over_best"]),
                label=f"{row['arm']}-seed{row['seed']}",
            )
    axis.set_xlabel("Early correction: F8/F0 (lower is better)")
    axis.set_ylabel("Late stability: F128/best (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "capacity_pareto.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    labels = ["original_v41"]
    labels.extend(
        sorted({str(row["model"]) for row in long_rows if row["model"].startswith(("small_seed", "large_seed"))})
    )
    for label in labels:
        curve = model_curve(long_rows, label)
        if curve:
            axis.plot(
                [int(row["iteration"]) for row in curve],
                [float(row["residual_norm_median"]) for row in curve],
                label=label,
                alpha=0.8,
            )
    axis.set_yscale("log")
    axis.set_xlabel("Frozen iteration")
    axis.set_ylabel("Median residual norm")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "long_rollout_comparison.png", dpi=160)
    plt.close(fig)


def combine_seed_csv(output: Path, seeds: list[int], filename: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        rows.extend(read_csv(output / f"seed_{seed}" / filename))
    write_csv(output / filename, rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v4.4 joint capacity ablation")
    parser.add_argument("--config", default="configs/v4_4_joint_capacity.json")
    args = parser.parse_args(argv)
    experiment = V44Config.from_json(args.config)
    base = V40Config.from_json(experiment.base_config)
    validate(experiment, base)
    if not torch.cuda.is_available():
        raise RuntimeError("IFI v4.4 requires CUDA")
    output = Path(experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "canonical_run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    except FileExistsError as error:
        raise RuntimeError("Refusing a second v4.4 canonical run") from error
    atomic_json(output / "config.json", experiment.to_dict())
    device = torch.device("cuda")
    atomic_json(output / "hardware.json", hardware(device))
    reference_checkpoint = torch.load(
        experiment.reference_checkpoint, map_location="cpu", weights_only=False
    )
    archive = torch.load(
        experiment.convergence_validation_pool, map_location="cpu", weights_only=False
    )
    trajectory_count = int(archive["trajectory_id"].max()) + 1
    validation_indices = torch.where(
        archive["trajectory_id"] >= int(0.8 * trajectory_count)
    )[0]
    reference_models = build_models(base, device)
    for name, model in reference_models.items():
        model.load_state_dict(reference_checkpoint["models"][name])
        model.eval().requires_grad_(False)
    reference_convergence = fixed_convergence_metrics(
        reference_models["query_history"],
        base,
        archive,
        validation_indices,
        device,
    )
    small_config = arm_config(base, experiment, "small", 11)
    large_config = arm_config(base, experiment, "large", 11)
    counts = {
        "small": parameter_count(PopulationController("query_history", small_config)),
        "large": parameter_count(PopulationController("query_history", large_config)),
    }
    counts["large_over_small"] = counts["large"] / counts["small"]
    atomic_json(output / "parameter_counts.json", counts)
    completed_arms: tuple[str, ...] = ("small", "large")
    seed_results = {}
    for seed_index, seed in enumerate(experiment.seeds):
        seed = int(seed)
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "status.json", {
            "phase": "training",
            "seed": seed,
            "seed_index": seed_index + 1,
            "seed_count": len(experiment.seeds),
            "arms": list(completed_arms),
        })
        trainer = JointSeedTrainer(
            experiment,
            base,
            seed,
            completed_arms,
            archive,
            validation_indices,
            reference_convergence["regression_fraction"],
            device,
            seed_output,
            enforce_smoke=seed_index == 0,
        )
        if seed_index == 0:
            sampler_gate = {
                arm: runtime.initial_sampler_gate
                for arm, runtime in trainer.arms.items()
            }
            sampler_gate["passed"] = all(
                gate["passed"]
                for arm, gate in sampler_gate.items()
                if arm != "passed"
            )
            atomic_json(output / "sampler_gate.json", sampler_gate)
            if not sampler_gate["passed"]:
                raise RuntimeError("v4.4 convergence sampler gate failed before training")
            initial_small = trainer.arms["small"].model
            initial_reference = reference_models["query_history"]
            scratch_difference = any(
                not torch.equal(initial_small.state_dict()[name].cpu(), initial_reference.state_dict()[name].cpu())
                for name in initial_small.state_dict()
            )
            integrity = {
                "small_parameter_count": counts["small"],
                "large_parameter_count": counts["large"],
                "large_over_small": counts["large_over_small"],
                "initialized_from_scratch": scratch_difference,
                "small_real_recovery_history_error": trainer.arms["small"].recovery.actual_history_max_error(),
                "large_real_recovery_history_error": trainer.arms["large"].recovery.actual_history_max_error(),
                "ordinary_per_batch": trainer.arms["small"].ordinary_count,
                "convergence_per_batch": trainer.arms["small"].recovery_count,
                "no_positional_parameters": not any(
                    "position" in name.lower()
                    for runtime in trainer.arms.values()
                    for name, _ in runtime.model.named_parameters()
                ),
                "controller_input_fields": [
                    "current", "previous_action", "history", "history_lengths"
                ],
                "no_bptt": True,
                "labels_hidden": True,
                "sampler_gate_passed": sampler_gate["passed"],
                "recovery_source": experiment.recovery_source,
            }
            integrity["passed"] = (
                integrity["initialized_from_scratch"]
                and integrity["small_real_recovery_history_error"] < 2e-6
                and integrity["large_real_recovery_history_error"] < 2e-6
                and integrity["ordinary_per_batch"] == 38
                and integrity["convergence_per_batch"] == 26
                and integrity["no_positional_parameters"]
                and integrity["sampler_gate_passed"]
            )
            atomic_json(output / "integrity_audit.json", integrity)
            if not integrity["passed"]:
                raise RuntimeError("v4.4 integrity audit failed")
        result = trainer.run()
        seed_results[seed] = result
        if seed_index == 0:
            atomic_json(output / "smoke_gate.json", result["gates"])
            completed_arms = tuple(result["completed_arms"])
            if not completed_arms:
                write_csv(output / "long_rollouts.csv", [])
                write_csv(output / "history_interventions.csv", [])
                gate_rows = [
                    {
                        "row_type": "smoke",
                        "arm": arm,
                        "passed": gate["passed"],
                        "f8_over_f0": gate["early"]["f8_over_f0_median"],
                        "convergence_q": gate["convergence"]["median_q"],
                        "convergence_regression_fraction": gate["convergence"]["regression_fraction"],
                    }
                    for arm, gate in result["gates"].items()
                ]
                write_csv(output / "capacity_comparison.csv", gate_rows)
                combine_seed_csv(output, [11], "train_metrics.csv")
                combine_seed_csv(output, [11], "regime_metrics.csv")
                combine_seed_csv(output, [11], "sampler_coverage.csv")
                (output / "report.md").write_text(
                    "# IFI v4.4 — Sampler-Gated Joint-from-Scratch Capacity Ablation\n\n"
                    "## CONVERGENCE SAMPLER GATE PASSED BEFORE TRAINING\n\n"
                    "The recovery source used only the genuine v4.3 train-split histories; "
                    "held-out trajectories remained exclusive to the smoke evaluation. "
                    "See `sampler_gate.json` and `sampler_coverage.csv`.\n\n"
                    "## GENERAL TRACK WEAKENED — NEITHER ARM PASSED THE JOINT SMOKE GATE\n\n"
                    "The run stopped at step 2,000 without post-result tuning. See `smoke_gate.json`.\n",
                    encoding="utf-8",
                )
                atomic_json(output / "status.json", {
                    "phase": "complete", "stopped_by_gate": True, "gates": result["gates"]
                })
                return
        del trainer
        torch.cuda.empty_cache()

    seeds = [int(seed) for seed in experiment.seeds]
    combine_seed_csv(output, seeds, "train_metrics.csv")
    combine_seed_csv(output, seeds, "regime_metrics.csv")
    combine_seed_csv(output, seeds, "sampler_coverage.csv")
    atomic_json(output / "status.json", {"phase": "evaluation", "arms": list(completed_arms)})
    fixtures = fixture_states(
        experiment,
        base,
        reference_checkpoint["populations"]["query_history"],
        device,
    )
    long_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
    baseline_specs = (
        ("original_v41", reference_models["query_history"], ALL_HISTORY),
        ("original_v41_fifo16", reference_models["query_history"], FIFO_16),
        ("current_only", reference_models["current_only"], ALL_HISTORY),
        ("ar_no_fresh", reference_models["ar_no_fresh"], ALL_HISTORY),
        ("analytic_oracle", None, ALL_HISTORY),
    )
    for fixture, base_state in fixtures.items():
        for label, model, policy in baseline_specs:
            long_rows.extend(
                rollout_summary(
                    experiment, base, model, label, "historical", fixture, base_state, policy
                )
            )
    intervention_rows.extend(
        history_intervention_summary(
            experiment,
            base,
            reference_models["query_history"],
            "original_v41",
            "historical",
            device,
        )
    )
    for seed in seeds:
        for arm in completed_arms:
            checkpoint_path = output / f"seed_{seed}" / f"{arm}_checkpoint_final.pt"
            model = load_trained_model(
                checkpoint_path, base, experiment, arm, seed, device
            )
            config = arm_config(base, experiment, arm, seed)
            label = f"{arm}_seed{seed}"
            for fixture, base_state in fixtures.items():
                for policy in (ALL_HISTORY, FIFO_16):
                    long_rows.extend(
                        rollout_summary(
                            experiment,
                            config,
                            model,
                            label,
                            seed,
                            fixture,
                            base_state,
                            policy,
                        )
                    )
            intervention_rows.extend(
                history_intervention_summary(
                    experiment, config, model, label, seed, device
                )
            )
            del model
            torch.cuda.empty_cache()
    capacity = capacity_rows(long_rows)
    result = verdicts(long_rows, capacity, intervention_rows)
    write_csv(output / "long_rollouts.csv", long_rows)
    write_csv(output / "history_interventions.csv", intervention_rows)
    write_csv(output / "capacity_comparison.csv", capacity)
    atomic_json(output / "verdict.json", result)
    write_report(output, long_rows, capacity, result)
    manifest = {
        "complete": True,
        "pid": os.getpid(),
        "seeds": seeds,
        "completed_arms": list(completed_arms),
        "seed_results": seed_results,
        "verdict": result,
    }
    atomic_json(output / "experiment_manifest.json", manifest)
    atomic_json(output / "status.json", {"phase": "complete", "manifest": manifest})
    print(json.dumps({"phase": "v44_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
