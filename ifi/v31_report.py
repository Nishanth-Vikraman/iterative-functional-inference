from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .v31_config import V31Config


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(values):
    return [float(value) for value in values if value not in {"", None} and math.isfinite(float(value))]


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_summary(records: list[dict[str, str]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["bucket"], row["severity"], row["method"], int(row["iteration"]))].append(
            float(row["query_mse"])
        )
    result = []
    for (bucket, severity, method, iteration), raw in sorted(grouped.items()):
        values = [value for value in raw if math.isfinite(value)]
        result.append({
            "bucket": bucket,
            "severity": severity,
            "method": method,
            "iteration": iteration,
            "query_mse_mean": mean(values) if values else math.nan,
            "query_mse_std": pstdev(values) if len(values) > 1 else 0.0,
            "n_finite": len(values),
            "n_total": len(raw),
        })
    return result


def _seed_summary(records: list[dict[str, str]], endpoint: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        if int(row["iteration"]) != endpoint:
            continue
        grouped[(row["bucket"], row["severity"], row["method"], int(row["seed"]))].append(
            float(row["query_mse"])
        )
    result = []
    for (bucket, severity, method, seed), raw in sorted(grouped.items()):
        values = [value for value in raw if math.isfinite(value)]
        result.append({
            "bucket": bucket,
            "severity": severity,
            "method": method,
            "seed": seed,
            "endpoint_iteration": endpoint,
            "query_mse_mean": mean(values) if values else math.nan,
            "query_mse_std": pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        })
    return result


def _projection_summary(rows: list[dict[str, str]], threshold: float) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["bucket"], row["severity"], row["method"])].append(row)
    result = []
    phi_keys = [f"phi_{index}" for index in range(12)]
    for (bucket, severity, method), values in sorted(grouped.items()):
        residuals = _finite(row["relative_projection_residual"] for row in values)
        conditions = _finite(row["condition_number"] for row in values)
        norms = _finite(row["phi_norm"] for row in values)
        components = _finite(row[key] for row in values for key in phi_keys)
        result.append({
            "bucket": bucket,
            "severity": severity,
            "method": method,
            "mean": mean(residuals),
            "median": _quantile(residuals, 0.5),
            "q90": _quantile(residuals, 0.9),
            "q95": _quantile(residuals, 0.95),
            "max": max(residuals),
            "fraction_outside": sum(value > threshold for value in residuals) / len(residuals),
            "condition_median": _quantile(conditions, 0.5),
            "condition_q95": _quantile(conditions, 0.95),
            "phi_norm_mean": mean(norms),
            "phi_norm_q95": _quantile(norms, 0.95),
            "phi_component_mean": mean(components),
            "phi_component_std": pstdev(components),
            "n": len(residuals),
        })
    return result


def _group_metrics(
    trajectories: list[dict], projections: list[dict], config: V31Config, train_rollout: int,
) -> tuple[list[dict], list[dict]]:
    curve = {
        (row["bucket"], row["severity"], row["method"], int(row["iteration"])):
        float(row["query_mse_mean"])
        for row in trajectories
    }
    projection = {
        (row["bucket"], row["severity"], row["method"]): row for row in projections
    }
    groups = sorted({(row["bucket"], row["severity"]) for row in trajectories})
    summaries = []
    classifications = []
    for bucket, severity in groups:
        def value(method, iteration):
            return curve.get((bucket, severity, method, iteration), math.nan)

        q_curve = [value("query_history", iteration) for iteration in range(config.rollout_steps + 1)]
        f0 = q_curve[0]
        endpoint = q_curve[-1]
        finite_curve = [(index, val) for index, val in enumerate(q_curve) if math.isfinite(val)]
        best_iteration, best_mse = min(finite_curve, key=lambda item: item[1])
        direct = value("direct_current", config.rollout_steps)
        ar = value("ar_no_feedback", config.rollout_steps)
        teacher = value("bayesian_oracle", config.rollout_steps)
        relative_improvement = (f0 - endpoint) / max(f0, 1e-12)
        advantage_direct = (direct - endpoint) / max(direct, 1e-12)
        advantage_ar = (ar - endpoint) / max(ar, 1e-12)
        teacher_improvement = (f0 - teacher) / max(f0, 1e-12)
        consecutive = 0
        first_degradation = ""
        for iteration in range(1, len(q_curve)):
            if q_curve[iteration] < q_curve[iteration - 1]:
                if first_degradation == "":
                    consecutive += 1
            elif first_degradation == "":
                first_degradation = iteration
        past_train = endpoint < q_curve[train_rollout]
        long_stable = math.isfinite(endpoint) and endpoint <= 1.05 * q_curve[train_rollout]
        proj = projection[(bucket, severity, "query_history")]
        expressive = float(proj["fraction_outside"]) < config.bucket_outside_fraction_threshold
        rollout_failure = (
            expressive
            and best_mse <= (1 - config.material_effect_threshold) * f0
            and endpoint > 1.05 * q_curve[train_rollout]
        )
        strong_transfer = (
            expressive
            and relative_improvement >= config.material_effect_threshold
            and advantage_direct >= config.material_effect_threshold
            and advantage_ar >= config.material_effect_threshold
        )
        if not expressive:
            classification = "OPERATOR-LANGUAGE OOD"
        elif rollout_failure:
            classification = "ROLLOUT-HORIZON FAILURE"
        elif (
            bucket in {"structural_shift", "compositional_shift"}
            and strong_transfer
            and past_train
        ):
            classification = "ROBUST TRANSFER"
        elif strong_transfer:
            classification = "LEARNING-PROCEDURE TRANSFER"
        else:
            classification = "INFERENCE GENERALIZATION FAILURE"
        row = {
            "bucket": bucket,
            "severity": severity,
            "projection_residual_mean": proj["mean"],
            "projection_residual_q95": proj["q95"],
            "projection_fraction_outside": proj["fraction_outside"],
            "query_history_endpoint": endpoint,
            "full_history_endpoint": value("full_history", config.rollout_steps),
            "direct_current_endpoint": direct,
            "ar_endpoint": ar,
            "teacher_endpoint": teacher,
            "query_history_relative_improvement": relative_improvement,
            "query_history_advantage_direct": advantage_direct,
            "query_history_advantage_ar": advantage_ar,
            "best_iteration": best_iteration,
            "best_mse": best_mse,
            "consecutive_improving_iterations": consecutive,
            "first_degradation_iteration": first_degradation,
            "improves_past_train_depth": past_train,
            "long_rollout_stable": long_stable,
            "classification": classification,
        }
        summaries.append(row)
        classifications.append({
            "bucket": bucket,
            "severity": severity,
            "classification": classification,
            "checks": {
                "operator_expressive": expressive,
                "teacher_strong_progress": teacher_improvement >= config.material_effect_threshold,
                "query_history_strong_progress": relative_improvement >= config.material_effect_threshold,
                "beats_direct_current": advantage_direct >= config.material_effect_threshold,
                "beats_ar": advantage_ar >= config.material_effect_threshold,
                "improves_past_train_depth": past_train,
                "rollout_failure": rollout_failure,
            },
        })
    return summaries, classifications


def _plot_curves(output: Path, trajectories: list[dict], bucket: str, filename: str) -> Path:
    rows = [row for row in trajectories if row["bucket"] == bucket]
    severities = sorted({row["severity"] for row in rows})
    figure, axes = plt.subplots(1, len(severities), figsize=(6 * len(severities), 4), squeeze=False)
    for axis, severity in zip(axes[0], severities):
        for method in ("query_history", "full_history", "direct_current", "ar_no_feedback", "bayesian_oracle"):
            values = sorted(
                (int(row["iteration"]), float(row["query_mse_mean"]))
                for row in rows if row["severity"] == severity and row["method"] == method
            )
            if values:
                axis.plot([item[0] for item in values], [item[1] for item in values], label=method)
        axis.set_title(f"{bucket}: {severity}")
        axis.set_xlabel("IFI iteration")
        axis.set_ylabel("query MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=8)
    figure.tight_layout()
    path = output / filename
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def make_plots(output: Path, trajectories: list[dict], summary: list[dict]) -> list[Path]:
    paths = [
        _plot_curves(output, trajectories, "heldout_combinations", "plot_a_heldout_mse.png"),
        _plot_curves(output, trajectories, "coefficient_extrapolation", "plot_b_coefficient_mse.png"),
        _plot_curves(output, trajectories, "structural_shift", "plot_c_structural_mse.png"),
        _plot_curves(output, trajectories, "compositional_shift", "plot_d_compositional_mse.png"),
    ]
    figure, axis = plt.subplots(figsize=(6, 4))
    for row in summary:
        axis.scatter(float(row["projection_residual_mean"]), float(row["query_history_endpoint"]), label=f"{row['bucket']}:{row['severity']}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("frozen O_phi projection residual")
    axis.set_ylabel("query-history endpoint MSE")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6)
    figure.tight_layout()
    path = output / "plot_e_error_vs_projection.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    order = {"baseline": 0.0, "near": 0.1, "medium": 0.25, "far": 0.5, "structural": 1.0, "mild": 1.5, "hard": 2.0}
    figure, axis = plt.subplots(figsize=(7, 4))
    ordered = sorted(summary, key=lambda row: order[row["severity"]])
    axis.plot(
        [order[row["severity"]] for row in ordered],
        [float(row["query_history_advantage_direct"]) for row in ordered],
        marker="o",
    )
    for row in ordered:
        axis.annotate(f"{row['bucket']}:{row['severity']}", (order[row["severity"]], float(row["query_history_advantage_direct"])), fontsize=6)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("preregistered OOD-distance order")
    axis.set_ylabel("query-history relative advantage over direct-current")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output / "plot_f_history_advantage_vs_ood.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    # Plot G: category-averaged maximum-depth trajectories.
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, bucket in zip(axes.flatten(), ("heldout_combinations", "coefficient_extrapolation", "structural_shift", "compositional_shift")):
        for method in ("query_history", "direct_current", "ar_no_feedback"):
            grouped = defaultdict(list)
            for row in trajectories:
                if row["bucket"] == bucket and row["method"] == method:
                    grouped[int(row["iteration"])].append(float(row["query_mse_mean"]))
            values = sorted((iteration, mean(items)) for iteration, items in grouped.items())
            axis.plot([item[0] for item in values], [item[1] for item in values], label=method)
        axis.set_title(bucket)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.set_xlabel("iteration")
        axis.set_ylabel("MSE")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    path = output / "plot_g_long_rollout.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def write_report(
    output_dir: str | Path,
    config: V31Config,
    train_rollout: int,
    phase_a_reference: dict,
) -> dict[str, object]:
    output = Path(output_dir)
    records = _read(output / "evaluation_records.csv")
    projection_rows = _read(output / "projection_records.csv")
    trajectories = _trajectory_summary(records)
    seeds = _seed_summary(records, config.rollout_steps)
    projections = _projection_summary(projection_rows, config.projection_span_threshold)
    summary, classifications = _group_metrics(trajectories, projections, config, train_rollout)
    _write_csv(output / "trajectory_summary.csv", trajectories)
    _write_csv(output / "seed_summary.csv", seeds)
    _write_csv(output / "projection_summary.csv", projections)
    _write_csv(output / "ood_distance_summary.csv", summary)
    (output / "classifications.json").write_text(
        json.dumps(classifications, indent=2), encoding="utf-8"
    )
    plots = make_plots(output, trajectories, summary)
    integrity = json.loads((output / "frozen_integrity.json").read_text(encoding="utf-8"))
    table = [
        "| OOD bucket | Severity | Projection mean | Q-history F18 | Direct F18 | AR F18 | Oracle F18 | Q advantage/direct | Best i | Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        table.append(
            f"| {row['bucket']} | {row['severity']} | {float(row['projection_residual_mean']):.4g} | "
            f"{float(row['query_history_endpoint']):.4g} | {float(row['direct_current_endpoint']):.4g} | "
            f"{float(row['ar_endpoint']):.4g} | {float(row['teacher_endpoint']):.4g} | "
            f"{float(row['query_history_advantage_direct']):+.1%} | {row['best_iteration']} | {row['classification']} |"
        )
    report = f"""# IFI v3.1 — Frozen-checkpoint OOD ladder

This is a pure evaluation of the completed v3.0 checkpoints. No weights were trained or updated.

## Frozen integrity

- Integrity gate passed: **{integrity['passed']}**
- Source Phase A SHA-256: `{integrity['checkpoints']['phase_a']['sha256']}`
- Source Phase B SHA-256: `{integrity['checkpoints']['phase_b']['sha256']}`
- Train rollout maximum: **{train_rollout}**; evaluated through **F{config.rollout_steps}**.
- Explicit-history overflow protocol: retain the most recent {config.history_capacity} steps, the exact frozen checkpoint capacity.
- The Bayesian OOD oracle is diagnostic-only and receives no path into any IFI controller.

## Locked classifications

{chr(10).join(table)}

## Frozen operator reference

Original v3.0 held-out Phase-A statistics:

```json
{json.dumps(phase_a_reference, indent=2)}
```

Per-bucket mean/median/q90/q95/max projection residuals, conditioning, and phi distributions are in `projection_summary.csv`. Individual-seed endpoints are in `seed_summary.csv`.

## Protocol notes

- OOD targets were deterministically locked in `target_registry.json` before evaluation.
- Each diagnostic oracle performs exact finite Bayesian inference only within its preregistered bucket/severity hypothesis set.
- IFI inputs contain observable noisy evidence, its own current predictions/actions, and allowed explicit history only.
- Query targets, target metadata, oracle posterior, and projected oracle actions never enter controller inputs.
- `relative_projection_residual` uses the same MSE/target-energy definition as v3.0.
- Oracle best iteration is diagnostic and is not a deployable stopping rule.

## Plots

{chr(10).join(f'- `{path.name}`' for path in plots)}
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return {
        "groups": len(summary),
        "classifications": classifications,
        "plots": [str(path) for path in plots],
    }
