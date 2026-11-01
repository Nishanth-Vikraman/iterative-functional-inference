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

from .v32_config import ARMS, V32Config
from .v32_training import LossCalibration


HISTORICAL_METHODS = {
    "query_history": "old_v31_query_history",
    "direct_current": "direct_current",
    "full_history": "full_history",
    "ar_no_feedback": "ar_no_feedback",
    "bayesian_oracle": "bayesian_oracle",
}
PRIMARY_METHODS = (
    "old_v31_query_history", "phi_mse", "normalized_functional",
    "normalized_mahalanobis",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2
        for position in range(index, end):
            result[order[position]] = average_rank
        index = end
    return result


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return math.nan
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = (
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    ) ** 0.5
    return numerator / denominator if denominator else math.nan


def _trajectory_rows(new_rows, historical_rows):
    grouped = defaultdict(list)
    for row in new_rows:
        grouped[(row["bucket"], row["severity"], row["arm"], int(row["iteration"]))].append(
            float(row["query_mse"])
        )
    for row in historical_rows:
        method = HISTORICAL_METHODS.get(row["method"])
        if method is None:
            continue
        grouped[(row["bucket"], row["severity"], method, int(row["iteration"]))].append(
            float(row["query_mse"])
        )
    result = []
    for (bucket, severity, method, iteration), raw in sorted(grouped.items()):
        values = [value for value in raw if math.isfinite(value)]
        result.append({
            "distribution": bucket,
            "severity": severity,
            "loss_or_baseline": method,
            "iteration": iteration,
            "mse_mean": mean(values),
            "mse_std": pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        })
    return result


def _seed_rows(new_rows, historical_rows, endpoint: int):
    grouped = defaultdict(list)
    for row in new_rows:
        if int(row["iteration"]) in {8, 12, 18, endpoint}:
            grouped[(row["bucket"], row["severity"], row["arm"], int(row["seed"]), int(row["iteration"]))].append(float(row["query_mse"]))
    for row in historical_rows:
        method = HISTORICAL_METHODS.get(row["method"])
        if method and int(row["iteration"]) in {8, 12, 18, endpoint}:
            grouped[(row["bucket"], row["severity"], method, int(row["seed"]), int(row["iteration"]))].append(float(row["query_mse"]))
    result = []
    for (bucket, severity, method, seed, iteration), values in sorted(grouped.items()):
        result.append({
            "distribution": bucket,
            "severity": severity,
            "loss_or_baseline": method,
            "seed": seed,
            "iteration": iteration,
            "mse_mean": mean(values),
            "mse_std": pstdev(values) if len(values) > 1 else 0.0,
            "mse_min": min(values),
            "mse_max": max(values),
            "n": len(values),
        })
    return result


def _projection_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["bucket"], row["severity"], row["arm"])].append(row)
    result = []
    for (bucket, severity, arm), values in sorted(grouped.items()):
        residual = [float(row["relative_projection_residual"]) for row in values]
        condition = [float(row["condition_number"]) for row in values]
        phi_star = [float(row["phi_star_norm"]) for row in values]
        phi_hat = [float(row["phi_hat_norm"]) for row in values]
        result.append({
            "distribution": bucket,
            "severity": severity,
            "loss": arm,
            "projection_mean": mean(residual),
            "projection_median": _quantile(residual, 0.5),
            "projection_q90": _quantile(residual, 0.9),
            "projection_q95": _quantile(residual, 0.95),
            "projection_max": max(residual),
            "outside_span_fraction": sum(item > 0.2 for item in residual) / len(residual),
            "condition_median": _quantile(condition, 0.5),
            "condition_q95": _quantile(condition, 0.95),
            "phi_star_norm_mean": mean(phi_star),
            "phi_hat_norm_mean": mean(phi_hat),
            "n": len(values),
        })
    return result


def _primary_comparison(trajectories, projections, endpoint):
    curve = {
        (row["distribution"], row["severity"], row["loss_or_baseline"], int(row["iteration"])): float(row["mse_mean"])
        for row in trajectories
    }
    projection = {
        (row["distribution"], row["severity"], row["loss"]): row
        for row in projections
    }
    groups = sorted({(row["distribution"], row["severity"]) for row in trajectories})
    result = []
    for distribution, severity in groups:
        for method in PRIMARY_METHODS:
            values = [curve.get((distribution, severity, method, iteration), math.nan) for iteration in range(endpoint + 1)]
            if not any(math.isfinite(value) for value in values):
                continue
            best_step, best = min(
                ((index, value) for index, value in enumerate(values) if math.isfinite(value)),
                key=lambda item: item[1],
            )
            f0 = values[0]
            endpoint_value = values[endpoint]
            if method in ARMS:
                proj = projection[(distribution, severity, method)]
                proj_value = proj["projection_mean"]
                condition = proj["condition_median"]
            else:
                proj_value = ""
                condition = ""
            result.append({
                "distribution": distribution,
                "severity": severity,
                "loss": method,
                "F0": f0,
                "F6": values[6],
                "F8": values[8],
                "F12": values[12],
                "F18": values[18],
                "best": best,
                "best_step": best_step,
                "endpoint_vs_F0": (endpoint_value - f0) / max(f0, 1e-12),
                "endpoint_over_best": endpoint_value / max(best, 1e-12),
                "projection_residual": proj_value,
                "condition_number": condition,
            })
    return result


def _ood_effects(primary, endpoint):
    lookup = {(row["distribution"], row["severity"], row["loss"]): row for row in primary}
    groups = sorted({(row["distribution"], row["severity"]) for row in primary})
    result = []
    for distribution, severity in groups:
        phi = lookup[(distribution, severity, "phi_mse")]
        nf = lookup[(distribution, severity, "normalized_functional")]
        nm = lookup[(distribution, severity, "normalized_mahalanobis")]
        for iteration in (8, 12, 18, endpoint):
            field = f"F{iteration}"
            result.append({
                "distribution": distribution,
                "severity": severity,
                "iteration": iteration,
                "phi_mse": phi[field],
                "normalized_functional": nf[field],
                "normalized_mahalanobis": nm[field],
                "NF_vs_phi_effect": (float(phi[field]) - float(nf[field])) / max(float(phi[field]), 1e-12),
                "NM_vs_phi_effect": (float(phi[field]) - float(nm[field])) / max(float(phi[field]), 1e-12),
                "NM_vs_NF_effect": (float(nf[field]) - float(nm[field])) / max(float(nf[field]), 1e-12),
            })
    return result


def _conditioning(new_rows, projection_rows):
    records = {
        (row["bucket"], row["severity"], row["target_id"], row["seed"], row["support_size"], row["episode"], row["arm"], int(row["iteration"]) - 1): row
        for row in new_rows if int(row["iteration"]) > 0
    }
    projections = {
        (row["bucket"], row["severity"], row["target_id"], row["seed"], row["support_size"], row["episode"], row["arm"], int(row["iteration"])): row
        for row in projection_rows
    }
    result = []
    for arm in ARMS:
        x, phi_error, functional_error, mse_change = [], [], [], []
        for key, projection in projections.items():
            if key[6] != arm or key not in records:
                continue
            condition = float(projection["condition_number"])
            row = records[key]
            if condition > 0 and math.isfinite(condition):
                x.append(math.log10(condition))
                phi_error.append(float(row["phi_prediction_mse"]))
                functional_error.append(float(row["normalized_functional_error"]))
                mse_change.append(float(row["next_mse_change"]))
        for metric, values in (
            ("phi_prediction_error", phi_error),
            ("normalized_functional_error", functional_error),
            ("next_step_mse_change", mse_change),
        ):
            result.append({
                "loss": arm,
                "metric": metric,
                "condition_transform": "log10",
                "pearson": _pearson(x, values),
                "spearman": _pearson(_rank(x), _rank(values)),
                "n": len(x),
            })
    # Paired per-state loss advantages use the phi-arm state's condition number.
    state_groups = defaultdict(dict)
    for key, row in records.items():
        state_key = key[:6] + (key[7],)
        state_groups[state_key][key[6]] = row
    phi_conditions = {}
    for key, row in projections.items():
        if key[6] == "phi_mse":
            phi_conditions[key[:6] + (key[7],)] = float(row["condition_number"])
    for comparison, arm in (("NF_advantage_over_phi", "normalized_functional"), ("NM_advantage_over_phi", "normalized_mahalanobis")):
        x, advantage = [], []
        for state, methods in state_groups.items():
            if "phi_mse" not in methods or arm not in methods or state not in phi_conditions:
                continue
            condition = phi_conditions[state]
            if condition <= 0 or not math.isfinite(condition):
                continue
            x.append(math.log10(condition))
            advantage.append(float(methods["phi_mse"]["query_mse"]) - float(methods[arm]["query_mse"]))
        result.append({
            "loss": arm,
            "metric": comparison,
            "condition_transform": "log10_phi_arm",
            "pearson": _pearson(x, advantage),
            "spearman": _pearson(_rank(x), _rank(advantage)),
            "n": len(x),
        })
    return result


def _geometry_summary(rows, calibration: LossCalibration):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["distribution"], row["severity"], row["arm"])].append(row)
    result = []
    for (distribution, severity, arm), values in sorted(grouped.items()):
        result.append({
            "distribution": distribution,
            "severity": severity,
            "loss": arm,
            "rank_mean": mean(float(row["rank"]) for row in values),
            "effective_rank_mean": mean(float(row["effective_rank"]) for row in values),
            "condition_before_median": _quantile([float(row["condition_before"]) for row in values if math.isfinite(float(row["condition_before"]))], 0.5),
            "condition_after_median": _quantile([float(row["condition_after"]) for row in values], 0.5),
            "top1_update_energy_mean": mean(float(row["top1_update_energy"]) for row in values),
            "top3_update_energy_mean": mean(float(row["top3_update_energy"]) for row in values),
            "top5_update_energy_mean": mean(float(row["top5_update_energy"]) for row in values),
            **{f"eigenvalue_{index}_mean": mean(float(row[f"eigenvalue_{index}"]) for row in values) for index in range(1, 8)},
            "n": len(values),
        })
    spectrum = calibration.diagnostics["leading_eigenvalue_median"]
    result.append({
        "distribution": "train_latents",
        "severity": "training_diagnostic",
        "loss": "teacher_geometry",
        "rank_mean": calibration.diagnostics["rank"]["median"],
        "effective_rank_mean": calibration.diagnostics["effective_rank_median"],
        "condition_before_median": "",
        "condition_after_median": calibration.diagnostics["regularized_condition_median"],
        "top1_update_energy_mean": "",
        "top3_update_energy_mean": "",
        "top5_update_energy_mean": "",
        **{f"eigenvalue_{index}_mean": spectrum[index - 1] for index in range(1, 8)},
        "n": calibration.diagnostics["states"],
    })
    return result


def _training_summary(output: Path):
    result = []
    for path in sorted(output.glob("training_history_seed_*.json")):
        seed = int(path.stem.rsplit("_", 1)[1])
        history = json.loads(path.read_text(encoding="utf-8"))
        for row in history:
            for arm in ARMS:
                result.append({
                    "seed": seed,
                    "step": int(row["step"]),
                    "loss": arm,
                    "objective": row[f"{arm}_loss"],
                    "numerator": row[f"{arm}_numerator"],
                    "denominator": row[f"{arm}_denominator"],
                    "phi_mse": row[f"{arm}_phi_mse"],
                    "functional_mse": row[f"{arm}_functional_mse"],
                    "action_norm": row[f"{arm}_action_norm"],
                    "gradient_norm": row[f"{arm}_gradient_norm"],
                    "gradient_clipped": row[f"{arm}_gradient_clipped"],
                    "elapsed_seconds": row["elapsed_seconds"],
                })
    return result


def _plots(output, trajectories, primary, projections, conditioning, geometry):
    paths = []
    bucket_files = (
        ("heldout_combinations", "plot_1_heldout_rollout.png"),
        ("coefficient_extrapolation", "plot_2_coefficient_ood.png"),
        ("structural_shift", "plot_3_structural_ood.png"),
        ("compositional_shift", "plot_4_compositional_ood.png"),
    )
    for bucket, filename in bucket_files:
        rows = [row for row in trajectories if row["distribution"] == bucket and row["loss_or_baseline"] in PRIMARY_METHODS]
        severities = sorted({row["severity"] for row in rows})
        fig, axes = plt.subplots(1, len(severities), figsize=(6 * len(severities), 4), squeeze=False)
        for axis, severity in zip(axes[0], severities):
            for method in PRIMARY_METHODS:
                points = sorted((int(row["iteration"]), float(row["mse_mean"])) for row in rows if row["severity"] == severity and row["loss_or_baseline"] == method)
                if points:
                    axis.plot([x for x, _ in points], [y for _, y in points], label=method)
            axis.set_title(f"{bucket}: {severity}")
            axis.set_yscale("log")
            axis.set_xlabel("iteration")
            axis.set_ylabel("MSE")
            axis.grid(alpha=0.25)
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    fig, axis = plt.subplots(figsize=(9, 4))
    labels = [f"{row['distribution']}:{row['severity']}:{row['loss']}" for row in primary if row["loss"] in ARMS]
    ratios = [float(row["endpoint_over_best"]) for row in primary if row["loss"] in ARMS]
    axis.bar(range(len(labels)), ratios)
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    axis.set_ylabel("F18 / best MSE")
    fig.tight_layout()
    path = output / "plot_5_endpoint_best_ratio.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    for metric, filename, ylabel in (
        ("projection_mean", "plot_6_projection_residual.png", "projection residual"),
        ("condition_median", "plot_7_condition_number.png", "median condition number"),
    ):
        fig, axis = plt.subplots(figsize=(9, 4))
        labels = [f"{row['distribution']}:{row['severity']}:{row['loss']}" for row in projections]
        values = [float(row[metric]) for row in projections]
        axis.bar(range(len(labels)), values)
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    fig, axis = plt.subplots(figsize=(7, 4))
    advantage_rows = [row for row in conditioning if "advantage" in row["metric"]]
    axis.bar([row["metric"] for row in advantage_rows], [float(row["spearman"]) for row in advantage_rows])
    axis.set_ylabel("Spearman rho vs log10(condition)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = output / "plot_8_loss_advantage_vs_condition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    fig, axis = plt.subplots(figsize=(8, 4))
    groups = [row for row in geometry if row["loss"] in {"normalized_mahalanobis", "teacher_geometry"}]
    for row in groups:
        eigenvalues = [float(row[f"eigenvalue_{index}_mean"]) for index in range(1, 8)]
        axis.plot(range(1, 8), eigenvalues, marker="o", label=f"{row['distribution']}:{row['severity']}")
    axis.set_yscale("log")
    axis.set_xlabel("eigenvalue index")
    axis.set_ylabel("mean/median covariance eigenvalue")
    axis.legend(fontsize=6)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = output / "plot_9_mahalanobis_eigenspectra.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_v32_report(
    config: V32Config,
    output_dir: str | Path,
    calibration: LossCalibration,
) -> dict[str, object]:
    output = Path(output_dir)
    new_rows = _read(output / "evaluation_records.csv")
    historical = _read(Path(config.source_v31_dir) / "evaluation_records.csv")
    raw_projection = _read(output / "projection_records.csv")
    raw_geometry = _read(output / "mahalanobis_geometry_records.csv")
    trajectories = _trajectory_rows(new_rows, historical)
    seeds = _seed_rows(new_rows, historical, config.evaluation_rollout)
    projections = _projection_summary(raw_projection)
    primary = _primary_comparison(trajectories, projections, config.evaluation_rollout)
    effects = _ood_effects(primary, config.evaluation_rollout)
    conditioning = _conditioning(new_rows, raw_projection)
    geometry = _geometry_summary(raw_geometry, calibration)
    training = _training_summary(output)
    _write(output / "training_summary.csv", training)
    _write(output / "trajectory_summary.csv", trajectories)
    _write(output / "seed_results.csv", seeds)
    _write(output / "ood_summary.csv", effects)
    _write(output / "projection_summary.csv", projections)
    _write(output / "conditioning_analysis.csv", conditioning)
    _write(output / "mahalanobis_geometry_summary.csv", geometry)
    _write(output / "primary_comparison.csv", primary)
    plots = _plots(output, trajectories, primary, projections, conditioning, geometry)
    lookup = {(row["distribution"], row["severity"], row["loss"]): row for row in primary}
    groups = sorted({(row["distribution"], row["severity"]) for row in primary})
    nf_effects = []
    nm_effects = []
    nm_nf_effects = []
    for group in groups:
        phi = lookup[group + ("phi_mse",)]
        nf = lookup[group + ("normalized_functional",)]
        nm = lookup[group + ("normalized_mahalanobis",)]
        nf_effects.append((float(phi["F18"]) - float(nf["F18"])) / max(float(phi["F18"]), 1e-12))
        nm_effects.append((float(phi["F18"]) - float(nm["F18"])) / max(float(phi["F18"]), 1e-12))
        nm_nf_effects.append((float(nf["F18"]) - float(nm["F18"])) / max(float(nf["F18"]), 1e-12))
    threshold = config.material_effect_threshold
    nf_wins = sum(effect >= threshold for effect in nf_effects)
    nm_wins = sum(effect >= threshold for effect in nm_effects)
    nm_over_nf = sum(effect >= threshold for effect in nm_nf_effects)
    endpoint_stable = {
        arm: sum(float(lookup[group + (arm,)]["endpoint_over_best"]) <= 1.05 for group in groups)
        for arm in ARMS
    }
    operator_expressive = all(float(row["outside_span_fraction"]) < 0.5 for row in projections)
    if (nf_wins or nm_wins) and max(endpoint_stable["normalized_functional"], endpoint_stable["normalized_mahalanobis"]) > endpoint_stable["phi_mse"]:
        primary_verdict = "COORDINATE-LOSS PATHOLOGY SUPPORTED"
    else:
        primary_verdict = "LOSS GEOMETRY INSUFFICIENT"
    if nf_wins >= nm_over_nf:
        geometry_verdict = "GENERIC FUNCTIONAL INVARIANCE AT LEAST AS SUFFICIENT AS TEACHER GEOMETRY"
    else:
        geometry_verdict = "TEACHER-INDUCED GEOMETRY SUPPORTED"
    questions = {
        "normalized_functional_outperforms_phi": nf_wins >= 4,
        "delays_or_eliminates_horizon_failure": endpoint_stable["normalized_functional"] > endpoint_stable["phi_mse"],
        "mahalanobis_improves_over_nf": nm_over_nf >= 4,
        "operator_remains_expressive": operator_expressive,
        "primary_verdict": primary_verdict,
        "geometry_verdict": geometry_verdict,
    }
    table = [
        "| Distribution | Loss | F0 | F6 | F8 | F12 | F18 | Best | Best step | F18/Best | Projection | Condition |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        table.append(
            f"| {row['distribution']}:{row['severity']} | {row['loss']} | {float(row['F0']):.4g} | {float(row['F6']):.4g} | {float(row['F8']):.4g} | {float(row['F12']):.4g} | {float(row['F18']):.4g} | {float(row['best']):.4g} | {row['best_step']} | {float(row['endpoint_over_best']):.3f} | {row['projection_residual']} | {row['condition_number']} |"
        )
    report = f"""# IFI v3.2 — Function-space Phase-B supervision

## Locked verdict

**{primary_verdict}**

Secondary geometry verdict: **{geometry_verdict}**

## Primary comparison

{chr(10).join(table)}

## Explicit answers

1. NF materially beats pure phi-MSE in {nf_wins}/7 F18 distributions.
2. NF improves endpoint stability in {endpoint_stable['normalized_functional']} groups versus {endpoint_stable['phi_mse']} for phi-MSE.
3. NM materially beats NF in {nm_over_nf}/7 F18 distributions.
4. Correlations with coordinate conditioning are reported in `conditioning_analysis.csv`; they are descriptive, not causal.
5–7. Coefficient, structural, and compositional effects are reported per horizon in `ood_summary.csv` and per seed in `seed_results.csv`.
8. Non-destructive continuation criterion (F18 within 5% of best): phi={endpoint_stable['phi_mse']}/7, NF={endpoint_stable['normalized_functional']}/7, NM={endpoint_stable['normalized_mahalanobis']}/7.
9. Frozen O_phi remains expressive: {operator_expressive}.
10. Primary remaining failure category: {primary_verdict}.

No stopping, damping, clipping change, OOD tuning, or oracle stopping was used. All OOD fixtures are the exact v3.1 registry.
"""
    (output / "verdict.md").write_text(report, encoding="utf-8")
    (output / "verdict.json").write_text(json.dumps({
        **questions,
        "nf_f18_effects": nf_effects,
        "nm_f18_effects": nm_effects,
        "nm_vs_nf_f18_effects": nm_nf_effects,
        "endpoint_stable_groups": endpoint_stable,
    }, indent=2), encoding="utf-8")
    return {"verdict": primary_verdict, "questions": questions, "plots": [str(path) for path in plots]}
