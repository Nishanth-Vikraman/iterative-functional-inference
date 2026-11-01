from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .v40_config import V40Config


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _rollout_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["bucket"], row["model"], int(row["iteration"]))].append(row)
    output = []
    for (bucket, model, iteration), selected in sorted(grouped.items()):
        norms = [float(row["residual_norm"]) for row in selected]
        q = [float(row["q"]) for row in selected if row["q"] != ""]
        output.append({
            "bucket": bucket,
            "model": model,
            "iteration": iteration,
            "residual_norm_median": _median(norms),
            "residual_norm_mean": statistics.fmean(norms),
            "q_median": _median(q),
            "q_fraction_below_one": (
                sum(value < 1 for value in q) / len(q) if q else float("nan")
            ),
            "converged_fraction": sum(row["converged"] == "True" for row in selected)
            / len(selected),
            "diverged_fraction": sum(row["diverged"] == "True" for row in selected)
            / len(selected),
            "action_norm_median": _median([float(row["action_norm"]) for row in selected]),
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _find(
    summary: list[dict[str, object]], bucket: str, model: str, iteration: int
) -> dict[str, object]:
    return next(
        row
        for row in summary
        if row["bucket"] == bucket
        and row["model"] == model
        and row["iteration"] == iteration
    )


def _plot(output: Path, summary: list[dict[str, object]]) -> None:
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for axis, bucket in zip(axes, ("same_family", "condition_4_6")):
        available = [row for row in summary if row["bucket"] == bucket]
        if not available:
            axis.set_visible(False)
            continue
        for model in sorted({str(row["model"]) for row in available}):
            rows = [row for row in available if row["model"] == model]
            axis.plot(
                [int(row["iteration"]) for row in rows],
                [float(row["residual_norm_median"]) for row in rows],
                label=model,
            )
        axis.set_yscale("log")
        axis.set_title(bucket)
        axis.set_xlabel("Frozen inference iteration")
        axis.set_ylabel("Median residual norm")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "rollout_generalization.png", dpi=160)
    plt.close(fig)


def write_report(config: V40Config, output: Path, stopped_by_gate: bool) -> dict[str, object]:
    gate = json.loads((output / "contraction_gate.json").read_text(encoding="utf-8"))
    rollout_rows = _read(output / "eval_rollouts.csv")
    intervention_rows = _read(output / "history_interventions.csv")
    summary = _rollout_summary(rollout_rows)
    pool_rows = _read(output / "pool_diagnostics.csv")
    _write_csv(output / "eval_summary.csv", summary)
    if stopped_by_gate or not summary:
        verdict = "NOT SUPPORTED — BASIC CONTRACTION GATE FAILED"
        local_supported = False
        rollout_supported = False
        history_supported = False
        report = [
            f"# IFI {config.experiment_version} — Population-Replay Residual Learner",
            "",
            f"## Verdict: {verdict}",
            "",
            "The preregistered step-2000 contraction gate failed, so training stopped without architecture tuning.",
            "",
            f"- Gate median q: {gate['median_q']:.6g} (required < {gate['median_q_threshold']})",
            f"- Gate F8/F0 median: {gate['median_endpoint_over_initial']:.6g} (required < {gate['endpoint_ratio_threshold']})",
        ]
    else:
        f0 = _find(summary, "same_family", "query_history", 0)
        f64 = _find(summary, "same_family", "query_history", config.evaluation_rollout)
        current64 = _find(summary, "same_family", "current_only", config.evaluation_rollout)
        ar64 = _find(summary, "same_family", "ar_no_fresh", config.evaluation_rollout)
        query_rows = [
            row
            for row in summary
            if row["bucket"] == "same_family"
            and row["model"] == "query_history"
            and int(row["iteration"]) > 0
        ]
        long_rows = [row for row in query_rows if int(row["iteration"]) > 8]
        query_norms = [float(row["residual_norm_median"]) for row in query_rows]
        first_systematic_degradation = None
        for index in range(2, len(query_norms)):
            if query_norms[index - 2] < query_norms[index - 1] < query_norms[index]:
                first_systematic_degradation = int(query_rows[index - 2]["iteration"])
                break
        local_supported = (
            float(f64["residual_norm_median"]) < 0.8 * float(f0["residual_norm_median"])
            and float(f64["residual_norm_median"])
            < 0.95 * float(current64["residual_norm_median"])
            and float(f64["residual_norm_median"])
            < 0.95 * float(ar64["residual_norm_median"])
            and _median([float(row["q_median"]) for row in query_rows]) < 1
            and _median([float(row["q_fraction_below_one"]) for row in query_rows]) > 0.6
        )
        f32 = _find(summary, "same_family", "query_history", 32)
        rollout_supported = (
            float(f32["residual_norm_median"]) < 0.5 * float(f0["residual_norm_median"])
            and float(f64["residual_norm_median"]) < float(f0["residual_norm_median"])
            and _median([float(row["q_median"]) for row in long_rows]) < 1
            and float(f64["diverged_fraction"]) < 0.01
        )
        intervention_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in intervention_rows:
            intervention_grouped[row["condition"]].append(row)
        intervention = {
            condition: {
                "one_step_q": _median([float(row["one_step_q"]) for row in rows]),
                "endpoint_ratio": _median([
                    float(row["endpoint_over_intervention_residual"]) for row in rows
                ]),
                "delta_phi": _median([float(row["delta_phi"]) for row in rows]),
            }
            for condition, rows in intervention_grouped.items()
        }
        correct = intervention["correct"]
        history_supported = all(
            correct["one_step_q"] < 0.95 * intervention[condition]["one_step_q"]
            and correct["endpoint_ratio"]
            < 0.95 * intervention[condition]["endpoint_ratio"]
            for condition in ("no_history", "wrong_world")
        )
        if local_supported and rollout_supported and history_supported:
            verdict = "LOCAL CORRECTION, ROLLOUT GENERALIZATION, AND HISTORY IDENTIFICATION SUPPORTED"
        elif local_supported and rollout_supported:
            verdict = "LOCAL CORRECTION AND ROLLOUT GENERALIZATION SUPPORTED; HISTORY NOT SUPPORTED"
        elif local_supported:
            verdict = "LOCAL CORRECTION LEARNED; LONG-HORIZON COMPOSITION NOT SUPPORTED"
        else:
            verdict = "NOT SUPPORTED"
        report = [
            f"# IFI {config.experiment_version} — Population-Replay Residual Learner",
            "",
            f"## Verdict: {verdict}",
            "",
            "All training gradients were local one-step contraction gradients from asynchronously sampled real population states. No teacher, action target, BPTT, position/age input, or oracle action entered training.",
            "",
            "## Same-family frozen rollout",
            "",
            "| Model | F0 median | F8 | F16 | F32 | F64 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model in ("query_history", "current_only", "ar_no_fresh", "fixed_linear", "analytic_oracle"):
            values = [
                float(_find(summary, "same_family", model, iteration)["residual_norm_median"])
                for iteration in (0, 8, 16, 32, 64)
            ]
            report.append(
                f"| {model} | " + " | ".join(f"{value:.6g}" for value in values) + " |"
            )
        report.extend([
            "",
            "## Preregistered criteria",
            "",
            f"- Local correction law learned: **{local_supported}**",
            f"- Rollout generalization supported: **{rollout_supported}**",
            f"- History-based system identification supported: **{history_supported}**",
            f"- Step-2000 contraction gate: **{gate['passed']}**",
            f"- First three-step systematic degradation onset: **{first_systematic_degradation}**",
            f"- Median q across F1–F64: **{_median([float(row['q_median']) for row in query_rows]):.6g}**",
            f"- Median fraction of episodes with q<1: **{_median([float(row['q_fraction_below_one']) for row in query_rows]):.6g}**",
            f"- Catastrophic divergence fraction at F64: **{float(f64['diverged_fraction']):.6g}**",
            "",
            "History interventions hold current residual fixed and alter only evidence history. The full per-episode results are in `history_interventions.csv`.",
        ])
        query_pool = [row for row in pool_rows if row["arm"] == "query_history"]
        if query_pool:
            final_pool = query_pool[-1]
            report.extend([
                "",
                "## Anti-shortcut diagnostic",
                "",
                f"- Final residual-norm/trajectory-age correlation: **{float(final_pool['residual_norm_age_correlation']):.6g}**",
                f"- Final population median age: **{float(final_pool['age_median']):.6g}**",
                f"- Final population q90 age: **{float(final_pool['age_q90']):.6g}**",
                "",
                "Age is logged only for diagnostics and is never supplied to a controller.",
            ])
        _plot(output, summary)
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    verdict_payload = {
        "verdict": verdict,
        "local_correction_law_learned": local_supported,
        "rollout_generalization_supported": rollout_supported,
        "history_system_identification_supported": history_supported,
        "stopped_by_gate": stopped_by_gate,
        "gate": gate,
        "first_systematic_degradation_step": (
            None if stopped_by_gate or not summary else first_systematic_degradation
        ),
    }
    (output / "verdict.json").write_text(
        json.dumps(verdict_payload, indent=2), encoding="utf-8"
    )
    return verdict_payload
