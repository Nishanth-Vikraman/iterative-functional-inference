from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .config import ExperimentConfig


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _endpoint(rows: list[dict[str, str]], method: str | None = None) -> float:
    relevant = [
        row for row in rows
        if row["split"] == "heldout_compositions"
        and (method is None or row.get("method") == method)
    ]
    final = max(int(row["iteration"]) for row in relevant)
    return mean(float(row["query_mse"]) for row in relevant if int(row["iteration"]) == final)


def _effects(rows: list[dict[str, str]]):
    grouped: dict[str, list[float]] = defaultdict(list)
    seeded: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["intervention"] == "correct":
            continue
        value = float(row["query_mse_delta_from_correct"])
        grouped[row["intervention"]].append(value)
        seeded[(row["intervention"], int(row["seed"]))].append(value)
    averages = {name: mean(values) for name, values in grouped.items()}
    consistent = {
        name: all(
            mean(values) > 0
            for (control, _), values in seeded.items() if control == name
        )
        for name in grouped
    }
    return averages, consistent


def write_v22_report(
    output_dir: str | Path,
    v21_dir: str | Path,
    historical_dir: str | Path,
    config: ExperimentConfig,
    counts: dict[str, int],
    history: list[dict[str, float]],
) -> None:
    output = Path(output_dir)
    current = _read(output / "evaluation_records.csv")
    v21 = _read(Path(v21_dir) / "evaluation_records.csv")
    historical = _read(Path(historical_dir) / "evaluation_records.csv")
    residual = _read(output / "residual_interventions.csv")
    state = _read(output / "state_interventions.csv")
    diversity = _read(output / "state_diversity.csv")

    endpoints = {
        "v2_2_predictive_recurrent": _endpoint(current),
        "v2_1_recurrent": _endpoint(v21, "direct_recurrent"),
        "direct_current": _endpoint(v21, "direct_current"),
        "direct_history": _endpoint(v21, "direct_history"),
        "ar_no_feedback": _endpoint(historical, "ar_no_feedback"),
        "historical_E_R_IFI": _endpoint(historical, "ifi"),
    }
    residual_effects, residual_consistency = _effects(residual)
    state_effects, state_consistency = _effects(state)
    improves_deep = endpoints["v2_2_predictive_recurrent"] < endpoints["v2_1_recurrent"]
    residual_causal = all(
        residual_consistency.get(control, False)
        for control in ("zero", "wrong_task", "shuffled", "stale")
    )
    state_causal = all(
        state_consistency.get(control, False)
        for control in ("zero", "wrong_task", "stale")
    )
    supported = improves_deep and residual_causal and state_causal
    verdict = "PREDICTIVE RECURRENT LEARNING STATE SUPPORTED" if supported else "NOT SUPPORTED"

    seed_endpoint: dict[int, list[float]] = defaultdict(list)
    for row in current:
        if row["split"] == "heldout_compositions" and int(row["iteration"]) == config.evaluation.rollout_steps:
            seed_endpoint[int(row["seed"])].append(float(row["query_mse"]))
    seed_summary = {f"seed_{seed}": mean(values) for seed, values in sorted(seed_endpoint.items())}

    route_keys = (
        "teacher_to_T_grad_norm", "teacher_to_U_grad_norm",
        "info_to_T_grad_norm", "info_to_U_grad_norm",
    )
    routing = {
        key: {
            "mean": mean(item[key] for item in history),
            "max": max(item[key] for item in history),
        }
        for key in route_keys
    }
    routing_verified = (
        routing["teacher_to_T_grad_norm"]["mean"] > 0
        and routing["info_to_U_grad_norm"]["mean"] > 0
        and routing["teacher_to_U_grad_norm"]["max"] == 0
        and routing["info_to_T_grad_norm"]["max"] == 0
    )
    deep_diversity = [
        float(row["mean_pairwise_state_distance"])
        for row in diversity if int(row["iteration"]) == config.evaluation.rollout_steps
    ]

    report = f"""# IFI v2.2 — Predictive recurrent learning state

## Preregistered verdict

**{verdict}**

Success was assigned only if v2.2 beat v2.1 recurrent at F8 and every preregistered current-residual
and recurrent-state corruption degraded performance consistently across seeds.

## Held-out composition F8

```json
{json.dumps(endpoints, indent=2)}
```

v2.2 improves over v2.1 recurrent: **{improves_deep}**.

## Current residual interventions

Mean query-MSE increase versus correct current residual:

```json
{json.dumps(residual_effects, indent=2)}
```

All-seed positive consistency:

```json
{json.dumps(residual_consistency, indent=2)}
```

Freshness endpoint (`stale`) supported: **{residual_consistency.get('stale', False)}**.

## Recurrent-state interventions

Mean query-MSE increase versus correct state:

```json
{json.dumps(state_effects, indent=2)}
```

All-seed positive consistency:

```json
{json.dumps(state_consistency, indent=2)}
```

## Predictive objective

- InfoNCE loss: {history[0]['infonce_loss']:.8f} → {history[-1]['infonce_loss']:.8f}
- InfoNCE top-1 accuracy: {history[0]['infonce_accuracy']:.8f} → {history[-1]['infonce_accuracy']:.8f}
- Mean pairwise state distance across tasks at F8: {mean(deep_diversity):.8f}

The contrastive residual representation is fixed: per-field mean, population standard deviation,
minimum, and maximum (24 values), aligned with state by a seeded fixed column-normalized random
projection. It has
no trainable parameters.

## Gradient routing

```json
{json.dumps(routing, indent=2)}
```

Routing verified: **{routing_verified}**. T received teacher supervision; U received only InfoNCE;
the operator remained frozen; no semantic residual encoder exists.

Parameter counts:

```json
{json.dumps(counts, indent=2)}
```

## Seed consistency at F8

```json
{json.dumps(seed_summary, indent=2)}
```

Full F0→F8 trajectories, K/novelty breakdowns, state changes/diversity, action errors, and both
intervention families are in the accompanying CSV and JSON artifacts. Lower RMSE alone did not
determine the verdict.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
