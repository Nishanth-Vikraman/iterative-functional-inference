from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .v30_config import V30Config


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mse(rows, method, iteration, split="heldout_latents", support=None, seed=None):
    values = [
        float(row["query_mse"])
        for row in rows
        if row["split"] == split
        and row["method"] == method
        and int(row["iteration"]) == iteration
        and (support is None or int(row["support_size"]) == support)
        and (seed is None or int(row["seed"]) == seed)
    ]
    return mean(values)


def _intervention_effects(rows, method, split="heldout_latents"):
    grouped = defaultdict(list)
    seeded = defaultdict(list)
    for row in rows:
        if row["split"] != split or row["method"] != method:
            continue
        if row["intervention"] == "correct":
            continue
        value = float(row["query_mse_delta_from_correct"])
        grouped[row["intervention"]].append(value)
        seeded[(row["intervention"], int(row["seed"]))].append(value)
    effects = {key: mean(values) for key, values in grouped.items()}
    consistent = {
        key: all(
            mean(values) > 0
            for (control, _), values in seeded.items()
            if control == key
        )
        for key in grouped
    }
    return effects, consistent


def write_v30_report(
    output_dir,
    config: V30Config,
    counts,
    phase_a_metrics,
    training_history,
):
    output = Path(output_dir)
    records = _read(output / "evaluation_records.csv")
    residual_rows = _read(output / "residual_interventions.csv")
    history_rows = _read(output / "history_interventions.csv")
    gate = json.loads((output / "gate" / "summary.json").read_text(encoding="utf-8"))
    primary = config.evaluation.rollout_steps
    extended = config.evaluation.extended_rollout_steps
    methods = (
        "direct_current", "full_history", "query_history",
        "ar_no_feedback", "bayesian_teacher",
    )
    endpoints = {
        f"{method}_F{primary}": _mse(records, method, primary)
        for method in methods
    }
    endpoints.update({
        f"{method}_F{extended}": _mse(records, method, extended)
        for method in methods
    })
    endpoints["F0"] = _mse(records, "f0", 0)
    trajectories = {}
    best = {}
    increments = {}
    accelerations = {}
    for method in methods:
        trajectory = [_mse(records, method, iteration) for iteration in range(1, extended + 1)]
        trajectory = [endpoints["F0"], *trajectory]
        trajectories[method] = trajectory
        best_iteration = min(range(len(trajectory)), key=lambda index: trajectory[index])
        best[method] = {
            "iteration": best_iteration,
            "mse": trajectory[best_iteration],
        }
        delta = [trajectory[index - 1] - trajectory[index] for index in range(1, len(trajectory))]
        acceleration = [delta[index] - delta[index - 1] for index in range(1, len(delta))]
        increments[method] = delta
        accelerations[method] = acceleration

    residual = {}
    residual_consistency = {}
    for method in ("direct_current", "full_history", "query_history"):
        residual[method], residual_consistency[method] = _intervention_effects(
            residual_rows, method
        )
    history = {}
    history_consistency = {}
    for method in ("full_history", "query_history"):
        history[method], history_consistency[method] = _intervention_effects(
            history_rows, method
        )

    direct_endpoint = endpoints[f"direct_current_F{primary}"]
    material = {}
    for method in ("full_history", "query_history"):
        endpoint = endpoints[f"{method}_F{primary}"]
        material[method] = (direct_endpoint - endpoint) / direct_endpoint >= 0.05
    key_history = (
        "remove_all", "wrong_task", "reversed",
        "shuffled_temporal", "break_correspondence",
    )
    key_residual = ("zero", "wrong_task", "shuffled")
    history_causal = {
        method: all(history_consistency[method].get(control, False) for control in key_history)
        for method in ("full_history", "query_history")
    }
    residual_causal = {
        method: all(residual_consistency[method].get(control, False) for control in key_residual)
        for method in ("full_history", "query_history")
    }
    supported_by = [
        method for method in ("full_history", "query_history")
        if material[method] and history_causal[method] and residual_causal[method]
    ]
    history_supported = gate["passed"] and bool(supported_by)
    a_endpoint = endpoints[f"full_history_F{primary}"]
    b_endpoint = endpoints[f"query_history_F{primary}"]
    a_history_effect = mean(history["full_history"].get(key, 0.0) for key in key_history)
    b_history_effect = mean(history["query_history"].get(key, 0.0) for key in key_history)
    query_supported = (
        history_supported
        and (a_endpoint - b_endpoint) / a_endpoint >= 0.05
        and b_history_effect > a_history_effect
        and "query_history" in supported_by
    )
    verdict = (
        "CURRENT-RESIDUAL QUERY HISTORY SUPPORTED"
        if query_supported
        else "HISTORY-DEPENDENT IFI SUPPORTED"
        if history_supported
        else "HISTORY NOT USED"
    )

    by_k = {
        f"{method}/K{support}/F{primary}": _mse(
            records, method, primary, support=support
        )
        for method in methods
        for support in config.tasks.support_sizes
    }
    by_seed = {
        f"{method}/seed{seed}/F{primary}": _mse(
            records, method, primary, seed=seed
        )
        for method in ("direct_current", "full_history", "query_history")
        for seed in config.evaluation.seeds
    }
    by_split = {
        f"{method}/{split}/F{primary}": _mse(
            records, method, primary, split=split
        )
        for method in methods
        for split in ("train_latents", "heldout_latents")
    }
    clipping = mean(row["gradient_clipped"] for row in training_history)
    operator_gradient = max(row["operator_gradient"] for row in training_history)
    report = f"""# IFI v3.0 — History-necessary Bayesian latent teacher

## Preregistered verdict

**{verdict}**

History-necessity gate: **{gate['verdict']}**.

## Gate

```json
{json.dumps(gate, indent=2)}
```

## Held-out latent endpoints

```json
{json.dumps(endpoints, indent=2)}
```

Models materially beating direct-current (locked 5% relative margin):

```json
{json.dumps(material, indent=2)}
```

## Full rollout trajectories

```json
{json.dumps(trajectories, indent=2)}
```

Best iterations:

```json
{json.dumps(best, indent=2)}
```

Learning increments:

```json
{json.dumps(increments, indent=2)}
```

Adaptation accelerations:

```json
{json.dumps(accelerations, indent=2)}
```

## Current-residual interventions

Mean held-out MSE increase:

```json
{json.dumps(residual, indent=2)}
```

All-seed directional consistency:

```json
{json.dumps(residual_consistency, indent=2)}
```

## Explicit-history interventions

Mean held-out MSE increase:

```json
{json.dumps(history, indent=2)}
```

All-seed directional consistency:

```json
{json.dumps(history_consistency, indent=2)}
```

History causality checks: `{json.dumps(history_causal)}`.
Residual causality checks: `{json.dumps(residual_causal)}`.

## K dependence

```json
{json.dumps(by_k, indent=2)}
```

## Latent novelty

```json
{json.dumps(by_split, indent=2)}
```

## Seed consistency

```json
{json.dumps(by_seed, indent=2)}
```

## Phase A and optimization integrity

```json
{json.dumps(phase_a_metrics, indent=2)}
```

Parameter counts:

```json
{json.dumps(counts, indent=2)}
```

Gradient-clipping fraction: `{clipping}`. Frozen-operator gradient maximum: `{operator_gradient}`.
No E_R, recurrent h, separate U, InfoNCE, posterior feature, or latent identity was supplied to
any controller. All negative outcomes are retained.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return verdict
