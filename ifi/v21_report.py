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


def _endpoint(rows: list[dict[str, str]], split: str) -> dict[str, float]:
    relevant = [row for row in rows if row["split"] == split]
    final_by_method = {
        method: max(int(row["iteration"]) for row in relevant if row["method"] == method)
        for method in {row["method"] for row in relevant}
    }
    values: dict[str, list[float]] = defaultdict(list)
    for row in relevant:
        if int(row["iteration"]) == final_by_method[row["method"]]:
            values[row["method"]].append(float(row["query_mse"]))
    return {method: mean(items) for method, items in values.items()}


def write_v21_report(
    output_dir: str | Path,
    historical_dir: str | Path,
    config: ExperimentConfig,
    parameter_counts: dict[str, int],
    history: list[dict[str, float]],
) -> None:
    output = Path(output_dir)
    direct = _read(output / "evaluation_records.csv")
    interventions = _read(output / "feedback_interventions.csv")
    historical = _read(Path(historical_dir) / "evaluation_records.csv")
    direct_endpoint = _endpoint(direct, "heldout_compositions")
    historical_endpoint = _endpoint(historical, "heldout_compositions")

    effects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seed_effects: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in interventions:
        if row["intervention"] == "correct":
            continue
        value = float(row["query_mse_delta_from_correct"])
        effects[row["variant"]][row["intervention"]].append(value)
        seed_effects[(row["variant"], row["intervention"], int(row["seed"]))].append(value)
    mean_effects = {
        variant: {corruption: mean(values) for corruption, values in corruptions.items()}
        for variant, corruptions in effects.items()
    }
    consistent = {
        variant: {
            corruption: all(
                mean(values) > 0
                for (item_variant, item_corruption, _), values in seed_effects.items()
                if item_variant == variant and item_corruption == corruption
            )
            for corruption in corruptions
        }
        for variant, corruptions in effects.items()
    }

    endpoint_by_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in direct:
        if row["split"] != "heldout_compositions" or int(row["iteration"]) != config.evaluation.rollout_steps:
            continue
        endpoint_by_seed[(row["method"], int(row["seed"]))].append(float(row["query_mse"]))
    seed_summary = {
        f"{method}/seed_{seed}": mean(values)
        for (method, seed), values in sorted(endpoint_by_seed.items())
    }

    stale_rows = [row for row in interventions if row["intervention"] == "stale"]
    residual_distances = [
        float(row["residual_distance_from_stale"])
        for row in stale_rows if row["residual_distance_from_stale"]
    ]
    teacher_changes = [
        float(row["teacher_action_change_from_previous"])
        for row in stale_rows if row["teacher_action_change_from_previous"]
    ]
    recurrent_rows = [
        row for row in direct
        if row["method"] == "direct_recurrent" and row["recurrent_state_norm"]
    ]

    clip_fraction = mean(item["gradient_clipped"] for item in history)
    grad_norm_mean = mean(item["gradient_norm"] for item in history)
    ar_endpoint = historical_endpoint["ar_no_feedback"]
    original_endpoint = historical_endpoint["ifi"]
    best_direct = min(direct_endpoint.values())
    direct_supported = any(
        direct_endpoint[variant] <= original_endpoint
        and direct_endpoint[variant] < ar_endpoint
        and all(consistent[variant].get(control, False) for control in (
            "zero", "wrong_task", "shuffled", "stale"
        ))
        for variant in direct_endpoint
    )
    feedback_without_freshness = any(
        all(consistent[variant].get(control, False) for control in (
            "zero", "wrong_task", "shuffled"
        )) and not consistent[variant].get("stale", False)
        for variant in direct_endpoint
    )
    deep_iteration = config.evaluation.rollout_steps
    recurrent_supported = (
        direct_endpoint["direct_recurrent"] < direct_endpoint["direct_current"]
        and direct_endpoint["direct_recurrent"] < direct_endpoint["direct_history"]
        and all(consistent["direct_recurrent"].get(control, False) for control in (
            "zero", "wrong_task", "shuffled", "stale"
        ))
    )
    if direct_supported:
        verdict = "DIRECT COUNTERFACTUAL IFI SUPPORTED"
    elif feedback_without_freshness:
        verdict = "RESIDUAL FEEDBACK SUPPORTED, FRESHNESS NOT SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"

    report = f"""# IFI v2.1 — Direct residual evidence report

## Preregistered verdict

**{verdict}**

Recurrent learning state supported: **{recurrent_supported}**. This requires C to beat A and B at
the unseen rollout endpoint F{deep_iteration} while preserving causal dependence on every current
residual control.

## Held-out composition endpoint

Direct variants:

```json
{json.dumps(direct_endpoint, indent=2)}
```

Frozen historical comparisons:

```json
{json.dumps(historical_endpoint, indent=2)}
```

Best direct endpoint: {best_direct:.8f}; original E_R IFI: {original_endpoint:.8f};
AR-no-feedback: {ar_endpoint:.8f}.

## Current-residual causal effects

Positive values mean corruption increased next-step query MSE relative to correct current R.

```json
{json.dumps(mean_effects, indent=2)}
```

All-seed sign consistency:

```json
{json.dumps(consistent, indent=2)}
```

Primary freshness endpoint is the `stale` entry above. Mean raw residual-token distance
`d_R(R_i,R_(i-1))` was {mean(residual_distances):.8f}; mean teacher desired-action change was
{mean(teacher_changes):.8f}.

## Parameter and optimization diagnostics

```json
{json.dumps(parameter_counts, indent=2)}
```

Mean pre-clip global gradient norm: {grad_norm_mean:.8f}; fraction of logged steps clipped:
{clip_fraction:.4f}. Mean C recurrent-state norm across evaluation records:
{mean(float(row['recurrent_state_norm']) for row in recurrent_rows):.8f}.

## Seed consistency at F{deep_iteration}

```json
{json.dumps(seed_summary, indent=2)}
```

Full trajectories by iteration/K/novelty, action error, state changes, interventions, and
fresh-minus-stale effects are in the accompanying CSV/JSON files. Overall RMSE alone was not used
to assign the verdict.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")

