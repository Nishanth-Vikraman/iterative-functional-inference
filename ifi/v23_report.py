from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mse_at(rows, iteration, method=None, split="heldout_compositions"):
    values = [
        float(row["query_mse"]) for row in rows
        if row["split"] == split and int(row["iteration"]) == iteration
        and (method is None or row.get("method") == method)
    ]
    return mean(values)


def _effects(rows):
    grouped: dict[str, list[float]] = defaultdict(list)
    seeded: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["intervention"] == "correct":
            continue
        value = float(row["query_mse_delta_from_correct"])
        grouped[row["intervention"]].append(value)
        seeded[(row["intervention"], int(row["seed"]))].append(value)
    averages = {key: mean(values) for key, values in grouped.items()}
    consistent = {
        key: all(
            mean(values) > 0
            for (control, _), values in seeded.items() if control == key
        ) for key in grouped
    }
    return averages, consistent


def write_v23_report(output_dir, v21_dir, v22_dir, historical_dir, config, counts, history):
    output = Path(output_dir)
    current = _read(output / "evaluation_records.csv")
    v21 = _read(Path(v21_dir) / "evaluation_records.csv")
    v22 = _read(Path(v22_dir) / "evaluation_records.csv")
    historical = _read(Path(historical_dir) / "evaluation_records.csv")
    residual = _read(output / "residual_interventions.csv")
    state = _read(output / "state_interventions.csv")
    diversity = _read(output / "state_diversity.csv")
    primary = config.evaluation.rollout_steps
    extended = config.forward_state.extended_rollout_steps
    endpoints = {
        "v2_3_F8": _mse_at(current, primary),
        "v2_3_F16": _mse_at(current, extended),
        "v2_2_F8": _mse_at(v22, primary),
        "v2_1_recurrent_F8": _mse_at(v21, primary, "direct_recurrent"),
        "direct_current_F8": _mse_at(v21, primary, "direct_current"),
        "direct_history_F8": _mse_at(v21, primary, "direct_history"),
        "ar_no_feedback_F8": _mse_at(historical, primary, "ar_no_feedback"),
        "historical_E_R_IFI_F8": _mse_at(historical, primary, "ifi"),
        "rkhs_teacher_F8": _mse_at(historical, primary, "rkhs_teacher"),
        "F0": _mse_at(historical, 0, "f0"),
    }
    trajectory = [_mse_at(current, iteration) for iteration in range(extended + 1)]
    increments = [trajectory[i - 1] - trajectory[i] for i in range(1, len(trajectory))]
    accelerations = [increments[i] - increments[i - 1] for i in range(1, len(increments))]
    continuous = all(value > 0 for value in increments)
    residual_effects, residual_consistency = _effects(residual)
    state_effects, state_consistency = _effects(state)
    improves_v21 = endpoints["v2_3_F8"] < endpoints["v2_1_recurrent_F8"]
    residual_causal = all(
        residual_consistency.get(control, False)
        for control in ("zero", "wrong_task", "shuffled")
    )
    state_causal = all(
        state_consistency.get(control, False)
        for control in ("zero", "wrong_task", "stale")
    )
    recurrent_supported = improves_v21 and continuous and residual_causal and state_causal
    freshness = residual_consistency.get("stale", False)
    verdict = (
        "FORWARD-TRAINED LEARNING STATE SUPPORTED"
        if recurrent_supported else "NOT SUPPORTED"
    )

    h_rows = [row for row in history if row["block_h"] == 1.0]
    phi_rows = [row for row in history if row["block_phi"] == 1.0]
    routing = {
        "H_h_grad_mean": mean(row["H_h_grad_norm"] for row in h_rows),
        "H_frozen_action_grad_max": max(row["H_frozen_action_grad_norm"] for row in h_rows),
        "H_phi_i_grad_max": max(row["H_phi_i_grad_norm"] for row in h_rows),
        "H_environment_grad_path_max": max(row["H_environment_grad_path"] for row in h_rows),
        "Phi_action_grad_mean": mean(row["Phi_action_grad_norm"] for row in phi_rows),
        "Phi_h_grad_max": max(row["Phi_h_grad_norm"] for row in phi_rows),
        "Phi_state_input_grad_max": max(row["Phi_state_inputs_require_grad"] for row in phi_rows),
        "H_clip_fraction": mean(row["gradient_clipped"] for row in h_rows),
        "Phi_clip_fraction": mean(row["gradient_clipped"] for row in phi_rows),
    }
    routing_verified = (
        routing["H_h_grad_mean"] > 0
        and routing["H_frozen_action_grad_max"] == 0
        and routing["H_phi_i_grad_max"] == 0
        and routing["H_environment_grad_path_max"] == 0
        and routing["Phi_action_grad_mean"] > 0
        and routing["Phi_h_grad_max"] == 0
        and routing["Phi_state_input_grad_max"] == 0
    )
    by_k = {
        f"K{support}/F{iteration}": mean(
            float(row["query_mse"]) for row in current
            if row["split"] == "heldout_compositions"
            and int(row["support_size"]) == support
            and int(row["iteration"]) == iteration
        )
        for support in config.tasks.support_sizes
        for iteration in (primary, extended)
    }
    by_split = {
        f"{split}/F{iteration}": _mse_at(current, iteration, split=split)
        for split in ("seen_parameters", "heldout_combinations", "heldout_compositions")
        for iteration in (primary, extended)
    }
    by_seed = {
        f"seed_{seed}/F{iteration}": mean(
            float(row["query_mse"]) for row in current
            if row["split"] == "heldout_compositions"
            and int(row["seed"]) == seed and int(row["iteration"]) == iteration
        )
        for seed in config.evaluation.seeds for iteration in (primary, extended)
    }
    deep_diversity = {
        f"F{iteration}": mean(
            float(row["mean_pairwise_state_distance"]) for row in diversity
            if int(row["iteration"]) == iteration
        ) for iteration in (primary, extended)
    }
    dynamics = {
        "mean_state_change": mean(
            float(row["recurrent_state_change"]) for row in current
            if row["recurrent_state_change"]
        ),
        "mean_state_cosine": mean(
            float(row["state_cosine_from_previous"]) for row in current
            if row["state_cosine_from_previous"]
        ),
        "mean_update_direction_cosine": mean(
            float(row["update_direction_cosine"]) for row in current
            if row["update_direction_cosine"]
        ),
        "pairwise_state_distance": deep_diversity,
    }
    acceleration_positive = [index + 2 for index, value in enumerate(accelerations) if value > 0]
    report = f"""# IFI v2.3 — Forward-trained recurrent learning state

## Preregistered verdict

**{verdict}**

Fresh iterative inference supported: **{recurrent_supported and freshness}**.

## Held-out composition endpoints

```json
{json.dumps(endpoints, indent=2)}
```

v2.3 improves over matched v2.1 recurrent at F8: **{improves_v21}**.
Continuous improvement through F{extended}: **{continuous}**.

## F0→F{extended} trajectory

```json
{json.dumps({"mse": trajectory, "increment": increments, "acceleration": accelerations}, indent=2)}
```

Iterations whose adaptation acceleration was positive: {acceleration_positive}.

## Current residual interventions

```json
{json.dumps({"mean_effect": residual_effects, "all_seed_positive": residual_consistency}, indent=2)}
```

Freshness (`stale`) supported: **{freshness}**.

## Recurrent-state interventions

```json
{json.dumps({"mean_effect": state_effects, "all_seed_positive": state_consistency}, indent=2)}
```

## Gradient routing audit

```json
{json.dumps(routing, indent=2)}
```

Routing verified: **{routing_verified}**. H blocks updated only the h-head through the frozen
next-step evaluator. Phi blocks updated shared/action parameters with detached carried states.
The operator and functional transition remained outside both gradient paths.

Parameter counts and fixed schedule:

```json
{json.dumps({"parameters": counts, "block_length": config.forward_state.block_length, "total_steps": config.phase_b.steps}, indent=2)}
```

## K dependence

```json
{json.dumps(by_k, indent=2)}
```

## Novelty splits

```json
{json.dumps(by_split, indent=2)}
```

## Seed consistency

```json
{json.dumps(by_seed, indent=2)}
```

## State dynamics

```json
{json.dumps(dynamics, indent=2)}
```

The verdict requires causal degradation from both state and current-residual corruption; lower
endpoint RMSE or positive acceleration alone is insufficient. All negative outcomes are retained.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
