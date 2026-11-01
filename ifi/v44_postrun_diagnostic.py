from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from .v40_config import V40Config
from .v40_evaluation import controller_batch, make_rollout_state
from .v40_models import ControllerBatch, PopulationController
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v44_config import V44Config
from .v44_training import arm_config


ALL_HISTORY = HistoryPolicy("all_history", 0, True, True)


def summarize(value: torch.Tensor) -> dict[str, float]:
    value = value.float().cpu()
    return {
        "median": float(value.median()),
        "mean": float(value.mean()),
        "q10": float(torch.quantile(value, 0.10)),
        "q90": float(torch.quantile(value, 0.90)),
    }


@torch.inference_mode()
def arm_diagnostic(
    experiment: V44Config,
    base: V40Config,
    arm: str,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = arm_config(base, experiment, arm, 11)
    model = PopulationController("query_history", config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    storage = replace(
        config,
        history_capacity=config.bootstrap_actions + experiment.evaluation_horizon,
    )
    selected_steps = {8, 16, 24, 32, 64}
    records: dict[int, dict[str, list[torch.Tensor]]] = {
        step: {
            "residual": [],
            "action": [],
            "q": [],
            "alignment": [],
            "action_scale_01": [],
            "action_scale_001": [],
            "action_zero_current": [],
            "action_zero_current_previous": [],
        }
        for step in selected_steps
    }
    for start in range(0, experiment.evaluation_episodes, config.evaluation_batch_size):
        count = min(config.evaluation_batch_size, experiment.evaluation_episodes - start)
        state = make_rollout_state(storage, count, config.seed + 600 + start, device)
        for step in range(1, experiment.evaluation_horizon + 1):
            before = state.current
            batch = controller_batch(config, state)
            action = model(batch)
            after = transition(before, state.matrices, action)
            if step in selected_steps:
                residual_norm = before.norm(dim=-1)
                action_norm = action.norm(dim=-1)
                q = after.norm(dim=-1) / residual_norm.clamp_min(config.numerical_epsilon)
                delta = after - before
                alignment = (
                    (-before * delta).sum(dim=-1)
                    / (
                        residual_norm
                        * delta.norm(dim=-1)
                    ).clamp_min(config.numerical_epsilon)
                )
                variants = {}
                for label, scale in (("01", 0.1), ("001", 0.01), ("zero", 0.0)):
                    modified = ControllerBatch(
                        current=batch.current * scale,
                        previous_action=batch.previous_action,
                        history=batch.history,
                        history_lengths=batch.history_lengths,
                    )
                    variants[label] = model(modified).norm(dim=-1)
                zero_both = ControllerBatch(
                    current=torch.zeros_like(batch.current),
                    previous_action=torch.zeros_like(batch.previous_action),
                    history=batch.history,
                    history_lengths=batch.history_lengths,
                )
                records[step]["residual"].append(residual_norm.cpu())
                records[step]["action"].append(action_norm.cpu())
                records[step]["q"].append(q.cpu())
                records[step]["alignment"].append(alignment.cpu())
                records[step]["action_scale_01"].append(variants["01"].cpu())
                records[step]["action_scale_001"].append(variants["001"].cpu())
                records[step]["action_zero_current"].append(variants["zero"].cpu())
                records[step]["action_zero_current_previous"].append(
                    model(zero_both).norm(dim=-1).cpu()
                )
            append_with_policy(
                state,
                before,
                action,
                after,
                ALL_HISTORY,
                config.bootstrap_actions,
                storage.history_capacity,
            )
    result = {}
    for step, values in records.items():
        result[str(step)] = {
            name: summarize(torch.cat(chunks))
            for name, chunks in values.items()
        }
        actual = result[str(step)]["action"]["median"]
        result[str(step)]["zero_current_over_actual_action"] = (
            result[str(step)]["action_zero_current"]["median"] / max(actual, 1e-12)
        )
        result[str(step)]["local_expansion_fraction"] = float(
            (torch.cat(values["q"]) > 1).float().mean()
        )
    recovery = checkpoint["recovery_population"]
    ordinary = checkpoint["ordinary_population"]
    result["saved_training_state"] = {
        "recovery_residual": summarize(recovery["current"].norm(dim=-1)),
        "ordinary_residual": summarize(ordinary["current"].norm(dim=-1)),
        "recovery_history_length": summarize(recovery["history_lengths"].float()),
        "ordinary_history_length": summarize(ordinary["history_lengths"].float()),
    }
    return result


def main() -> None:
    experiment = V44Config.from_json(
        "configs/v4_4_joint_capacity_sampler_gated.json"
    )
    base = V40Config.from_json(experiment.base_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("artifacts/v4_4_joint_capacity_step_count_run")
    output = {
        arm: arm_diagnostic(
            experiment,
            base,
            arm,
            root / f"{arm}_checkpoint_step_12000.pt",
            device,
        )
        for arm in ("small", "large")
    }
    path = root / "postrun_diagnostic.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        arm: {
            step: {
                "residual": data["residual"]["median"],
                "action": data["action"]["median"],
                "q": data["q"]["median"],
                "expansion_fraction": data["local_expansion_fraction"],
                "zero_current_action": data["action_zero_current"]["median"],
                "zero_over_actual": data["zero_current_over_actual_action"],
                "alignment": data["alignment"]["median"],
            }
            for step, data in result.items()
            if step.isdigit()
        }
        for arm, result in output.items()
    }, indent=2))


if __name__ == "__main__":
    main()
