from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_models import PopulationController
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v44_evaluation import fresh_state_from_matrices
from .v45q_config import V45QConfig


CAPPED_HISTORY = HistoryPolicy("all_available_capped_128", 128, False)


def clone_slice(state: RolloutState, start: int, stop: int) -> RolloutState:
    return RolloutState(**{
        name: value[start:stop].clone() for name, value in state.__dict__.items()
    })


@torch.inference_mode()
def rollout_mode(
    experiment: V45QConfig,
    config: V40Config,
    model: PopulationController,
    base_state: RolloutState,
    distribution: str,
    mode: str,
) -> list[dict[str, object]]:
    accumulators: dict[int, list[Tensor]] = {
        step: [] for step in range(experiment.rollout_horizon + 1)
    }
    first_rejections = []
    accepted_counts = []
    harmful_blocked = []
    accepted_q = []
    batch_size = config.evaluation_batch_size
    for start in range(0, base_state.current.shape[0], batch_size):
        state = clone_slice(
            base_state, start, min(start + batch_size, base_state.current.shape[0])
        )
        count = state.current.shape[0]
        stopped = torch.zeros(count, dtype=torch.bool, device=state.current.device)
        first_reject = torch.full(
            (count,), experiment.rollout_horizon + 1, dtype=torch.int64
        )
        accepted = torch.zeros(count, dtype=torch.int64)
        blocked = torch.zeros(count, dtype=torch.int64)
        accumulators[0].append(state.current.norm(dim=-1).cpu())
        for step in range(1, experiment.rollout_horizon + 1):
            before = state.current
            if mode == "no_update":
                action = torch.zeros_like(before)
                after = before
            else:
                proposed = model(controller_batch(config, state))
                proposed_after = transition(before, state.matrices, proposed)
                q_true = proposed_after.norm(dim=-1) / before.norm(dim=-1).clamp_min(
                    config.numerical_epsilon
                )
                if mode == "t_only":
                    accept = ~stopped
                elif mode == "oracle_hard_stop":
                    accept = (~stopped) & (q_true < 1)
                    reject = (~stopped) & (~accept)
                    first_reject[reject.cpu()] = step
                    blocked += reject.cpu().to(torch.int64)
                    stopped |= reject
                else:
                    raise ValueError(mode)
                accepted += accept.cpu().to(torch.int64)
                if accept.any():
                    accepted_q.append(q_true[accept].cpu())
                action = torch.where(accept[:, None], proposed, torch.zeros_like(proposed))
                after = torch.where(accept[:, None], proposed_after, before)
            append_with_policy(
                state,
                before,
                action,
                after,
                CAPPED_HISTORY,
                config.bootstrap_actions,
                experiment.max_history,
            )
            accumulators[step].append(after.norm(dim=-1).cpu())
        first_rejections.append(first_reject)
        accepted_counts.append(accepted)
        harmful_blocked.append(blocked)
    norms = {step: torch.cat(values) for step, values in accumulators.items()}
    first_rejection = torch.cat(first_rejections)
    accepted = torch.cat(accepted_counts)
    blocked = torch.cat(harmful_blocked)
    best_step = min(norms, key=lambda step: float(norms[step].median()))
    best = float(norms[best_step].median())
    rows = []
    for step in range(experiment.rollout_horizon + 1):
        rows.append({
            "distribution": distribution,
            "mode": mode,
            "iteration": step,
            "residual_median": float(norms[step].median()),
            "residual_mean": float(norms[step].mean()),
            "normalized_f_over_f0": float(
                (norms[step] / norms[0].clamp_min(config.numerical_epsilon)).median()
            ),
            "best_residual": best,
            "best_step": best_step,
            "final_over_best": float(norms[experiment.rollout_horizon].median()) / best,
            "accepted_action_fraction": float(accepted.float().mean())
            / experiment.rollout_horizon,
            "first_rejected_step_median": (
                ""
                if mode != "oracle_hard_stop"
                else float(first_rejection.float().median())
            ),
            "median_q_accepted": (
                "" if not accepted_q else float(torch.cat(accepted_q).median())
            ),
            "harmful_proposals_blocked_mean": float(blocked.float().mean()),
        })
    return rows


def build_fixtures(
    experiment: V45QConfig,
    config: V40Config,
    v41: dict[str, object],
    device: torch.device,
) -> dict[str, RolloutState]:
    storage = replace(config, history_capacity=experiment.max_history)
    population = v41["populations"]["query_history"]
    return {
        "training_family_fresh": fresh_state_from_matrices(
            config,
            population["matrices"][: experiment.oracle_episodes],
            population["world_seeds"][: experiment.oracle_episodes],
            experiment.max_history,
            experiment.seed + 910,
            device,
        ),
        "heldout_same_family": make_rollout_state(
            storage,
            experiment.oracle_episodes,
            experiment.seed + 920,
            device,
        ),
        "condition_4_6": make_rollout_state(
            storage,
            experiment.oracle_episodes,
            experiment.seed + 930,
            device,
            harder=True,
        ),
    }


def oracle_gate_verdict(
    experiment: V45QConfig,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    primary = "heldout_same_family"
    lookup = {
        (str(row["mode"]), int(row["iteration"])): row
        for row in rows
        if row["distribution"] == primary
    }
    ungated = lookup[("t_only", experiment.rollout_horizon)]
    oracle = lookup[("oracle_hard_stop", experiment.rollout_horizon)]
    t_useful = float(ungated["best_residual"]) <= 0.75 * float(
        lookup[("t_only", 0)]["residual_median"]
    )
    oracle_stable = float(oracle["final_over_best"]) <= experiment.oracle_final_best_ratio_max
    oracle_useful = float(oracle["residual_median"]) <= (
        experiment.oracle_endpoint_f0_ratio_max
        * float(lookup[("oracle_hard_stop", 0)]["residual_median"])
    )
    improved = float(oracle["residual_median"]) < float(ungated["residual_median"])
    passed = t_useful and oracle_stable and oracle_useful and improved
    return {
        "passed": passed,
        "verdict": (
            "RUNTIME ACCEPTANCE BOTTLENECK SUPPORTED"
            if passed
            else "GATING INSUFFICIENT"
        ),
        "checks": {
            "t_has_useful_contraction": t_useful,
            "oracle_final_near_best": oracle_stable,
            "oracle_preserves_meaningful_contraction": oracle_useful,
            "oracle_improves_f128": improved,
        },
        "primary_distribution": primary,
        "ungated_f128": ungated["residual_median"],
        "ungated_best": ungated["best_residual"],
        "oracle_f128": oracle["residual_median"],
        "oracle_best": oracle["best_residual"],
        "oracle_final_over_best": oracle["final_over_best"],
        "oracle_first_rejection_median": oracle["first_rejected_step_median"],
    }


def load_frozen_t(
    experiment: V45QConfig,
    device: torch.device,
) -> tuple[V40Config, PopulationController]:
    source = torch.load(
        experiment.t_source_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint = torch.load(experiment.t_checkpoint, map_location="cpu", weights_only=False)
    config = V40Config(**source["config"])
    model = PopulationController("query_history", config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return config, model


def run_oracle_phase(
    experiment: V45QConfig,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config, model = load_frozen_t(experiment, device)
    v41 = torch.load(experiment.v41_checkpoint, map_location="cpu", weights_only=False)
    rows = []
    for distribution, fixture in build_fixtures(
        experiment, config, v41, device
    ).items():
        for mode in ("t_only", "oracle_hard_stop", "no_update"):
            rows.extend(
                rollout_mode(
                    experiment,
                    config,
                    model,
                    fixture,
                    distribution,
                    mode,
                )
            )
    return rows, oracle_gate_verdict(experiment, rows)


def write_oracle_outputs(
    output: Path,
    rows: list[dict[str, object]],
    verdict: dict[str, object],
) -> None:
    import csv

    with (output / "oracle_gate_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "phase_a_gate.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )


def main() -> None:
    experiment = V45QConfig.from_json("configs/v4_5_q_gate.json")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = Path(experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, verdict = run_oracle_phase(experiment, torch.device("cuda"))
    write_oracle_outputs(output, rows, verdict)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
