from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from .v40_config import V40Config
from .v40_evaluation import make_rollout_state
from .v40_models import PopulationController
from .v44_evaluation import FIFO_16, fresh_state_from_matrices, rollout_summary
from .v45_config import V45Config
from .v45_onpolicy_history_aggregation import (
    CAPPED_ALL_128,
    alignment_rollout,
    atomic_json,
    history_interventions,
    write_csv,
    write_report,
)
from .v45_replay import collect_current_policy


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete v4.5 frozen evaluation")
    parser.add_argument("--config", default="configs/v4_5_onpolicy_history_aggregation.json")
    args = parser.parse_args()
    experiment = V45Config.from_json(args.config)
    base = V40Config.from_json(experiment.base_config)
    output = Path(experiment.output_dir)
    source = torch.load(experiment.initial_checkpoint, map_location="cpu", weights_only=False)
    config = V40Config(**source["config"])
    final = torch.load(
        output / f"checkpoint_cycle_{experiment.cycles}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(final["cycle"]) != experiment.cycles:
        raise RuntimeError("Final v4.5 cycle checkpoint is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = PopulationController("query_history", config).to(device)
    model.load_state_dict(final["model"])
    model.eval().requires_grad_(False)
    v41 = torch.load(experiment.v41_checkpoint, map_location="cpu", weights_only=False)
    anchor = v41["populations"]["query_history"]
    latest = collect_current_policy(
        experiment,
        config,
        model,
        experiment.cycles,
        device,
    )
    storage = replace(config, history_capacity=experiment.max_history)
    unseen = make_rollout_state(
        storage,
        experiment.evaluation_episodes,
        experiment.seed + 600,
        device,
    )
    familiar = fresh_state_from_matrices(
        config,
        anchor["matrices"][: experiment.evaluation_episodes],
        anchor["world_seeds"][: experiment.evaluation_episodes],
        experiment.max_history,
        experiment.seed + 610,
        device,
    )
    fixtures = {
        "training_B_fresh_residual": familiar,
        "unseen_same_family": unseen,
        "stored_long_policy_states": latest.deepest_rollout_state(
            experiment.evaluation_episodes, device
        ),
        "condition_4_6": make_rollout_state(
            storage,
            experiment.evaluation_episodes,
            experiment.seed + 620,
            device,
            harder=True,
        ),
    }
    rows = []
    for fixture, state in fixtures.items():
        for policy in (CAPPED_ALL_128, FIFO_16):
            rows.extend(
                rollout_summary(
                    experiment,
                    config,
                    model,
                    "v45_onpolicy",
                    experiment.seed,
                    fixture,
                    state,
                    policy,
                )
            )
    v44 = PopulationController("query_history", config).to(device)
    v44.load_state_dict(source["model"])
    v44.eval().requires_grad_(False)
    v41_model = PopulationController("query_history", base).to(device)
    v41_model.load_state_dict(v41["models"]["query_history"])
    v41_model.eval().requires_grad_(False)
    for label, baseline_model, baseline_config in (
        ("v44_large_12k", v44, config),
        ("v41_original", v41_model, base),
    ):
        rows.extend(
            rollout_summary(
                experiment,
                baseline_config,
                baseline_model,
                label,
                "historical",
                "unseen_same_family",
                unseen,
                CAPPED_ALL_128,
            )
        )
    interventions = history_interventions(experiment, config, model, device)
    alignment = alignment_rollout(
        experiment, config, model, "v45_onpolicy", device
    )
    alignment.extend(
        alignment_rollout(experiment, config, v44, "v44_large_12k", device)
    )
    cycle_rows = read_csv(output / "rollout_diagnostics.csv")
    write_csv(output / "long_rollouts.csv", rows)
    write_csv(output / "history_interventions.csv", interventions)
    write_csv(output / "action_alignment.csv", alignment)
    verdict = write_report(output, cycle_rows, rows)
    manifest = {
        "complete": True,
        "cycles": experiment.cycles,
        "states_per_collection": latest.state_count,
        "history_policy": CAPPED_ALL_128.name,
        "evaluation_recovered_from_cycle_checkpoint": True,
        "verdict": verdict,
    }
    atomic_json(output / "experiment_manifest.json", manifest)
    atomic_json(output / "status.json", {"phase": "complete", **manifest})
    print(json.dumps({"phase": "v45_evaluation_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
