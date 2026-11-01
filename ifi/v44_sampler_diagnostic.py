from __future__ import annotations

import argparse
import json

import torch

from .v40_config import V40Config
from .v40_models import PopulationController
from .v44_config import V44Config
from .v44_populations import RecoveryPopulation
from .v44_training import arm_config


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a v4.4 recovery sampler")
    parser.add_argument(
        "--config",
        default="configs/v4_4_joint_capacity_sampler_gated.json",
    )
    args = parser.parse_args()
    experiment = V44Config.from_json(args.config)
    base = V40Config.from_json(experiment.base_config)
    archive = torch.load(
        experiment.convergence_validation_pool,
        map_location="cpu",
        weights_only=False,
    )
    trajectory_count = int(archive["trajectory_id"].max()) + 1
    cutoff = int(0.8 * trajectory_count)
    train_indices = torch.where(archive["trajectory_id"] < cutoff)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for arm in ("small", "large"):
        config = arm_config(base, experiment, arm, 11)
        torch.manual_seed(111)
        model = PopulationController("query_history", config).to(device)
        population = RecoveryPopulation(
            experiment,
            config,
            model,
            11,
            device,
            archive,
            train_indices,
        )
        results[arm] = population.enforce_sufficiency(model)
    results["passed"] = all(results[arm]["passed"] for arm in ("small", "large"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
