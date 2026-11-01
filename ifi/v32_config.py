from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V32Config:
    source_v30_run: str = "runs/v3_0_bayesian"
    source_v31_dir: str = "artifacts/v3_1"
    output_dir: str = "artifacts/v3_2"
    device: str = "cuda"
    require_cuda: bool = True
    training_seeds: tuple[int, ...] = (11, 23, 47)
    training_steps: int = 2000
    functional_grid_size: int = 128
    posterior_perturbation_scale: float = 0.10
    covariance_samples: int = 8
    covariance_rank_tolerance: float = 1e-7
    covariance_lambda_multiplier: float = 0.10
    covariance_lambda_quantile: float = 0.90
    epsilon_quantile: float = 0.50
    epsilon_multiplier: float = 0.01
    minimum_epsilon: float = 1e-10
    calibration_episodes: int = 64
    evaluation_rollout: int = 18
    history_capacity: int = 12
    material_effect_threshold: float = 0.05
    diagnostic_geometry_states_per_group: int = 24

    @classmethod
    def from_json(cls, path: str | Path) -> "V32Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V32Config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ARMS = ("phi_mse", "normalized_functional", "normalized_mahalanobis")
