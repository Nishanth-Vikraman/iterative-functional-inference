from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class V43Config:
    output_dir: str = "artifacts/v4_3_convergence_replay"
    base_config: str = "artifacts/v4_1/config.json"
    base_checkpoint: str = "artifacts/v4_1/checkpoint_final.pt"
    seed: int = 11
    harvest_worlds: int = 2048
    harvest_batch_size: int = 256
    harvest_rollout: int = 64
    harvest_before_minimum: int = 2
    harvest_after_minimum: int = 8
    harvest_train_fraction: float = 0.80
    convergence_fraction: float = 0.70
    batch_size: int = 64
    training_steps: int = 8000
    smoke_step: int = 2000
    learning_rate: float = 2e-5
    lambda_safe: float = 1.0
    log_every: int = 100
    checkpoint_every: int = 500
    early_retention_ratio_max: float = 1.15
    convergence_q_max: float = 0.995
    convergence_regression_fraction_max: float = 0.45
    evaluation_episodes: int = 512
    evaluation_horizon: int = 128
    intervention_episodes: int = 256
    intervention_at: int = 8
    intervention_followup: int = 32

    @classmethod
    def from_json(cls, path: str | Path) -> "V43Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V43Config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
