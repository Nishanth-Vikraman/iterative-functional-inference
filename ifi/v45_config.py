from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class V45Config:
    output_dir: str = "artifacts/v4_5_onpolicy_history_aggregation"
    base_config: str = "artifacts/v4_1/config.json"
    initial_checkpoint: str = (
        "artifacts/v4_4_joint_capacity_step_count_run/large_checkpoint_step_12000.pt"
    )
    v41_checkpoint: str = "artifacts/v4_1/checkpoint_final.pt"
    seed: int = 11
    max_history: int = 128
    collection_worlds: int = 384
    collection_batch_size: int = 128
    collection_depths: tuple[int, ...] = (16, 32, 64)
    cycles: int = 5
    steps_per_cycle: int = 1000
    batch_size: int = 64
    latest_batch: int = 45
    older_batch: int = 13
    anchor_batch: int = 6
    older_versions_keep: int = 4
    learning_rate: float = 1e-4
    lambda_safe: float = 1.0
    log_every: int = 100
    diagnostic_episodes: int = 256
    diagnostic_horizon: int = 64
    smoke_early_retention_factor: float = 1.20
    smoke_q_upper_bound: float = 1.50
    evaluation_episodes: int = 512
    evaluation_horizon: int = 128
    intervention_episodes: int = 256
    intervention_at: int = 8
    intervention_horizon: int = 64

    @classmethod
    def from_json(cls, path: str | Path) -> "V45Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V45Config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
