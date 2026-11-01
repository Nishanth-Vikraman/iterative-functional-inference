from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V31Config:
    source_run: str = "runs/v3_0_bayesian"
    output_dir: str = "artifacts/v3_1"
    device: str = "cuda"
    require_cuda: bool = True
    seeds: tuple[int, ...] = (11, 23, 47)
    support_sizes: tuple[int, ...] = (2, 4, 8, 16)
    episodes_per_target: int = 4
    query_size: int = 256
    projection_size: int = 128
    rollout_steps: int = 18
    history_capacity: int = 12
    projection_span_threshold: float = 0.20
    bucket_outside_fraction_threshold: float = 0.50
    material_effect_threshold: float = 0.05
    checkpoint_every_cells: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> "V31Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V31Config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, train_rollout: int, checkpoint_capacity: int) -> None:
        if self.rollout_steps < 3 * train_rollout:
            raise ValueError("v3.1 must evaluate through at least 3x the v3.0 train rollout")
        if self.history_capacity != checkpoint_capacity:
            raise ValueError("history_capacity must equal the frozen checkpoint capacity")
        if self.episodes_per_target < 1 or self.query_size < 2 or self.projection_size < 2:
            raise ValueError("invalid v3.1 evaluation sizes")
        if not 0 < self.projection_span_threshold < 1:
            raise ValueError("projection_span_threshold must be in (0, 1)")
