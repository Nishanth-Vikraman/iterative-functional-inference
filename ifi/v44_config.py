from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class V44Config:
    output_dir: str = "artifacts/v4_4_joint_capacity"
    base_config: str = "artifacts/v4_1/config.json"
    reference_checkpoint: str = "artifacts/v4_1/checkpoint_final.pt"
    convergence_validation_pool: str = "artifacts/v4_3_convergence_replay/convergence_pool.pt"
    seeds: tuple[int, ...] = (11, 23, 47)
    small_model_dimension: int = 64
    small_heads: int = 4
    small_layers: int = 2
    small_ffn: int = 128
    large_model_dimension: int = 96
    large_heads: int = 4
    large_layers: int = 3
    large_ffn: int = 192
    ordinary_pool_size: int = 4096
    recovery_pool_size: int = 2048
    recovery_history_capacity: int = 64
    recovery_harvest_rollout: int = 32
    recovery_before_minimum: int = 2
    recovery_after_minimum: int = 8
    recovery_refresh_every: int = 500
    recovery_refresh_count: int = 512
    recovery_source: str = "current_policy"
    recovery_gate_every: int = 100
    sampler_gate_norm_median_max: float = 0.30
    sampler_gate_norm_q90_max: float = 1.00
    sampler_gate_low_norm_threshold: float = 0.20
    sampler_gate_low_fraction_min: float = 0.30
    sampler_gate_overshoot_q: float = 1.05
    sampler_gate_overshoot_fraction_min: float = 0.12
    sampler_gate_recovery_q: float = 0.95
    sampler_gate_recovery_fraction_min: float = 0.15
    sampler_gate_post_min_fraction_min: float = 0.60
    sampler_gate_unique_worlds_min: int = 1000
    sampler_gate_history_error_max: float = 2e-6
    ordinary_fraction: float = 0.60
    batch_size: int = 64
    training_steps: int = 12000
    smoke_step: int = 2000
    learning_rate_small: float = 1e-4
    learning_rate_large: float = 1e-4
    lambda_safe: float = 1.0
    log_every: int = 100
    checkpoint_every: int = 500
    smoke_f8_ratio_max: float = 0.5700954794883728
    smoke_q_max: float = 0.95
    smoke_convergence_q_max: float = 0.995
    smoke_regression_improvement_factor: float = 0.80
    evaluation_episodes: int = 512
    evaluation_horizon: int = 128
    intervention_episodes: int = 256
    intervention_at: int = 8
    intervention_horizon: int = 128

    @classmethod
    def from_json(cls, path: str | Path) -> "V44Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V44Config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
