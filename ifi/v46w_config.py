from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class V46WConfig:
    output_dir: str = "artifacts/v4_6_residual_world_model"
    t_checkpoint: str = "artifacts/v4_5_onpolicy_history_aggregation/checkpoint_cycle_5.pt"
    t_source_checkpoint: str = (
        "artifacts/v4_4_joint_capacity_step_count_run/large_checkpoint_step_12000.pt"
    )
    old_q_checkpoint: str = "artifacts/v4_5_q_gate/q_checkpoint.pt"
    old_q_config: str = "configs/v4_5_q_gate.json"
    seed: int = 11
    canonical_seeds: tuple[int, ...] = (11, 23, 47)
    max_history: int = 128
    train_worlds: int = 1536
    validation_worlds: int = 384
    heldout_worlds: int = 512
    ood_worlds: int = 512
    collection_depths: tuple[int, ...] = (32, 64, 128)
    collection_batch_size: int = 128
    model_dimension: int = 96
    heads: int = 4
    layers: int = 2
    ffn_dimension: int = 192
    batch_size: int = 128
    smoke_steps: int = 2000
    canonical_steps: int = 10000
    learning_rate: float = 2e-4
    log_every: int = 100
    delta_floor: float = 1e-3
    actual_action_fraction: float = 0.40
    scaled_action_fraction: float = 0.20
    random_action_fraction: float = 0.20
    orthogonal_action_fraction: float = 0.20
    scaled_factors: tuple[float, ...] = (0.25, 0.5, 1.5, 2.0)
    random_action_norm_min: float = 0.02
    random_action_norm_max: float = 0.70
    evaluation_states: int = 4096
    intervention_states: int = 4096
    recent_history: int = 16
    smoke_relative_baseline_max: float = 0.80
    smoke_cosine_min: float = 0.80
    smoke_history_degradation_min: float = 1.20
    final_heldout_false_safe_max: float = 0.15
    final_ood_false_safe_max: float = 0.20
    gate_thresholds: tuple[float, ...] = (1.0, 0.98, 0.95)
    primary_gate_threshold: float = 0.98
    gate_rollout_episodes: int = 512
    gate_rollout_horizon: int = 128

    @classmethod
    def from_json(cls, path: str | Path) -> "V46WConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V46WConfig keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
