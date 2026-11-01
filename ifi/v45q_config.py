from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class V45QConfig:
    output_dir: str = "artifacts/v4_5_q_gate"
    base_config: str = "artifacts/v4_1/config.json"
    t_checkpoint: str = "artifacts/v4_5_onpolicy_history_aggregation/checkpoint_cycle_5.pt"
    t_source_checkpoint: str = (
        "artifacts/v4_4_joint_capacity_step_count_run/large_checkpoint_step_12000.pt"
    )
    v41_checkpoint: str = "artifacts/v4_1/checkpoint_final.pt"
    seed: int = 11
    max_history: int = 128
    oracle_episodes: int = 512
    rollout_horizon: int = 128
    oracle_final_best_ratio_max: float = 1.20
    oracle_endpoint_f0_ratio_max: float = 0.75
    dataset_train_worlds: int = 1536
    dataset_validation_worlds: int = 384
    dataset_test_worlds: int = 512
    dataset_ood_worlds: int = 512
    dataset_depths: tuple[int, ...] = (32, 64, 128)
    collection_batch_size: int = 128
    q_model_dimension: int = 64
    q_heads: int = 4
    q_layers: int = 2
    q_ffn: int = 128
    q_batch_size: int = 128
    q_training_steps: int = 4000
    q_learning_rate: float = 2e-4
    q_classification_weight: float = 1.0
    q_log_every: int = 100
    primary_gate_threshold: float = 0.98
    gate_thresholds: tuple[float, ...] = (1.0, 0.98, 0.95)
    offline_false_safe_max: float = 0.15
    offline_auroc_min: float = 0.80
    intervention_samples: int = 4096

    @classmethod
    def from_json(cls, path: str | Path) -> "V45QConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V45QConfig keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
