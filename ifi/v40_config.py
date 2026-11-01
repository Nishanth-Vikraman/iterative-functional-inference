from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V40Config:
    experiment_version: str = "v4.0"
    output_dir: str = "artifacts/v4_0"
    device: str = "cuda"
    require_cuda: bool = True
    seed: int = 11
    canonical_seeds: tuple[int, ...] = (11,)
    residual_dimension: int = 8
    action_dimension: int = 8
    model_dimension: int = 64
    attention_heads: int = 4
    attention_layers: int = 2
    ffn_dimension: int = 128
    history_capacity: int = 16
    pool_size: int = 4096
    batch_size: int = 64
    training_steps: int = 12000
    learning_rate: float = 3e-4
    gradient_clip: float = 1.0
    action_max: float = 2.0
    action_regularization: float = 1e-3
    loss_kind: str = "contraction_squared"
    progress_tau: float = 0.95
    exploration_std: float = 0.02
    bootstrap_actions: int = 2
    bootstrap_action_norm: float = 0.35
    bootstrap_orthogonal: bool = False
    training_residual_norm_min: float = 0.5
    training_residual_norm_max: float = 2.0
    training_residual_is_post_bootstrap: bool = False
    evaluation_residual_norm_min: float = 0.5
    evaluation_residual_norm_max: float = 2.0
    evaluation_residual_is_post_bootstrap: bool = False
    stratified_magnitude_sampling: bool = False
    magnitude_bin_log10_edges: tuple[float, ...] = (
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    )
    singular_min: float = 0.5
    singular_max: float = 2.0
    magnitude_scale: float = 1.0
    numerical_epsilon: float = 1e-8
    convergence_threshold: float = 0.03
    divergence_threshold: float = 20.0
    max_age: int = 64
    random_reset_probability: float = 0.002
    log_every: int = 100
    checkpoint_every: int = 500
    gate_step: int = 2000
    gate_episodes: int = 192
    gate_rollout: int = 8
    gate_median_q_max: float = 0.98
    gate_endpoint_ratio_max: float = 0.90
    evaluation_episodes: int = 512
    evaluation_rollout: int = 64
    evaluation_batch_size: int = 128
    intervention_episodes: int = 256
    intervention_at: int = 8
    intervention_followup: int = 16
    fixed_eta_candidates: tuple[float, ...] = (
        -0.8,
        -0.4,
        -0.2,
        -0.1,
        -0.05,
        0.0,
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
    )
    fixed_eta_calibration_worlds: int = 4096
    harder_bucket_enabled: bool = True
    harder_singular_min: float = 1.0 / 3.0
    harder_singular_max: float = 2.0
    harder_condition_min: float = 4.0
    harder_condition_max: float = 6.0

    @classmethod
    def from_json(cls, path: str | Path) -> "V40Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V40Config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LEARNED_MODELS = ("query_history", "current_only", "ar_no_fresh")
