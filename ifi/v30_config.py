from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    valid = {item.name for item in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


@dataclass
class V30TaskConfig:
    x_min: float = -2.0
    x_max: float = 2.0
    support_sizes: tuple[int, ...] = (2, 4, 8, 16)
    sensor_centers: tuple[float, ...] = (-1.5, -0.75, 0.0, 0.75, 1.5)
    sensor_width: float = 0.08
    observation_noise: float = 0.10
    likelihood_noise: float = 0.10
    train_sequence_length: int = 6
    evaluation_sequence_length: int = 12
    coefficients_a: tuple[float, ...] = (-1.0, -0.35, 0.35, 1.0)
    coefficients_b: tuple[float, ...] = (-0.65, 0.65)
    coefficients_c: tuple[float, ...] = (-0.85, 0.85)
    frequencies: tuple[float, ...] = (1.0, 2.0)


@dataclass
class V30GateConfig:
    seed: int = 3001
    episodes_per_k: int = 384
    pair_samples: int = 24000
    probe_size: int = 128
    minimum_incremental_history_r2: float = 0.05
    minimum_alias_prevalence: float = 0.05
    minimum_nearest_teacher_ratio: float = 0.35


@dataclass
class V30OperatorConfig:
    dimension: int = 32
    action_dimension: int = 12
    hidden_dimension: int = 96
    basis_depth: int = 2
    projection_ridge: float = 1e-4


@dataclass
class V30PhaseAConfig:
    steps: int = 3000
    batch_size: int = 24
    probe_size: int = 96
    learning_rate: float = 2e-3
    conditioning_weight: float = 0.05
    phi_weight: float = 1e-5
    span_threshold: float = 0.20
    validation_episodes: int = 256
    log_every: int = 100


@dataclass
class V30PhaseBConfig:
    steps: int = 5000
    batch_size: int = 8
    rollout_max: int = 6
    variable_rollout: bool = True
    probe_size: int = 64
    model_dimension: int = 64
    learning_rate: float = 1e-3
    action_metric: str = "functional"
    lambda_functional: float = 0.25
    gradient_clip: float = 1.0
    log_every: int = 100


@dataclass
class V30EvalConfig:
    episodes_per_cell: int = 64
    query_size: int = 256
    rollout_steps: int = 8
    extended_rollout_steps: int = 12
    seeds: tuple[int, ...] = (11, 23, 47)


@dataclass
class V30Config:
    seed: int = 30
    device: str = "cuda"
    require_cuda: bool = True
    output_dir: str = "runs/v3_0_bayesian"
    tasks: V30TaskConfig = field(default_factory=V30TaskConfig)
    gate: V30GateConfig = field(default_factory=V30GateConfig)
    operator: V30OperatorConfig = field(default_factory=V30OperatorConfig)
    phase_a: V30PhaseAConfig = field(default_factory=V30PhaseAConfig)
    phase_b: V30PhaseBConfig = field(default_factory=V30PhaseBConfig)
    evaluation: V30EvalConfig = field(default_factory=V30EvalConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "V30Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown V30Config keys: {sorted(unknown)}")
        nested = {"tasks", "gate", "operator", "phase_a", "phase_b", "evaluation"}
        scalars = {key: value for key, value in raw.items() if key not in nested}
        return cls(
            **scalars,
            tasks=_construct(V30TaskConfig, raw.get("tasks")),
            gate=_construct(V30GateConfig, raw.get("gate")),
            operator=_construct(V30OperatorConfig, raw.get("operator")),
            phase_a=_construct(V30PhaseAConfig, raw.get("phase_a")),
            phase_b=_construct(V30PhaseBConfig, raw.get("phase_b")),
            evaluation=_construct(V30EvalConfig, raw.get("evaluation")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
