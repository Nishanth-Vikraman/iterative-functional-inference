from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


@dataclass
class TaskConfig:
    x_min: float = -2.0
    x_max: float = 2.0
    support_sizes: tuple[int, ...] = (2, 4, 8, 16)
    noise_std: float = 0.0
    families: tuple[str, ...] = ("polynomial", "sinusoid", "gaussian", "mixture")


@dataclass
class TeacherConfig:
    kernel: str = "rbf"
    bandwidth: float = 0.45
    step_size: float = 0.8
    normalize_by_support: bool = True
    update_epsilon: float = 1e-5
    improvement_epsilon: float = 1e-7


@dataclass
class OperatorConfig:
    dimension: int = 32
    action_dimension: int = 12
    hidden_dimension: int = 96
    basis_depth: int = 2
    projection_ridge: float = 1e-4


@dataclass
class PhaseAConfig:
    steps: int = 6000
    batch_size: int = 32
    probe_size: int = 64
    learning_rate: float = 2e-3
    conditioning_weight: float = 0.05
    phi_weight: float = 1e-5
    span_threshold: float = 0.20
    validation_episodes: int = 256
    log_every: int = 100


@dataclass
class PhaseBConfig:
    steps: int = 12000
    batch_size: int = 16
    rollout_max: int = 5
    variable_rollout: bool = True
    probe_size: int = 64
    latent_dimension: int = 96
    controller_hidden_dimension: int = 128
    learning_rate: float = 1e-3
    action_metric: str = "functional"
    lambda_functional: float = 0.25
    lambda_differential: float = 0.0
    gradient_clip: float = 1.0
    log_every: int = 100


@dataclass
class EvalConfig:
    episodes_per_cell: int = 128
    query_size: int = 256
    rollout_steps: int = 8
    seeds: tuple[int, ...] = (11, 23, 47)


@dataclass
class PredictiveStateConfig:
    temperature: float = 0.1
    information_weight: float = 1.0
    projection_seed: int = 2202
    updater_layers: int = 1


@dataclass
class ForwardStateConfig:
    block_length: int = 100
    extended_rollout_steps: int = 16


@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "cuda"
    require_cuda: bool = True
    output_dir: str = "runs/canonical"
    tasks: TaskConfig = field(default_factory=TaskConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    phase_a: PhaseAConfig = field(default_factory=PhaseAConfig)
    phase_b: PhaseBConfig = field(default_factory=PhaseBConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    predictive_state: PredictiveStateConfig = field(default_factory=PredictiveStateConfig)
    forward_state: ForwardStateConfig = field(default_factory=ForwardStateConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        top = {f.name for f in fields(cls)}
        unknown = set(raw) - top
        if unknown:
            raise ValueError(f"Unknown ExperimentConfig keys: {sorted(unknown)}")
        scalars = {k: v for k, v in raw.items() if k not in {
            "tasks", "teacher", "operator", "phase_a", "phase_b", "evaluation",
            "predictive_state",
            "forward_state",
        }}
        return cls(
            **scalars,
            tasks=_construct(TaskConfig, raw.get("tasks")),
            teacher=_construct(TeacherConfig, raw.get("teacher")),
            operator=_construct(OperatorConfig, raw.get("operator")),
            phase_a=_construct(PhaseAConfig, raw.get("phase_a")),
            phase_b=_construct(PhaseBConfig, raw.get("phase_b")),
            evaluation=_construct(EvalConfig, raw.get("evaluation")),
            predictive_state=_construct(PredictiveStateConfig, raw.get("predictive_state")),
            forward_state=_construct(ForwardStateConfig, raw.get("forward_state")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
