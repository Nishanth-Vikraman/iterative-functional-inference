from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .functions import Functional
from .teachers import FunctionalUpdate
from .v30_config import V30Config
from .v30_tasks import build_latent_library, latent_split_indices


@dataclass(frozen=True)
class V31Target:
    target_id: str
    bucket: str
    severity: str
    formula: str
    parameters: dict[str, float]
    ood_distance: float
    factor_count: int = 0

    def evaluate(self, x: Tensor) -> Tensor:
        p = self.parameters
        a, b, c, w = (p.get(key, 0.0) for key in ("a", "b", "c", "omega"))
        if self.formula == "original":
            return a * x + b * x.square() + c * torch.sin(w * x)
        if self.formula == "cubic":
            return a * x + b * x.pow(3) + c * torch.sin(w * x)
        if self.formula == "abs_quadratic":
            return a * x + b * x.abs() * x + c * torch.sin(w * x)
        if self.formula == "gaussian_bump":
            return a * x + b * torch.exp(-p["gamma"] * x.square()) + c * torch.sin(w * x)
        if self.formula == "cosine":
            return a * x + b * x.square() + c * torch.cos(w * x)
        if self.formula == "sin_quadratic":
            return a * torch.sin(w * x.square()) + b * x + c * x.square()
        if self.formula == "x_sin":
            return a * x * torch.sin(w * x) + b * x.square() + c * x
        if self.formula == "sin_poly":
            return a * torch.sin(w * (x + b * x.square())) + c * x
        if self.formula == "composed_tanh":
            return a * torch.tanh(b * (x + c * torch.sin(w * x)))
        if self.formula == "deep_sin_poly":
            inner = x + b * x.square() + c * torch.sin(x)
            return a * torch.sin(w * inner)
        if self.formula == "deep_sin_sin":
            return a * torch.sin(w * torch.sin(x + b * x.square())) + c * x
        if self.formula == "deep_interaction":
            return a * x * torch.sin(w * (x + b * x.square())) + c * x.square()
        if self.formula == "depth3_composition":
            return a * torch.tanh(b * torch.sin(w * (x + c * x.square())))
        raise ValueError(f"Unknown v3.1 target formula: {self.formula}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "V31Target":
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class V31EvidenceBatch:
    x: Tensor
    y: Tensor
    sensor_index: int


@dataclass(frozen=True)
class V31Episode:
    target: V31Target
    batches: tuple[V31EvidenceBatch, ...]
    episode_id: str

    def cumulative(self, iteration: int) -> tuple[Tensor, Tensor]:
        selected = self.batches[: iteration + 1]
        return (
            torch.cat([batch.x for batch in selected]),
            torch.cat([batch.y for batch in selected]),
        )


class V31OracleTeacher:
    """Exact finite Bayesian diagnostic oracle over a preregistered OOD group."""

    def __init__(self, hypotheses: tuple[V31Target, ...], likelihood_noise: float):
        if not hypotheses:
            raise ValueError("diagnostic oracle requires at least one hypothesis")
        self.hypotheses = hypotheses
        self.likelihood_noise = likelihood_noise

    def posterior(self, support: tuple[Tensor, Tensor]) -> Tensor:
        x, y = support
        predictions = torch.stack([target.evaluate(x) for target in self.hypotheses])
        residual = y.unsqueeze(0) - predictions
        log_likelihood = -0.5 * residual.square().flatten(1).sum(dim=-1) / (
            self.likelihood_noise ** 2
        )
        return torch.softmax(log_likelihood, dim=0)

    def predictive(self, posterior: Tensor, x: Tensor) -> Tensor:
        values = torch.stack([target.evaluate(x) for target in self.hypotheses])
        return torch.einsum("h,hnd->nd", posterior, values)

    def propose_update(
        self,
        current_function: Functional,
        support_set: tuple[Tensor, Tensor],
        probe_inputs: Tensor,
    ) -> FunctionalUpdate:
        del probe_inputs
        posterior = self.posterior(support_set).detach()

        def updated(x: Tensor) -> Tensor:
            return self.predictive(posterior, x)

        return FunctionalUpdate(current_function, updated, {
            "teacher": "V31OracleTeacher",
            "hypothesis_count": len(self.hypotheses),
            "target_identity_exposed_to_ifi": False,
        })


def _ranges(values: tuple[float, ...]) -> dict[str, object]:
    return {"values": list(values), "min": min(values), "max": max(values), "width": max(values) - min(values)}


def reconstruct_training_support(config: V30Config, device: torch.device) -> dict[str, object]:
    library = build_latent_library(config.tasks, device)
    splits = latent_split_indices(library)

    def combinations(indices: tuple[int, ...]) -> list[dict[str, float]]:
        keys = ("a", "b", "c", "omega")
        return [
            dict(zip(keys, map(float, library.coefficients[index].tolist())))
            for index in indices
        ]

    return {
        "source": "persisted runs/v3_0_bayesian/config.resolved.json and checkpoints",
        "functional_form": "a*x + b*x^2 + c*sin(omega*x)",
        "coefficient_support": {
            "a": _ranges(config.tasks.coefficients_a),
            "b": _ranges(config.tasks.coefficients_b),
            "c": _ranges(config.tasks.coefficients_c),
            "omega": _ranges(config.tasks.frequencies),
        },
        "latent_library_size": library.size,
        "train_combinations": combinations(splits["train_latents"]),
        "heldout_combinations": combinations(splits["heldout_latents"]),
        "split_rule": "checkerboard: held out iff sum(discrete coefficient indices) mod 4 == 0",
        "x_range": [config.tasks.x_min, config.tasks.x_max],
        "sensor_centers": list(config.tasks.sensor_centers),
        "sensor_width": config.tasks.sensor_width,
        "observation_noise": {"distribution": "Gaussian", "std": config.tasks.observation_noise},
        "teacher_likelihood_noise": config.tasks.likelihood_noise,
        "support_sizes": list(config.tasks.support_sizes),
        "train_rollout_max": config.phase_b.rollout_max,
        "train_rollout_variable": config.phase_b.variable_rollout,
        "train_sequence_length": config.tasks.train_sequence_length,
        "v3_evaluation_rollouts": {
            "primary": config.evaluation.rollout_steps,
            "extended": config.evaluation.extended_rollout_steps,
        },
        "normalization": "none",
        "function_representation": {
            "dimension": config.operator.dimension,
            "fixed_features": "[initial_value, x, sin(x*f), cos(x*f), optional constant]",
            "fixed_feature_frequencies": [0.5, 4.0],
            "decoder": "first/value channel",
        },
        "residual_tokens": "[x,y,F(x),e,abs(e),e^2]",
    }


def build_target_registry(training_support: dict[str, object]) -> tuple[V31Target, ...]:
    targets: list[V31Target] = []
    heldout = training_support["heldout_combinations"]
    assert isinstance(heldout, list)
    for index, parameters in enumerate(heldout):
        targets.append(V31Target(
            f"heldout_{index:02d}", "heldout_combinations", "baseline", "original",
            dict(parameters), 0.0,
        ))

    support = training_support["coefficient_support"]
    assert isinstance(support, dict)
    base = {"a": 0.35, "b": 0.65, "c": 0.85, "omega": 2.0}
    severities = (("near", 0.10), ("medium", 0.25), ("far", 0.50))
    patterns = (
        ("a",), ("b",), ("c",), ("omega",), ("a", "b"), ("c", "omega"),
        ("a", "b", "c", "omega"),
    )
    for severity, distance in severities:
        for pattern in patterns:
            parameters = dict(base)
            for key in pattern:
                item = support[key]
                assert isinstance(item, dict)
                parameters[key] = float(item["max"]) + distance * float(item["width"])
            label = "_".join(pattern)
            targets.append(V31Target(
                f"coeff_{severity}_{label}", "coefficient_extrapolation", severity,
                "original", parameters, distance, len(pattern),
            ))

    templates = (
        {"a": -0.35, "b": 0.65, "c": 0.85, "omega": 1.0},
        {"a": 1.0, "b": -0.65, "c": -0.85, "omega": 2.0},
    )
    for formula in ("cubic", "abs_quadratic", "gaussian_bump", "cosine"):
        for index, template in enumerate(templates):
            parameters = dict(template)
            if formula == "gaussian_bump":
                parameters["gamma"] = 0.75 if index == 0 else 1.5
            targets.append(V31Target(
                f"struct_{formula}_{index}", "structural_shift", "structural", formula,
                parameters, 1.0,
            ))

    mild = (
        ("sin_quadratic", {"a": 0.85, "b": 0.35, "c": -0.35, "omega": 1.0}),
        ("x_sin", {"a": 0.85, "b": -0.65, "c": 0.35, "omega": 2.0}),
        ("sin_poly", {"a": 1.0, "b": 0.35, "c": -0.35, "omega": 1.0}),
        ("composed_tanh", {"a": 1.0, "b": 0.65, "c": 0.85, "omega": 2.0}),
    )
    hard = (
        ("deep_sin_poly", {"a": 0.85, "b": 0.35, "c": 0.65, "omega": 2.0}),
        ("deep_sin_sin", {"a": 1.0, "b": 0.65, "c": -0.35, "omega": 2.0}),
        ("deep_interaction", {"a": 0.85, "b": -0.35, "c": 0.35, "omega": 1.0}),
        ("depth3_composition", {"a": 1.0, "b": 0.65, "c": 0.35, "omega": 2.0}),
    )
    for severity, distance, definitions in (("mild", 1.5, mild), ("hard", 2.0, hard)):
        for index, (formula, parameters) in enumerate(definitions):
            targets.append(V31Target(
                f"composition_{severity}_{index}_{formula}", "compositional_shift",
                severity, formula, parameters, distance,
            ))
    return tuple(targets)


def registry_groups(targets: tuple[V31Target, ...]) -> dict[tuple[str, str], tuple[V31Target, ...]]:
    result: dict[tuple[str, str], list[V31Target]] = {}
    for target in targets:
        result.setdefault((target.bucket, target.severity), []).append(target)
    return {key: tuple(values) for key, values in result.items()}


def episode_seed(seed: int, support_size: int, target_id: str, episode_index: int) -> int:
    payload = f"{seed}:{support_size}:{target_id}:{episode_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def sample_episode(
    target: V31Target,
    train_config: V30Config,
    support_size: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> V31Episode:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    task = train_config.tasks
    for _ in range(steps):
        sensor_index = int(torch.randint(len(task.sensor_centers), (), generator=generator))
        center = task.sensor_centers[sensor_index]
        offsets = (
            torch.linspace(-task.sensor_width, task.sensor_width, support_size)
            if support_size > 1 else torch.zeros(1)
        )
        x = (center + offsets).unsqueeze(-1).to(device)
        noise = torch.randn((support_size, 1), generator=generator).to(device)
        y = target.evaluate(x) + task.observation_noise * noise
        batches.append(V31EvidenceBatch(x, y, sensor_index))
    return V31Episode(target, tuple(batches), str(seed))


def write_registry(path: str | Path, targets: tuple[V31Target, ...]) -> None:
    payload = {
        "locked_before_evaluation": True,
        "selection_depends_on_model_performance": False,
        "targets": [target.to_dict() for target in targets],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_registry(path: str | Path) -> tuple[V31Target, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(V31Target.from_dict(item) for item in payload["targets"])
