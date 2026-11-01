from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor

from .config import TaskConfig


Split = Literal["train", "seen_parameters", "heldout_combinations", "heldout_compositions"]


@dataclass(frozen=True)
class TargetFunction:
    family: str
    parameters: dict[str, float]
    fn: Callable[[Tensor], Tensor]

    def __call__(self, x: Tensor) -> Tensor:
        return self.fn(x)


@dataclass(frozen=True)
class Episode:
    target: TargetFunction
    support_x: Tensor
    support_y: Tensor
    split: Split
    episode_id: int


def _uniform(generator: torch.Generator, low: float, high: float, device: torch.device) -> float:
    value = torch.rand((), generator=generator, device="cpu").item()
    return low + (high - low) * value


class SyntheticFunctionBenchmark:
    """Explicit novelty splits for 1D regression learning-rule evaluation."""

    SPLITS: tuple[Split, ...] = (
        "train", "seen_parameters", "heldout_combinations", "heldout_compositions"
    )

    def __init__(self, config: TaskConfig, seed: int = 0, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._next_id = 0

    def sample_inputs(self, n: int) -> Tensor:
        x = torch.rand((n, 1), generator=self.generator)
        x = self.config.x_min + (self.config.x_max - self.config.x_min) * x
        return x.to(self.device)

    def _primitive(self, family: str, heldout: bool) -> TargetFunction:
        g, dev = self.generator, self.device
        if family == "polynomial":
            scale = (1.0, 1.7) if heldout else (0.2, 1.0)
            a = _uniform(g, *scale, dev) * (-1 if _uniform(g, 0, 1, dev) < 0.5 else 1)
            b = _uniform(g, -1.2, 1.2, dev)
            c = _uniform(g, -0.6, 0.6, dev)
            return TargetFunction(family, {"a": a, "b": b, "c": c}, lambda x: a*x*x+b*x+c)
        if family == "sinusoid":
            freq = _uniform(g, 2.2, 3.8, dev) if heldout else _uniform(g, 0.6, 2.2, dev)
            amp = _uniform(g, 0.3, 1.3, dev)
            phase = _uniform(g, -3.14, 3.14, dev)
            return TargetFunction(family, {"amp": amp, "freq": freq, "phase": phase},
                                  lambda x: amp * torch.sin(freq*x+phase))
        if family == "gaussian":
            width = _uniform(g, 0.12, 0.28, dev) if heldout else _uniform(g, 0.28, 0.8, dev)
            center = _uniform(g, -1.4, 1.4, dev)
            amp = _uniform(g, -1.3, 1.3, dev)
            return TargetFunction(family, {"amp": amp, "center": center, "width": width},
                                  lambda x: amp*torch.exp(-0.5*((x-center)/width)**2))
        raise ValueError(f"Unknown primitive family: {family}")

    def sample_target(self, split: Split = "train") -> TargetFunction:
        if split == "heldout_compositions":
            base = self._primitive("sinusoid", heldout=True)
            bump = self._primitive("gaussian", heldout=False)
            return TargetFunction("composition", {}, lambda x: base(x + 0.25*bump(x)))
        primitive_names = [f for f in self.config.families if f != "mixture"]
        if split == "heldout_combinations" or (
            split == "train" and "mixture" in self.config.families
            and _uniform(self.generator, 0, 1, self.device) < 0.35
        ):
            left = self._primitive(primitive_names[0], heldout=False)
            right_name = primitive_names[1 + int(_uniform(
                self.generator, 0, max(1, len(primitive_names)-1), self.device
            )) % max(1, len(primitive_names)-1)]
            right = self._primitive(right_name, heldout=False)
            if split == "heldout_combinations":
                return TargetFunction("heldout_mixture", {}, lambda x: left(x) * (0.5 + right(x)))
            return TargetFunction("mixture", {}, lambda x: 0.6*left(x) + 0.6*right(x))
        index = int(_uniform(self.generator, 0, len(primitive_names), self.device))
        family = primitive_names[min(index, len(primitive_names) - 1)]
        return self._primitive(family, heldout=split == "seen_parameters")

    def sample_episode(self, support_size: int, split: Split = "train") -> Episode:
        target = self.sample_target(split)
        x = self.sample_inputs(support_size)
        y = target(x)
        if self.config.noise_std > 0:
            noise = torch.randn(y.shape, generator=self.generator).to(self.device)
            y = y + self.config.noise_std * noise
        result = Episode(target, x, y, split, self._next_id)
        self._next_id += 1
        return result

    def sample_support_size(self) -> int:
        choices = self.config.support_sizes
        index = int(torch.randint(len(choices), (), generator=self.generator).item())
        return choices[index]

