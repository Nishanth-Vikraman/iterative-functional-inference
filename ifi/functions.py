from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .operators import FunctionalOperator


class Functional(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


class FixedFeatureEncoder(nn.Module):
    """Fixed x features with a protected scalar value channel at index zero."""

    def __init__(self, dimension: int):
        super().__init__()
        if dimension < 6:
            raise ValueError("dimension must be at least 6")
        frequencies = torch.linspace(0.5, 4.0, (dimension - 2) // 2)
        self.register_buffer("frequencies", frequencies)
        self.dimension = dimension

    def forward(self, x: Tensor, initial_value: Tensor | None = None) -> Tensor:
        value = torch.zeros_like(x) if initial_value is None else initial_value
        pieces = [value, x]
        angles = x * self.frequencies
        pieces.extend([torch.sin(angles), torch.cos(angles)])
        z = torch.cat(pieces, dim=-1)
        if z.shape[-1] < self.dimension:
            z = torch.cat([z, torch.ones_like(x)], dim=-1)
        return z[..., : self.dimension]


class FixedValueDecoder(nn.Module):
    def forward(self, z: Tensor) -> Tensor:
        return z[..., :1]


class FunctionSystem(nn.Module):
    """Shared fixed encoder/decoder defining the d-dimensional function representation."""

    def __init__(self, dimension: int):
        super().__init__()
        self.encoder = FixedFeatureEncoder(dimension)
        self.decoder = FixedValueDecoder()
        self.dimension = dimension

    def encode(self, x: Tensor, initial_value: Tensor | None = None) -> Tensor:
        return self.encoder(x, initial_value)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


@dataclass(frozen=True)
class ComposedFunction:
    system: FunctionSystem
    operator: "FunctionalOperator"
    actions: tuple[Tensor, ...] = ()
    initial_function: Callable[[Tensor], Tensor] | None = None

    def latent(self, x: Tensor) -> Tensor:
        initial = None if self.initial_function is None else self.initial_function(x)
        z = self.system.encode(x, initial)
        for phi in self.actions:
            z = self.operator(z, phi)
        return z

    def __call__(self, x: Tensor) -> Tensor:
        return self.system.decode(self.latent(x))

    def append(self, phi: Tensor, *, detach: bool = False) -> "ComposedFunction":
        action = phi.detach() if detach else phi
        return ComposedFunction(
            self.system, self.operator, self.actions + (action,), self.initial_function
        )

    def detached(self) -> "ComposedFunction":
        return ComposedFunction(
            self.system, self.operator, tuple(a.detach() for a in self.actions), self.initial_function
        )

    @property
    def depth(self) -> int:
        return len(self.actions)

