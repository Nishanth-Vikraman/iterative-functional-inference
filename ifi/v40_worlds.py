from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class WorldBatch:
    matrices: Tensor
    singular_values: Tensor
    seeds: Tensor


def _orthogonal(batch: int, dimension: int, generator: torch.Generator) -> Tensor:
    raw = torch.randn(batch, dimension, dimension, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(-2)


def sample_well_conditioned_worlds(
    count: int,
    dimension: int,
    singular_min: float,
    singular_max: float,
    generator: torch.Generator,
    first_seed: int,
    condition_range: tuple[float, float] | None = None,
) -> WorldBatch:
    q1 = _orthogonal(count, dimension, generator)
    q2 = _orthogonal(count, dimension, generator)
    if condition_range is None:
        singular = singular_min + (singular_max - singular_min) * torch.rand(
            count, dimension, generator=generator
        )
    else:
        condition_min, condition_max = condition_range
        condition = condition_min + (condition_max - condition_min) * torch.rand(
            count, generator=generator
        )
        low = singular_min + (singular_max / condition - singular_min) * torch.rand(
            count, generator=generator
        )
        high = low * condition
        interior = low[:, None] + (high - low)[:, None] * torch.rand(
            count, dimension, generator=generator
        )
        interior[:, 0] = low
        interior[:, 1] = high
        order = torch.argsort(torch.rand(count, dimension, generator=generator), dim=-1)
        singular = torch.gather(interior, 1, order)
    matrices = q1 @ torch.diag_embed(singular) @ q2.transpose(-1, -2)
    seeds = torch.arange(first_seed, first_seed + count, dtype=torch.int64)
    return WorldBatch(matrices, singular, seeds)


def sample_initial_residuals(
    count: int,
    dimension: int,
    generator: torch.Generator,
    minimum_norm: float = 0.5,
    maximum_norm: float = 2.0,
) -> Tensor:
    direction = torch.randn(count, dimension, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    log_min = torch.tensor(minimum_norm).log()
    log_max = torch.tensor(maximum_norm).log()
    magnitude = torch.exp(log_min + (log_max - log_min) * torch.rand(count, generator=generator))
    return direction * magnitude[:, None]


def sample_probe_actions(
    count: int,
    probes: int,
    dimension: int,
    action_norm: float,
    generator: torch.Generator,
    orthogonal: bool = False,
) -> Tensor:
    if orthogonal:
        if probes > dimension:
            raise ValueError("Cannot create more mutually orthogonal probes than dimensions")
        basis = _orthogonal(count, dimension, generator)
        return action_norm * basis[:, :probes]
    actions = torch.randn(count, probes, dimension, generator=generator)
    return action_norm * actions / actions.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def transition(residual: Tensor, matrix: Tensor, action: Tensor) -> Tensor:
    return residual + torch.bmm(matrix, action.unsqueeze(-1)).squeeze(-1)


def analytic_action(matrix: Tensor, residual: Tensor) -> Tensor:
    return torch.linalg.solve(matrix, -residual.unsqueeze(-1)).squeeze(-1)
