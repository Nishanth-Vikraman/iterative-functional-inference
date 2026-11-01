from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import OperatorConfig
from .functions import ComposedFunction, FunctionSystem
from .teachers import FunctionalUpdate


class FunctionalOperator(nn.Module):
    """O_phi(z) = z + sum_k phi_k B_k(z), with O_0 exactly the identity."""

    def __init__(self, config: OperatorConfig):
        super().__init__()
        layers: list[nn.Module] = []
        width = config.dimension
        for _ in range(config.basis_depth):
            layers.extend([nn.Linear(width, config.hidden_dimension), nn.SiLU()])
            width = config.hidden_dimension
        layers.append(nn.Linear(width, config.action_dimension * config.dimension))
        self.bases = nn.Sequential(*layers)
        self.dimension = config.dimension
        self.action_dimension = config.action_dimension

    def basis_values(self, z: Tensor) -> Tensor:
        values = self.bases(z)
        return values.reshape(*z.shape[:-1], self.action_dimension, self.dimension)

    def forward(self, z: Tensor, phi: Tensor) -> Tensor:
        basis = self.basis_values(z)
        if phi.ndim == 1:
            update = torch.einsum("...rd,r->...d", basis, phi)
        else:
            while phi.ndim < basis.ndim - 1:
                phi = phi.unsqueeze(-2)
            update = torch.einsum("...rd,...r->...d", basis, phi)
        return z + update

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.parameters())


@dataclass(frozen=True)
class ProjectionResult:
    phi: Tensor
    reconstruction_mse: Tensor
    target_energy: Tensor
    relative_residual: Tensor
    gram: Tensor
    condition_number: Tensor


def action_design(
    operator: FunctionalOperator,
    function: ComposedFunction,
    probes: Tensor,
) -> Tensor:
    """Scalar decoded displacement produced by every unit action."""
    latent = function.latent(probes)
    basis = operator.basis_values(latent)
    return basis[..., 0]  # fixed decoder selects value channel


def project_update(
    operator: FunctionalOperator,
    function: ComposedFunction,
    update: FunctionalUpdate,
    probes: Tensor,
    ridge: float,
) -> ProjectionResult:
    design = action_design(operator, function, probes).reshape(-1, operator.action_dimension)
    target = update.displacement(probes).reshape(-1, 1).detach()
    gram = design.transpose(0, 1) @ design / max(1, design.shape[0])
    rhs = design.transpose(0, 1) @ target / max(1, design.shape[0])
    identity = torch.eye(operator.action_dimension, device=design.device, dtype=design.dtype)
    phi = torch.linalg.solve(gram + ridge * identity, rhs).squeeze(-1)
    residual = design @ phi.unsqueeze(-1) - target
    mse = residual.square().mean()
    energy = target.square().mean()
    relative = mse / energy.clamp_min(1e-10)
    eigenvalues = torch.linalg.eigvalsh(gram.detach())
    condition = eigenvalues[-1] / eigenvalues[0].clamp_min(1e-10)
    return ProjectionResult(phi, mse, energy, relative, gram, condition)


def build_function_system(config: OperatorConfig, device: torch.device) -> tuple[FunctionSystem, FunctionalOperator]:
    system = FunctionSystem(config.dimension).to(device)
    operator = FunctionalOperator(config).to(device)
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    return system, operator

