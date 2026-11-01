from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LowRankFunctionalGeometry:
    """Low-rank teacher-induced covariance plus an isotropic floor."""

    eigenvalues: Tensor
    eigenvectors: Tensor
    regularization: float
    raw_rank: int
    effective_rank: float
    condition_before: float
    condition_after: float

    def quadratic(self, vector: Tensor) -> Tensor:
        flat = vector.reshape(-1)
        regularization = torch.as_tensor(
            self.regularization, device=flat.device, dtype=flat.dtype
        )
        total = flat.square().sum() / regularization
        if self.eigenvalues.numel() == 0:
            return total
        eigenvalues = self.eigenvalues.to(flat)
        eigenvectors = self.eigenvectors.to(flat)
        coordinates = eigenvectors.transpose(0, 1) @ flat
        correction = coordinates.square() * (
            1.0 / regularization - 1.0 / (eigenvalues + regularization)
        )
        return total - correction.sum()


def functional_update(candidate_values: Tensor, current_values: Tensor) -> Tensor:
    return candidate_values - current_values


def normalized_functional_loss(
    predicted_update: Tensor,
    target_update: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    numerator = (predicted_update - target_update).square().mean()
    denominator = target_update.square().mean()
    return numerator / (denominator + epsilon), numerator, denominator


def normalized_mahalanobis_loss(
    predicted_update: Tensor,
    target_update: Tensor,
    geometry: LowRankFunctionalGeometry,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    error = predicted_update - target_update
    numerator = geometry.quadratic(error) / error.numel()
    denominator = geometry.quadratic(target_update) / target_update.numel()
    return numerator / (denominator + epsilon), numerator, denominator


def teacher_geometry(
    perturbations: Tensor,
    regularization: float,
    rank_tolerance: float,
) -> LowRankFunctionalGeometry:
    """Build covariance geometry from [sample, functional-grid] perturbations."""
    if perturbations.ndim != 2 or perturbations.shape[0] < 2:
        raise ValueError("teacher perturbations must be [M, grid] with M >= 2")
    centered = perturbations - perturbations.mean(dim=0, keepdim=True)
    scaled = centered / (perturbations.shape[0] - 1) ** 0.5
    _, singular_values, vh = torch.linalg.svd(scaled, full_matrices=False)
    eigenvalues = singular_values.square()
    maximum = eigenvalues.max().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    keep = eigenvalues > rank_tolerance * maximum
    retained = eigenvalues[keep].detach()
    eigenvectors = vh[keep].transpose(0, 1).detach()
    raw_rank = int(keep.sum())
    if retained.numel():
        probabilities = retained / retained.sum()
        effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
        minimum = float(retained.min())
        maximum_value = float(retained.max())
        condition_before = maximum_value / max(minimum, 1e-30)
        condition_after = (maximum_value + regularization) / regularization
    else:
        effective_rank = 0.0
        condition_before = float("inf")
        condition_after = 1.0
    return LowRankFunctionalGeometry(
        retained,
        eigenvectors,
        float(regularization),
        raw_rank,
        effective_rank,
        condition_before,
        condition_after,
    )


def posterior_function_perturbations(
    posterior: Tensor,
    hypothesis_values: Tensor,
    noise: Tensor,
) -> Tensor:
    """Perturb valid Bayesian log-posterior coordinates and return function changes."""
    logits = posterior.clamp_min(1e-30).log()
    perturbed = torch.softmax(logits.unsqueeze(0) + noise, dim=-1)
    base = posterior @ hypothesis_values
    return perturbed @ hypothesis_values - base.unsqueeze(0)


def antithetic_logit_noise(
    samples: int,
    hypotheses: int,
    scale: float,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    if samples % 2:
        raise ValueError("covariance_samples must be even for antithetic perturbations")
    half = torch.randn((samples // 2, hypotheses), generator=generator)
    return torch.cat([half, -half], dim=0).to(device) * scale
