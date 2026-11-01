from __future__ import annotations

import torch
from torch import Tensor

from ..config import TeacherConfig
from ..functions import Functional
from .base import FunctionalUpdate, Teacher


def rbf_kernel(x: Tensor, centers: Tensor, bandwidth: float) -> Tensor:
    squared_distance = (x.unsqueeze(-2) - centers.unsqueeze(-3)).square().sum(dim=-1)
    return torch.exp(-0.5 * squared_distance / (bandwidth * bandwidth))


class RKHSFunctionalGradientTeacher(Teacher):
    """One deterministic empirical squared-loss functional-gradient step."""

    def __init__(self, config: TeacherConfig):
        if config.kernel != "rbf":
            raise ValueError(f"Unsupported kernel: {config.kernel}")
        if config.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.config = config

    def propose_update(
        self,
        current_function: Functional,
        support_set: tuple[Tensor, Tensor],
        probe_inputs: Tensor,
    ) -> FunctionalUpdate:
        del probe_inputs  # interface symmetry; the teacher uses support data only
        support_x, support_y = support_set
        with torch.no_grad():
            residual = support_y - current_function(support_x)
            scale = self.config.step_size
            if self.config.normalize_by_support:
                scale /= support_x.shape[-2]
            coefficients = scale * residual

        def updated(x: Tensor) -> Tensor:
            kernel = rbf_kernel(x, support_x, self.config.bandwidth)
            return current_function(x) + kernel @ coefficients

        support_before = residual.square().mean()
        support_after = (support_y - updated(support_x)).square().mean()
        metadata = {
            "kernel": self.config.kernel,
            "bandwidth": self.config.bandwidth,
            "support_size": support_x.shape[-2],
            "support_mse_before": float(support_before.detach()),
            "support_mse_after": float(support_after.detach()),
            "predicted_improvement": float((support_before - support_after).detach()),
        }
        return FunctionalUpdate(current_function, updated, metadata)
