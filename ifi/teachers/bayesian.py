from __future__ import annotations

import torch
from torch import Tensor

from ..functions import Functional
from ..v30_tasks import LatentLibrary
from .base import FunctionalUpdate, Teacher


class BayesianLatentSystemTeacher(Teacher):
    """Exact finite-hypothesis Bayesian posterior-predictive teacher."""

    def __init__(self, library: LatentLibrary, likelihood_noise: float):
        if likelihood_noise <= 0:
            raise ValueError("likelihood_noise must be positive")
        self.library = library
        self.likelihood_noise = likelihood_noise

    def posterior(self, support_set: tuple[Tensor, Tensor]) -> Tensor:
        x, y = support_set
        predictions = self.library.evaluate(x).squeeze(-1)
        residual = y.squeeze(-1).unsqueeze(0) - predictions
        log_likelihood = -0.5 * residual.square().sum(dim=-1) / (
            self.likelihood_noise ** 2
        )
        return torch.softmax(log_likelihood, dim=0)

    def predictive(self, posterior: Tensor, x: Tensor) -> Tensor:
        return torch.einsum("h,hnd->nd", posterior, self.library.evaluate(x))

    def propose_update(
        self,
        current_function: Functional,
        support_set: tuple[Tensor, Tensor],
        probe_inputs: Tensor,
    ) -> FunctionalUpdate:
        del probe_inputs
        with torch.no_grad():
            posterior = self.posterior(support_set)

        def updated(x: Tensor) -> Tensor:
            return self.predictive(posterior, x)

        entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum()
        metadata = {
            "teacher": "BayesianLatentSystemTeacher",
            "posterior_entropy": float(entropy),
            "effective_hypotheses": float(entropy.exp()),
            "support_size": support_set[0].shape[0],
        }
        return FunctionalUpdate(current_function, updated, metadata)
