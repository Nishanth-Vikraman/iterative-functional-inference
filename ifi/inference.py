from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .functions import ComposedFunction
from .models import IFIPolicy, residual_features


Intervention = Literal["correct", "zero", "shuffled", "stale", "wrong_task"]


@dataclass(frozen=True)
class InferenceTrace:
    functions: tuple[ComposedFunction, ...]
    actions: tuple[Tensor, ...]
    residual_embeddings: tuple[Tensor, ...]


@torch.no_grad()
def adapt_without_gradients(
    policy: IFIPolicy,
    initial_function: ComposedFunction,
    support_x: Tensor,
    support_y: Tensor,
    steps: int,
    intervention: Intervention = "correct",
    wrong_support: tuple[Tensor, Tensor] | None = None,
) -> InferenceTrace:
    """The complete teacher-free test path. No optimizer or query labels are accepted."""
    policy.eval()
    z_data = policy.encode_data(support_x, support_y)
    previous_action, state = policy.initial(z_data)
    current = initial_function
    functions = [current]
    actions: list[Tensor] = []
    embeddings: list[Tensor] = []
    stale_features: Tensor | None = None
    for _ in range(steps):
        prediction = current(support_x)
        features = residual_features(support_x, support_y, prediction)
        if intervention == "zero":
            z_residual = prediction.new_zeros(policy.latent_dimension)
        elif intervention == "shuffled":
            permutation = torch.arange(support_x.shape[0] - 1, -1, -1, device=support_x.device)
            corrupted = features.clone()
            corrupted[:, 3:] = corrupted[permutation, 3:]
            z_residual = policy.residual_encoder(corrupted)
        elif intervention == "stale" and stale_features is not None:
            z_residual = policy.residual_encoder(stale_features)
        elif intervention == "wrong_task":
            if wrong_support is None:
                raise ValueError("wrong_task intervention requires wrong_support")
            wrong_x, wrong_y = wrong_support
            wrong_prediction = current(wrong_x)
            z_residual = policy.residual_encoder(
                residual_features(wrong_x, wrong_y, wrong_prediction)
            )
        else:
            z_residual = policy.encode_residual(support_x, support_y, prediction)
        phi, state = policy.controller(z_data, z_residual, previous_action, state)
        current = current.append(phi.detach(), detach=True)
        functions.append(current)
        actions.append(phi.detach())
        embeddings.append(z_residual.detach())
        previous_action = phi.detach()
        stale_features = features
    return InferenceTrace(tuple(functions), tuple(actions), tuple(embeddings))

