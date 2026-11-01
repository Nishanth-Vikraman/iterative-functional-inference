from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepSetEncoder(nn.Module):
    def __init__(self, input_dimension: int, latent_dimension: int):
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(input_dimension, latent_dimension),
            nn.SiLU(),
            nn.Linear(latent_dimension, latent_dimension),
            nn.SiLU(),
        )
        self.aggregate = nn.Sequential(
            nn.Linear(latent_dimension, latent_dimension), nn.LayerNorm(latent_dimension), nn.SiLU()
        )

    def forward(self, elements: Tensor) -> Tensor:
        return self.aggregate(self.element(elements).mean(dim=-2))


def residual_features(x: Tensor, y: Tensor, prediction: Tensor) -> Tensor:
    residual = y - prediction
    return torch.cat(
        [x, y, prediction, residual, residual.abs(), residual.square()], dim=-1
    )


class RecurrentController(nn.Module):
    """Small autoregressive GRU controller, replaceable behind the same interface."""

    def __init__(self, latent_dimension: int, action_dimension: int, hidden_dimension: int):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.cell = nn.GRUCell(2 * latent_dimension + action_dimension, hidden_dimension)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, action_dimension),
        )

    def initial_state(self, reference: Tensor) -> Tensor:
        return reference.new_zeros(self.hidden_dimension)

    def forward(
        self, z_data: Tensor, z_residual: Tensor, previous_action: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        controller_input = torch.cat([z_data, z_residual, previous_action], dim=-1)
        next_state = self.cell(controller_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
        return self.head(next_state), next_state


class IFIPolicy(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        action_dimension: int,
        controller_hidden_dimension: int,
        use_residual_feedback: bool = True,
    ):
        super().__init__()
        self.data_encoder = DeepSetEncoder(2, latent_dimension)
        self.residual_encoder = DeepSetEncoder(6, latent_dimension)
        self.controller = RecurrentController(
            latent_dimension, action_dimension, controller_hidden_dimension
        )
        self.action_dimension = action_dimension
        self.latent_dimension = latent_dimension
        self.use_residual_feedback = use_residual_feedback

    def encode_data(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        return self.data_encoder(torch.cat([support_x, support_y], dim=-1))

    def encode_residual(self, support_x: Tensor, support_y: Tensor, prediction: Tensor) -> Tensor:
        if not self.use_residual_feedback:
            return prediction.new_zeros(self.latent_dimension)
        return self.residual_encoder(residual_features(support_x, support_y, prediction))

    def initial(self, z_data: Tensor) -> tuple[Tensor, Tensor]:
        return z_data.new_zeros(self.action_dimension), self.controller.initial_state(z_data)


class StaticHypernetwork(nn.Module):
    """Matched non-iterative baseline: support set directly predicts a full action sequence."""

    def __init__(self, latent_dimension: int, action_dimension: int, rollout_max: int):
        super().__init__()
        self.data_encoder = DeepSetEncoder(2, latent_dimension)
        self.head = nn.Sequential(
            nn.Linear(latent_dimension, latent_dimension),
            nn.SiLU(),
            nn.Linear(latent_dimension, rollout_max * action_dimension),
        )
        self.action_dimension = action_dimension
        self.rollout_max = rollout_max

    def forward(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        z = self.data_encoder(torch.cat([support_x, support_y], dim=-1))
        return self.head(z).reshape(self.rollout_max, self.action_dimension)

