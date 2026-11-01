from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .models import DeepSetEncoder, residual_features


@dataclass(frozen=True)
class ForwardStep:
    action: Tensor
    next_state: Tensor


class ForwardCreditTransformer(nn.Module):
    """One recurrent T with a shared evidence trunk and separate action/state heads."""

    def __init__(
        self,
        model_dimension: int,
        action_dimension: int,
        max_support_size: int,
        max_history: int = 32,
    ):
        super().__init__()
        self.raw_residual_projection = nn.Linear(6, model_dimension)
        self.data_projection = nn.Linear(model_dimension, model_dimension)
        self.state_projection = nn.Linear(model_dimension, model_dimension)
        self.history_projection = nn.Linear(action_dimension, model_dimension)
        self.null_history = nn.Parameter(torch.zeros(model_dimension))
        self.type_embedding = nn.Embedding(4, model_dimension)
        self.position_embedding = nn.Parameter(
            torch.zeros(max_support_size + max_history + 3, model_dimension)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=4,
            dim_feedforward=2 * model_dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(model_dimension)
        )
        self.phi_head = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Linear(model_dimension, model_dimension),
            nn.GELU(),
            nn.Linear(model_dimension, action_dimension),
        )
        # Only this head belongs to theta_h. Its output is not norm-constrained.
        self.h_delta = nn.Linear(model_dimension + action_dimension, model_dimension)
        self.h_gate = nn.Linear(model_dimension + action_dimension, model_dimension)

    def _encode(
        self,
        raw_tokens: Tensor,
        z_data: Tensor,
        state: Tensor,
        history: tuple[Tensor, ...],
    ) -> Tensor:
        device = raw_tokens.device
        data = self.data_projection(z_data).unsqueeze(0)
        data = data + self.type_embedding(torch.tensor(0, device=device))
        state_token = self.state_projection(state).unsqueeze(0)
        state_token = state_token + self.type_embedding(torch.tensor(1, device=device))
        if history:
            action_tokens = self.history_projection(torch.stack(history))
        else:
            action_tokens = self.null_history.unsqueeze(0)
        action_tokens = action_tokens + self.type_embedding(torch.tensor(2, device=device))
        residual_tokens = self.raw_residual_projection(raw_tokens)
        residual_tokens = residual_tokens + self.type_embedding(torch.tensor(3, device=device))
        tokens = torch.cat([data, state_token, action_tokens, residual_tokens], dim=0)
        if tokens.shape[0] > self.position_embedding.shape[0]:
            raise ValueError("Token sequence exceeds v2.3 controller capacity")
        return self.trunk(
            (tokens + self.position_embedding[: tokens.shape[0]]).unsqueeze(0)
        ).squeeze(0)

    def action_only(
        self,
        raw_tokens: Tensor,
        z_data: Tensor,
        state: Tensor,
        history: tuple[Tensor, ...],
    ) -> Tensor:
        encoded = self._encode(raw_tokens, z_data, state, history)
        return self.phi_head(encoded[0])

    def forward(
        self,
        raw_tokens: Tensor,
        z_data: Tensor,
        state: Tensor,
        history: tuple[Tensor, ...],
    ) -> ForwardStep:
        encoded = self._encode(raw_tokens, z_data, state, history)
        action = self.phi_head(encoded[0])
        h_input = torch.cat([encoded[1], action.detach()], dim=-1)
        delta = torch.tanh(self.h_delta(h_input))
        gate = torch.sigmoid(self.h_gate(h_input))
        return ForwardStep(action, state + gate * delta)


class ForwardTrainedPolicy(nn.Module):
    def __init__(self, model_dimension: int, action_dimension: int, max_support_size: int):
        super().__init__()
        self.data_encoder = DeepSetEncoder(2, model_dimension)
        self.controller = ForwardCreditTransformer(
            model_dimension, action_dimension, max_support_size
        )
        self.model_dimension = model_dimension
        self.action_dimension = action_dimension

    def h_parameters(self) -> list[nn.Parameter]:
        return list(self.controller.h_delta.parameters()) + list(self.controller.h_gate.parameters())

    def action_parameters(self) -> list[nn.Parameter]:
        h_ids = {id(parameter) for parameter in self.h_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in h_ids]

    def set_training_block(self, block: str) -> None:
        if block not in {"h", "phi"}:
            raise ValueError(f"Unknown coordinate block: {block}")
        for parameter in self.action_parameters():
            parameter.requires_grad_(block == "phi")
        for parameter in self.h_parameters():
            parameter.requires_grad_(block == "h")

    def encode_data(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        return self.data_encoder(torch.cat([support_x, support_y], dim=-1))

    def raw_residual_tokens(
        self, support_x: Tensor, support_y: Tensor, prediction: Tensor
    ) -> Tensor:
        return residual_features(support_x, support_y, prediction)

    def initial_state(self, reference: Tensor) -> Tensor:
        return reference.new_zeros(self.model_dimension)

    def step(
        self,
        raw_tokens: Tensor,
        z_data: Tensor,
        state: Tensor,
        history: tuple[Tensor, ...],
    ) -> ForwardStep:
        return self.controller(raw_tokens, z_data, state, history)

    def predict_action(
        self,
        raw_tokens: Tensor,
        z_data: Tensor,
        state: Tensor,
        history: tuple[Tensor, ...],
    ) -> Tensor:
        return self.controller.action_only(raw_tokens, z_data, state, history)


def parameter_counts(policy: ForwardTrainedPolicy) -> dict[str, int]:
    phi = sum(parameter.numel() for parameter in policy.action_parameters())
    h = sum(parameter.numel() for parameter in policy.h_parameters())
    return {"shared_and_phi": phi, "h_head": h, "total": phi + h}
