from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .models import DeepSetEncoder, residual_features


DirectVariant = Literal["direct_current", "direct_history", "direct_recurrent"]


@dataclass(frozen=True)
class DirectPolicyStep:
    action: Tensor
    next_state: Tensor | None


class DirectResidualTransformer(nn.Module):
    """Controller T: token-level residual interpretation, action history, and optional memory."""

    def __init__(
        self,
        model_dimension: int,
        action_dimension: int,
        max_support_size: int,
        max_history: int = 32,
        heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        if model_dimension % heads:
            raise ValueError("model_dimension must be divisible by heads")
        self.raw_residual_projection = nn.Linear(6, model_dimension)
        self.data_projection = nn.Linear(model_dimension, model_dimension)
        self.history_projection = nn.Linear(action_dimension, model_dimension)
        self.state_projection = nn.Linear(model_dimension, model_dimension)
        self.type_embedding = nn.Embedding(4, model_dimension)
        self.position_embedding = nn.Parameter(
            torch.zeros(max_support_size + max_history + 3, model_dimension)
        )
        self.null_history = nn.Parameter(torch.zeros(model_dimension))
        self.null_state = nn.Parameter(torch.zeros(model_dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=heads,
            dim_feedforward=2 * model_dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(model_dimension)
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Linear(model_dimension, model_dimension),
            nn.GELU(),
            nn.Linear(model_dimension, action_dimension),
        )
        self.action_to_state = nn.Linear(action_dimension, model_dimension)
        self.state_norm = nn.LayerNorm(model_dimension)
        self.model_dimension = model_dimension

    def forward(
        self,
        raw_residual_tokens: Tensor,
        z_data: Tensor,
        action_history: tuple[Tensor, ...],
        state: Tensor | None,
        *,
        use_history: bool,
        use_recurrent_state: bool,
    ) -> DirectPolicyStep:
        device = raw_residual_tokens.device
        data_token = self.data_projection(z_data).unsqueeze(0) + self.type_embedding(
            torch.tensor(0, device=device)
        )
        if use_recurrent_state and state is not None:
            state_value = self.state_projection(state)
        else:
            state_value = self.null_state
        state_token = state_value.unsqueeze(0) + self.type_embedding(torch.tensor(1, device=device))
        if use_history and action_history:
            history = self.history_projection(torch.stack(action_history))
        else:
            history = self.null_history.unsqueeze(0)
        history = history + self.type_embedding(torch.tensor(2, device=device))
        residual = self.raw_residual_projection(raw_residual_tokens)
        residual = residual + self.type_embedding(torch.tensor(3, device=device))
        tokens = torch.cat([data_token, state_token, history, residual], dim=0)
        if tokens.shape[0] > self.position_embedding.shape[0]:
            raise ValueError("Token sequence exceeds configured controller capacity")
        encoded = self.transformer(
            (tokens + self.position_embedding[: tokens.shape[0]]).unsqueeze(0)
        ).squeeze(0)
        action = self.action_head(encoded[0])
        next_state = None
        if use_recurrent_state:
            next_state = self.state_norm(encoded[1] + self.action_to_state(action))
        return DirectPolicyStep(action, next_state)


class DirectIFIPolicy(nn.Module):
    """Matched A/B/C policy with no standalone learned residual encoder."""

    def __init__(
        self,
        variant: DirectVariant,
        model_dimension: int,
        action_dimension: int,
        max_support_size: int,
    ):
        super().__init__()
        self.variant = variant
        self.data_encoder = DeepSetEncoder(2, model_dimension)
        self.controller = DirectResidualTransformer(
            model_dimension, action_dimension, max_support_size
        )
        self.action_dimension = action_dimension
        self.model_dimension = model_dimension

    @property
    def uses_history(self) -> bool:
        return self.variant in {"direct_history", "direct_recurrent"}

    @property
    def uses_recurrent_state(self) -> bool:
        return self.variant == "direct_recurrent"

    def encode_data(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        return self.data_encoder(torch.cat([support_x, support_y], dim=-1))

    def raw_residual_tokens(
        self, support_x: Tensor, support_y: Tensor, prediction: Tensor
    ) -> Tensor:
        return residual_features(support_x, support_y, prediction)

    def initial_state(self, reference: Tensor) -> Tensor | None:
        if not self.uses_recurrent_state:
            return None
        return reference.new_zeros(self.model_dimension)

    def step(
        self,
        raw_residual_tokens: Tensor,
        z_data: Tensor,
        action_history: tuple[Tensor, ...],
        state: Tensor | None,
    ) -> DirectPolicyStep:
        return self.controller(
            raw_residual_tokens,
            z_data,
            action_history,
            state,
            use_history=self.uses_history,
            use_recurrent_state=self.uses_recurrent_state,
        )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
