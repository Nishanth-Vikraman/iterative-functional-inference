from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import DeepSetEncoder, residual_features


def _transformer(model_dimension: int, heads: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=model_dimension,
        nhead=heads,
        dim_feedforward=2 * model_dimension,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(model_dimension))


class PredictiveActionController(nn.Module):
    """T: directly attends to current raw residuals, state, data, and action history."""

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
        self.transformer = _transformer(model_dimension, heads=4, layers=2)
        self.action_head = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Linear(model_dimension, model_dimension),
            nn.GELU(),
            nn.Linear(model_dimension, action_dimension),
        )

    def forward(
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
        encoded = self.transformer(
            (tokens + self.position_embedding[: tokens.shape[0]]).unsqueeze(0)
        ).squeeze(0)
        return self.action_head(encoded[0])


class PredictiveStateUpdater(nn.Module):
    """U: retains history predictive of the next residual; never consumes teacher targets."""

    def __init__(
        self,
        model_dimension: int,
        action_dimension: int,
        max_support_size: int,
        layers: int,
    ):
        super().__init__()
        self.raw_residual_projection = nn.Linear(6, model_dimension)
        self.state_projection = nn.Linear(model_dimension, model_dimension)
        self.action_projection = nn.Linear(action_dimension, model_dimension)
        self.type_embedding = nn.Embedding(3, model_dimension)
        self.position_embedding = nn.Parameter(
            torch.zeros(max_support_size + 2, model_dimension)
        )
        self.transformer = _transformer(model_dimension, heads=4, layers=layers)
        self.delta_head = nn.Linear(model_dimension, model_dimension)
        self.gate_head = nn.Linear(model_dimension, model_dimension)

    def forward(self, state: Tensor, raw_tokens: Tensor, action: Tensor) -> Tensor:
        device = raw_tokens.device
        state_token = self.state_projection(state).unsqueeze(0)
        state_token = state_token + self.type_embedding(torch.tensor(0, device=device))
        action_token = self.action_projection(action).unsqueeze(0)
        action_token = action_token + self.type_embedding(torch.tensor(1, device=device))
        residual_tokens = self.raw_residual_projection(raw_tokens)
        residual_tokens = residual_tokens + self.type_embedding(torch.tensor(2, device=device))
        tokens = torch.cat([state_token, action_token, residual_tokens], dim=0)
        encoded = self.transformer(
            (tokens + self.position_embedding[: tokens.shape[0]]).unsqueeze(0)
        ).squeeze(0)[0]
        delta = torch.tanh(self.delta_head(encoded))
        gate = torch.sigmoid(self.gate_head(encoded))
        return state + gate * delta


@dataclass(frozen=True)
class PredictiveStep:
    action: Tensor
    next_state: Tensor


class PredictiveRecurrentPolicy(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        action_dimension: int,
        max_support_size: int,
        updater_layers: int = 1,
    ):
        super().__init__()
        self.data_encoder = DeepSetEncoder(2, model_dimension)
        self.action_controller = PredictiveActionController(
            model_dimension, action_dimension, max_support_size
        )
        self.state_updater = PredictiveStateUpdater(
            model_dimension, action_dimension, max_support_size, updater_layers
        )
        self.model_dimension = model_dimension
        self.action_dimension = action_dimension

    def t_parameters(self) -> list[nn.Parameter]:
        return list(self.data_encoder.parameters()) + list(self.action_controller.parameters())

    def u_parameters(self) -> list[nn.Parameter]:
        return list(self.state_updater.parameters())

    def encode_data(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        return self.data_encoder(torch.cat([support_x, support_y], dim=-1))

    def raw_residual_tokens(
        self, support_x: Tensor, support_y: Tensor, prediction: Tensor
    ) -> Tensor:
        return residual_features(support_x, support_y, prediction)

    def initial_state(self, reference: Tensor) -> Tensor:
        return reference.new_zeros(self.model_dimension)

    def predict_action(
        self,
        raw_tokens: Tensor,
        z_data: Tensor,
        state: Tensor,
        history: tuple[Tensor, ...],
    ) -> Tensor:
        # This detach is the hard boundary preventing teacher-action gradients from reaching U.
        return self.action_controller(raw_tokens, z_data, state.detach(), history)

    def update_state(self, state: Tensor, raw_tokens: Tensor, action: Tensor) -> Tensor:
        # These detaches prevent predictive-state gradients from reaching T or prior rollout steps.
        return self.state_updater(state.detach(), raw_tokens.detach(), action.detach())


class FixedResidualContrast(nn.Module):
    """Fixed moment signature and fixed random state projection; contains no trainable parameters."""

    signature_dimension = 24

    def __init__(self, state_dimension: int, seed: int, temperature: float):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_matrix = torch.randn(
            state_dimension, self.signature_dimension, generator=generator
        )
        projection = F.normalize(random_matrix, dim=0)
        self.register_buffer("state_projection", projection)
        self.temperature = temperature

    @staticmethod
    def residual_signature(raw_tokens: Tensor) -> Tensor:
        return torch.cat([
            raw_tokens.mean(dim=0),
            raw_tokens.std(dim=0, unbiased=False),
            raw_tokens.amin(dim=0),
            raw_tokens.amax(dim=0),
        ])

    def forward(
        self, next_states: Tensor, future_raw_tokens: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        state_codes = F.normalize(next_states @ self.state_projection, dim=-1)
        residual_codes = F.normalize(
            torch.stack([self.residual_signature(tokens) for tokens in future_raw_tokens]),
            dim=-1,
        )
        logits = state_codes @ residual_codes.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, accuracy


def parameter_counts(policy: PredictiveRecurrentPolicy) -> dict[str, int]:
    t_count = sum(parameter.numel() for parameter in policy.t_parameters())
    u_count = sum(parameter.numel() for parameter in policy.u_parameters())
    return {"T": t_count, "U": u_count, "total": t_count + u_count}
