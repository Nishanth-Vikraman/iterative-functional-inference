from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .models import DeepSetEncoder, residual_features


V30Variant = Literal[
    "direct_current", "full_history", "query_history", "ar_no_feedback"
]


@dataclass(frozen=True)
class HistoryItem:
    residual: Tensor | None
    action: Tensor | None


class FullHistoryController(nn.Module):
    """Causal self-attention over explicit residual/action history and ACTION."""

    def __init__(
        self,
        dimension: int,
        action_dimension: int,
        max_support: int,
        max_steps: int,
    ):
        super().__init__()
        self.raw_projection = nn.Linear(6, dimension)
        self.data_projection = nn.Linear(dimension, dimension)
        self.action_projection = nn.Linear(action_dimension, dimension)
        self.type_embedding = nn.Embedding(4, dimension)
        self.step_embedding = nn.Embedding(max_steps + 2, dimension)
        maximum_tokens = 2 + (max_support + 1) * (max_steps + 1)
        self.position_embedding = nn.Parameter(torch.zeros(maximum_tokens, dimension))
        self.action_token = nn.Parameter(torch.zeros(dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=2 * dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # Four self-attention layers approximately match three cross-attention layers in B.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=4, norm=nn.LayerNorm(dimension)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, action_dimension),
        )

    def forward(
        self,
        current_raw: Tensor | None,
        z_data: Tensor,
        history: tuple[HistoryItem, ...],
    ) -> Tensor:
        device = z_data.device
        tokens = [
            self.data_projection(z_data)
            + self.type_embedding(torch.tensor(0, device=device))
            + self.step_embedding(torch.tensor(0, device=device))
        ]
        for step, item in enumerate(history):
            step_index = torch.tensor(step, device=device)
            if item.residual is not None:
                values = self.raw_projection(item.residual)
                values = values + self.type_embedding(torch.tensor(1, device=device))
                values = values + self.step_embedding(step_index)
                tokens.extend(values.unbind(0))
            if item.action is not None:
                tokens.append(
                    self.action_projection(item.action)
                    + self.type_embedding(torch.tensor(2, device=device))
                    + self.step_embedding(step_index)
                )
        current_step = torch.tensor(len(history), device=device)
        if current_raw is not None:
            values = self.raw_projection(current_raw)
            values = values + self.type_embedding(torch.tensor(1, device=device))
            values = values + self.step_embedding(current_step)
            tokens.extend(values.unbind(0))
        tokens.append(
            self.action_token
            + self.type_embedding(torch.tensor(3, device=device))
            + self.step_embedding(current_step)
        )
        sequence = torch.stack(tokens)
        if sequence.shape[0] > self.position_embedding.shape[0]:
            raise ValueError("v3.0 explicit history exceeds configured token capacity")
        sequence = sequence + self.position_embedding[: sequence.shape[0]]
        mask = torch.triu(torch.ones(
            sequence.shape[0], sequence.shape[0], dtype=torch.bool, device=device
        ), diagonal=1)
        encoded = self.transformer(sequence.unsqueeze(0), mask=mask).squeeze(0)
        return self.head(encoded[-1])


class QueryHistoryController(nn.Module):
    """Current residual tokens query explicit prior residual/action memory."""

    def __init__(
        self,
        dimension: int,
        action_dimension: int,
        max_support: int,
        max_steps: int,
    ):
        super().__init__()
        self.raw_projection = nn.Linear(6, dimension)
        self.data_projection = nn.Linear(dimension, dimension)
        self.action_projection = nn.Linear(action_dimension, dimension)
        self.type_embedding = nn.Embedding(4, dimension)
        self.step_embedding = nn.Embedding(max_steps + 2, dimension)
        maximum_memory = 1 + (max_support + 1) * max_steps
        self.memory_position = nn.Parameter(torch.zeros(maximum_memory, dimension))
        self.query_position = nn.Parameter(torch.zeros(max_support + 1, dimension))
        self.action_query = nn.Parameter(torch.zeros(dimension))
        layer = nn.TransformerDecoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=2 * dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=3, norm=nn.LayerNorm(dimension)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, action_dimension),
        )

    def forward(
        self,
        current_raw: Tensor,
        z_data: Tensor,
        history: tuple[HistoryItem, ...],
    ) -> Tensor:
        device = z_data.device
        memory = [
            self.data_projection(z_data)
            + self.type_embedding(torch.tensor(0, device=device))
            + self.step_embedding(torch.tensor(0, device=device))
        ]
        for step, item in enumerate(history):
            step_index = torch.tensor(step, device=device)
            if item.residual is not None:
                values = self.raw_projection(item.residual)
                values = values + self.type_embedding(torch.tensor(1, device=device))
                values = values + self.step_embedding(step_index)
                memory.extend(values.unbind(0))
            if item.action is not None:
                memory.append(
                    self.action_projection(item.action)
                    + self.type_embedding(torch.tensor(2, device=device))
                    + self.step_embedding(step_index)
                )
        memory_tensor = torch.stack(memory)
        if memory_tensor.shape[0] > self.memory_position.shape[0]:
            raise ValueError("v3.0 query-history memory exceeds configured capacity")
        memory_tensor = memory_tensor + self.memory_position[: memory_tensor.shape[0]]
        current_step = torch.tensor(len(history), device=device)
        query = self.raw_projection(current_raw)
        query = query + self.type_embedding(torch.tensor(1, device=device))
        query = query + self.step_embedding(current_step)
        query = torch.cat([
            query,
            (
                self.action_query
                + self.type_embedding(torch.tensor(3, device=device))
                + self.step_embedding(current_step)
            ).unsqueeze(0),
        ])
        query = query + self.query_position[: query.shape[0]]
        # Decoder residuals are the only direct current-R path; the ACTION representation is
        # produced by self-attending current evidence and cross-attending explicit history.
        encoded = self.decoder(
            query.unsqueeze(0), memory_tensor.unsqueeze(0)
        ).squeeze(0)
        return self.head(encoded[-1])


class V30Policy(nn.Module):
    def __init__(
        self,
        variant: V30Variant,
        dimension: int,
        action_dimension: int,
        max_support: int,
        max_steps: int,
    ):
        super().__init__()
        self.variant = variant
        self.data_encoder = DeepSetEncoder(2, dimension)
        if variant == "query_history":
            self.controller = QueryHistoryController(
                dimension, action_dimension, max_support, max_steps
            )
        else:
            self.controller = FullHistoryController(
                dimension, action_dimension, max_support, max_steps
            )
        self.action_dimension = action_dimension

    def encode_data(self, x: Tensor, y: Tensor) -> Tensor:
        return self.data_encoder(torch.cat([x, y], dim=-1))

    @staticmethod
    def raw_residual_tokens(x: Tensor, y: Tensor, prediction: Tensor) -> Tensor:
        return residual_features(x, y, prediction)

    def predict(
        self,
        current_raw: Tensor,
        z_data: Tensor,
        history: tuple[HistoryItem, ...],
    ) -> Tensor:
        if self.variant == "query_history":
            return self.controller(current_raw, z_data, history)
        if self.variant == "direct_current":
            return self.controller(current_raw, z_data, ())
        if self.variant == "ar_no_feedback":
            action_only = tuple(HistoryItem(None, item.action) for item in history)
            return self.controller(None, z_data, action_only)
        return self.controller(current_raw, z_data, history)


def parameter_counts(policies: dict[str, V30Policy]) -> dict[str, int]:
    return {
        name: sum(parameter.numel() for parameter in policy.parameters())
        for name, policy in policies.items()
    }
