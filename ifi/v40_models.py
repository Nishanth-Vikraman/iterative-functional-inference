from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .v40_config import V40Config


def normalized_residual(
    residual: Tensor,
    magnitude_scale: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    norm = residual.norm(dim=-1, keepdim=True)
    direction = residual / (norm + epsilon)
    magnitude = norm / (norm + magnitude_scale)
    return direction, magnitude


def transition_tokens(
    before: Tensor,
    action: Tensor,
    after: Tensor,
    magnitude_scale: float,
    epsilon: float,
) -> Tensor:
    before_direction, before_magnitude = normalized_residual(
        before, magnitude_scale, epsilon
    )
    after_direction, after_magnitude = normalized_residual(after, magnitude_scale, epsilon)
    delta = after - before
    delta_direction = delta / (delta.norm(dim=-1, keepdim=True) + epsilon)
    return torch.cat(
        [
            before_direction,
            before_magnitude,
            action,
            after_direction,
            after_magnitude,
            delta_direction,
        ],
        dim=-1,
    )


class CrossAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn_dimension: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dimension),
            nn.GELU(),
            nn.Linear(ffn_dimension, dimension),
        )

    def forward(self, query: Tensor, memory: Tensor, padding_mask: Tensor) -> Tensor:
        normalized_query = self.query_norm(query)
        attended, _ = self.attention(
            normalized_query,
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class ResidualFFNBlock(nn.Module):
    def __init__(self, dimension: int, ffn_dimension: int):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dimension),
            nn.GELU(),
            nn.Linear(ffn_dimension, dimension),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.ffn(self.norm(value))


@dataclass(frozen=True)
class ControllerBatch:
    current: Tensor
    previous_action: Tensor
    history: Tensor
    history_lengths: Tensor


class PopulationController(nn.Module):
    """Controller with no positional, age, task, seed, or world-geometry input."""

    def __init__(self, kind: str, config: V40Config):
        super().__init__()
        if kind not in {"query_history", "current_only", "ar_no_fresh"}:
            raise ValueError(f"Unknown v4.0 controller: {kind}")
        self.kind = kind
        self.config = config
        residual_features = config.residual_dimension + 1 + config.action_dimension
        history_features = 4 * config.residual_dimension + 2
        if kind == "query_history":
            self.query_projection = nn.Linear(residual_features, config.model_dimension)
        elif kind == "ar_no_fresh":
            self.query_projection = nn.Linear(config.action_dimension, config.model_dimension)
        else:
            self.query_projection = nn.Linear(residual_features, config.model_dimension)
        self.history_projection = (
            nn.Linear(history_features, config.model_dimension)
            if kind != "current_only"
            else None
        )
        self.null_history = (
            nn.Parameter(torch.zeros(1, 1, config.model_dimension))
            if kind != "current_only"
            else None
        )
        if kind == "current_only":
            self.blocks = nn.ModuleList(
                [
                    ResidualFFNBlock(config.model_dimension, config.ffn_dimension)
                    for _ in range(4)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        config.model_dimension,
                        config.attention_heads,
                        config.ffn_dimension,
                    )
                    for _ in range(config.attention_layers)
                ]
            )
        self.output_norm = nn.LayerNorm(config.model_dimension)
        self.action_head = nn.Linear(config.model_dimension, config.action_dimension)

    def forward(self, batch: ControllerBatch) -> Tensor:
        current_direction, current_magnitude = normalized_residual(
            batch.current,
            self.config.magnitude_scale,
            self.config.numerical_epsilon,
        )
        if self.kind == "ar_no_fresh":
            query_input = batch.previous_action
        else:
            query_input = torch.cat(
                [current_direction, current_magnitude, batch.previous_action], dim=-1
            )
        query = self.query_projection(query_input).unsqueeze(1)
        if self.kind == "current_only":
            for block in self.blocks:
                query = block(query)
        else:
            memory = self.history_projection(batch.history)
            batch_size = memory.shape[0]
            null = self.null_history.expand(batch_size, -1, -1)
            memory = torch.cat([null, memory], dim=1)
            positions = torch.arange(batch.history.shape[1], device=memory.device)[None]
            padding = positions >= batch.history_lengths[:, None]
            padding = torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=memory.device), padding],
                dim=1,
            )
            for block in self.blocks:
                query = block(query, memory, padding)
        raw_action = self.action_head(self.output_norm(query.squeeze(1)))
        return self.config.action_max * torch.tanh(raw_action)


def parameter_counts(models: dict[str, nn.Module]) -> dict[str, int]:
    return {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
