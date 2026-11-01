from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .v40_config import V40Config
from .v40_models import ControllerBatch, CrossAttentionBlock, normalized_residual
from .v45q_config import V45QConfig


@dataclass(frozen=True)
class CriticBatch:
    state: ControllerBatch
    candidate_action: Tensor


class ContractionCritic(nn.Module):
    """Position-free q critic with no world, depth, or policy-version input."""

    def __init__(self, base: V40Config, experiment: V45QConfig):
        super().__init__()
        self.base = base
        self.experiment = experiment
        query_features = 3 * base.residual_dimension + 1
        history_features = 4 * base.residual_dimension + 2
        self.query_projection = nn.Linear(query_features, experiment.q_model_dimension)
        self.history_projection = nn.Linear(history_features, experiment.q_model_dimension)
        self.null_history = nn.Parameter(
            torch.zeros(1, 1, experiment.q_model_dimension)
        )
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                experiment.q_model_dimension,
                experiment.q_heads,
                experiment.q_ffn,
            )
            for _ in range(experiment.q_layers)
        ])
        self.output_norm = nn.LayerNorm(experiment.q_model_dimension)
        self.log_q_head = nn.Linear(experiment.q_model_dimension, 1)
        self.harm_head = nn.Linear(experiment.q_model_dimension, 1)

    def forward(self, batch: CriticBatch) -> tuple[Tensor, Tensor]:
        direction, magnitude = normalized_residual(
            batch.state.current,
            self.base.magnitude_scale,
            self.base.numerical_epsilon,
        )
        query = self.query_projection(torch.cat([
            direction,
            magnitude,
            batch.state.previous_action,
            batch.candidate_action,
        ], dim=-1)).unsqueeze(1)
        memory = self.history_projection(batch.state.history)
        batch_size = memory.shape[0]
        memory = torch.cat(
            [self.null_history.expand(batch_size, -1, -1), memory], dim=1
        )
        positions = torch.arange(
            batch.state.history.shape[1], device=memory.device
        )[None]
        padding = positions >= batch.state.history_lengths[:, None]
        padding = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=memory.device),
            padding,
        ], dim=1)
        for block in self.blocks:
            query = block(query, memory, padding)
        representation = self.output_norm(query.squeeze(1))
        return (
            self.log_q_head(representation).squeeze(-1),
            self.harm_head(representation).squeeze(-1),
        )
