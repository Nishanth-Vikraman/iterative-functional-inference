from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .v40_config import V40Config
from .v40_models import ControllerBatch, CrossAttentionBlock, normalized_residual
from .v46w_config import V46WConfig


@dataclass(frozen=True)
class WorldModelBatch:
    state: ControllerBatch
    candidate_action: Tensor


class ResidualWorldModel(nn.Module):
    """Position-free observable-history model for (R, H, u) -> delta R."""

    def __init__(self, base: V40Config, experiment: V46WConfig):
        super().__init__()
        self.base = base
        self.experiment = experiment
        query_features = 2 * base.residual_dimension + 1
        history_features = 4 * base.residual_dimension + 2
        self.query_projection = nn.Linear(query_features, experiment.model_dimension)
        self.history_projection = nn.Linear(history_features, experiment.model_dimension)
        self.null_history = nn.Parameter(torch.zeros(1, 1, experiment.model_dimension))
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                experiment.model_dimension,
                experiment.heads,
                experiment.ffn_dimension,
            )
            for _ in range(experiment.layers)
        ])
        self.output_norm = nn.LayerNorm(experiment.model_dimension)
        self.delta_head = nn.Linear(experiment.model_dimension, base.residual_dimension)

    def forward(self, batch: WorldModelBatch) -> Tensor:
        direction, magnitude = normalized_residual(
            batch.state.current,
            self.base.magnitude_scale,
            self.base.numerical_epsilon,
        )
        query = self.query_projection(torch.cat([
            direction,
            magnitude,
            batch.candidate_action,
        ], dim=-1)).unsqueeze(1)
        memory = self.history_projection(batch.state.history)
        batch_size = memory.shape[0]
        memory = torch.cat(
            [self.null_history.expand(batch_size, -1, -1), memory], dim=1
        )
        positions = torch.arange(batch.state.history.shape[1], device=memory.device)[None]
        padding = positions >= batch.state.history_lengths[:, None]
        padding = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=memory.device),
            padding,
        ], dim=1)
        for block in self.blocks:
            query = block(query, memory, padding)
        return self.delta_head(self.output_norm(query.squeeze(1)))
