from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_models import ControllerBatch, PopulationController, transition_tokens
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v45_config import V45Config


ALL_HISTORY = HistoryPolicy("all_history", 0, True, True)


@dataclass
class RawReplayBatch:
    matrices: Tensor
    current: Tensor
    previous_action: Tensor
    history_before: Tensor
    history_action: Tensor
    history_after: Tensor
    history_lengths: Tensor
    depth: Tensor
    policy_version: Tensor
    source: Tensor


@dataclass
class TrajectoryCollection:
    matrices: Tensor
    world_seeds: Tensor
    bootstrap_before: Tensor
    bootstrap_action: Tensor
    bootstrap_after: Tensor
    residuals: Tensor
    actions: Tensor
    horizons: Tensor
    policy_version: int
    bootstrap_actions: int
    max_history: int

    def __post_init__(self) -> None:
        episodes = torch.arange(self.horizons.numel())
        self.flat_episode = torch.repeat_interleave(episodes, self.horizons + 1)
        self.flat_depth = torch.cat(
            [torch.arange(int(horizon) + 1) for horizon in self.horizons]
        )

    @property
    def state_count(self) -> int:
        return int(self.flat_episode.numel())

    def sample(
        self,
        count: int,
        generator: torch.Generator,
        source_id: int,
    ) -> RawReplayBatch:
        positions = torch.randint(self.state_count, (count,), generator=generator)
        return self.batch_from_positions(positions, source_id)

    def batch_from_positions(
        self,
        positions: Tensor,
        source_id: int,
    ) -> RawReplayBatch:
        count = int(positions.numel())
        episode = self.flat_episode[positions]
        depth = self.flat_depth[positions]
        dimension = self.current_dimension
        before = torch.zeros(count, self.max_history, dimension)
        action_history = torch.zeros(count, self.max_history, dimension)
        after = torch.zeros(count, self.max_history, dimension)
        before[:, : self.bootstrap_actions] = self.bootstrap_before[episode]
        action_history[:, : self.bootstrap_actions] = self.bootstrap_action[episode]
        after[:, : self.bootstrap_actions] = self.bootstrap_after[episode]
        previous = torch.empty(count, dimension)
        for row, (episode_index, state_depth) in enumerate(
            zip(episode.tolist(), depth.tolist(), strict=True)
        ):
            bootstrap_keep = min(self.bootstrap_actions, max(0, self.max_history - state_depth))
            learned_keep = min(state_depth, self.max_history)
            if bootstrap_keep:
                bootstrap_first = self.bootstrap_actions - bootstrap_keep
                before[row, :bootstrap_keep] = self.bootstrap_before[
                    episode_index, bootstrap_first:
                ]
                action_history[row, :bootstrap_keep] = self.bootstrap_action[
                    episode_index, bootstrap_first:
                ]
                after[row, :bootstrap_keep] = self.bootstrap_after[
                    episode_index, bootstrap_first:
                ]
            if learned_keep:
                learned_first = state_depth - learned_keep
                first = bootstrap_keep
                last = first + learned_keep
                before[row, first:last] = self.residuals[
                    episode_index, learned_first:state_depth
                ]
                action_history[row, first:last] = self.actions[
                    episode_index, learned_first:state_depth
                ]
                after[row, first:last] = self.residuals[
                    episode_index, learned_first + 1 : state_depth + 1
                ]
            if state_depth:
                previous[row] = self.actions[episode_index, state_depth - 1]
            else:
                previous[row] = self.bootstrap_action[episode_index, -1]
        return RawReplayBatch(
            matrices=self.matrices[episode],
            current=self.residuals[episode, depth],
            previous_action=previous,
            history_before=before,
            history_action=action_history,
            history_after=after,
            history_lengths=(self.bootstrap_actions + depth).clamp_max(self.max_history),
            depth=depth,
            policy_version=torch.full((count,), self.policy_version, dtype=torch.int64),
            source=torch.full((count,), source_id, dtype=torch.int64),
        )

    @property
    def current_dimension(self) -> int:
        return int(self.residuals.shape[-1])

    def history_error(self, sample_size: int = 128) -> float:
        generator = torch.Generator().manual_seed(451)
        sample = self.sample(min(sample_size, self.state_count), generator, 0)
        capacity = sample.history_before.shape[1]
        valid = torch.arange(capacity)[None] < sample.history_lengths[:, None]
        matrices = sample.matrices.repeat_interleave(capacity, dim=0)
        expected = sample.history_before + torch.bmm(
            matrices,
            sample.history_action.reshape(-1, self.current_dimension, 1),
        ).reshape(sample.history_before.shape)
        return float((expected - sample.history_after).abs()[valid].max())

    def deepest_rollout_state(
        self,
        count: int,
        device: torch.device,
    ) -> RolloutState:
        candidates = torch.where(self.horizons == self.horizons.max())[0]
        selected = candidates[: min(count, candidates.numel())]
        depth = int(self.horizons.max())
        before = torch.zeros(selected.numel(), self.max_history, self.current_dimension)
        action = torch.zeros_like(before)
        after = torch.zeros_like(before)
        before[:, : self.bootstrap_actions] = self.bootstrap_before[selected]
        action[:, : self.bootstrap_actions] = self.bootstrap_action[selected]
        after[:, : self.bootstrap_actions] = self.bootstrap_after[selected]
        first = self.bootstrap_actions
        last = first + depth
        before[:, first:last] = self.residuals[selected, :depth]
        action[:, first:last] = self.actions[selected, :depth]
        after[:, first:last] = self.residuals[selected, 1 : depth + 1]
        return RolloutState(
            matrices=self.matrices[selected].to(device),
            singular_values=torch.linalg.svdvals(self.matrices[selected]).to(device),
            world_seeds=self.world_seeds[selected].to(device),
            current=self.residuals[selected, depth].to(device),
            previous_action=self.actions[selected, depth - 1].to(device),
            history_before=before.to(device),
            history_action=action.to(device),
            history_after=after.to(device),
            history_lengths=torch.full(
                (selected.numel(),),
                self.bootstrap_actions + depth,
                dtype=torch.int64,
                device=device,
            ),
        )


@torch.inference_mode()
def collect_current_policy(
    experiment: V45Config,
    config: V40Config,
    model: PopulationController,
    policy_version: int,
    device: torch.device,
    harder: bool = False,
) -> TrajectoryCollection:
    maximum_depth = max(experiment.collection_depths)
    storage = replace(config, history_capacity=experiment.max_history)
    matrices = []
    world_seeds = []
    bootstrap_before = []
    bootstrap_action = []
    bootstrap_after = []
    residuals = []
    actions = []
    horizons = torch.tensor(
        [
            experiment.collection_depths[index % len(experiment.collection_depths)]
            for index in range(experiment.collection_worlds)
        ],
        dtype=torch.int64,
    )
    history_policy = (
        ALL_HISTORY
        if config.bootstrap_actions + maximum_depth <= experiment.max_history
        else HistoryPolicy("all_available_capped", experiment.max_history, False)
    )
    was_training = model.training
    model.eval()
    for start in range(0, experiment.collection_worlds, experiment.collection_batch_size):
        count = min(
            experiment.collection_batch_size,
            experiment.collection_worlds - start,
        )
        state = make_rollout_state(
            storage,
            count,
            experiment.seed + 4500 + policy_version * 1000 + start,
            device,
            harder=harder,
        )
        matrices.append(state.matrices.cpu())
        world_seeds.append(state.world_seeds.cpu())
        bootstrap_before.append(
            state.history_before[:, : config.bootstrap_actions].cpu()
        )
        bootstrap_action.append(
            state.history_action[:, : config.bootstrap_actions].cpu()
        )
        bootstrap_after.append(
            state.history_after[:, : config.bootstrap_actions].cpu()
        )
        chunk_residuals = [state.current.cpu()]
        chunk_actions = []
        for _ in range(maximum_depth):
            before = state.current
            action = model(controller_batch(config, state))
            after = transition(before, state.matrices, action)
            chunk_actions.append(action.cpu())
            chunk_residuals.append(after.cpu())
            append_with_policy(
                state,
                before,
                action,
                after,
                history_policy,
                config.bootstrap_actions,
                experiment.max_history,
            )
        residuals.append(torch.stack(chunk_residuals, dim=1))
        actions.append(torch.stack(chunk_actions, dim=1))
    if was_training:
        model.train()
    return TrajectoryCollection(
        matrices=torch.cat(matrices),
        world_seeds=torch.cat(world_seeds),
        bootstrap_before=torch.cat(bootstrap_before),
        bootstrap_action=torch.cat(bootstrap_action),
        bootstrap_after=torch.cat(bootstrap_after),
        residuals=torch.cat(residuals),
        actions=torch.cat(actions),
        horizons=horizons,
        policy_version=policy_version,
        bootstrap_actions=config.bootstrap_actions,
        max_history=experiment.max_history,
    )


def anchor_sample(
    population: dict[str, object],
    config: V40Config,
    experiment: V45Config,
    count: int,
    generator: torch.Generator,
) -> RawReplayBatch:
    indices = torch.randint(config.pool_size, (count,), generator=generator)
    before = torch.zeros(count, experiment.max_history, config.residual_dimension)
    action = torch.zeros(count, experiment.max_history, config.action_dimension)
    after = torch.zeros(count, experiment.max_history, config.residual_dimension)
    before[:, : config.history_capacity] = population["history_before"][indices]
    action[:, : config.history_capacity] = population["history_action"][indices]
    after[:, : config.history_capacity] = population["history_after"][indices]
    return RawReplayBatch(
        matrices=population["matrices"][indices],
        current=population["current"][indices],
        previous_action=population["previous_action"][indices],
        history_before=before,
        history_action=action,
        history_after=after,
        history_lengths=population["history_lengths"][indices],
        depth=population["ages"][indices],
        policy_version=torch.full((count,), -1, dtype=torch.int64),
        source=torch.full((count,), 2, dtype=torch.int64),
    )


def concatenate_raw(batches: list[RawReplayBatch]) -> RawReplayBatch:
    return RawReplayBatch(**{
        name: torch.cat([getattr(batch, name) for batch in batches])
        for name in RawReplayBatch.__dataclass_fields__
    })


def mixed_replay_batch(
    experiment: V45Config,
    config: V40Config,
    latest: TrajectoryCollection,
    older: list[TrajectoryCollection],
    anchor_population: dict[str, object],
    generator: torch.Generator,
    device: torch.device,
) -> tuple[ControllerBatch, Tensor, dict[str, Tensor]]:
    batches = [latest.sample(experiment.latest_batch, generator, 0)]
    old_source = older if older else [latest]
    assignments = torch.randint(
        len(old_source), (experiment.older_batch,), generator=generator
    )
    old_batches = []
    for collection_index in assignments.unique(sorted=True).tolist():
        count = int((assignments == collection_index).sum())
        old_batches.append(old_source[collection_index].sample(count, generator, 1))
    batches.append(concatenate_raw(old_batches))
    batches.append(
        anchor_sample(
            anchor_population,
            config,
            experiment,
            experiment.anchor_batch,
            generator,
        )
    )
    raw = concatenate_raw(batches)
    permutation = torch.randperm(experiment.batch_size, generator=generator)
    controller = ControllerBatch(
        current=raw.current[permutation].to(device).detach(),
        previous_action=raw.previous_action[permutation].to(device).detach(),
        history=transition_tokens(
            raw.history_before[permutation].to(device).detach(),
            raw.history_action[permutation].to(device).detach(),
            raw.history_after[permutation].to(device).detach(),
            config.magnitude_scale,
            config.numerical_epsilon,
        ).detach(),
        history_lengths=raw.history_lengths[permutation].to(device).detach(),
    )
    metadata = {
        "source": raw.source[permutation],
        "depth": raw.depth[permutation],
        "policy_version": raw.policy_version[permutation],
        "history_lengths": raw.history_lengths[permutation],
    }
    return controller, raw.matrices[permutation].to(device).detach(), metadata
