from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from .v40_config import V40Config
from .v40_evaluation import controller_batch, make_rollout_state
from .v40_models import PopulationController
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v44_config import V44Config


ALL_HISTORY = HistoryPolicy("all_history", 0, True, True)


@torch.inference_mode()
def generate_recovery_batch(
    experiment: V44Config,
    base_config: V40Config,
    model: PopulationController,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, Tensor]:
    rollout = experiment.recovery_harvest_rollout
    harvest_capacity = base_config.bootstrap_actions + rollout
    storage = replace(base_config, history_capacity=harvest_capacity)
    state = make_rollout_state(storage, count, seed, device)
    bootstrap_before = state.history_before[:, : base_config.bootstrap_actions].cpu()
    bootstrap_action = state.history_action[:, : base_config.bootstrap_actions].cpu()
    bootstrap_after = state.history_after[:, : base_config.bootstrap_actions].cpu()
    residuals = [state.current.cpu()]
    actions = []
    was_training = model.training
    model.eval()
    for _ in range(rollout):
        before = state.current
        action = model(controller_batch(base_config, state))
        after = transition(before, state.matrices, action)
        residuals.append(after.cpu())
        actions.append(action.cpu())
        append_with_policy(
            state,
            before,
            action,
            after,
            ALL_HISTORY,
            base_config.bootstrap_actions,
            harvest_capacity,
        )
    if was_training:
        model.train()
    residual = torch.stack(residuals)
    action = torch.stack(actions)
    minima = residual.norm(dim=-1).argmin(dim=0)
    generator = torch.Generator().manual_seed(seed + 44)
    offsets = torch.randint(
        -experiment.recovery_before_minimum,
        experiment.recovery_after_minimum + 1,
        (count,),
        generator=generator,
    )
    depths = (minima.cpu() + offsets).clamp(0, rollout)
    capacity = experiment.recovery_history_capacity
    history_before = torch.zeros(count, capacity, base_config.residual_dimension)
    history_action = torch.zeros(count, capacity, base_config.action_dimension)
    history_after = torch.zeros(count, capacity, base_config.residual_dimension)
    history_before[:, : base_config.bootstrap_actions] = bootstrap_before
    history_action[:, : base_config.bootstrap_actions] = bootstrap_action
    history_after[:, : base_config.bootstrap_actions] = bootstrap_after
    current = torch.empty(count, base_config.residual_dimension)
    previous = torch.empty(count, base_config.action_dimension)
    lengths = torch.empty(count, dtype=torch.int64)
    relation = torch.empty(count, dtype=torch.int64)
    for episode in range(count):
        depth = int(depths[episode])
        if depth:
            history_before[
                episode,
                base_config.bootstrap_actions : base_config.bootstrap_actions + depth,
            ] = residual[:depth, episode]
            history_action[
                episode,
                base_config.bootstrap_actions : base_config.bootstrap_actions + depth,
            ] = action[:depth, episode]
            history_after[
                episode,
                base_config.bootstrap_actions : base_config.bootstrap_actions + depth,
            ] = residual[1 : depth + 1, episode]
            previous[episode] = action[depth - 1, episode]
        else:
            previous[episode] = bootstrap_action[episode, -1]
        current[episode] = residual[depth, episode]
        lengths[episode] = base_config.bootstrap_actions + depth
        delta = depth - int(minima[episode])
        relation[episode] = 0 if delta < 0 else 1 if delta == 0 else 2 if delta <= 3 else 3
    return {
        "matrices": state.matrices.cpu(),
        "current": current,
        "previous_action": previous,
        "history_before": history_before,
        "history_action": history_action,
        "history_after": history_after,
        "history_lengths": lengths,
        "depth": depths,
        "minimum_depth": minima.cpu(),
        "relation": relation,
        "world_seeds": state.world_seeds.cpu(),
    }


class RecoveryPopulation:
    """Evolving real recovery states; diagnostic labels never enter controller batches."""

    def __init__(
        self,
        experiment: V44Config,
        base_config: V40Config,
        model: PopulationController,
        seed: int,
        device: torch.device,
        archive: dict[str, Tensor] | None = None,
        train_indices: Tensor | None = None,
    ):
        self.experiment = experiment
        self.base_config = base_config
        self.seed = seed
        self.device = device
        self.generator = torch.Generator().manual_seed(seed + 4400)
        self.refresh_round = 0
        self.archive = archive
        self.train_indices = train_indices
        if experiment.recovery_source not in {"current_policy", "genuine_archive"}:
            raise ValueError(f"Unknown recovery source: {experiment.recovery_source}")
        if experiment.recovery_source == "genuine_archive" and (
            archive is None or train_indices is None
        ):
            raise ValueError("genuine_archive recovery requires an archive train split")
        count = experiment.recovery_pool_size
        dimension = base_config.residual_dimension
        capacity = experiment.recovery_history_capacity
        self.matrices = torch.empty(count, dimension, dimension)
        self.current = torch.empty(count, dimension)
        self.previous_action = torch.empty(count, dimension)
        self.history_before = torch.zeros(count, capacity, dimension)
        self.history_action = torch.zeros(count, capacity, dimension)
        self.history_after = torch.zeros(count, capacity, dimension)
        self.history_lengths = torch.zeros(count, dtype=torch.int64)
        self.depth = torch.zeros(count, dtype=torch.int64)
        self.minimum_depth = torch.zeros(count, dtype=torch.int64)
        self.relation = torch.zeros(count, dtype=torch.int64)
        self.world_seeds = torch.zeros(count, dtype=torch.int64)
        self.source_trajectory_id = torch.full((count,), -1, dtype=torch.int64)
        self.refresh(torch.arange(count), model)

    def refresh(self, indices: Tensor, model: PopulationController) -> None:
        indices = indices.cpu().to(torch.int64)
        if self.experiment.recovery_source == "genuine_archive":
            self._refresh_from_archive(indices)
            self.refresh_round += 1
            return
        generated = 0
        while generated < indices.numel():
            count = min(256, indices.numel() - generated)
            target = indices[generated : generated + count]
            seed = self.seed + 9000 + self.refresh_round * 10000 + generated
            batch = generate_recovery_batch(
                self.experiment,
                self.base_config,
                model,
                count,
                seed,
                self.device,
            )
            for name, value in batch.items():
                getattr(self, name)[target] = value
            generated += count
        self.refresh_round += 1

    def _refresh_from_archive(self, indices: Tensor) -> None:
        assert self.archive is not None
        assert self.train_indices is not None
        positions = torch.randint(
            self.train_indices.numel(),
            (indices.numel(),),
            generator=self.generator,
        )
        source = self.train_indices[positions]
        archive = self.archive
        capacity = self.experiment.recovery_history_capacity
        source_capacity = archive["history_before"].shape[1]
        self.history_before[indices] = 0
        self.history_action[indices] = 0
        self.history_after[indices] = 0
        for target_row, source_row in zip(indices.tolist(), source.tolist(), strict=True):
            source_length = int(archive["history_lengths"][source_row])
            retained = min(source_length, capacity)
            first = max(0, source_length - capacity)
            self.history_before[target_row, :retained] = archive["history_before"][
                source_row, first : first + retained
            ]
            self.history_action[target_row, :retained] = archive["history_action"][
                source_row, first : first + retained
            ]
            self.history_after[target_row, :retained] = archive["history_after"][
                source_row, first : first + retained
            ]
            self.history_lengths[target_row] = retained
        self.matrices[indices] = archive["matrices"][source]
        self.current[indices] = archive["current"][source]
        self.previous_action[indices] = archive["previous_action"][source]
        self.depth[indices] = archive["depth"][source]
        self.minimum_depth[indices] = archive["minimum_depth"][source]
        self.relation[indices] = archive["relation"][source]
        self.world_seeds[indices] = archive["world_seed"][source]
        self.source_trajectory_id[indices] = archive["trajectory_id"][source]
        if source_capacity < capacity:
            raise RuntimeError("Archive history capacity is smaller than recovery capacity")

    def periodic_refresh(self, model: PopulationController) -> None:
        indices = torch.randperm(
            self.experiment.recovery_pool_size, generator=self.generator
        )[: self.experiment.recovery_refresh_count]
        self.refresh(indices, model)

    def enforce_sufficiency(
        self,
        model: PopulationController,
        max_attempts: int = 10,
    ) -> dict[str, object]:
        attempts = []
        for attempt in range(max_attempts):
            result = self.sufficiency_metrics()
            attempts.append(result)
            if result["passed"]:
                return {"passed": True, "attempts": attempt + 1, "result": result}
            self.refresh(
                torch.arange(self.experiment.recovery_pool_size),
                model,
            )
        return {"passed": False, "attempts": max_attempts, "result": attempts[-1]}

    def sufficiency_metrics(self) -> dict[str, object]:
        norms = self.current.norm(dim=-1)
        rows = torch.arange(norms.numel())
        last = (self.history_lengths - 1).clamp_min(0)
        previous_norm = self.history_before[rows, last].norm(dim=-1)
        last_q = norms / previous_norm.clamp_min(self.base_config.numerical_epsilon)
        capacity = self.experiment.recovery_history_capacity
        valid = torch.arange(capacity)[None] < self.history_lengths[:, None]
        transition_q = self.history_after.norm(dim=-1) / self.history_before.norm(
            dim=-1
        ).clamp_min(self.base_config.numerical_epsilon)
        prior_overshoot = (
            (transition_q >= self.experiment.sampler_gate_overshoot_q) & valid
        ).any(dim=1)
        recovery = (
            last_q <= self.experiment.sampler_gate_recovery_q
        ) & prior_overshoot
        unique_worlds = int(self.source_trajectory_id.unique().numel())
        metrics: dict[str, object] = {
            "count": int(norms.numel()),
            "norm_median": float(norms.median()),
            "norm_q10": float(torch.quantile(norms, 0.10)),
            "norm_q90": float(torch.quantile(norms, 0.90)),
            "low_norm_fraction": float(
                (norms <= self.experiment.sampler_gate_low_norm_threshold).float().mean()
            ),
            "overshoot_fraction": float(
                (last_q >= self.experiment.sampler_gate_overshoot_q).float().mean()
            ),
            "recovery_fraction": float(recovery.float().mean()),
            "post_min_fraction": float((self.relation >= 2).float().mean()),
            "unique_source_worlds": unique_worlds,
            "history_error_max": self.actual_history_max_error(
                sample_size=self.experiment.recovery_pool_size
            ),
        }
        checks = {
            "median_is_near_convergence": metrics["norm_median"]
            <= self.experiment.sampler_gate_norm_median_max,
            "q90_is_bounded": metrics["norm_q90"]
            <= self.experiment.sampler_gate_norm_q90_max,
            "low_residual_prevalence": metrics["low_norm_fraction"]
            >= self.experiment.sampler_gate_low_fraction_min,
            "overshoot_prevalence": metrics["overshoot_fraction"]
            >= self.experiment.sampler_gate_overshoot_fraction_min,
            "recovery_prevalence": metrics["recovery_fraction"]
            >= self.experiment.sampler_gate_recovery_fraction_min,
            "post_min_prevalence": metrics["post_min_fraction"]
            >= self.experiment.sampler_gate_post_min_fraction_min,
            "world_diversity": metrics["unique_source_worlds"]
            >= self.experiment.sampler_gate_unique_worlds_min,
            "histories_are_real": metrics["history_error_max"]
            <= self.experiment.sampler_gate_history_error_max,
        }
        return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}

    def sample_indices(self, count: int) -> Tensor:
        return torch.randperm(
            self.experiment.recovery_pool_size, generator=self.generator
        )[:count]

    def append(self, indices: Tensor, before: Tensor, action: Tensor, after: Tensor) -> None:
        indices = indices.cpu()
        before = before.detach().cpu()
        action = action.detach().cpu()
        after = after.detach().cpu()
        lengths = self.history_lengths[indices]
        full = lengths >= self.experiment.recovery_history_capacity
        if full.any():
            selected = indices[full]
            self.history_before[selected, :-1] = self.history_before[selected, 1:].clone()
            self.history_action[selected, :-1] = self.history_action[selected, 1:].clone()
            self.history_after[selected, :-1] = self.history_after[selected, 1:].clone()
        positions = lengths.clamp_max(self.experiment.recovery_history_capacity - 1)
        self.history_before[indices, positions] = before
        self.history_action[indices, positions] = action
        self.history_after[indices, positions] = after
        self.history_lengths[indices] = (lengths + 1).clamp_max(
            self.experiment.recovery_history_capacity
        )
        self.current[indices] = after
        self.previous_action[indices] = action
        self.depth[indices] += 1

    def actual_history_max_error(self, sample_size: int = 128) -> float:
        indices = torch.arange(min(sample_size, self.experiment.recovery_pool_size))
        capacity = self.experiment.recovery_history_capacity
        valid = torch.arange(capacity)[None] < self.history_lengths[indices, None]
        matrices = self.matrices[indices].repeat_interleave(capacity, dim=0)
        actions = self.history_action[indices].reshape(-1, self.base_config.action_dimension, 1)
        expected = self.history_before[indices] + torch.bmm(matrices, actions).reshape(
            indices.numel(), capacity, self.base_config.residual_dimension
        )
        return float((expected - self.history_after[indices]).abs()[valid].max())

    def state_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "matrices",
                "current",
                "previous_action",
                "history_before",
                "history_action",
                "history_after",
                "history_lengths",
                "depth",
                "minimum_depth",
                "relation",
                "world_seeds",
                "source_trajectory_id",
            )
        } | {
            "generator_state": self.generator.get_state(),
            "refresh_round": self.refresh_round,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            if name in {"generator_state", "refresh_round"}:
                continue
            setattr(self, name, value)
        self.generator.set_state(state["generator_state"])
        self.refresh_round = int(state["refresh_round"])
