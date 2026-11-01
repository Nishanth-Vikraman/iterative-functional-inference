from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .v40_config import V40Config
from .v40_models import ControllerBatch, transition_tokens
from .v40_worlds import (
    sample_initial_residuals,
    sample_probe_actions,
    sample_well_conditioned_worlds,
    transition,
)


@dataclass(frozen=True)
class PopulationUpdate:
    convergence_resets: int
    divergence_resets: int
    age_resets: int
    random_resets: int

    @property
    def total_resets(self) -> int:
        return (
            self.convergence_resets
            + self.divergence_resets
            + self.age_resets
            + self.random_resets
        )


class ReplayPopulation:
    """CPU population whose histories contain only executed world transitions."""

    def __init__(self, config: V40Config, arm: str, seed: int):
        self.config = config
        self.arm = arm
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.world_counter = seed * 1_000_000
        self.registry: list[dict[str, object]] = []
        size = config.pool_size
        dimension = config.residual_dimension
        capacity = config.history_capacity
        self.matrices = torch.empty(size, dimension, dimension)
        self.singular_values = torch.empty(size, dimension)
        self.current = torch.empty(size, dimension)
        self.previous_action = torch.empty(size, config.action_dimension)
        self.history_before = torch.zeros(size, capacity, dimension)
        self.history_action = torch.zeros(size, capacity, config.action_dimension)
        self.history_after = torch.zeros(size, capacity, dimension)
        self.history_lengths = torch.zeros(size, dtype=torch.int64)
        self.ages = torch.zeros(size, dtype=torch.int64)
        self.world_seeds = torch.zeros(size, dtype=torch.int64)
        self.policy_versions = torch.zeros(size, dtype=torch.int64)
        self._reset_indices(torch.arange(size), step=0)

    def _sample_worlds(self, count: int, step: int):
        batch = sample_well_conditioned_worlds(
            count,
            self.config.residual_dimension,
            self.config.singular_min,
            self.config.singular_max,
            self.generator,
            self.world_counter,
        )
        sampled_residual = sample_initial_residuals(
            count,
            self.config.residual_dimension,
            self.generator,
            self.config.training_residual_norm_min,
            self.config.training_residual_norm_max,
        )
        probes = sample_probe_actions(
            count,
            self.config.bootstrap_actions,
            self.config.action_dimension,
            self.config.bootstrap_action_norm,
            self.generator,
            self.config.bootstrap_orthogonal,
        )
        if self.config.training_residual_is_post_bootstrap:
            total_probe_effect = torch.bmm(
                batch.matrices,
                probes.sum(dim=1).unsqueeze(-1),
            ).squeeze(-1)
            initial = sampled_residual - total_probe_effect
        else:
            initial = sampled_residual
        residual = initial
        before = []
        after = []
        for probe_index in range(self.config.bootstrap_actions):
            before.append(residual)
            residual = transition(residual, batch.matrices, probes[:, probe_index])
            after.append(residual)
        for index in range(count):
            row: dict[str, object] = {
                "arm": self.arm,
                "world_seed": int(batch.seeds[index]),
                "created_at_optimizer_step": step,
            }
            for singular_index, value in enumerate(batch.singular_values[index]):
                row[f"singular_{singular_index}"] = float(value)
            for matrix_index, value in enumerate(batch.matrices[index].flatten()):
                row[f"B_{matrix_index}"] = float(value)
            self.registry.append(row)
        self.world_counter += count
        return batch, residual, probes, torch.stack(before, 1), torch.stack(after, 1)

    def _reset_indices(self, indices: Tensor, step: int) -> None:
        indices = indices.to(dtype=torch.int64, device="cpu")
        if indices.numel() == 0:
            return
        worlds, current, probes, before, after = self._sample_worlds(indices.numel(), step)
        self.matrices[indices] = worlds.matrices
        self.singular_values[indices] = worlds.singular_values
        self.current[indices] = current
        self.previous_action[indices] = probes[:, -1]
        self.history_before[indices] = 0
        self.history_action[indices] = 0
        self.history_after[indices] = 0
        self.history_before[indices, : self.config.bootstrap_actions] = before
        self.history_action[indices, : self.config.bootstrap_actions] = probes
        self.history_after[indices, : self.config.bootstrap_actions] = after
        self.history_lengths[indices] = self.config.bootstrap_actions
        self.ages[indices] = 0
        self.world_seeds[indices] = worlds.seeds
        self.policy_versions[indices] = step

    def magnitude_bin_ids(self, residual: Tensor | None = None) -> Tensor:
        values = self.current if residual is None else residual
        log_norm = values.norm(dim=-1).clamp_min(1e-12).log10()
        inner_edges = torch.tensor(
            self.config.magnitude_bin_log10_edges[1:-1], dtype=log_norm.dtype
        )
        return torch.bucketize(log_norm.cpu(), inner_edges)

    def magnitude_bin_occupancy(self) -> Tensor:
        bins = self.magnitude_bin_ids()
        return torch.bincount(
            bins, minlength=len(self.config.magnitude_bin_log10_edges) - 1
        )

    def sample_indices(self, batch_size: int) -> Tensor:
        if self.config.stratified_magnitude_sampling:
            bin_ids = self.magnitude_bin_ids()
            bin_count = len(self.config.magnitude_bin_log10_edges) - 1
            base = batch_size // bin_count
            remainder = batch_size % bin_count
            selected = []
            selected_mask = torch.zeros(self.config.pool_size, dtype=torch.bool)
            for bin_index in range(bin_count):
                quota = base + int(bin_index < remainder)
                candidates = torch.where(bin_ids == bin_index)[0]
                if candidates.numel():
                    order = torch.randperm(candidates.numel(), generator=self.generator)
                    chosen = candidates[order[:quota]]
                    selected.append(chosen)
                    selected_mask[chosen] = True
            chosen_count = sum(item.numel() for item in selected)
            if chosen_count < batch_size:
                remaining = torch.where(~selected_mask)[0]
                order = torch.randperm(remaining.numel(), generator=self.generator)
                selected.append(remaining[order[: batch_size - chosen_count]])
            combined = torch.cat(selected)
            order = torch.randperm(combined.numel(), generator=self.generator)
            return combined[order]
        return torch.randperm(self.config.pool_size, generator=self.generator)[:batch_size]

    def controller_batch(self, indices: Tensor, device: torch.device) -> ControllerBatch:
        before = self.history_before[indices].to(device, non_blocking=True).detach()
        action = self.history_action[indices].to(device, non_blocking=True).detach()
        after = self.history_after[indices].to(device, non_blocking=True).detach()
        history = transition_tokens(
            before,
            action,
            after,
            self.config.magnitude_scale,
            self.config.numerical_epsilon,
        ).detach()
        return ControllerBatch(
            current=self.current[indices].to(device, non_blocking=True).detach(),
            previous_action=self.previous_action[indices]
            .to(device, non_blocking=True)
            .detach(),
            history=history,
            history_lengths=self.history_lengths[indices]
            .to(device, non_blocking=True)
            .detach(),
        )

    def matrix_batch(self, indices: Tensor, device: torch.device) -> Tensor:
        return self.matrices[indices].to(device, non_blocking=True).detach()

    def _append(self, indices: Tensor, before: Tensor, action: Tensor, after: Tensor) -> None:
        lengths = self.history_lengths[indices]
        full = lengths >= self.config.history_capacity
        if full.any():
            full_indices = indices[full]
            self.history_before[full_indices, :-1] = self.history_before[
                full_indices, 1:
            ].clone()
            self.history_action[full_indices, :-1] = self.history_action[
                full_indices, 1:
            ].clone()
            self.history_after[full_indices, :-1] = self.history_after[
                full_indices, 1:
            ].clone()
        positions = torch.minimum(
            lengths, torch.tensor(self.config.history_capacity - 1, dtype=torch.int64)
        )
        self.history_before[indices, positions] = before
        self.history_action[indices, positions] = action
        self.history_after[indices, positions] = after
        self.history_lengths[indices] = torch.minimum(
            lengths + 1, torch.tensor(self.config.history_capacity, dtype=torch.int64)
        )

    def apply(
        self,
        indices: Tensor,
        executed_action: Tensor,
        next_residual: Tensor,
        step: int,
    ) -> PopulationUpdate:
        indices = indices.to(device="cpu", dtype=torch.int64)
        action = executed_action.detach().cpu()
        after = next_residual.detach().cpu()
        before = self.current[indices].clone()
        self._append(indices, before, action, after)
        self.current[indices] = after
        self.previous_action[indices] = action
        self.ages[indices] += 1
        self.policy_versions[indices] = step
        norm = after.norm(dim=-1)
        convergence = norm < self.config.convergence_threshold
        divergence = norm > self.config.divergence_threshold
        age = self.ages[indices] >= self.config.max_age
        random_draw = torch.rand(indices.numel(), generator=self.generator)
        random_reset = random_draw < self.config.random_reset_probability
        random_reset &= ~(convergence | divergence | age)
        age &= ~(convergence | divergence)
        divergence &= ~convergence
        reset = convergence | divergence | age | random_reset
        self._reset_indices(indices[reset], step)
        return PopulationUpdate(
            int(convergence.sum()),
            int(divergence.sum()),
            int(age.sum()),
            int(random_reset.sum()),
        )

    def diagnostics(self, step: int) -> dict[str, float]:
        norms = self.current.norm(dim=-1)
        age_float = self.ages.float()
        norm_centered = norms - norms.mean()
        age_centered = age_float - age_float.mean()
        correlation = (norm_centered * age_centered).mean() / (
            norm_centered.std(unbiased=False) * age_centered.std(unbiased=False) + 1e-12
        )
        return {
            "step": float(step),
            "arm": self.arm,
            "age_mean": float(age_float.mean()),
            "age_std": float(age_float.std(unbiased=False)),
            "age_min": float(age_float.min()),
            "age_median": float(age_float.median()),
            "age_q90": float(torch.quantile(age_float, 0.9)),
            "age_max": float(age_float.max()),
            "history_length_mean": float(self.history_lengths.float().mean()),
            "residual_norm_mean": float(norms.mean()),
            "residual_norm_median": float(norms.median()),
            "residual_norm_q90": float(torch.quantile(norms, 0.9)),
            "residual_norm_age_correlation": float(correlation),
            "policy_staleness_mean": float((step - self.policy_versions).float().mean()),
        }

    def actual_history_max_error(self, sample_size: int = 128) -> float:
        indices = torch.arange(min(sample_size, self.config.pool_size))
        lengths = self.history_lengths[indices]
        positions = torch.arange(self.config.history_capacity)[None]
        valid = positions < lengths[:, None]
        expected = self.history_before[indices] + torch.bmm(
            self.matrices[indices].repeat_interleave(self.config.history_capacity, 0),
            self.history_action[indices].reshape(-1, self.config.action_dimension, 1),
        ).reshape(indices.numel(), self.config.history_capacity, self.config.residual_dimension)
        error = (expected - self.history_after[indices]).abs()
        return float(error[valid].max())

    def state_dict(self) -> dict[str, object]:
        names = (
            "matrices",
            "singular_values",
            "current",
            "previous_action",
            "history_before",
            "history_action",
            "history_after",
            "history_lengths",
            "ages",
            "world_seeds",
            "policy_versions",
        )
        return {
            **{name: getattr(self, name) for name in names},
            "generator_state": self.generator.get_state(),
            "world_counter": self.world_counter,
            "registry": self.registry,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            if name in {"generator_state", "world_counter", "registry"}:
                continue
            setattr(self, name, value)
        self.generator.set_state(state["generator_state"])
        self.world_counter = int(state["world_counter"])
        self.registry = list(state["registry"])
