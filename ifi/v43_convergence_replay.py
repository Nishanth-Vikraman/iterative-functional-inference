from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_models import ControllerBatch, PopulationController, transition_tokens
from .v40_training import build_models
from .v40_worlds import analytic_action, transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v43_config import V43Config


ALL_HISTORY = HistoryPolicy("all_history", 0, True, True)
FIFO_16 = HistoryPolicy("fifo_16", 16, False)
RELATIONS = {0: "pre_min", 1: "minimum", 2: "mild_post_min", 3: "later_post_min"}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def clone_model(config: V40Config, state: dict[str, Tensor], device: torch.device) -> PopulationController:
    model = PopulationController("query_history", config).to(device)
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def harvest_convergence_pool(
    experiment: V43Config,
    base_config: V40Config,
    frozen_model: PopulationController,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[dict[str, object]]]:
    maximum_history = base_config.bootstrap_actions + experiment.harvest_rollout
    storage_config = replace(base_config, history_capacity=maximum_history)
    tensors: dict[str, list[Tensor]] = defaultdict(list)
    trajectory_rows: list[dict[str, object]] = []
    generated = 0
    while generated < experiment.harvest_worlds:
        count = min(experiment.harvest_batch_size, experiment.harvest_worlds - generated)
        state = make_rollout_state(
            storage_config,
            count,
            base_config.seed + 800 + generated,
            device,
        )
        bootstrap_before = state.history_before[:, : base_config.bootstrap_actions].cpu()
        bootstrap_action = state.history_action[:, : base_config.bootstrap_actions].cpu()
        bootstrap_after = state.history_after[:, : base_config.bootstrap_actions].cpu()
        residuals = [state.current.cpu()]
        actions: list[Tensor] = []
        for _ in range(experiment.harvest_rollout):
            before = state.current
            action = frozen_model(controller_batch(base_config, state))
            after = transition(before, state.matrices, action)
            actions.append(action.cpu())
            residuals.append(after.cpu())
            append_with_policy(
                state,
                before,
                action,
                after,
                ALL_HISTORY,
                base_config.bootstrap_actions,
                maximum_history,
            )
        residual_trajectory = torch.stack(residuals)
        action_trajectory = torch.stack(actions)
        norms = residual_trajectory.norm(dim=-1)
        minima = norms.argmin(dim=0)
        matrices = state.matrices.cpu()
        world_seeds = state.world_seeds.cpu()
        for episode in range(count):
            minimum = int(minima[episode])
            first = max(0, minimum - experiment.harvest_before_minimum)
            last = min(experiment.harvest_rollout, minimum + experiment.harvest_after_minimum)
            depths = list(range(first, last + 1))
            number = len(depths)
            before_history = torch.zeros(number, maximum_history, base_config.residual_dimension)
            action_history = torch.zeros(number, maximum_history, base_config.action_dimension)
            after_history = torch.zeros(number, maximum_history, base_config.residual_dimension)
            before_history[:, : base_config.bootstrap_actions] = bootstrap_before[episode]
            action_history[:, : base_config.bootstrap_actions] = bootstrap_action[episode]
            after_history[:, : base_config.bootstrap_actions] = bootstrap_after[episode]
            previous = []
            current = []
            lengths = []
            relation = []
            for row, depth in enumerate(depths):
                if depth:
                    before_history[row, base_config.bootstrap_actions : base_config.bootstrap_actions + depth] = residual_trajectory[:depth, episode]
                    action_history[row, base_config.bootstrap_actions : base_config.bootstrap_actions + depth] = action_trajectory[:depth, episode]
                    after_history[row, base_config.bootstrap_actions : base_config.bootstrap_actions + depth] = residual_trajectory[1 : depth + 1, episode]
                    previous.append(action_trajectory[depth - 1, episode])
                else:
                    previous.append(bootstrap_action[episode, -1])
                current.append(residual_trajectory[depth, episode])
                lengths.append(base_config.bootstrap_actions + depth)
                offset = depth - minimum
                relation.append(0 if offset < 0 else 1 if offset == 0 else 2 if offset <= 3 else 3)
            global_trajectory = generated + episode
            tensors["matrices"].append(matrices[episode].expand(number, -1, -1).clone())
            tensors["current"].append(torch.stack(current))
            tensors["previous_action"].append(torch.stack(previous))
            tensors["history_before"].append(before_history)
            tensors["history_action"].append(action_history)
            tensors["history_after"].append(after_history)
            tensors["history_lengths"].append(torch.tensor(lengths, dtype=torch.int64))
            tensors["trajectory_id"].append(torch.full((number,), global_trajectory, dtype=torch.int64))
            tensors["world_seed"].append(torch.full((number,), int(world_seeds[episode]), dtype=torch.int64))
            tensors["depth"].append(torch.tensor(depths, dtype=torch.int64))
            tensors["minimum_depth"].append(torch.full((number,), minimum, dtype=torch.int64))
            tensors["relation"].append(torch.tensor(relation, dtype=torch.int64))
            trajectory_rows.append({
                "trajectory_id": global_trajectory,
                "world_seed": int(world_seeds[episode]),
                "minimum_depth": minimum,
                "minimum_residual_norm": float(norms[minimum, episode]),
                "initial_residual_norm": float(norms[0, episode]),
                "harvested_states": number,
                "first_harvest_depth": first,
                "last_harvest_depth": last,
            })
        generated += count
    archive = {name: torch.cat(values) for name, values in tensors.items()}
    return archive, trajectory_rows


def archive_history_error(archive: dict[str, Tensor], sample_size: int = 256) -> float:
    count = min(sample_size, archive["current"].shape[0])
    indices = torch.arange(count)
    capacity = archive["history_before"].shape[1]
    valid = torch.arange(capacity)[None] < archive["history_lengths"][indices, None]
    matrices = archive["matrices"][indices].repeat_interleave(capacity, dim=0)
    actions = archive["history_action"][indices].reshape(-1, matrices.shape[-1], 1)
    expected = archive["history_before"][indices] + torch.bmm(matrices, actions).reshape(
        count, capacity, -1
    )
    error = (expected - archive["history_after"][indices]).abs()
    return float(error[valid].max())


class MixedReplay:
    def __init__(
        self,
        experiment: V43Config,
        base_config: V40Config,
        archive: dict[str, Tensor],
        ordinary: dict[str, object],
    ):
        self.experiment = experiment
        self.base_config = base_config
        self.archive = archive
        self.ordinary = ordinary
        cutoff = math.floor(experiment.harvest_worlds * experiment.harvest_train_fraction)
        self.train_indices = torch.where(archive["trajectory_id"] < cutoff)[0]
        self.validation_indices = torch.where(archive["trajectory_id"] >= cutoff)[0]
        self.generator = torch.Generator().manual_seed(experiment.seed + 4300)
        self.convergence_batch = round(experiment.batch_size * experiment.convergence_fraction)
        self.ordinary_batch = experiment.batch_size - self.convergence_batch
        self.capacity = archive["history_before"].shape[1]

    def _sample_indices(self, available: Tensor, count: int) -> Tensor:
        positions = torch.randint(available.numel(), (count,), generator=self.generator)
        return available[positions]

    def sample(self, device: torch.device) -> tuple[ControllerBatch, Tensor, Tensor, dict[str, Tensor]]:
        convergence_indices = self._sample_indices(self.train_indices, self.convergence_batch)
        ordinary_indices = torch.randint(
            self.base_config.pool_size,
            (self.ordinary_batch,),
            generator=self.generator,
        )
        histories = {}
        for name in ("history_before", "history_action", "history_after"):
            convergence_value = self.archive[name][convergence_indices]
            ordinary_value = self.ordinary[name][ordinary_indices]
            padded = torch.zeros(
                self.ordinary_batch,
                self.capacity,
                ordinary_value.shape[-1],
            )
            padded[:, : self.base_config.history_capacity] = ordinary_value
            histories[name] = torch.cat((convergence_value, padded))
        current = torch.cat((
            self.archive["current"][convergence_indices],
            self.ordinary["current"][ordinary_indices],
        ))
        previous = torch.cat((
            self.archive["previous_action"][convergence_indices],
            self.ordinary["previous_action"][ordinary_indices],
        ))
        lengths = torch.cat((
            self.archive["history_lengths"][convergence_indices],
            self.ordinary["history_lengths"][ordinary_indices],
        ))
        matrices = torch.cat((
            self.archive["matrices"][convergence_indices],
            self.ordinary["matrices"][ordinary_indices],
        ))
        source = torch.cat((
            torch.ones(self.convergence_batch, dtype=torch.bool),
            torch.zeros(self.ordinary_batch, dtype=torch.bool),
        ))
        permutation = torch.randperm(self.experiment.batch_size, generator=self.generator)
        current = current[permutation].to(device).detach()
        previous = previous[permutation].to(device).detach()
        lengths = lengths[permutation].to(device).detach()
        matrices = matrices[permutation].to(device).detach()
        source = source[permutation].to(device)
        for name in histories:
            histories[name] = histories[name][permutation].to(device).detach()
        batch = ControllerBatch(
            current=current,
            previous_action=previous,
            history=transition_tokens(
                histories["history_before"],
                histories["history_action"],
                histories["history_after"],
                self.base_config.magnitude_scale,
                self.base_config.numerical_epsilon,
            ).detach(),
            history_lengths=lengths,
        )
        logging = {
            "depth": torch.cat((
                self.archive["depth"][convergence_indices],
                self.ordinary["ages"][ordinary_indices],
            ))[permutation],
            "relation": torch.cat((
                self.archive["relation"][convergence_indices],
                torch.full((self.ordinary_batch,), -1, dtype=torch.int64),
            ))[permutation],
        }
        return batch, matrices, source, logging

    def validation_batch(self, count: int, device: torch.device) -> tuple[ControllerBatch, Tensor]:
        indices = self.validation_indices[: min(count, self.validation_indices.numel())]
        batch = ControllerBatch(
            current=self.archive["current"][indices].to(device),
            previous_action=self.archive["previous_action"][indices].to(device),
            history=transition_tokens(
                self.archive["history_before"][indices].to(device),
                self.archive["history_action"][indices].to(device),
                self.archive["history_after"][indices].to(device),
                self.base_config.magnitude_scale,
                self.base_config.numerical_epsilon,
            ),
            history_lengths=self.archive["history_lengths"][indices].to(device),
        )
        return batch, self.archive["matrices"][indices].to(device)


def gradient_norm(parameters: list[nn.Parameter]) -> float:
    values = [parameter.grad.square().sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


@torch.inference_mode()
def short_rollout_endpoint(
    model: PopulationController,
    config: V40Config,
    device: torch.device,
    episodes: int = 192,
    horizon: int = 8,
) -> float:
    storage = replace(config, history_capacity=config.bootstrap_actions + horizon)
    state = make_rollout_state(storage, episodes, config.seed + 600, device)
    for _ in range(horizon):
        before = state.current
        action = model(controller_batch(config, state))
        after = transition(before, state.matrices, action)
        append_with_policy(
            state,
            before,
            action,
            after,
            ALL_HISTORY,
            config.bootstrap_actions,
            storage.history_capacity,
        )
    return float(state.current.norm(dim=-1).median())


@torch.inference_mode()
def convergence_validation(
    model: PopulationController,
    replay: MixedReplay,
    device: torch.device,
) -> dict[str, float]:
    batch, matrices = replay.validation_batch(2048, device)
    action = model(batch)
    after = transition(batch.current, matrices, action)
    before_norm = batch.current.norm(dim=-1)
    q = after.norm(dim=-1) / before_norm.clamp_min(1e-12)
    return {
        "median_q": float(q.median()),
        "regression_fraction": float((q > 1).float().mean()),
        "median_action_norm": float(action.norm(dim=-1).median()),
        "median_residual_norm": float(before_norm.median()),
        "median_action_residual_ratio": float(
            (action.norm(dim=-1) / before_norm.clamp_min(1e-12)).median()
        ),
    }


class PostTrainer:
    def __init__(
        self,
        experiment: V43Config,
        base_config: V40Config,
        model: PopulationController,
        original: PopulationController,
        replay: MixedReplay,
        device: torch.device,
        output: Path,
    ):
        self.experiment = experiment
        self.base_config = base_config
        self.model = model
        self.original = original
        self.replay = replay
        self.device = device
        self.output = output
        self.optimizer = torch.optim.Adam(model.parameters(), lr=experiment.learning_rate)
        self.generator = torch.Generator(device=device).manual_seed(experiment.seed + 4301)
        self.rows: list[dict[str, object]] = []
        self.started = time.time()
        self.gate: dict[str, object] | None = None

    def step(self, step: int) -> list[dict[str, object]]:
        batch, matrices, source, logging = self.replay.sample(self.device)
        action = self.model(batch)
        exploration = self.base_config.exploration_std * torch.randn(
            action.shape, generator=self.generator, device=self.device
        )
        executed = (action + exploration).clamp(
            -self.base_config.action_max, self.base_config.action_max
        )
        after = transition(batch.current, matrices, executed)
        before_norm = batch.current.norm(dim=-1)
        after_norm = after.norm(dim=-1)
        q = after_norm / before_norm.clamp_min(self.base_config.numerical_epsilon)
        base_loss = F.softplus(q - self.base_config.progress_tau)
        safe_loss = F.relu(after.square().sum(-1) - batch.current.square().sum(-1))
        per_example = base_loss + self.experiment.lambda_safe * safe_loss
        loss = per_example.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite v4.3 loss at step {step}")
        self.optimizer.zero_grad(set_to_none=True)
        source_gradient = {}
        for name, mask in (("convergence", source), ("ordinary", ~source)):
            gradient = torch.autograd.grad(per_example[mask].mean(), action, retain_graph=True)[0]
            source_gradient[name] = float(gradient.norm())
        loss.backward()
        parameters = list(self.model.parameters())
        total_gradient = gradient_norm(parameters)
        if not math.isfinite(total_gradient):
            raise RuntimeError(f"Non-finite v4.3 gradient at step {step}")
        self.optimizer.step()
        output = []
        for name, mask in (("convergence", source), ("ordinary", ~source)):
            ratios = action.detach().norm(dim=-1) / before_norm.clamp_min(1e-12)
            output.append({
                "step": step,
                "source": name,
                "loss": float(per_example.detach()[mask].mean()),
                "v41_loss": float(base_loss.detach()[mask].mean()),
                "safe_loss": float(safe_loss.detach()[mask].mean()),
                "q_mean": float(q.detach()[mask].mean()),
                "q_median": float(q.detach()[mask].median()),
                "residual_norm_median": float(before_norm.detach()[mask].median()),
                "action_norm_median": float(action.detach().norm(dim=-1)[mask].median()),
                "action_residual_ratio_median": float(ratios[mask].median()),
                "gradient_norm": total_gradient,
                "source_action_gradient_norm": source_gradient[name],
                "gradient_clipping_fraction": 0.0,
                "contraction_fraction": float((q.detach()[mask] < 1).float().mean()),
                "regression_fraction": float((q.detach()[mask] > 1).float().mean()),
                "trajectory_depth_median_logging_only": float(logging["depth"][mask.cpu()].float().median()),
            })
        return output

    def save(self, step: int, complete: bool, stopped_by_gate: bool = False) -> None:
        payload = {
            "experiment": "IFI v4.3 convergence replay post-training",
            "step": step,
            "complete": complete,
            "stopped_by_gate": stopped_by_gate,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gate": self.gate,
            "base_checkpoint_sha256": sha256(Path(self.experiment.base_checkpoint)),
        }
        path = self.output / ("checkpoint_final.pt" if complete else "checkpoint_progress.pt")
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_csv(self.output / "train_metrics.csv", self.rows)

    def smoke_gate(self, step: int) -> dict[str, object]:
        self.model.eval()
        post_f8 = short_rollout_endpoint(self.model, self.base_config, self.device)
        original_f8 = short_rollout_endpoint(self.original, self.base_config, self.device)
        convergence = convergence_validation(self.model, self.replay, self.device)
        self.model.train()
        checks = {
            "early_v41_behavior_preserved": post_f8 <= self.experiment.early_retention_ratio_max * original_f8,
            "heldout_convergence_q": convergence["median_q"] <= self.experiment.convergence_q_max,
            "heldout_convergence_regression_fraction": convergence["regression_fraction"] <= self.experiment.convergence_regression_fraction_max,
        }
        return {
            "step": step,
            "passed": all(checks.values()),
            "checks": checks,
            "posttrained_f8": post_f8,
            "original_f8": original_f8,
            "post_over_original_f8": post_f8 / original_f8,
            "early_retention_ratio_max": self.experiment.early_retention_ratio_max,
            "convergence_validation": convergence,
            "convergence_q_max": self.experiment.convergence_q_max,
            "convergence_regression_fraction_max": self.experiment.convergence_regression_fraction_max,
        }

    def run(self) -> dict[str, object]:
        for step in range(1, self.experiment.training_steps + 1):
            metrics = self.step(step)
            if step % self.experiment.log_every == 0:
                self.rows.extend(metrics)
                elapsed = time.time() - self.started
                write_csv(self.output / "train_metrics.csv", self.rows)
                atomic_json(self.output / "progress.json", {
                    "phase": "post_training",
                    "step": step,
                    "total_steps": self.experiment.training_steps,
                    "steps_per_second": step / elapsed,
                    "eta_seconds": (self.experiment.training_steps - step) / (step / elapsed),
                    "gate": self.gate,
                })
                print(json.dumps({"phase": "v43_training", "step": step, "metrics": metrics}), flush=True)
            if step % self.experiment.checkpoint_every == 0:
                self.save(step, complete=False)
            if step == self.experiment.smoke_step:
                self.gate = self.smoke_gate(step)
                atomic_json(self.output / "smoke_gate.json", self.gate)
                if not self.gate["passed"]:
                    self.save(step, complete=True, stopped_by_gate=True)
                    return {"stopped_by_gate": True, "step": step, "gate": self.gate}
        self.save(self.experiment.training_steps, complete=True)
        return {"stopped_by_gate": False, "step": self.experiment.training_steps, "gate": self.gate}


def slice_state(state: RolloutState, start: int, stop: int) -> RolloutState:
    return RolloutState(**{name: value[start:stop].clone() for name, value in state.__dict__.items()})


@torch.inference_mode()
def evaluate_models(
    experiment: V43Config,
    config: V40Config,
    models: dict[str, PopulationController],
    device: torch.device,
) -> list[dict[str, object]]:
    specifications = (
        ("original_query_all", "original_query", ALL_HISTORY),
        ("original_query_fifo16", "original_query", FIFO_16),
        ("v43_query_all", "v43_query", ALL_HISTORY),
        ("v43_query_fifo16", "v43_query", FIFO_16),
        ("current_only", "current_only", ALL_HISTORY),
        ("ar_no_fresh_all", "ar_no_fresh", ALL_HISTORY),
        ("analytic_oracle", "analytic_oracle", ALL_HISTORY),
    )
    storage_config = replace(
        config, history_capacity=config.bootstrap_actions + experiment.evaluation_horizon
    )
    records: list[dict[str, object]] = []
    for bucket_index, (bucket, harder) in enumerate((("same_family", False), ("condition_4_6", True))):
        base = make_rollout_state(
            storage_config,
            experiment.evaluation_episodes,
            config.seed + 600 + bucket_index,
            device,
            harder,
        )
        for label, model_name, policy in specifications:
            for start in range(0, experiment.evaluation_episodes, config.evaluation_batch_size):
                state = slice_state(base, start, min(start + config.evaluation_batch_size, experiment.evaluation_episodes))
                initial = state.current.norm(dim=-1)
                previous = initial
                seeds = state.world_seeds.cpu().tolist()
                for local, norm in enumerate(initial.cpu().tolist()):
                    records.append({
                        "bucket": bucket,
                        "model": label,
                        "episode": start + local,
                        "world_seed": int(seeds[local]),
                        "iteration": 0,
                        "residual_norm": norm,
                        "q": "",
                        "action_norm": 0.0,
                        "action_residual_ratio": 0.0,
                        "converged": norm < config.convergence_threshold,
                        "diverged": norm > config.divergence_threshold,
                    })
                for iteration in range(1, experiment.evaluation_horizon + 1):
                    before = state.current
                    if model_name == "analytic_oracle":
                        action = analytic_action(state.matrices, before)
                    else:
                        action = models[model_name](controller_batch(config, state))
                    after = transition(before, state.matrices, action)
                    norm = after.norm(dim=-1)
                    q = norm / previous.clamp_min(1e-12)
                    ratio = action.norm(dim=-1) / previous.clamp_min(1e-12)
                    for local, values in enumerate(
                        zip(norm.cpu().tolist(), q.cpu().tolist(), action.norm(dim=-1).cpu().tolist(), ratio.cpu().tolist())
                    ):
                        records.append({
                            "bucket": bucket,
                            "model": label,
                            "episode": start + local,
                            "world_seed": int(seeds[local]),
                            "iteration": iteration,
                            "residual_norm": values[0],
                            "q": values[1],
                            "action_norm": values[2],
                            "action_residual_ratio": values[3],
                            "converged": values[0] < config.convergence_threshold,
                            "diverged": values[0] > config.divergence_threshold,
                        })
                    append_with_policy(
                        state,
                        before,
                        action,
                        after,
                        policy,
                        config.bootstrap_actions,
                        storage_config.history_capacity,
                    )
                    previous = norm
    return records


def intervened_batch(
    config: V40Config,
    state: RolloutState,
    condition: str,
    wrong: RolloutState,
) -> ControllerBatch:
    modified = state.clone()
    if condition == "no_history":
        modified.history_lengths.zero_()
    elif condition == "wrong_world":
        modified.history_before = wrong.history_before.roll(1, 0)
        modified.history_action = wrong.history_action.roll(1, 0)
        modified.history_after = wrong.history_after.roll(1, 0)
        modified.history_lengths = wrong.history_lengths.roll(1, 0)
    elif condition != "correct":
        raise ValueError(condition)
    return controller_batch(config, modified)


@torch.inference_mode()
def history_interventions(
    experiment: V43Config,
    config: V40Config,
    models: dict[str, PopulationController],
    device: torch.device,
) -> list[dict[str, object]]:
    capacity = config.bootstrap_actions + experiment.intervention_at + experiment.intervention_followup
    storage = replace(config, history_capacity=capacity)
    records = []
    for label in ("original_query", "v43_query"):
        model = models[label]
        state = make_rollout_state(storage, experiment.intervention_episodes, config.seed + 700, device)
        wrong = make_rollout_state(storage, experiment.intervention_episodes, config.seed + 701, device)
        for _ in range(experiment.intervention_at):
            for active in (state, wrong):
                before = active.current
                action = model(controller_batch(config, active))
                after = transition(before, active.matrices, action)
                append_with_policy(active, before, action, after, ALL_HISTORY, config.bootstrap_actions, capacity)
        initial = state.current.norm(dim=-1)
        for condition in ("correct", "no_history", "wrong_world"):
            rollout = state.clone()
            action = model(intervened_batch(config, rollout, condition, wrong))
            before = rollout.current
            after = transition(before, rollout.matrices, action)
            one_step_q = after.norm(dim=-1) / initial.clamp_min(1e-12)
            append_with_policy(rollout, before, action, after, ALL_HISTORY, config.bootstrap_actions, capacity)
            for _ in range(1, experiment.intervention_followup):
                before = rollout.current
                action = model(controller_batch(config, rollout))
                after = transition(before, rollout.matrices, action)
                append_with_policy(rollout, before, action, after, ALL_HISTORY, config.bootstrap_actions, capacity)
            endpoint = rollout.current.norm(dim=-1) / initial.clamp_min(1e-12)
            for episode, values in enumerate(zip(one_step_q.cpu().tolist(), endpoint.cpu().tolist())):
                records.append({
                    "model": label,
                    "condition": condition,
                    "episode": episode,
                    "one_step_q": values[0],
                    "endpoint_ratio": values[1],
                    "followup_steps": experiment.intervention_followup,
                })
    return records


def summarize_rollouts(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[(str(row["bucket"]), str(row["model"]), int(row["iteration"]))].append(row)
    output = []
    for (bucket, model, iteration), rows in sorted(groups.items()):
        norms = [float(row["residual_norm"]) for row in rows]
        q = [float(row["q"]) for row in rows if row["q"] != ""]
        output.append({
            "bucket": bucket,
            "model": model,
            "iteration": iteration,
            "residual_norm_median": statistics.median(norms),
            "residual_norm_mean": statistics.fmean(norms),
            "q_median": statistics.median(q) if q else "",
            "q_fraction_below_one": sum(value < 1 for value in q) / len(q) if q else "",
            "action_norm_median": statistics.median(float(row["action_norm"]) for row in rows),
            "action_residual_ratio_median": statistics.median(float(row["action_residual_ratio"]) for row in rows),
            "diverged_fraction": sum(bool(row["diverged"]) for row in rows) / len(rows),
        })
    return output


def summarize_interventions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[(str(row["model"]), str(row["condition"]))].append(row)
    return [
        {
            "model": model,
            "condition": condition,
            "one_step_q_median": statistics.median(float(row["one_step_q"]) for row in rows),
            "endpoint_ratio_median": statistics.median(float(row["endpoint_ratio"]) for row in rows),
        }
        for (model, condition), rows in sorted(groups.items())
    ]


def find(summary: list[dict[str, object]], model: str, iteration: int, bucket: str = "same_family") -> dict[str, object]:
    return next(row for row in summary if row["bucket"] == bucket and row["model"] == model and row["iteration"] == iteration)


def scientific_verdict(
    summary: list[dict[str, object]],
    intervention: list[dict[str, object]],
) -> dict[str, object]:
    curves = {
        model: [row for row in summary if row["bucket"] == "same_family" and row["model"] == model]
        for model in ("original_query_all", "v43_query_all")
    }
    best = {model: min(rows, key=lambda row: float(row["residual_norm_median"])) for model, rows in curves.items()}
    ratios = {
        model: {
            64: float(find(summary, model, 64)["residual_norm_median"]) / float(best[model]["residual_norm_median"]),
            128: float(find(summary, model, 128)["residual_norm_median"]) / float(best[model]["residual_norm_median"]),
        }
        for model in curves
    }
    late_q = {
        model: {
            "33_64": statistics.median(float(row["q_median"]) for row in rows if 33 <= int(row["iteration"]) <= 64),
            "65_128": statistics.median(float(row["q_median"]) for row in rows if 65 <= int(row["iteration"]) <= 128),
        }
        for model, rows in curves.items()
    }
    intervention_map = {(row["model"], row["condition"]): row for row in intervention}
    correct = intervention_map[("v43_query", "correct")]
    history = all(
        float(correct["one_step_q_median"]) <= 0.95 * float(intervention_map[("v43_query", condition)]["one_step_q_median"])
        and float(correct["endpoint_ratio_median"]) <= 0.95 * float(intervention_map[("v43_query", condition)]["endpoint_ratio_median"])
        for condition in ("no_history", "wrong_world")
    )
    early = float(find(summary, "v43_query_all", 8)["residual_norm_median"]) <= 1.15 * float(find(summary, "original_query_all", 8)["residual_norm_median"])
    reduced = ratios["v43_query_all"][128] <= 0.5 * ratios["original_query_all"][128]
    near_best = ratios["v43_query_all"][64] <= 1.5 and ratios["v43_query_all"][128] <= 1.5
    stable_q = late_q["v43_query_all"]["33_64"] <= 1 and late_q["v43_query_all"]["65_128"] <= 1
    checks = {
        "early_system_identification_preserved": early,
        "best_step_degradation_materially_reduced": reduced,
        "late_q_not_expansive": stable_q,
        "f64_f128_remain_near_best": near_best,
        "history_causality_preserved": history,
    }
    return {
        "verdict": "CONVERGENCE LAW SUPPORTED" if all(checks.values()) else "FIXED-POINT STABILITY NOT SUPPORTED",
        "supported": all(checks.values()),
        "checks": checks,
        "endpoint_over_best": ratios,
        "late_q": late_q,
        "best": {
            model: {"iteration": int(row["iteration"]), "residual": float(row["residual_norm_median"])}
            for model, row in best.items()
        },
    }


def write_report(
    output: Path,
    summary: list[dict[str, object]],
    intervention: list[dict[str, object]],
    verdict: dict[str, object],
) -> None:
    lines = [
        "# IFI v4.3 — Convergence Replay Post-Training",
        "",
        f"## Verdict: {verdict['verdict']}",
        "",
        "| Model | F0 | F8 | F16 | F32 | F64 | F128 | Best | Best step | F64/best | F128/best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("original_query_all", "v43_query_all", "v43_query_fifo16", "current_only", "ar_no_fresh_all", "analytic_oracle"):
        curve = [row for row in summary if row["bucket"] == "same_family" and row["model"] == model]
        best = min(curve, key=lambda row: float(row["residual_norm_median"]))
        values = [float(find(summary, model, step)["residual_norm_median"]) for step in (0, 8, 16, 32, 64, 128)]
        lines.append(
            f"| {model} | " + " | ".join(f"{value:.6g}" for value in values)
            + f" | {float(best['residual_norm_median']):.6g} | {best['iteration']}"
            + f" | {values[-2] / float(best['residual_norm_median']):.6g}"
            + f" | {values[-1] / float(best['residual_norm_median']):.6g} |"
        )
    lines.extend(["", "## Locked criteria", ""])
    for name, passed in verdict["checks"].items():
        lines.append(f"- {name.replace('_', ' ')}: **{passed}**")
    lines.extend([
        "",
        "Post-training used a 70/30 convergence/ordinary mixture, randomly sampled one-step consequences, and genuine detached histories. Neither harvesting depth nor relation-to-minimum labels entered the controller.",
        "",
        "History intervention summaries are saved in `history_intervention_summary.csv`.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axis = plt.subplots(figsize=(9, 5))
    for model in ("original_query_all", "v43_query_all", "v43_query_fifo16", "current_only", "ar_no_fresh_all"):
        curve = [row for row in summary if row["bucket"] == "same_family" and row["model"] == model]
        axis.plot([int(row["iteration"]) for row in curve], [float(row["residual_norm_median"]) for row in curve], label=model)
    axis.set_yscale("log")
    axis.set_xlabel("Frozen iteration")
    axis.set_ylabel("Median residual norm")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "rollout_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in ("original_query_all", "v43_query_all"):
        curve = [row for row in summary if row["bucket"] == "same_family" and row["model"] == model]
        axes[0].plot([int(row["iteration"]) for row in curve], [float(row["action_norm_median"]) for row in curve], label=model)
        axes[1].plot([int(row["iteration"]) for row in curve], [float(row["action_residual_ratio_median"]) for row in curve], label=model)
    axes[0].set_ylabel("Median action norm")
    axes[1].set_ylabel("Median action / residual norm")
    for axis in axes:
        axis.set_xlabel("Frozen iteration")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "action_scaling_diagnostic.png", dpi=160)
    plt.close(fig)


def validate_configs(experiment: V43Config, base: V40Config) -> None:
    if experiment.seed != base.seed:
        raise ValueError("v4.3 must preserve the canonical v4.1 seed")
    if not 0 < experiment.harvest_train_fraction < 1:
        raise ValueError("Harvest train fraction must be in (0,1)")
    expected = round(experiment.batch_size * experiment.convergence_fraction)
    if (expected, experiment.batch_size - expected) != (45, 19):
        raise ValueError("Locked mixed replay batch must contain 45 convergence and 19 ordinary states")
    if experiment.learning_rate >= base.learning_rate:
        raise ValueError("Post-training learning rate must be lower than v4.1")
    if experiment.smoke_step >= experiment.training_steps:
        raise ValueError("Smoke must precede canonical completion")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v4.3 convergence replay")
    parser.add_argument("--config", default="configs/v4_3_convergence_replay.json")
    args = parser.parse_args(argv)
    experiment = V43Config.from_json(args.config)
    base_config = V40Config.from_json(experiment.base_config)
    validate_configs(experiment, base_config)
    if not torch.cuda.is_available():
        raise RuntimeError("IFI v4.3 requires CUDA")
    output = Path(experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "canonical_run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    except FileExistsError as error:
        raise RuntimeError("Refusing a second v4.3 canonical run") from error
    atomic_json(output / "config.json", experiment.to_dict())
    checkpoint_path = Path(experiment.base_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("complete") or int(checkpoint["step"]) != base_config.training_steps:
        raise RuntimeError("v4.1 initialization checkpoint is incomplete")
    device = torch.device("cuda")
    seed_all(experiment.seed)
    original = clone_model(base_config, checkpoint["models"]["query_history"], device)
    original.eval().requires_grad_(False)
    atomic_json(output / "status.json", {"phase": "harvesting", "worlds": experiment.harvest_worlds})
    archive, trajectory_rows = harvest_convergence_pool(experiment, base_config, original, device)
    torch.save(archive, output / "convergence_pool.pt")
    write_csv(output / "harvest_trajectories.csv", trajectory_rows)
    replay = MixedReplay(
        experiment,
        base_config,
        archive,
        checkpoint["populations"]["query_history"],
    )
    model = clone_model(base_config, checkpoint["models"]["query_history"], device)
    audit_batch, audit_matrices, _, _ = replay.sample(device)
    audit_action = model(audit_batch)
    audit_after = transition(audit_batch.current, audit_matrices, audit_action)
    audit_loss = F.softplus(
        audit_after.norm(dim=-1) / audit_batch.current.norm(dim=-1).clamp_min(1e-12)
        - base_config.progress_tau
    ).mean()
    model.zero_grad(set_to_none=True)
    audit_loss.backward()
    audit = {
        "base_checkpoint_sha256": sha256(checkpoint_path),
        "base_checkpoint_step": int(checkpoint["step"]),
        "archive_states": int(archive["current"].shape[0]),
        "archive_train_states": int(replay.train_indices.numel()),
        "archive_validation_states": int(replay.validation_indices.numel()),
        "real_history_max_error": archive_history_error(archive),
        "history_detached": not audit_batch.history.requires_grad,
        "world_matrix_detached": not audit_matrices.requires_grad,
        "controller_gradient_nonzero": gradient_norm(list(model.parameters())) > 0,
        "controller_inputs": list(audit_batch.__dict__),
        "forbidden_logging_fields_not_in_controller": all(
            name not in audit_batch.__dict__ for name in ("depth", "relation", "world_seed", "matrix")
        ),
        "convergence_batch": replay.convergence_batch,
        "ordinary_batch": replay.ordinary_batch,
        "no_bptt": True,
        "no_teacher_or_oracle_action_training": True,
        "architecture_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    audit["passed"] = (
        audit["real_history_max_error"] < 2e-6
        and audit["history_detached"]
        and audit["world_matrix_detached"]
        and audit["controller_gradient_nonzero"]
        and audit["forbidden_logging_fields_not_in_controller"]
        and audit["convergence_batch"] == 45
        and audit["ordinary_batch"] == 19
    )
    atomic_json(output / "integrity_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("v4.3 integrity audit failed")
    model.zero_grad(set_to_none=True)
    atomic_json(output / "status.json", {"phase": "post_training", "step": 0})
    trainer = PostTrainer(experiment, base_config, model, original, replay, device, output)
    training_result = trainer.run()
    if training_result["stopped_by_gate"]:
        result = {
            "verdict": "FIXED-POINT STABILITY NOT SUPPORTED — SMOKE GATE FAILED",
            "training": training_result,
        }
        atomic_json(output / "verdict.json", result)
        (output / "report.md").write_text(
            "# IFI v4.3 — Convergence Replay Post-Training\n\n"
            "## Verdict: FIXED-POINT STABILITY NOT SUPPORTED — SMOKE GATE FAILED\n\n"
            "Post-training stopped at the locked smoke gate without architecture tuning.\n",
            encoding="utf-8",
        )
        atomic_json(output / "status.json", {"phase": "complete", "result": result})
        return
    atomic_json(output / "status.json", {"phase": "evaluation"})
    base_models = build_models(base_config, device)
    for name, base_model in base_models.items():
        base_model.load_state_dict(checkpoint["models"][name])
        base_model.eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    models = {
        "original_query": base_models["query_history"],
        "v43_query": model,
        "current_only": base_models["current_only"],
        "ar_no_fresh": base_models["ar_no_fresh"],
    }
    rollouts = evaluate_models(experiment, base_config, models, device)
    interventions = history_interventions(experiment, base_config, models, device)
    rollout_summary = summarize_rollouts(rollouts)
    intervention_summary = summarize_interventions(interventions)
    result = scientific_verdict(rollout_summary, intervention_summary)
    write_csv(output / "eval_rollouts.csv", rollouts)
    write_csv(output / "eval_summary.csv", rollout_summary)
    write_csv(output / "history_interventions.csv", interventions)
    write_csv(output / "history_intervention_summary.csv", intervention_summary)
    atomic_json(output / "verdict.json", result)
    write_report(output, rollout_summary, intervention_summary, result)
    manifest = {
        "complete": True,
        "pid": os.getpid(),
        "base_checkpoint_sha256": sha256(checkpoint_path),
        "posttrained_checkpoint_sha256": sha256(output / "checkpoint_final.pt"),
        "training": training_result,
        "verdict": result,
    }
    atomic_json(output / "experiment_manifest.json", manifest)
    atomic_json(output / "status.json", {"phase": "complete", "manifest": manifest})
    print(json.dumps({"phase": "v43_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
