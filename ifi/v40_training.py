from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .v40_config import LEARNED_MODELS, V40Config
from .v40_evaluation import quick_gate
from .v40_models import PopulationController, parameter_counts
from .v40_population import PopulationUpdate, ReplayPopulation
from .v40_worlds import transition


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(config: V40Config, device: torch.device) -> dict[str, PopulationController]:
    models = {}
    for offset, name in enumerate(LEARNED_MODELS):
        seed_all(config.seed + 100 + offset)
        models[name] = PopulationController(name, config).to(device)
    return models


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return 0.0 if not squares else float(torch.stack(squares).sum().sqrt())


def _atomic_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def integrity_audit(
    config: V40Config,
    models: dict[str, PopulationController],
    populations: dict[str, ReplayPopulation],
    device: torch.device,
) -> dict[str, object]:
    query_model = models["query_history"]
    population = populations["query_history"]
    indices = population.sample_indices(min(16, config.batch_size))
    batch = population.controller_batch(indices, device)
    matrix = population.matrix_batch(indices, device)
    query_model.eval()
    with torch.no_grad():
        normal = query_model(batch)
        permuted_history = batch.history.clone()
        for row, length in enumerate(batch.history_lengths.tolist()):
            permutation = torch.arange(length - 1, -1, -1, device=device)
            permuted_history[row, :length] = batch.history[row, permutation]
        permuted = query_model(type(batch)(
            batch.current,
            batch.previous_action,
            permuted_history,
            batch.history_lengths,
        ))
    query_model.train()
    action = query_model(batch)
    next_residual = transition(batch.current, matrix, action)
    if config.loss_kind == "softplus_progress":
        local_q = next_residual.norm(dim=-1) / batch.current.norm(dim=-1).clamp_min(
            config.numerical_epsilon
        )
        local_loss = F.softplus(local_q - config.progress_tau).mean()
    else:
        local_loss = (
            next_residual.square().sum(-1)
            / (batch.current.square().sum(-1) + config.numerical_epsilon)
        ).mean()
    query_model.zero_grad(set_to_none=True)
    local_loss.backward()
    parameter_gradient = _gradient_norm(list(query_model.parameters()))
    checks = {
        "no_positional_parameters": not any(
            "position" in name.lower() or "step_embedding" in name.lower()
            for model in models.values()
            for name, _ in model.named_parameters()
        ),
        "permutation_invariant_history": bool(
            torch.allclose(normal, permuted, atol=2e-6, rtol=2e-6)
        ),
        "stored_history_detached": not batch.history.requires_grad,
        "current_state_detached": not batch.current.requires_grad,
        "world_matrix_detached": not matrix.requires_grad,
        "local_controller_gradient_nonzero": parameter_gradient > 0,
        "real_history_transition_identity": population.actual_history_max_error() < 2e-6,
        "batch_slots_distinct": indices.unique().numel() == indices.numel(),
        "bounded_action_head": float(action.detach().abs().max()) <= config.action_max,
        "controller_interface_excludes_world_metadata": set(batch.__dict__) == {
            "current",
            "previous_action",
            "history",
            "history_lengths",
        },
    }
    if config.stratified_magnitude_sampling:
        sampled_counts = torch.bincount(
            population.magnitude_bin_ids(population.current[indices]),
            minlength=len(config.magnitude_bin_log10_edges) - 1,
        )
        occupancy = population.magnitude_bin_occupancy()
        checks["every_magnitude_bin_populated"] = bool((occupancy > 0).all())
        checks["stratified_batch_balanced"] = bool(
            int(sampled_counts.max() - sampled_counts.min()) <= 1
        )
        if config.training_residual_is_post_bootstrap:
            population_norm = population.current.norm(dim=-1)
            checks["post_bootstrap_residual_range_realized"] = bool(
                float(population_norm.min())
                >= config.training_residual_norm_min * (1 - 1e-4)
                and float(population_norm.max())
                <= config.training_residual_norm_max * (1 + 1e-4)
            )
    query_model.zero_grad(set_to_none=True)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "parameter_counts": parameter_counts(models),
            "local_controller_gradient_norm": parameter_gradient,
            "history_permutation_max_action_difference": float(
                (normal - permuted).abs().max()
            ),
            "actual_history_max_transition_error": population.actual_history_max_error(),
            "initial_batch_unique_ages": int(population.ages[indices].unique().numel()),
            "initial_magnitude_bin_occupancy": population.magnitude_bin_occupancy().tolist(),
        },
        "forbidden_inputs": [
            "B/xi",
            "world_seed",
            "episode_identity",
            "rollout_age",
            "optimizer_step",
            "trajectory_position",
            "oracle_action",
            "future/test_residual",
        ],
        "gradient_route": "T -> action -> one world transition -> contraction loss only",
    }


class PopulationTrainer:
    def __init__(self, config: V40Config, device: torch.device, output: Path):
        self.config = config
        self.device = device
        self.output = output
        self.models = build_models(config, device)
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            for name, model in self.models.items()
        }
        self.populations = {
            name: ReplayPopulation(config, name, config.seed + 200)
            for name in LEARNED_MODELS
        }
        self.action_generators = {
            name: torch.Generator(device=device).manual_seed(config.seed + 300 + offset)
            for offset, name in enumerate(LEARNED_MODELS)
        }
        self.train_rows: list[dict[str, object]] = []
        self.pool_rows: list[dict[str, object]] = []
        self.bin_rows: list[dict[str, object]] = []
        self.latest_bin_rows: dict[str, list[dict[str, object]]] = {}
        self.gate: dict[str, object] | None = None
        self.start_step = 0
        self.started_at = time.time()
        self.interval_resets = {
            name: PopulationUpdate(0, 0, 0, 0) for name in LEARNED_MODELS
        }
        self.interval_clips = {name: 0 for name in LEARNED_MODELS}

    def _checkpoint_payload(self, step: int, complete: bool) -> dict[str, object]:
        return {
            "experiment": (
                f"IFI {self.config.experiment_version} population-replay residual learner"
            ),
            "step": step,
            "complete": complete,
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
            "populations": {
                name: population.state_dict()
                for name, population in self.populations.items()
            },
            "action_generator_states": {
                name: generator.get_state()
                for name, generator in self.action_generators.items()
            },
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "train_rows": self.train_rows,
            "pool_rows": self.pool_rows,
            "bin_rows": self.bin_rows,
            "gate": self.gate,
        }

    def save_checkpoint(self, step: int, complete: bool = False) -> None:
        _atomic_save(
            self.output / ("checkpoint_final.pt" if complete else "checkpoint_progress.pt"),
            self._checkpoint_payload(step, complete),
        )
        _write_csv(self.output / "train_metrics.csv", self.train_rows)
        _write_csv(self.output / "pool_diagnostics.csv", self.pool_rows)
        _write_csv(self.output / "bin_metrics.csv", self.bin_rows)

    def _merge_resets(
        self, left: PopulationUpdate, right: PopulationUpdate
    ) -> PopulationUpdate:
        return PopulationUpdate(
            left.convergence_resets + right.convergence_resets,
            left.divergence_resets + right.divergence_resets,
            left.age_resets + right.age_resets,
            left.random_resets + right.random_resets,
        )

    def train_step(self, step: int) -> dict[str, dict[str, float]]:
        step_metrics = {}
        for name in LEARNED_MODELS:
            model = self.models[name]
            population = self.populations[name]
            indices = population.sample_indices(self.config.batch_size)
            sampled_ages = population.ages[indices].clone()
            sampled_history_lengths = population.history_lengths[indices].clone()
            sampled_bin_ids = population.magnitude_bin_ids(
                population.current[indices]
            )
            bin_occupancy = population.magnitude_bin_occupancy()
            batch = population.controller_batch(indices, self.device)
            matrix = population.matrix_batch(indices, self.device)
            action = model(batch)
            exploration = self.config.exploration_std * torch.randn(
                action.shape,
                generator=self.action_generators[name],
                device=self.device,
            )
            executed = (action + exploration).clamp(
                -self.config.action_max, self.config.action_max
            )
            next_residual = transition(batch.current, matrix, executed)
            denominator = batch.current.square().sum(-1) + self.config.numerical_epsilon
            contraction = next_residual.square().sum(-1) / denominator
            q = next_residual.norm(dim=-1) / batch.current.norm(
                dim=-1
            ).clamp_min(self.config.numerical_epsilon)
            action_energy = action.square().sum(-1) / denominator
            if self.config.loss_kind == "contraction_squared":
                primary_loss = contraction
            elif self.config.loss_kind == "softplus_progress":
                primary_loss = F.softplus(q - self.config.progress_tau)
            else:
                raise ValueError(f"Unknown v4 loss: {self.config.loss_kind}")
            loss = (
                primary_loss + self.config.action_regularization * action_energy
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {name} loss at step {step}")
            optimizer = self.optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            if step % self.config.log_every == 0:
                edges = self.config.magnitude_bin_log10_edges
                bin_rows = []
                for bin_index in range(len(edges) - 1):
                    cpu_mask = sampled_bin_ids == bin_index
                    mask = cpu_mask.to(self.device)
                    if mask.any():
                        action_gradient = torch.autograd.grad(
                            primary_loss[mask].mean(),
                            action,
                            retain_graph=True,
                        )[0]
                        selected_q = q.detach()[mask]
                        selected_action = action.detach()[mask].norm(dim=-1)
                        history = sampled_history_lengths[cpu_mask].float()
                        row = {
                            "bin_index": bin_index,
                            "log10_rho_lower": edges[bin_index],
                            "log10_rho_upper": edges[bin_index + 1],
                            "sample_count": int(mask.sum()),
                            "q_mean": float(selected_q.mean()),
                            "q_median": float(selected_q.median()),
                            "contraction_fraction": float(
                                (selected_q < 1).float().mean()
                            ),
                            "action_norm_mean": float(selected_action.mean()),
                            "action_gradient_norm": float(action_gradient.norm()),
                            "population_occupancy": int(bin_occupancy[bin_index]),
                            "population_occupancy_fraction": float(
                                bin_occupancy[bin_index] / self.config.pool_size
                            ),
                            "history_length_mean": float(history.mean()),
                        }
                    else:
                        row = {
                            "bin_index": bin_index,
                            "log10_rho_lower": edges[bin_index],
                            "log10_rho_upper": edges[bin_index + 1],
                            "sample_count": 0,
                            "q_mean": float("nan"),
                            "q_median": float("nan"),
                            "contraction_fraction": float("nan"),
                            "action_norm_mean": float("nan"),
                            "action_gradient_norm": float("nan"),
                            "population_occupancy": int(bin_occupancy[bin_index]),
                            "population_occupancy_fraction": float(
                                bin_occupancy[bin_index] / self.config.pool_size
                            ),
                            "history_length_mean": float("nan"),
                        }
                    bin_rows.append(row)
                self.latest_bin_rows[name] = bin_rows
            loss.backward()
            parameters = list(model.parameters())
            gradient_norm = _gradient_norm(parameters)
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite {name} gradient at step {step}")
            if self.config.gradient_clip > 0:
                clipped_norm = float(
                    nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
                )
                was_clipped = clipped_norm > self.config.gradient_clip
            else:
                clipped_norm = gradient_norm
                was_clipped = False
            self.interval_clips[name] += int(was_clipped)
            optimizer.step()
            q = q.detach()
            update = population.apply(indices, executed, next_residual, step)
            self.interval_resets[name] = self._merge_resets(
                self.interval_resets[name], update
            )
            step_metrics[name] = {
                "loss": float(loss.detach()),
                "contraction_loss": float(contraction.detach().mean()),
                "progress_loss": float(primary_loss.detach().mean()),
                "action_loss": float(action_energy.detach().mean()),
                "q_mean": float(q.mean()),
                "q_median": float(q.median()),
                "q_fraction_below_one": float((q < 1).float().mean()),
                "action_norm_mean": float(action.detach().norm(dim=-1).mean()),
                "gradient_norm": gradient_norm,
                "gradient_clipped": float(was_clipped),
                "sampled_age_mean": float(sampled_ages.float().mean()),
                "sampled_age_unique": float(sampled_ages.unique().numel()),
            }
        return step_metrics

    def _log(self, step: int, metrics: dict[str, dict[str, float]]) -> None:
        elapsed = time.time() - self.started_at
        for name, values in metrics.items():
            resets = self.interval_resets[name]
            attempts = self.config.log_every * self.config.batch_size
            row: dict[str, object] = {
                "step": step,
                "arm": name,
                **values,
                "convergence_reset_fraction": resets.convergence_resets / attempts,
                "divergence_reset_fraction": resets.divergence_resets / attempts,
                "age_reset_fraction": resets.age_resets / attempts,
                "random_reset_fraction": resets.random_resets / attempts,
                "gradient_clipping_fraction": self.interval_clips[name]
                / self.config.log_every,
                "elapsed_seconds": elapsed,
                "steps_per_second": step / max(elapsed, 1e-9),
            }
            self.train_rows.append(row)
            self.pool_rows.append(self.populations[name].diagnostics(step))
            self.interval_resets[name] = PopulationUpdate(0, 0, 0, 0)
            self.interval_clips[name] = 0
            for bin_row in self.latest_bin_rows.get(name, []):
                self.bin_rows.append({"step": step, "arm": name, **bin_row})
        _write_csv(self.output / "train_metrics.csv", self.train_rows)
        _write_csv(self.output / "pool_diagnostics.csv", self.pool_rows)
        _write_csv(self.output / "bin_metrics.csv", self.bin_rows)
        progress = {
            "phase": "training",
            "step": step,
            "total_steps": self.config.training_steps,
            "fraction": step / self.config.training_steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": step / max(elapsed, 1e-9),
            "eta_seconds": (self.config.training_steps - step)
            / max(step / max(elapsed, 1e-9), 1e-9),
            "gate": self.gate,
        }
        temporary = self.output / "progress.json.tmp"
        temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        temporary.replace(self.output / "progress.json")
        print(json.dumps({"phase": "v4_training", **progress, "metrics": metrics}), flush=True)

    def run(self, stop_on_gate_failure: bool = True) -> dict[str, object]:
        audit = integrity_audit(
            self.config, self.models, self.populations, self.device
        )
        (self.output / "integrity_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        if not audit["passed"]:
            raise RuntimeError("v4.0 integrity audit failed")
        stopped_by_gate = False
        for step in range(1, self.config.training_steps + 1):
            metrics = self.train_step(step)
            if step % self.config.log_every == 0:
                self._log(step, metrics)
            if step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)
            if step == self.config.gate_step:
                model = self.models["query_history"]
                model.eval()
                self.gate = quick_gate(model, self.config, self.device, step)
                model.train()
                (self.output / "contraction_gate.json").write_text(
                    json.dumps(self.gate, indent=2), encoding="utf-8"
                )
                if not self.gate["passed"] and stop_on_gate_failure:
                    stopped_by_gate = True
                    self.save_checkpoint(step, complete=True)
                    break
        final_step = self.config.gate_step if stopped_by_gate else self.config.training_steps
        if not stopped_by_gate:
            self.save_checkpoint(final_step, complete=True)
        return {
            "stopped_by_gate": stopped_by_gate,
            "final_step": final_step,
            "gate": self.gate,
            "models": self.models,
            "populations": self.populations,
        }


def write_world_registry(path: Path, populations: dict[str, ReplayPopulation]) -> None:
    rows = [row for population in populations.values() for row in population.registry]
    _write_csv(path, rows)
