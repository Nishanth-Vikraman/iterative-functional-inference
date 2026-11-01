from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .v40_config import V40Config
from .v40_evaluation import controller_batch, make_rollout_state
from .v40_models import ControllerBatch, PopulationController, transition_tokens
from .v40_population import PopulationUpdate, ReplayPopulation
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy
from .v44_config import V44Config
from .v44_populations import RecoveryPopulation


ALL_HISTORY = HistoryPolicy("all_history", 0, True, True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arm_config(base: V40Config, experiment: V44Config, arm: str, seed: int) -> V40Config:
    if arm == "small":
        values = (
            experiment.small_model_dimension,
            experiment.small_heads,
            experiment.small_layers,
            experiment.small_ffn,
        )
    elif arm == "large":
        values = (
            experiment.large_model_dimension,
            experiment.large_heads,
            experiment.large_layers,
            experiment.large_ffn,
        )
    else:
        raise ValueError(arm)
    return replace(
        base,
        seed=seed,
        model_dimension=values[0],
        attention_heads=values[1],
        attention_layers=values[2],
        ffn_dimension=values[3],
        pool_size=experiment.ordinary_pool_size,
        batch_size=experiment.batch_size,
        training_steps=experiment.training_steps,
        learning_rate=(
            experiment.learning_rate_small if arm == "small" else experiment.learning_rate_large
        ),
        log_every=experiment.log_every,
        checkpoint_every=experiment.checkpoint_every,
        gate_step=experiment.smoke_step,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def gradient_norm(parameters: list[nn.Parameter]) -> float:
    values = [parameter.grad.square().sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ArmRuntime:
    def __init__(
        self,
        experiment: V44Config,
        base: V40Config,
        arm: str,
        seed: int,
        device: torch.device,
        archive: dict[str, Tensor] | None = None,
        archive_train_indices: Tensor | None = None,
    ):
        self.experiment = experiment
        self.arm = arm
        self.seed = seed
        self.device = device
        self.config = arm_config(base, experiment, arm, seed)
        seed_all(seed + 100)
        self.model = PopulationController("query_history", self.config).to(device)
        learning_rate = (
            experiment.learning_rate_small if arm == "small" else experiment.learning_rate_large
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ordinary = ReplayPopulation(self.config, f"v44_{arm}", seed + 200)
        self.recovery = RecoveryPopulation(
            experiment,
            self.config,
            self.model,
            seed,
            device,
            archive,
            archive_train_indices,
        )
        if experiment.recovery_source == "genuine_archive":
            self.initial_sampler_gate = self.recovery.enforce_sufficiency(self.model)
            if not self.initial_sampler_gate["passed"]:
                raise RuntimeError(f"{arm} convergence sampler sufficiency gate failed")
        else:
            self.initial_sampler_gate = {
                "passed": True,
                "attempts": 0,
                "result": {"checks": {}, "metrics": {}},
                "not_applicable": True,
            }
        self.generator = torch.Generator(device=device).manual_seed(seed + 300)
        self.sample_generator = torch.Generator().manual_seed(seed + 301)
        self.ordinary_count = round(experiment.batch_size * experiment.ordinary_fraction)
        self.recovery_count = experiment.batch_size - self.ordinary_count
        self.interval_resets = PopulationUpdate(0, 0, 0, 0)

    def mixed_batch(
        self,
    ) -> tuple[ControllerBatch, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        ordinary_indices = self.ordinary.sample_indices(self.ordinary_count)
        recovery_indices = self.recovery.sample_indices(self.recovery_count)
        capacity = self.experiment.recovery_history_capacity
        ordinary_before = torch.zeros(
            self.ordinary_count, capacity, self.config.residual_dimension
        )
        ordinary_action = torch.zeros(
            self.ordinary_count, capacity, self.config.action_dimension
        )
        ordinary_after = torch.zeros(
            self.ordinary_count, capacity, self.config.residual_dimension
        )
        ordinary_before[:, : self.config.history_capacity] = self.ordinary.history_before[
            ordinary_indices
        ]
        ordinary_action[:, : self.config.history_capacity] = self.ordinary.history_action[
            ordinary_indices
        ]
        ordinary_after[:, : self.config.history_capacity] = self.ordinary.history_after[
            ordinary_indices
        ]
        before = torch.cat((ordinary_before, self.recovery.history_before[recovery_indices]))
        history_action = torch.cat((
            ordinary_action,
            self.recovery.history_action[recovery_indices],
        ))
        history_after = torch.cat((ordinary_after, self.recovery.history_after[recovery_indices]))
        current = torch.cat((
            self.ordinary.current[ordinary_indices],
            self.recovery.current[recovery_indices],
        ))
        previous = torch.cat((
            self.ordinary.previous_action[ordinary_indices],
            self.recovery.previous_action[recovery_indices],
        ))
        lengths = torch.cat((
            self.ordinary.history_lengths[ordinary_indices],
            self.recovery.history_lengths[recovery_indices],
        ))
        matrices = torch.cat((
            self.ordinary.matrices[ordinary_indices],
            self.recovery.matrices[recovery_indices],
        ))
        source = torch.cat((
            torch.zeros(self.ordinary_count, dtype=torch.int64),
            torch.ones(self.recovery_count, dtype=torch.int64),
        ))
        logging_depth = torch.cat((
            self.ordinary.ages[ordinary_indices],
            self.recovery.depth[recovery_indices],
        ))
        permutation = torch.randperm(
            self.experiment.batch_size, generator=self.sample_generator
        )
        batch = ControllerBatch(
            current=current[permutation].to(self.device).detach(),
            previous_action=previous[permutation].to(self.device).detach(),
            history=transition_tokens(
                before[permutation].to(self.device).detach(),
                history_action[permutation].to(self.device).detach(),
                history_after[permutation].to(self.device).detach(),
                self.config.magnitude_scale,
                self.config.numerical_epsilon,
            ).detach(),
            history_lengths=lengths[permutation].to(self.device).detach(),
        )
        return (
            batch,
            matrices[permutation].to(self.device).detach(),
            source[permutation].to(self.device),
            ordinary_indices,
            recovery_indices,
            {
                "permutation": permutation,
                "depth": logging_depth[permutation],
            },
        )

    def train_step(self, step: int) -> tuple[dict[str, object], list[dict[str, object]]]:
        (
            batch,
            matrices,
            source,
            ordinary_indices,
            recovery_indices,
            logging,
        ) = self.mixed_batch()
        action = self.model(batch)
        exploration = self.config.exploration_std * torch.randn(
            action.shape, generator=self.generator, device=self.device
        )
        executed = (action + exploration).clamp(-self.config.action_max, self.config.action_max)
        after = transition(batch.current, matrices, executed)
        before_norm = batch.current.norm(dim=-1)
        after_norm = after.norm(dim=-1)
        q = after_norm / before_norm.clamp_min(self.config.numerical_epsilon)
        base_loss = F.softplus(q - self.config.progress_tau)
        safety = F.relu(after.square().sum(-1) - batch.current.square().sum(-1))
        per_example = base_loss + self.experiment.lambda_safe * safety
        loss = per_example.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite v4.4 {self.arm} loss at step {step}")
        self.optimizer.zero_grad(set_to_none=True)
        source_gradients = {}
        for source_id, name in ((0, "ordinary"), (1, "convergence")):
            mask = source == source_id
            gradient = torch.autograd.grad(per_example[mask].mean(), action, retain_graph=True)[0]
            source_gradients[name] = float(gradient.norm())
        loss.backward()
        parameters = list(self.model.parameters())
        total_gradient = gradient_norm(parameters)
        if not math.isfinite(total_gradient):
            raise RuntimeError(f"Non-finite v4.4 {self.arm} gradient at step {step}")
        self.optimizer.step()
        inverse = torch.argsort(logging["permutation"]).to(self.device)
        executed_ordered = executed.detach()[inverse]
        after_ordered = after.detach()[inverse]
        ordinary_update = self.ordinary.apply(
            ordinary_indices,
            executed_ordered[: self.ordinary_count],
            after_ordered[: self.ordinary_count],
            step,
        )
        self.interval_resets = PopulationUpdate(
            self.interval_resets.convergence_resets + ordinary_update.convergence_resets,
            self.interval_resets.divergence_resets + ordinary_update.divergence_resets,
            self.interval_resets.age_resets + ordinary_update.age_resets,
            self.interval_resets.random_resets + ordinary_update.random_resets,
        )
        recovery_before = self.recovery.current[recovery_indices].clone()
        self.recovery.append(
            recovery_indices,
            recovery_before,
            executed_ordered[self.ordinary_count :],
            after_ordered[self.ordinary_count :],
        )
        overall = {
            "step": step,
            "seed": self.seed,
            "arm": self.arm,
            "loss": float(loss.detach()),
            "q_median": float(q.detach().median()),
            "q_mean": float(q.detach().mean()),
            "contraction_fraction": float((q.detach() < 1).float().mean()),
            "regression_fraction": float((q.detach() > 1).float().mean()),
            "action_norm_median": float(action.detach().norm(dim=-1).median()),
            "gradient_norm": total_gradient,
            "gradient_clipping_fraction": 0.0,
        }
        regimes = []
        for source_id, name in ((0, "ordinary"), (1, "convergence")):
            mask = source == source_id
            ratio = action.detach().norm(dim=-1) / before_norm.clamp_min(1e-12)
            regimes.append({
                "step": step,
                "seed": self.seed,
                "arm": self.arm,
                "regime": name,
                "loss": float(per_example.detach()[mask].mean()),
                "v41_loss": float(base_loss.detach()[mask].mean()),
                "safe_loss": float(safety.detach()[mask].mean()),
                "q_median": float(q.detach()[mask].median()),
                "q_mean": float(q.detach()[mask].mean()),
                "residual_norm_median": float(before_norm.detach()[mask].median()),
                "action_norm_median": float(action.detach().norm(dim=-1)[mask].median()),
                "action_residual_ratio_median": float(ratio[mask].median()),
                "contraction_fraction": float((q.detach()[mask] < 1).float().mean()),
                "regression_fraction": float((q.detach()[mask] > 1).float().mean()),
                "gradient_norm": total_gradient,
                "source_action_gradient_norm": source_gradients[name],
                "gradient_clipping_fraction": 0.0,
                "trajectory_depth_median_logging_only": float(
                    logging["depth"][mask.cpu()].float().median()
                ),
            })
        return overall, regimes

    def refresh_if_due(self, step: int) -> dict[str, object] | None:
        if step % self.experiment.recovery_refresh_every == 0:
            self.recovery.periodic_refresh(self.model)
        if (
            self.experiment.recovery_source == "genuine_archive"
            and step % self.experiment.recovery_gate_every == 0
        ):
            result = self.recovery.enforce_sufficiency(self.model)
            if not result["passed"]:
                raise RuntimeError(
                    f"{self.arm} convergence sampler lost sufficient coverage at step {step}"
                )
            return {
                "step": step,
                "seed": self.seed,
                "arm": self.arm,
                "gate_refresh_attempts": result["attempts"],
                "full_reset_required": result["attempts"] > 1,
                **result["result"]["metrics"],
            }
        return None

    def state_dict(self) -> dict[str, object]:
        ordinary = self.ordinary.state_dict()
        ordinary.pop("registry", None)
        return {
            "arm": self.arm,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ordinary_population": ordinary,
            "recovery_population": self.recovery.state_dict(),
            "action_generator_state": self.generator.get_state(),
            "sample_generator_state": self.sample_generator.get_state(),
        }

    def load_checkpoint_state(
        self,
        checkpoint: dict[str, object],
        completed_steps: int,
    ) -> dict[str, object]:
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        ordinary_state = dict(checkpoint["ordinary_population"])
        ordinary_state.setdefault("registry", [])
        self.ordinary.load_state_dict(ordinary_state)
        self.recovery.load_state_dict(checkpoint["recovery_population"])
        exact_action_state = "action_generator_state" in checkpoint
        exact_sample_state = "sample_generator_state" in checkpoint
        if exact_action_state:
            self.generator.set_state(checkpoint["action_generator_state"])
        else:
            for _ in range(completed_steps):
                torch.randn(
                    self.experiment.batch_size,
                    self.config.action_dimension,
                    generator=self.generator,
                    device=self.device,
                )
        if exact_sample_state:
            self.sample_generator.set_state(checkpoint["sample_generator_state"])
        else:
            for _ in range(completed_steps):
                torch.randperm(
                    self.experiment.batch_size,
                    generator=self.sample_generator,
                )
        return {
            "model_optimizer_populations_restored": True,
            "action_generator_exactly_saved": exact_action_state,
            "sample_generator_exactly_saved": exact_sample_state,
            "missing_streams_reconstructed_by_draw_count": not (
                exact_action_state and exact_sample_state
            ),
            "completed_steps": completed_steps,
        }


@torch.inference_mode()
def quick_early_metrics(
    model: PopulationController,
    config: V40Config,
    device: torch.device,
    episodes: int = 192,
) -> dict[str, float]:
    horizon = 8
    storage = replace(config, history_capacity=config.bootstrap_actions + horizon)
    state = make_rollout_state(storage, episodes, config.seed + 500, device)
    initial = state.current.norm(dim=-1)
    q_values = []
    for _ in range(horizon):
        before = state.current
        action = model(controller_batch(config, state))
        after = transition(before, state.matrices, action)
        q_values.append(after.norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12))
        append_with_policy(
            state,
            before,
            action,
            after,
            ALL_HISTORY,
            config.bootstrap_actions,
            storage.history_capacity,
        )
    return {
        "f8_over_f0_median": float((state.current.norm(dim=-1) / initial).median()),
        "median_q": float(torch.cat(q_values).median()),
        "f8_residual_median": float(state.current.norm(dim=-1).median()),
    }


@torch.inference_mode()
def fixed_convergence_metrics(
    model: PopulationController,
    config: V40Config,
    archive: dict[str, Tensor],
    indices: Tensor,
    device: torch.device,
) -> dict[str, float]:
    selected = indices[: min(2048, indices.numel())]
    batch = ControllerBatch(
        current=archive["current"][selected].to(device),
        previous_action=archive["previous_action"][selected].to(device),
        history=transition_tokens(
            archive["history_before"][selected].to(device),
            archive["history_action"][selected].to(device),
            archive["history_after"][selected].to(device),
            config.magnitude_scale,
            config.numerical_epsilon,
        ),
        history_lengths=archive["history_lengths"][selected].to(device),
    )
    action = model(batch)
    after = transition(batch.current, archive["matrices"][selected].to(device), action)
    q = after.norm(dim=-1) / batch.current.norm(dim=-1).clamp_min(1e-12)
    return {
        "median_q": float(q.median()),
        "regression_fraction": float((q > 1).float().mean()),
        "action_norm_median": float(action.norm(dim=-1).median()),
    }


def smoke_gate(
    runtime: ArmRuntime,
    archive: dict[str, Tensor],
    validation_indices: Tensor,
    reference_regression_fraction: float,
) -> dict[str, object]:
    runtime.model.eval()
    early = quick_early_metrics(runtime.model, runtime.config, runtime.device)
    convergence = fixed_convergence_metrics(
        runtime.model,
        runtime.config,
        archive,
        validation_indices,
        runtime.device,
    )
    runtime.model.train()
    checks = {
        "early_f8_stage_matched": early["f8_over_f0_median"] <= runtime.experiment.smoke_f8_ratio_max,
        "early_median_q": early["median_q"] <= runtime.experiment.smoke_q_max,
        "convergence_median_q": convergence["median_q"] <= runtime.experiment.smoke_convergence_q_max,
        "convergence_regression_improved": convergence["regression_fraction"]
        <= runtime.experiment.smoke_regression_improvement_factor * reference_regression_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "early": early,
        "convergence": convergence,
        "thresholds": {
            "f8_over_f0": runtime.experiment.smoke_f8_ratio_max,
            "early_median_q": runtime.experiment.smoke_q_max,
            "convergence_median_q": runtime.experiment.smoke_convergence_q_max,
            "reference_regression_fraction": reference_regression_fraction,
            "regression_improvement_factor": runtime.experiment.smoke_regression_improvement_factor,
        },
    }


class JointSeedTrainer:
    def __init__(
        self,
        experiment: V44Config,
        base: V40Config,
        seed: int,
        arms: tuple[str, ...],
        archive: dict[str, Tensor],
        validation_indices: Tensor,
        reference_regression_fraction: float,
        device: torch.device,
        output: Path,
        enforce_smoke: bool,
    ):
        self.experiment = experiment
        self.seed = seed
        self.arms = {
            arm: ArmRuntime(
                experiment,
                base,
                arm,
                seed,
                device,
                archive,
                torch.where(
                    archive["trajectory_id"]
                    < int(archive["trajectory_id"][validation_indices].min())
                )[0],
            )
            for arm in arms
        }
        self.archive = archive
        self.validation_indices = validation_indices
        self.reference_regression_fraction = reference_regression_fraction
        self.output = output
        self.enforce_smoke = enforce_smoke
        self.train_rows: list[dict[str, object]] = []
        self.regime_rows: list[dict[str, object]] = []
        self.sampler_rows: list[dict[str, object]] = []
        self.gates: dict[str, dict[str, object]] = {}
        self.started = time.time()

    def save_arm(self, arm: str, step: int, complete: bool, passed_smoke: bool) -> None:
        payload = {
            "experiment": "IFI v4.4 joint from scratch",
            "step": step,
            "complete": complete,
            "passed_smoke": passed_smoke,
            "gate": self.gates.get(arm),
            **self.arms[arm].state_dict(),
        }
        path = self.output / f"{arm}_checkpoint_{'final' if complete else 'progress'}.pt"
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def run(self) -> dict[str, object]:
        active = set(self.arms)
        passed_at_smoke: set[str] = set()
        for step in range(1, self.experiment.training_steps + 1):
            for arm in tuple(active):
                overall, regimes = self.arms[arm].train_step(step)
                if step % self.experiment.log_every == 0:
                    self.train_rows.append(overall)
                    self.regime_rows.extend(regimes)
                sampler = self.arms[arm].refresh_if_due(step)
                if sampler is not None:
                    self.sampler_rows.append(sampler)
            if step % self.experiment.log_every == 0:
                elapsed = time.time() - self.started
                write_csv(self.output / "train_metrics.csv", self.train_rows)
                write_csv(self.output / "regime_metrics.csv", self.regime_rows)
                write_csv(self.output / "sampler_coverage.csv", self.sampler_rows)
                status = {
                    "phase": "training",
                    "seed": self.seed,
                    "step": step,
                    "active_arms": sorted(active),
                    "steps_per_second": step / elapsed,
                    "eta_seconds": (self.experiment.training_steps - step) / (step / elapsed),
                }
                (self.output / "progress.json").write_text(
                    json.dumps(status, indent=2), encoding="utf-8"
                )
                print(json.dumps({"phase": "v44_training", **status}), flush=True)
            if step % self.experiment.checkpoint_every == 0:
                for arm in active:
                    self.save_arm(arm, step, complete=False, passed_smoke=arm in passed_at_smoke)
            if step == self.experiment.smoke_step:
                self.gates = {
                    arm: smoke_gate(
                        runtime,
                        self.archive,
                        self.validation_indices,
                        self.reference_regression_fraction,
                    )
                    for arm, runtime in self.arms.items()
                }
                passed_at_smoke = {arm for arm, gate in self.gates.items() if gate["passed"]}
                (self.output / "smoke_gate.json").write_text(
                    json.dumps(self.gates, indent=2), encoding="utf-8"
                )
                if self.enforce_smoke:
                    for arm in active - passed_at_smoke:
                        self.save_arm(arm, step, complete=True, passed_smoke=False)
                    active &= passed_at_smoke
                    if not active:
                        return {
                            "stopped_by_gate": True,
                            "step": step,
                            "gates": self.gates,
                            "completed_arms": [],
                        }
            if not active:
                break
        for arm in active:
            self.save_arm(
                arm,
                self.experiment.training_steps,
                complete=True,
                passed_smoke=arm in passed_at_smoke,
            )
        return {
            "stopped_by_gate": False,
            "step": self.experiment.training_steps,
            "gates": self.gates,
            "completed_arms": sorted(active),
        }
