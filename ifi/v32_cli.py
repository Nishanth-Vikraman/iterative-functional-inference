from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .operators import build_function_system
from .v30_config import V30Config
from .v30_models import V30Policy, parameter_counts
from .v30_tasks import BayesianLatentBenchmark
from .v31_targets import load_registry
from .v32_config import ARMS, V32Config
from .v32_evaluation import V32Evaluator
from .v32_losses import (
    LowRankFunctionalGeometry,
    functional_update,
    normalized_functional_loss,
    normalized_mahalanobis_loss,
    teacher_geometry,
)
from .v32_report import write_v32_report
from .v32_training import (
    LossCalibration,
    V32Trainer,
    atomic_training_checkpoint,
    calibrate_losses,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: V32Config) -> torch.device:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("IFI v3.2 requires CUDA")
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def _locked_json(path: Path, payload: dict) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if json.dumps(current, sort_keys=True) != json.dumps(payload, sort_keys=True):
            raise RuntimeError(f"Refusing to modify locked v3.2 artifact: {path}")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_paths(config: V32Config) -> dict[str, Path]:
    v30 = Path(config.source_v30_run)
    v31 = Path(config.source_v31_dir)
    paths = {
        "v30_config": v30 / "config.resolved.json",
        "phase_a": v30 / "phase_a.pt",
        "phase_b_historical": v30 / "phase_b.pt",
        "teacher_source": Path("ifi/teachers/bayesian.py"),
        "controller_source": Path("ifi/v30_models.py"),
        "v31_config": v31 / "config.resolved.json",
        "v31_training_support": v31 / "training_support.json",
        "v31_integrity": v31 / "frozen_integrity.json",
        "v31_target_registry": v31 / "target_registry.json",
        "v31_evaluation_records": v31 / "evaluation_records.csv",
        "v31_projection_records": v31 / "projection_records.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing authoritative v3.0/v3.1 artifact: {missing}")
    return paths


def _load_operator(config: V32Config, device: torch.device):
    paths = _source_paths(config)
    train_config = V30Config.from_json(paths["v30_config"])
    phase_a = torch.load(paths["phase_a"], map_location=device, weights_only=False)
    system, operator = build_function_system(train_config.operator, device)
    system.load_state_dict(phase_a["system"])
    operator.load_state_dict(phase_a["operator"])
    operator.freeze()
    system.eval()
    system.requires_grad_(False)
    if config.functional_grid_size != json.loads(paths["v31_config"].read_text())["projection_size"] and config.output_dir == "artifacts/v3_2":
        raise RuntimeError("Canonical v3.2 functional grid must exactly match v3.1 projection grid")
    if config.evaluation_rollout != json.loads(paths["v31_config"].read_text())["rollout_steps"]:
        raise RuntimeError("v3.2 must reuse the exact v3.1 rollout horizon")
    return train_config, system, operator, paths


def _new_policies(train_config: V30Config, seed: int, device: torch.device):
    _seed(seed + 200)
    template = V30Policy(
        "query_history",
        train_config.phase_b.model_dimension,
        train_config.operator.action_dimension,
        max(train_config.tasks.support_sizes),
        train_config.evaluation.extended_rollout_steps,
    ).to(device)
    initial = copy.deepcopy(template.state_dict())
    policies = {}
    for arm in ARMS:
        policy = V30Policy(
            "query_history",
            train_config.phase_b.model_dimension,
            train_config.operator.action_dimension,
            max(train_config.tasks.support_sizes),
            train_config.evaluation.extended_rollout_steps,
        ).to(device)
        policy.load_state_dict(initial)
        policies[arm] = policy
    return policies


def _recovered_protocol(config, train_config, paths, operator, targets):
    v31_config = json.loads(paths["v31_config"].read_text(encoding="utf-8"))
    v31_integrity = json.loads(paths["v31_integrity"].read_text(encoding="utf-8"))
    registry_hash = _sha256(paths["v31_target_registry"])
    if registry_hash != v31_integrity["target_registry"]["sha256"]:
        raise RuntimeError("v3.1 OOD registry hash no longer matches its frozen integrity artifact")
    return {
        "recovered_from_repository_not_prompt": True,
        "source_hashes": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "phase_a": {
            "checkpoint": str(paths["phase_a"].resolve()),
            "operator_frozen": operator.frozen,
            "operator": train_config.operator.__dict__,
        },
        "teacher": {
            "class": "BayesianLatentSystemTeacher",
            "finite_hypotheses": 32,
            "state_for_mahalanobis": "explicit finite log-posterior coordinates",
            "definition_unchanged": True,
        },
        "benchmark": train_config.tasks.__dict__,
        "latent_split": {
            "definition": "v30_tasks.latent_split_indices checkerboard",
            "training_support_artifact": str(paths["v31_training_support"].resolve()),
        },
        "phase_b_authoritative": train_config.phase_b.__dict__,
        "controller": {
            "class": "V30Policy(query_history) / QueryHistoryController",
            "parameter_count": 185420,
            "architecture_unchanged": True,
            "history_capacity": config.history_capacity,
        },
        "fresh_training": {
            "seeds": list(config.training_seeds),
            "same_initial_state_within_seed_across_arms": True,
            "optimizer": "Adam",
            "learning_rate": train_config.phase_b.learning_rate,
            "source_steps": train_config.phase_b.steps,
            "steps": config.training_steps,
            "budget_override_user_requested": config.training_steps != train_config.phase_b.steps,
            "batch_size": train_config.phase_b.batch_size,
            "rollout_max": train_config.phase_b.rollout_max,
            "variable_rollout": train_config.phase_b.variable_rollout,
            "gradient_clip": train_config.phase_b.gradient_clip,
            "learning_rate_schedule": "none",
        },
        "objective_arms": {
            "phi_mse": "pure mean squared phi-coordinate imitation as explicitly specified for v3.2",
            "normalized_functional": "functional update MSE / (teacher update energy + locked epsilon)",
            "normalized_mahalanobis": "teacher-induced functional Mahalanobis error / teacher-update Mahalanobis energy",
            "no_auxiliary_or_mixed_loss": True,
        },
        "historical_objective_discrepancy": {
            "v30_config_action_metric": train_config.phase_b.action_metric,
            "v30_lambda_functional": train_config.phase_b.lambda_functional,
            "note": "Frozen old v3.1 query-history remains the historical reference; new Arm A follows the user-locked pure phi-MSE equation and does not silently preserve v3.0's auxiliary term.",
        },
        "functional_grid": {
            "points": config.functional_grid_size,
            "construction": "deterministic linspace over persisted x_min/x_max",
            "matches_v31_projection_grid": config.functional_grid_size == v31_config["projection_size"],
        },
        "v31_ood_fixtures": {
            "registry_path": str(paths["v31_target_registry"].resolve()),
            "registry_sha256": registry_hash,
            "loaded_directly_not_regenerated": True,
            "target_count": len(targets),
            "groups": sorted({f"{target.bucket}:{target.severity}" for target in targets}),
            "episodes_per_target": v31_config["episodes_per_target"],
            "support_sizes": v31_config["support_sizes"],
            "evaluation_seeds": v31_config["seeds"],
            "query_size": v31_config["query_size"],
            "rollout": v31_config["rollout_steps"],
            "history_overflow": v31_integrity["history_overflow_protocol"],
        },
        "rollout_extension": {
            "evaluated": 18,
            "F24_F32_not_run": "v3.1 fixture protocol is locked to 18 evidence batches and was already unstable; extension would modify fixtures",
        },
        "normalization": {
            "benchmark": "none",
            "residual_tokens": "[x,y,F,e,abs(e),e^2]",
            "functional_losses": "per-grid mean; locked training-diagnostic denominators",
        },
    }


def _loss_audit(operator, system, calibration: LossCalibration, device):
    grid = torch.linspace(-2, 2, 32, device=device).unsqueeze(-1)
    current = torch.zeros_like(grid)
    latent = system.encode(grid)
    audit_generator = torch.Generator(device=device).manual_seed(3204)
    action = (
        torch.randn(
            operator.action_dimension,
            generator=audit_generator,
            device=device,
        )
        * 0.05
    )
    action.requires_grad_()
    candidate = system.decode(operator(latent, action))
    realized = functional_update(candidate, current)
    direct = candidate - current
    target = torch.linspace(-0.5, 0.5, grid.shape[0], device=device).unsqueeze(-1)
    nf, _, _ = normalized_functional_loss(
        realized, target, calibration.normalized_functional_epsilon
    )
    nf.backward(retain_graph=True)
    action_gradient = float(action.grad.norm()) if action.grad is not None else 0.0
    operator_has_gradient = any(parameter.grad is not None for parameter in operator.parameters())
    zero_nf = float(normalized_functional_loss(target, target, 1e-10)[0])
    identity = torch.eye(2, device=device)
    synthetic_geometry = LowRankFunctionalGeometry(
        torch.tensor([10.0, 0.1], device=device), identity, 0.01, 2, 2.0, 100.0, 1001.0
    )
    synthetic_target = torch.tensor([0.5, -0.5], device=device)
    zero_nm = float(normalized_mahalanobis_loss(
        synthetic_target, synthetic_target, synthetic_geometry, 1e-10
    )[0])
    high_error = synthetic_geometry.quadratic(torch.tensor([1.0, 0.0], device=device))
    low_error = synthetic_geometry.quadratic(torch.tensor([0.0, 1.0], device=device))
    first = normalized_functional_loss(target * 1.2, target, 1e-12)[0]
    second = normalized_functional_loss(target * 3.6, target * 3.0, 1e-12)[0]
    perturbations = torch.tensor(
        [[1.0, 0.0, 0.5], [-1.0, 0.0, -0.5], [0.5, 0.2, 0.0], [-0.5, -0.2, 0.0]],
        device=device,
    )
    geometry = teacher_geometry(
        perturbations,
        calibration.covariance_regularization,
        1e-7,
    )
    covariance = torch.cov(perturbations.transpose(0, 1))
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    checks = {
        "operator_frozen_no_gradient": operator.frozen and not operator_has_gradient,
        "action_gradient_nonzero": action_gradient > 0 and math.isfinite(action_gradient),
        "functional_update_exact": bool(torch.allclose(realized, direct, atol=1e-7)),
        "normalized_functional_zero_error": zero_nf < 1e-10,
        "normalized_mahalanobis_zero_error": zero_nm < 1e-10,
        "normalized_scale_invariance": bool(torch.allclose(first, second, rtol=1e-5, atol=1e-7)),
        "mahalanobis_low_variance_penalized_more": float(low_error) > float(high_error),
        "covariance_psd": float(covariance_eigenvalues.min()) >= -1e-6,
        "regularized_geometry_finite_positive": geometry.regularization > 0 and math.isfinite(geometry.condition_after),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "action_gradient_norm": action_gradient,
            "high_variance_direction_penalty": float(high_error),
            "low_variance_direction_penalty": float(low_error),
            "covariance_min_eigenvalue": float(covariance_eigenvalues.min()),
            "geometry_rank": geometry.raw_rank,
            "geometry_effective_rank": geometry.effective_rank,
            "condition_before": geometry.condition_before,
            "condition_after": geometry.condition_after,
        },
        "calibration": calibration.to_dict(),
    }


def prepare(config: V32Config, device: torch.device):
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_config, system, operator, paths = _load_operator(config, device)
    targets = load_registry(paths["v31_target_registry"])
    protocol = _recovered_protocol(config, train_config, paths, operator, targets)
    _locked_json(output / "recovered_protocol.json", protocol)
    calibration_path = output / "loss_calibration.json"
    if calibration_path.exists():
        calibration = LossCalibration.from_dict(json.loads(calibration_path.read_text()))
    else:
        _seed(3201)
        benchmark = BayesianLatentBenchmark(train_config.tasks, 3202, device)
        grid = torch.linspace(
            train_config.tasks.x_min,
            train_config.tasks.x_max,
            config.functional_grid_size,
            device=device,
        ).unsqueeze(-1)
        calibration = calibrate_losses(config, train_config, benchmark, grid, 3203)
        _locked_json(calibration_path, calibration.to_dict())
    audit = _loss_audit(operator, system, calibration, device)
    _locked_json(output / "loss_audit.json", audit)
    _locked_json(output / "config.resolved.json", config.to_dict())
    if not audit["passed"]:
        raise RuntimeError("v3.2 loss/gradient audit failed; expensive run forbidden")
    print(json.dumps({
        "phase": "v3_2_prepared",
        "audit_passed": True,
        "targets_reused": len(targets),
        "target_registry_sha256": _sha256(paths["v31_target_registry"]),
        "calibration": calibration.to_dict(),
    }), flush=True)
    return train_config, system, operator, targets, calibration, paths


def _train_seed(config, train_config, system, operator, calibration, seed, steps):
    policies = _new_policies(train_config, seed, next(operator.parameters()).device)
    if len(set(parameter_counts({arm: policy})[arm] for arm, policy in policies.items())) != 1:
        raise RuntimeError("v3.2 arms have mismatched parameter counts")
    benchmark = BayesianLatentBenchmark(train_config.tasks, seed + 201, next(operator.parameters()).device)
    trainer = V32Trainer(
        config, train_config, system, operator, benchmark, policies, calibration, seed
    )
    output = Path(config.output_dir)
    completed_path = output / f"training_seed_{seed}.pt"
    if completed_path.exists():
        completed = torch.load(completed_path, map_location="cpu", weights_only=False)
        if completed.get("complete") and int(completed["step"]) == steps:
            return list(completed["history"])
        raise RuntimeError(f"Incompatible completed seed checkpoint: {completed_path}")

    start_step = 0
    history = []
    resume_rng_exact = True
    progress_path = output / f"training_seed_{seed}_progress.pt"
    if progress_path.exists():
        resumed = torch.load(progress_path, map_location=next(operator.parameters()).device, weights_only=False)
        if int(resumed["seed"]) != seed or int(resumed["step"]) >= steps:
            raise RuntimeError(f"Incompatible partial seed checkpoint: {progress_path}")
        for arm, policy in policies.items():
            policy.load_state_dict(resumed["arms"][arm])
            trainer.optimizers[arm].load_state_dict(resumed["optimizers"][arm])
        start_step = int(resumed["step"])
        history = list(resumed["history"])
        rng = resumed.get("rng")
        resume_rng_exact = rng is not None
        if rng is not None:
            trainer.episode_rng.setstate(rng["episode_rng"])
            trainer.geometry_generator.set_state(rng["geometry_generator"])
            trainer.benchmark.generator.set_state(rng["benchmark_generator"])
            trainer.benchmark._next_episode = int(rng["benchmark_next_episode"])
            torch.set_rng_state(rng["torch_cpu"].cpu())
            if torch.cuda.is_available() and rng["torch_cuda"]:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
        (output / f"resume_provenance_seed_{seed}.json").write_text(
            json.dumps({
                "resumed_from": str(progress_path.resolve()),
                "resumed_step": start_step,
                "target_total_steps": steps,
                "model_and_optimizer_restored": True,
                "rng_state_restored_exactly": resume_rng_exact,
                "rng_note": (
                    "Exact RNG state restored from checkpoint."
                    if resume_rng_exact
                    else "Legacy checkpoint predated RNG-state persistence; subsequent sampling restarted from the seed while model and Adam states were retained."
                ),
            }, indent=2),
            encoding="utf-8",
        )

    def checkpoint(step, history):
        atomic_training_checkpoint(
            output / f"training_seed_{seed}_progress.pt",
            step, policies, trainer, history, seed, False,
        )
        (output / f"training_history_seed_{seed}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        elapsed = history[-1]["elapsed_seconds"]
        rate = step / max(elapsed, 1e-9)
        progress = {
            "phase": "training",
            "seed": seed,
            "step": step,
            "steps_per_seed": steps,
            "seed_index": list(config.training_seeds).index(seed) + 1,
            "total_seeds": len(config.training_seeds),
            "steps_per_second": rate,
            "current_seed_eta_seconds": (steps - step) / rate,
            "resumed_from_step": start_step,
            "resume_rng_exact": resume_rng_exact,
        }
        temporary = output / "progress.json.tmp"
        temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        temporary.replace(output / "progress.json")

    history = trainer.run(
        steps,
        on_log=checkpoint,
        start_step=start_step,
        history=history,
    )
    atomic_training_checkpoint(
        output / f"training_seed_{seed}.pt",
        steps, policies, trainer, history, seed, True,
    )
    return history


def run_experiment(config: V32Config, device: torch.device, smoke: bool = False):
    train_config, system, operator, targets, calibration, paths = prepare(config, device)
    steps = 2 if smoke else config.training_steps
    for seed in config.training_seeds:
        _train_seed(config, train_config, system, operator, calibration, seed, steps)
    evaluator = V32Evaluator(
        config, train_config, system, operator, targets, calibration, config.output_dir
    )

    def policy_loader(seed):
        policies = _new_policies(train_config, seed, device)
        checkpoint = torch.load(
            Path(config.output_dir) / f"training_seed_{seed}.pt",
            map_location=device,
            weights_only=False,
        )
        for arm, policy in policies.items():
            policy.load_state_dict(checkpoint["arms"][arm])
            policy.eval()
            policy.requires_grad_(False)
        return policies

    evaluation = evaluator.run(policy_loader, max_cells=7 if smoke else None)
    report = write_v32_report(config, config.output_dir, calibration)
    manifest = {
        "experiment": "IFI v3.2 function-space Phase-B objectives",
        "complete": True,
        "smoke": smoke,
        "pid": os.getpid(),
        "source_phase_a_sha256": _sha256(paths["phase_a"]),
        "source_v31_registry_sha256": _sha256(paths["v31_target_registry"]),
        "training_steps_per_seed": steps,
        "training_seeds": list(config.training_seeds),
        "evaluation": evaluation,
        "report": report,
    }
    (Path(config.output_dir) / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"phase": "v3_2_complete", **manifest}), flush=True)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="IFI v3.2 function-space losses")
    parser.add_argument("command", choices=("prepare", "smoke", "run"))
    parser.add_argument("--config", default="configs/v3_2_functional.json")
    args = parser.parse_args(argv)
    config = V32Config.from_json(args.config)
    device = _device(config)
    if args.command == "prepare":
        prepare(config, device)
        return
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ("canonical_run.lock" if args.command == "run" else "smoke_run.lock")
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
    except FileExistsError as error:
        raise RuntimeError(f"Refusing a second v3.2 {args.command}") from error
    run_experiment(config, device, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
