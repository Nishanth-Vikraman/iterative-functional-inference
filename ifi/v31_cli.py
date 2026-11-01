from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .operators import build_function_system
from .v30_config import V30Config
from .v30_models import V30Policy, parameter_counts
from .v31_config import V31Config
from .v31_evaluation import METHODS, V31Evaluator
from .v31_report import write_report
from .v31_targets import (
    build_target_registry,
    reconstruct_training_support,
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


def _json_normalized(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _device(config: V31Config) -> torch.device:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("IFI v3.1 requires CUDA")
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def _source_paths(config: V31Config) -> dict[str, Path]:
    source = Path(config.source_run)
    paths = {
        "config": source / "config.resolved.json",
        "phase_a": source / "phase_a.pt",
        "phase_b": source / "phase_b.pt",
        "phase_a_metrics": source / "phase_a_metrics.json",
        "manifest": source / "experiment_manifest.json",
        "report": source / "REPORT.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing completed v3.0 artifact(s); retraining forbidden: {missing}")
    return paths


def _load_frozen(config: V31Config, device: torch.device):
    paths = _source_paths(config)
    train_config = V30Config.from_json(paths["config"])
    config.validate(train_config.phase_b.rollout_max, train_config.evaluation.extended_rollout_steps)
    phase_a = torch.load(paths["phase_a"], map_location=device, weights_only=False)
    phase_b = torch.load(paths["phase_b"], map_location=device, weights_only=False)
    if int(phase_b.get("step", -1)) != train_config.phase_b.steps:
        raise RuntimeError("v3.0 Phase-B checkpoint is not the completed canonical step")
    if _json_normalized(phase_b.get("config")) != _json_normalized(train_config.to_dict()):
        raise RuntimeError("v3.0 Phase-B checkpoint/config provenance mismatch")
    system, operator = build_function_system(train_config.operator, device)
    system.load_state_dict(phase_a["system"])
    operator.load_state_dict(phase_a["operator"])
    operator.freeze()
    policies = {
        name: V30Policy(
            name,
            train_config.phase_b.model_dimension,
            train_config.operator.action_dimension,
            max(train_config.tasks.support_sizes),
            train_config.evaluation.extended_rollout_steps,
        ).to(device)
        for name in METHODS
    }
    for name, policy in policies.items():
        policy.load_state_dict(phase_b["policies"][name], strict=True)
        policy.eval()
        policy.requires_grad_(False)
    system.eval()
    system.requires_grad_(False)
    return train_config, system, operator, policies, phase_a, phase_b, paths


def _locked_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if _json_normalized(current) != _json_normalized(payload):
            raise RuntimeError(f"Refusing to change preregistered artifact: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def prepare(config: V31Config, device: torch.device):
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_config, system, operator, policies, phase_a, phase_b, paths = _load_frozen(
        config, device
    )
    training_support = reconstruct_training_support(train_config, device)
    _locked_json(output / "training_support.json", training_support)
    targets = build_target_registry(training_support)
    registry_path = output / "target_registry.json"
    registry_payload = {
        "locked_before_evaluation": True,
        "selection_depends_on_model_performance": False,
        "targets": [target.to_dict() for target in targets],
    }
    _locked_json(registry_path, registry_payload)
    grid = torch.linspace(
        train_config.tasks.x_min, train_config.tasks.x_max, 1024, device=device
    ).unsqueeze(-1)
    target_validation = {
        target.target_id: {
            "finite": bool(torch.isfinite(target.evaluate(grid)).all()),
            "max_abs_on_domain": float(target.evaluate(grid).abs().max()),
        }
        for target in targets
    }
    if not all(item["finite"] for item in target_validation.values()):
        raise RuntimeError("A preregistered OOD target is numerically nonfinite")
    _locked_json(output / "target_validation.json", target_validation)
    source_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    actual_hashes = {name: _sha256(path) for name, path in paths.items()}
    manifest_hash_match = (
        source_manifest.get("phase_a", {}).get("sha256") == actual_hashes["phase_a"]
        and source_manifest.get("phase_b", {}).get("sha256") == actual_hashes["phase_b"]
    )
    frozen_objects = {"operator": operator, "system": system, **policies}
    requires_grad = {
        name: [parameter.requires_grad for parameter in module.parameters()]
        for name, module in frozen_objects.items()
    }
    all_frozen = all(not flag for values in requires_grad.values() for flag in values)
    integrity = {
        "passed": bool(all_frozen and operator.frozen and manifest_hash_match),
        "source_run_complete": True,
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": actual_hashes[name]}
            for name, path in paths.items()
        },
        "source_manifest_checkpoint_hashes_match": manifest_hash_match,
        "phase_b_step": int(phase_b["step"]),
        "expected_phase_b_step": train_config.phase_b.steps,
        "all_parameters_require_grad_false": all_frozen,
        "requires_grad_audit": requires_grad,
        "operator_frozen": operator.frozen,
        "optimizer_constructed": False,
        "optimizer_step_calls": 0,
        "adaptation": "torch.inference_mode forward-only functional composition",
        "teacher_information_enters_model_inputs": False,
        "model_input_schema": "raw current residual tokens, current observable z_D, own explicit residual/action history",
        "forbidden_model_inputs": [
            "target identity", "target coefficients", "oracle posterior", "oracle phi*",
            "query targets", "teacher metadata",
        ],
        "target_registry": {"path": str(registry_path.resolve()), "sha256": _sha256(registry_path)},
        "normalization_changed": False,
        "stopping_rule": "fixed rollout only; best iteration diagnostic",
        "history_overflow_protocol": {
            "checkpoint_capacity": config.history_capacity,
            "policy": "retain most recent checkpoint-capacity history entries",
            "reason": "evaluate F18 without changing frozen positional parameters",
        },
        "diagnostic_oracle": {
            "type": "exact finite Bayesian posterior within each locked bucket/severity",
            "used_for": ["oracle baseline", "desired functional update", "projection diagnostic"],
            "used_by_ifi_during_adaptation": False,
        },
        "parameter_counts": parameter_counts(policies),
    }
    _locked_json(output / "frozen_integrity.json", integrity)
    _locked_json(output / "config.resolved.json", config.to_dict())
    if not integrity["passed"]:
        raise RuntimeError("v3.1 no-retraining integrity gate failed")
    print(json.dumps({
        "phase": "v3_1_prepared",
        "targets": len(targets),
        "target_groups": len({(target.bucket, target.severity) for target in targets}),
        "integrity": True,
        "checkpoint_hashes": {key: actual_hashes[key] for key in ("phase_a", "phase_b")},
    }), flush=True)
    return train_config, system, operator, policies, targets, phase_a, paths


def evaluate(config: V31Config, device: torch.device, max_cells: int | None = None) -> None:
    _seed(3100)
    output = Path(config.output_dir)
    train_config, system, operator, policies, targets, phase_a, paths = prepare(config, device)
    evaluator = V31Evaluator(
        config, train_config, system, operator, policies, targets, output
    )
    torch.set_grad_enabled(False)
    result = evaluator.run(max_cells=max_cells)
    phase_a_reference = json.loads(paths["phase_a_metrics"].read_text(encoding="utf-8"))[
        "heldout_latents"
    ]
    report = write_report(
        output, config, train_config.phase_b.rollout_max, phase_a_reference
    )
    manifest = {
        "experiment": "IFI v3.1 frozen-checkpoint OOD ladder",
        "completed": True,
        "canonical": max_cells is None,
        "source_phase_a_sha256": _sha256(paths["phase_a"]),
        "source_phase_b_sha256": _sha256(paths["phase_b"]),
        "target_registry_sha256": _sha256(output / "target_registry.json"),
        "result": result,
        "report": report,
        "no_retraining": True,
        "pid": os.getpid(),
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"phase": "evaluation_v3_1_complete", **manifest}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v3.1 frozen-checkpoint OOD ladder")
    parser.add_argument("command", choices=("prepare", "smoke", "run"))
    parser.add_argument("--config", default="configs/v3_1_ood.json")
    parser.add_argument("--smoke-cells", type=int, default=7)
    args = parser.parse_args(argv)
    config = V31Config.from_json(args.config)
    device = _device(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(config, device)
        return
    lock = output / ("canonical_evaluation.lock" if args.command == "run" else "smoke.lock")
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "experiment": "IFI v3.1", "pid": os.getpid(), "started_unix": time.time(),
                "canonical": args.command == "run",
            }))
    except FileExistsError as error:
        raise RuntimeError(f"Refusing a second v3.1 {args.command} evaluation") from error
    evaluate(config, device, None if args.command == "run" else args.smoke_cells)


if __name__ == "__main__":
    main()
