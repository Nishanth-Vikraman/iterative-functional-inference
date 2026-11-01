from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .operators import build_function_system
from .tasks import SyntheticFunctionBenchmark
from .v21_evaluation import V21Evaluator, make_v21_plots, write_v21_outputs
from .v21_models import DirectIFIPolicy, trainable_parameter_count
from .v21_phase_b import DirectPhaseBTrainer, rng_state_payload
from .v21_report import write_v21_report


VARIANTS = ("direct_current", "direct_history", "direct_recurrent")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(config: ExperimentConfig) -> torch.device:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("IFI v2.1 canonical configuration requires CUDA")
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def _load_frozen_phase_a(path: Path, config: ExperimentConfig, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    system, operator = build_function_system(config.operator, device)
    system.load_state_dict(payload["system"])
    operator.load_state_dict(payload["operator"])
    operator.freeze()
    return system, operator


def _policies(config: ExperimentConfig, device: torch.device) -> dict[str, DirectIFIPolicy]:
    maximum_support = max(config.tasks.support_sizes)
    return {
        name: DirectIFIPolicy(
            name, config.phase_b.latent_dimension,
            config.operator.action_dimension, maximum_support,
        ).to(device)
        for name in VARIANTS
    }


def _save_checkpoint(
    path: Path,
    config: ExperimentConfig,
    policies: dict[str, DirectIFIPolicy],
    trainer: DirectPhaseBTrainer,
    step: int,
    history: list[dict[str, float]],
) -> None:
    payload = {
        "config": config.to_dict(),
        "step": step,
        "policies": {name: policy.state_dict() for name, policy in policies.items()},
        "optimizer": trainer.optimizer.state_dict(),
        "history": history,
        "rng": rng_state_payload(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_policies(
    path: Path,
    policies: dict[str, DirectIFIPolicy],
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    for name, policy in policies.items():
        policy.load_state_dict(payload["policies"][name])
    return payload


def train(
    config: ExperimentConfig,
    output: Path,
    phase_a_checkpoint: Path,
    device: torch.device,
):
    _seed_everything(config.seed + 1000)
    system, operator = _load_frozen_phase_a(phase_a_checkpoint, config, device)
    policies = _policies(config, device)
    counts = {name: trainable_parameter_count(policy) for name, policy in policies.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"A/B/C parameter counts are not matched: {counts}")
    (output / "parameter_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    benchmark = SyntheticFunctionBenchmark(config.tasks, config.seed + 1, device)
    trainer = DirectPhaseBTrainer(config, system, operator, benchmark, policies)

    def save_progress(step: int, history: list[dict[str, float]]) -> None:
        _save_checkpoint(
            output / "phase_b_progress.pt", config, policies, trainer, step, history
        )
        (output / "phase_b_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    history = trainer.run(on_log=save_progress)
    _save_checkpoint(
        output / "phase_b.pt", config, policies, trainer,
        config.phase_b.steps, history,
    )
    print(json.dumps({"phase": "B_v2_1_complete", "parameter_counts": counts}), flush=True)
    return system, operator, policies, history, counts


def evaluate(
    config: ExperimentConfig,
    output: Path,
    phase_a_checkpoint: Path,
    historical_dir: Path,
    device: torch.device,
) -> None:
    _seed_everything(config.seed + 2000)
    system, operator = _load_frozen_phase_a(phase_a_checkpoint, config, device)
    policies = _policies(config, device)
    checkpoint = _load_policies(output / "phase_b.pt", policies, device)
    counts = {name: trainable_parameter_count(policy) for name, policy in policies.items()}
    evaluator = V21Evaluator(config, system, operator, policies, output)
    records, interventions, trajectories = evaluator.run()
    write_v21_outputs(output, records, interventions, trajectories)
    plots = make_v21_plots(output)
    history = checkpoint["history"]
    write_v21_report(output, historical_dir, config, counts, history)  # type: ignore[arg-type]
    print(json.dumps({
        "phase": "evaluation_v2_1_complete",
        "records": len(records),
        "interventions": len(interventions),
        "plots": [str(path) for path in plots],
    }), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v2.1 direct residual experiment")
    parser.add_argument("command", choices=("phase-b", "evaluate", "all"))
    parser.add_argument("--config", default="configs/v2_1_direct.json")
    parser.add_argument("--phase-a-checkpoint", default="runs/canonical/phase_a.pt")
    parser.add_argument("--historical-run", default="runs/canonical")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    device = _device(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    phase_a_checkpoint = Path(args.phase_a_checkpoint)
    historical_dir = Path(args.historical_run)
    manifest = {
        "experiment": "IFI v2.1 direct residual evidence + recurrent learning-state ablation",
        "phase_a_checkpoint": str(phase_a_checkpoint.resolve()),
        "phase_a_sha256": _sha256(phase_a_checkpoint),
        "historical_phase_b_checkpoint": str((historical_dir / "phase_b.pt").resolve()),
        "historical_phase_b_sha256": _sha256(historical_dir / "phase_b.pt"),
        "variants": list(VARIANTS),
        "phase_a_frozen": True,
        "teacher_unchanged": True,
        "projection_unchanged": True,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.command in {"phase-b", "all"}:
        train(config, output, phase_a_checkpoint, device)
    if args.command in {"evaluate", "all"}:
        evaluate(config, output, phase_a_checkpoint, historical_dir, device)


if __name__ == "__main__":
    main()
