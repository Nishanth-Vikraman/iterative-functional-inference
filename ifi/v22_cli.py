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
from .v21_phase_b import rng_state_payload
from .v22_evaluation import V22Evaluator, make_v22_plots, write_v22_outputs
from .v22_models import PredictiveRecurrentPolicy, parameter_counts
from .v22_phase_b import PredictiveStateTrainer
from .v22_report import write_v22_report


def _seed(seed: int) -> None:
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
        raise RuntimeError("IFI v2.2 canonical run requires CUDA")
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def _load_phase_a(path: Path, config: ExperimentConfig, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    system, operator = build_function_system(config.operator, device)
    system.load_state_dict(payload["system"])
    operator.load_state_dict(payload["operator"])
    operator.freeze()
    return system, operator


def _policy(config: ExperimentConfig, device: torch.device) -> PredictiveRecurrentPolicy:
    return PredictiveRecurrentPolicy(
        config.phase_b.latent_dimension,
        config.operator.action_dimension,
        max(config.tasks.support_sizes),
        config.predictive_state.updater_layers,
    ).to(device)


def _save(path, config, policy, trainer, step, history):
    payload = {
        "config": config.to_dict(), "step": step,
        "policy": policy.state_dict(), "optimizer": trainer.optimizer.state_dict(),
        "history": history, "rng": rng_state_payload(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, output, phase_a_path, device):
    _seed(config.seed + 1000)
    system, operator = _load_phase_a(phase_a_path, config, device)
    policy = _policy(config, device)
    counts = parameter_counts(policy)
    (output / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    benchmark = SyntheticFunctionBenchmark(config.tasks, config.seed + 1, device)
    trainer = PredictiveStateTrainer(config, system, operator, benchmark, policy)

    def checkpoint(step, history):
        _save(output / "phase_b_progress.pt", config, policy, trainer, step, history)
        (output / "phase_b_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    history = trainer.run(on_log=checkpoint)
    _save(output / "phase_b.pt", config, policy, trainer, config.phase_b.steps, history)
    print(json.dumps({"phase": "B_v2_2_complete", "parameter_counts": counts}), flush=True)
    return system, operator, policy, history, counts


def evaluate(config, output, phase_a_path, v21_dir, historical_dir, device):
    _seed(config.seed + 2000)
    system, operator = _load_phase_a(phase_a_path, config, device)
    policy = _policy(config, device)
    checkpoint = torch.load(output / "phase_b.pt", map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    evaluator = V22Evaluator(config, system, operator, policy, output)
    records, residual, state, trajectories, diversity = evaluator.run()
    write_v22_outputs(output, records, residual, state, trajectories, diversity)
    plots = make_v22_plots(output, v21_dir)
    counts = parameter_counts(policy)
    write_v22_report(
        output, v21_dir, historical_dir, config, counts, checkpoint["history"]
    )
    print(json.dumps({
        "phase": "evaluation_v2_2_complete", "records": len(records),
        "residual_interventions": len(residual), "state_interventions": len(state),
        "plots": [str(path) for path in plots],
    }), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v2.2 predictive recurrent state")
    parser.add_argument("command", choices=("phase-b", "evaluate", "all"))
    parser.add_argument("--config", default="configs/v2_2_predictive.json")
    parser.add_argument("--phase-a-checkpoint", default="runs/canonical/phase_a.pt")
    parser.add_argument("--v21-run", default="runs/v2_1_direct")
    parser.add_argument("--historical-run", default="runs/canonical")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    device = _device(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase_a = Path(args.phase_a_checkpoint)
    v21_dir = Path(args.v21_run)
    historical_dir = Path(args.historical_run)

    if args.command in {"phase-b", "all"}:
        lock = output / "canonical_run.lock"
        try:
            with lock.open("x", encoding="utf-8") as handle:
                handle.write("IFI v2.2 canonical run: one-shot lock\n")
        except FileExistsError as error:
            raise RuntimeError("Refusing a second v2.2 training run") from error

    (output / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    manifest = {
        "experiment": "IFI v2.2 predictive recurrent learning state",
        "phase_a": {"path": str(phase_a.resolve()), "sha256": _sha256(phase_a)},
        "v2_1": {"path": str((v21_dir / "phase_b.pt").resolve()),
                  "sha256": _sha256(v21_dir / "phase_b.pt")},
        "historical": {"path": str((historical_dir / "phase_b.pt").resolve()),
                       "sha256": _sha256(historical_dir / "phase_b.pt")},
        "fixed_residual_signature": "mean,std,min,max per six raw token fields",
        "fixed_alignment": "seeded column-normalized random projection",
        "semantic_residual_encoder": False,
        "phase_a_frozen": True,
        "one_shot": True,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.command in {"phase-b", "all"}:
        train(config, output, phase_a, device)
    if args.command in {"evaluate", "all"}:
        evaluate(config, output, phase_a, v21_dir, historical_dir, device)


if __name__ == "__main__":
    main()
