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
from .v23_evaluation import V23Evaluator, make_v23_plots, write_v23_outputs
from .v23_models import ForwardTrainedPolicy, parameter_counts
from .v23_phase_b import ForwardCreditTrainer
from .v23_report import write_v23_report


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(config):
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("IFI v2.3 canonical run requires CUDA")
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def _load_phase_a(path, config, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    system, operator = build_function_system(config.operator, device)
    system.load_state_dict(payload["system"])
    operator.load_state_dict(payload["operator"])
    operator.freeze()
    return system, operator


def _policy(config, device):
    return ForwardTrainedPolicy(
        config.phase_b.latent_dimension,
        config.operator.action_dimension,
        max(config.tasks.support_sizes),
    ).to(device)


def _save(path, config, policy, trainer, step, history):
    payload = {
        "config": config.to_dict(), "step": step, "policy": policy.state_dict(),
        "optimizer": trainer.optimizer.state_dict(), "history": history,
        "rng": rng_state_payload(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, output, phase_a, device):
    _seed(config.seed + 1000)
    system, operator = _load_phase_a(phase_a, config, device)
    policy = _policy(config, device)
    counts = parameter_counts(policy)
    (output / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    benchmark = SyntheticFunctionBenchmark(config.tasks, config.seed + 1, device)
    trainer = ForwardCreditTrainer(config, system, operator, benchmark, policy)

    def checkpoint(step, history):
        _save(output / "phase_b_progress.pt", config, policy, trainer, step, history)
        (output / "phase_b_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    history = trainer.run(on_log=checkpoint)
    _save(output / "phase_b.pt", config, policy, trainer, config.phase_b.steps, history)
    print(json.dumps({"phase": "B_v2_3_complete", "parameter_counts": counts}), flush=True)


def evaluate(config, output, phase_a, v21, v22, historical, device):
    _seed(config.seed + 2000)
    system, operator = _load_phase_a(phase_a, config, device)
    policy = _policy(config, device)
    checkpoint = torch.load(output / "phase_b.pt", map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    policy.set_training_block("phi")
    policy.eval()
    evaluator = V23Evaluator(config, system, operator, policy, output)
    records, residual, state, trajectories, diversity = evaluator.run()
    write_v23_outputs(output, records, residual, state, trajectories, diversity)
    plots = make_v23_plots(output, v21, v22)
    write_v23_report(
        output, v21, v22, historical, config,
        parameter_counts(policy), checkpoint["history"],
    )
    print(json.dumps({
        "phase": "evaluation_v2_3_complete", "records": len(records),
        "residual_interventions": len(residual), "state_interventions": len(state),
        "plots": [str(path) for path in plots],
    }), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="IFI v2.3 forward-trained state")
    parser.add_argument("command", choices=("phase-b", "evaluate", "all"))
    parser.add_argument("--config", default="configs/v2_3_forward.json")
    parser.add_argument("--phase-a-checkpoint", default="runs/canonical/phase_a.pt")
    parser.add_argument("--v21-run", default="runs/v2_1_direct")
    parser.add_argument("--v22-run", default="runs/v2_2_predictive")
    parser.add_argument("--historical-run", default="runs/canonical")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    device = _device(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase_a, v21, v22, historical = map(Path, (
        args.phase_a_checkpoint, args.v21_run, args.v22_run, args.historical_run
    ))
    if args.command in {"phase-b", "all"}:
        try:
            with (output / "canonical_run.lock").open("x", encoding="utf-8") as handle:
                handle.write("IFI v2.3 canonical one-shot lock\n")
        except FileExistsError as error:
            raise RuntimeError("Refusing a second v2.3 training run") from error
    (output / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    manifest = {
        "experiment": "IFI v2.3 forward-trained recurrent learning state",
        "phase_a": {"path": str(phase_a.resolve()), "sha256": _sha(phase_a)},
        "v2_1": {"path": str((v21 / "phase_b.pt").resolve()), "sha256": _sha(v21 / "phase_b.pt")},
        "v2_2": {"path": str((v22 / "phase_b.pt").resolve()), "sha256": _sha(v22 / "phase_b.pt")},
        "historical": {"path": str((historical / "phase_b.pt").resolve()), "sha256": _sha(historical / "phase_b.pt")},
        "block_length": config.forward_state.block_length,
        "total_optimizer_steps": config.phase_b.steps,
        "primary_rollout": config.evaluation.rollout_steps,
        "extended_rollout": config.forward_state.extended_rollout_steps,
        "semantic_residual_encoder": False, "separate_state_updater": False,
        "phase_a_frozen": True, "one_shot": True,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.command in {"phase-b", "all"}:
        train(config, output, phase_a, device)
    if args.command in {"evaluate", "all"}:
        evaluate(config, output, phase_a, v21, v22, historical, device)


if __name__ == "__main__":
    main()

