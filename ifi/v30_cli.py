from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .operators import build_function_system
from .v30_config import V30Config
from .v30_evaluation import V30Evaluator, make_v30_plots, write_v30_outputs
from .v30_models import V30Policy, parameter_counts
from .v30_phase_a import V30PhaseATrainer, save_v30_phase_a
from .v30_phase_b import VARIANTS, V30PhaseBTrainer
from .v30_report import write_v30_report
from .v30_tasks import BayesianLatentBenchmark


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(config: V30Config) -> torch.device:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("IFI v3.0 canonical run requires CUDA")
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def _atomic_save(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _policies(config: V30Config, device: torch.device):
    return {
        name: V30Policy(
            name,
            config.phase_b.model_dimension,
            config.operator.action_dimension,
            max(config.tasks.support_sizes),
            config.evaluation.extended_rollout_steps,
        ).to(device)
        for name in VARIANTS
    }


def phase_a(config: V30Config, output: Path, device: torch.device):
    _seed(config.seed + 100)
    system, operator = build_function_system(config.operator, device)
    benchmark = BayesianLatentBenchmark(config.tasks, config.seed + 101, device)
    trainer = V30PhaseATrainer(config, system, operator, benchmark)

    def checkpoint(step, history):
        _atomic_save(output / "phase_a_progress.pt", {
            "step": step,
            "system": system.state_dict(),
            "operator": operator.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "history": history,
        })
        (output / "phase_a_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    history = trainer.run(on_log=checkpoint)
    metrics = {
        split: trainer.evaluate(split)
        for split in ("train_latents", "heldout_latents")
    }
    save_v30_phase_a(output / "phase_a.pt", config, system, operator, metrics)
    metric_payload = {key: asdict(value) for key, value in metrics.items()}
    (output / "phase_a_metrics.json").write_text(
        json.dumps(metric_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "phase": "A_v3_0_complete", "metrics": metric_payload,
    }), flush=True)
    return history


def _load_phase_a(config: V30Config, output: Path, device: torch.device):
    checkpoint = torch.load(output / "phase_a.pt", map_location=device, weights_only=False)
    system, operator = build_function_system(config.operator, device)
    system.load_state_dict(checkpoint["system"])
    operator.load_state_dict(checkpoint["operator"])
    operator.freeze()
    return system, operator, checkpoint["metrics"]


def phase_b(config: V30Config, output: Path, device: torch.device):
    _seed(config.seed + 200)
    system, operator, _ = _load_phase_a(config, output, device)
    policies = _policies(config, device)
    counts = parameter_counts(policies)
    (output / "parameter_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    benchmark = BayesianLatentBenchmark(config.tasks, config.seed + 201, device)
    trainer = V30PhaseBTrainer(config, system, operator, benchmark, policies)

    def checkpoint(step, history):
        _atomic_save(output / "phase_b_progress.pt", {
            "step": step,
            "policies": {name: policy.state_dict() for name, policy in policies.items()},
            "optimizer": trainer.optimizer.state_dict(),
            "history": history,
        })
        (output / "phase_b_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    history = trainer.run(on_log=checkpoint)
    _atomic_save(output / "phase_b.pt", {
        "step": config.phase_b.steps,
        "config": config.to_dict(),
        "policies": {name: policy.state_dict() for name, policy in policies.items()},
        "optimizer": trainer.optimizer.state_dict(),
        "history": history,
    })
    print(json.dumps({
        "phase": "B_v3_0_complete", "parameter_counts": counts,
    }), flush=True)
    return history


def evaluate(config: V30Config, output: Path, device: torch.device):
    _seed(config.seed + 300)
    system, operator, phase_a_metrics = _load_phase_a(config, output, device)
    policies = _policies(config, device)
    checkpoint = torch.load(output / "phase_b.pt", map_location=device, weights_only=False)
    for name, policy in policies.items():
        policy.load_state_dict(checkpoint["policies"][name])
        policy.eval()
    evaluator = V30Evaluator(config, system, operator, policies, output)
    records, residual, history = evaluator.run()
    write_v30_outputs(output, records, residual, history)
    plots = make_v30_plots(output, config)
    verdict = write_v30_report(
        output,
        config,
        parameter_counts(policies),
        phase_a_metrics,
        checkpoint["history"],
    )
    print(json.dumps({
        "phase": "evaluation_v3_0_complete",
        "records": len(records),
        "residual_interventions": len(residual),
        "history_interventions": len(history),
        "plots": [str(path) for path in plots],
        "verdict": verdict,
    }), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v3.0 Bayesian latent teacher")
    parser.add_argument("command", choices=("phase-a", "phase-b", "evaluate", "all"))
    parser.add_argument("--config", default="configs/v3_0_bayesian.json")
    parser.add_argument("--validated-gate")
    args = parser.parse_args(argv)
    config = V30Config.from_json(args.config)
    device = _device(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gate_path = Path(args.validated_gate) if args.validated_gate else output / "gate" / "summary.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Refusing v3.0 training because the history-necessity gate failed")
    local_gate = output / "gate" / "summary.json"
    if gate_path.resolve() != local_gate.resolve():
        local_gate.parent.mkdir(parents=True, exist_ok=True)
        local_gate.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if args.command == "all":
        try:
            with (output / "canonical_run.lock").open("x", encoding="utf-8") as handle:
                handle.write("IFI v3.0 one-shot Phase-A/Phase-B/evaluation run\n")
        except FileExistsError as error:
            raise RuntimeError("Refusing a second v3.0 all-stage run") from error
    (output / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    manifest = {
        "experiment": "IFI v3.0 history-necessary Bayesian latent teacher",
        "gate": {"path": str(gate_path.resolve()), "sha256": _sha256(gate_path)},
        "teacher": "BayesianLatentSystemTeacher",
        "finite_hypotheses": 32,
        "latent_exposed_to_ifi": False,
        "posterior_exposed_to_ifi": False,
        "semantic_residual_encoder": False,
        "recurrent_state": False,
        "separate_state_updater": False,
        "infonce": False,
        "phase_a_phase_b_separated": True,
        "variant_b_direct_path": "Transformer decoder residual connections only",
        "material_improvement_threshold": 0.05,
        "one_shot": args.command == "all",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.command in {"phase-a", "all"}:
        phase_a(config, output, device)
    if args.command in {"phase-b", "all"}:
        phase_b(config, output, device)
    if args.command in {"evaluate", "all"}:
        evaluate(config, output, device)
    if (output / "phase_a.pt").exists():
        manifest["phase_a"] = {
            "path": str((output / "phase_a.pt").resolve()),
            "sha256": _sha256(output / "phase_a.pt"),
            "frozen_for_phase_b": True,
        }
    if (output / "phase_b.pt").exists():
        manifest["phase_b"] = {
            "path": str((output / "phase_b.pt").resolve()),
            "sha256": _sha256(output / "phase_b.pt"),
        }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
