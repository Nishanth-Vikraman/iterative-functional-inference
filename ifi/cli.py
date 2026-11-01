from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_policy_states, save_phase_b
from .config import ExperimentConfig
from .evaluation import Evaluator, write_evaluation
from .models import IFIPolicy, StaticHypernetwork
from .operators import build_function_system
from .phase_a import PhaseATrainer, save_phase_a, write_phase_a_report
from .phase_b import PhaseBTrainer
from .plots import make_plots
from .tasks import SyntheticFunctionBenchmark


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: ExperimentConfig) -> torch.device:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "This configuration requires a CUDA GPU, but torch.cuda.is_available() is false. "
            "Use configs/smoke.json only for CPU smoke verification."
        )
    requested = config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _models(config: ExperimentConfig, device: torch.device):
    phase = config.phase_b
    action = config.operator.action_dimension
    ifi = IFIPolicy(
        phase.latent_dimension, action, phase.controller_hidden_dimension, True
    ).to(device)
    ar = IFIPolicy(
        phase.latent_dimension, action, phase.controller_hidden_dimension, False
    ).to(device)
    static = StaticHypernetwork(
        phase.latent_dimension, action, phase.rollout_max
    ).to(device)
    return ifi, ar, static


def _load_phase_a(path: Path, config: ExperimentConfig, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Phase-A checkpoint not found: {path}")
    system, operator = build_function_system(config.operator, device)
    payload = torch.load(path, map_location=device, weights_only=False)
    system.load_state_dict(payload["system"])
    operator.load_state_dict(payload["operator"])
    operator.freeze()
    return system, operator


def run_phase_a(config: ExperimentConfig, output: Path, device: torch.device):
    _seed_everything(config.seed)
    system, operator = build_function_system(config.operator, device)
    benchmark = SyntheticFunctionBenchmark(config.tasks, config.seed, device)
    trainer = PhaseATrainer(config, system, operator, benchmark)
    history = trainer.run()
    metrics_by_split = {
        split: trainer.evaluate(split=split)
        for split in ("seen_parameters", "heldout_combinations", "heldout_compositions")
    }
    metrics = metrics_by_split["seen_parameters"]
    operator.freeze()
    save_phase_a(output / "phase_a.pt", config, system, operator, metrics)
    write_phase_a_report(output / "phase_a_metrics.json", metrics)
    (output / "phase_a_metrics_by_split.json").write_text(
        json.dumps(
            {split: values.__dict__ for split, values in metrics_by_split.items()}, indent=2
        ),
        encoding="utf-8",
    )
    (output / "phase_a_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({
        "phase": "A_complete",
        "seen_parameters": metrics.__dict__,
        "heldout_combinations": metrics_by_split["heldout_combinations"].__dict__,
        "heldout_compositions": metrics_by_split["heldout_compositions"].__dict__,
    }))
    return system, operator


def run_phase_b(config: ExperimentConfig, output: Path, device: torch.device):
    _seed_everything(config.seed + 1000)
    system, operator = _load_phase_a(output / "phase_a.pt", config, device)
    benchmark = SyntheticFunctionBenchmark(config.tasks, config.seed + 1, device)
    ifi, ar, static = _models(config, device)
    trainer = PhaseBTrainer(config, system, operator, benchmark, ifi, ar, static)
    history = trainer.run()
    save_phase_b(output / "phase_b.pt", config, ifi, ar, static)
    (output / "phase_b_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return system, operator, ifi, ar, static


def run_evaluation(config: ExperimentConfig, output: Path, device: torch.device):
    _seed_everything(config.seed + 2000)
    system, operator = _load_phase_a(output / "phase_a.pt", config, device)
    ifi, ar, static = _models(config, device)
    load_policy_states(output / "phase_b.pt", ifi, ar, static, device)
    evaluator = Evaluator(config, system, operator, ifi, ar, static)
    records, interventions = evaluator.run()
    summary = write_evaluation(output, records, interventions, evaluator.trajectory_rows)
    plots = make_plots(output)
    _write_scientific_report(output, config, summary, interventions)
    print(json.dumps({
        "phase": "evaluation_complete", "records": len(records),
        "interventions": len(interventions), "plots": [str(path) for path in plots]
    }))


def _write_scientific_report(
    output: Path,
    config: ExperimentConfig,
    summary: list[dict[str, object]],
    interventions: list[dict[str, float | str | int]],
) -> None:
    final_iteration = config.evaluation.rollout_steps
    heldout = [row for row in summary if row["split"] == "heldout_compositions"]
    final = [row for row in heldout if row["iteration"] == final_iteration]
    method_values: dict[str, list[float]] = {}
    for row in final:
        method_values.setdefault(str(row["method"]), []).append(float(row["query_mse_mean"]))
    averages = {key: sum(value) / len(value) for key, value in method_values.items()}
    feedback: dict[str, list[float]] = {}
    for row in interventions:
        feedback.setdefault(str(row["intervention"]), []).append(
            float(row["query_mse_delta_from_correct"])
        )
    feedback_mean = {key: sum(value) / len(value) for key, value in feedback.items()}
    ifi_beats_ar = averages.get("ifi", float("inf")) < averages.get("ar_no_feedback", float("inf"))
    corruptions = [key for key in ("zero", "wrong_task", "shuffled", "stale") if key in feedback_mean]
    causal = bool(corruptions) and all(feedback_mean[key] > 0 for key in corruptions)
    report = f"""# IFI experiment report

Configuration: `{config.output_dir}`; seeds: {list(config.evaluation.seeds)}.

## Held-out composition endpoint

```json
{json.dumps(averages, indent=2)}
```

## Feedback interventions

Positive numbers mean corrupted current residual evidence increased query MSE relative to the
correct residual while holding the current function, data embedding, action history, and recurrent
state fixed.

```json
{json.dumps(feedback_mean, indent=2)}
```

## Pre-registered mechanism checks

- IFI beats AR-no-feedback on held-out compositions: **{ifi_beats_ar}**
- Every measured residual corruption is worse than correct residual feedback: **{causal}**

These booleans are descriptive checks, not significance tests. See `evaluation_records.csv`,
`feedback_interventions.csv`, and `phase_a_metrics.json` for the underlying measurements.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Iterative Functional Inference experiments")
    parser.add_argument("command", choices=("phase-a", "phase-b", "evaluate", "all"))
    parser.add_argument("--config", default="configs/canonical.json")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    _seed_everything(config.seed)
    device = _device(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    if args.command in ("phase-a", "all"):
        run_phase_a(config, output, device)
    if args.command in ("phase-b", "all"):
        run_phase_b(config, output, device)
    if args.command in ("evaluate", "all"):
        run_evaluation(config, output, device)


if __name__ == "__main__":
    main()
