from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models import residual_features
from .v30_config import V30Config
from .v30_tasks import BayesianLatentBenchmark


@dataclass
class GateState:
    episode: int
    support_size: int
    iteration: int
    sensor: int
    residual: np.ndarray
    history: np.ndarray
    update: np.ndarray


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _posterior_from_batches(benchmark, batches):
    if not batches:
        return torch.full(
            (benchmark.library.size,),
            1.0 / benchmark.library.size,
            device=benchmark.device,
        )
    return benchmark.posterior((
        torch.cat([batch.x for batch in batches]),
        torch.cat([batch.y for batch in batches]),
    ))


@torch.no_grad()
def collect_gate_states(config: V30Config, device: torch.device) -> list[GateState]:
    gate = config.gate
    benchmark = BayesianLatentBenchmark(config.tasks, gate.seed, device)
    probes = torch.linspace(
        config.tasks.x_min,
        config.tasks.x_max,
        gate.probe_size,
        device=device,
    ).unsqueeze(-1)
    states = []
    maximum_history = (
        (config.tasks.train_sequence_length - 1)
        * max(config.tasks.support_sizes)
        * 2
    )
    for support_size in config.tasks.support_sizes:
        for _ in range(gate.episodes_per_k):
            episode = benchmark.sample_episode(
                support_size, "train_latents", config.tasks.train_sequence_length
            )
            for iteration in range(1, config.tasks.train_sequence_length):
                prior = _posterior_from_batches(benchmark, episode.batches[:iteration])
                batch = episode.batches[iteration]
                prediction = benchmark.posterior_predictive(prior, batch.x)
                raw = residual_features(batch.x, batch.y, prediction)
                posterior = _posterior_from_batches(
                    benchmark, episode.batches[: iteration + 1]
                )
                before = benchmark.posterior_predictive(prior, probes)
                after = benchmark.posterior_predictive(posterior, probes)
                history = torch.zeros(maximum_history, device=device)
                observed = torch.cat([
                    torch.cat([item.x, item.y], dim=-1).reshape(-1)
                    for item in episode.batches[:iteration]
                ])
                history[: observed.numel()] = observed
                states.append(GateState(
                    episode.episode_id,
                    support_size,
                    iteration,
                    batch.sensor_index,
                    raw.cpu().numpy(),
                    history.cpu().numpy(),
                    (after - before).reshape(-1).cpu().numpy(),
                ))
    return states


def _r2(target: np.ndarray, columns: list[np.ndarray]) -> float:
    design = np.column_stack([np.ones_like(target), *columns])
    prediction = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    denominator = np.square(target - target.mean()).sum()
    return 0.0 if denominator == 0 else 1.0 - np.square(target - prediction).sum() / denominator


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
    }


def analyse_gate(states: list[GateState], config: V30Config):
    generator = np.random.default_rng(config.gate.seed + 1)
    pair_rows = []
    nearest_rows = []
    grouped = {}
    for state in states:
        grouped.setdefault(
            (state.support_size, state.iteration, state.sensor), []
        ).append(state)
    for group in grouped.values():
        if len(group) < 3:
            continue
        residual_scale = np.maximum(
            np.concatenate([state.residual for state in group]).std(axis=0), 1e-5
        )
        history_scale = max(
            float(np.stack([state.history for state in group]).std()), 1e-5
        )
        update_scale = max(
            float(np.stack([state.update for state in group]).std()), 1e-5
        )
        residual_vectors = np.stack([
            (state.residual / residual_scale).reshape(-1) for state in group
        ])
        residual_matrix = np.sqrt(np.mean(np.square(
            residual_vectors[:, None] - residual_vectors[None, :, :]
        ), axis=-1))
        np.fill_diagonal(residual_matrix, np.inf)
        nearest = residual_matrix.argmin(axis=1)
        for index, neighbor in enumerate(nearest):
            left, right = group[index], group[int(neighbor)]
            nearest_rows.append({
                "support_size": left.support_size,
                "iteration": left.iteration,
                "sensor": left.sensor,
                "residual_distance": float(residual_matrix[index, neighbor]),
                "history_distance": float(np.sqrt(np.mean(np.square(
                    (left.history - right.history) / history_scale
                )))),
                "teacher_update_distance": float(np.sqrt(np.mean(np.square(
                    (left.update - right.update) / update_scale
                )))),
            })
        count = min(config.gate.pair_samples // max(1, len(grouped)), len(group) ** 2)
        for _ in range(count):
            left_index, right_index = generator.choice(len(group), size=2, replace=False)
            left, right = group[int(left_index)], group[int(right_index)]
            pair_rows.append({
                "support_size": left.support_size,
                "iteration": left.iteration,
                "sensor": left.sensor,
                "residual_distance": float(np.sqrt(np.mean(np.square(
                    (left.residual - right.residual) / residual_scale
                )))),
                "history_distance": float(np.sqrt(np.mean(np.square(
                    (left.history - right.history) / history_scale
                )))),
                "teacher_update_distance": float(np.sqrt(np.mean(np.square(
                    (left.update - right.update) / update_scale
                )))),
            })

    residual = np.array([row["residual_distance"] for row in pair_rows])
    history = np.array([row["history_distance"] for row in pair_rows])
    teacher = np.array([row["teacher_update_distance"] for row in pair_rows])
    nearest_residual = np.array([row["residual_distance"] for row in nearest_rows])
    nearest_history = np.array([row["history_distance"] for row in nearest_rows])
    nearest_teacher = np.array([row["teacher_update_distance"] for row in nearest_rows])
    small_threshold = np.quantile(residual, 0.10)
    history_threshold = np.quantile(history, 0.50)
    teacher_threshold = np.quantile(teacher, 0.50)
    aliases = (
        (nearest_residual <= small_threshold)
        & (nearest_history >= history_threshold)
        & (nearest_teacher >= teacher_threshold)
    )
    residual_r2 = _r2(teacher, [residual])
    combined_r2 = _r2(teacher, [residual, history])
    incremental = combined_r2 - residual_r2
    teacher_ratio = float(np.median(nearest_teacher) / np.median(teacher))
    alias_prevalence = float(aliases.mean())
    criteria = {
        "incremental_history_r2": bool(
            incremental >= config.gate.minimum_incremental_history_r2
        ),
        "alias_prevalence": bool(
            alias_prevalence >= config.gate.minimum_alias_prevalence
        ),
        "nearest_teacher_ratio": bool(
            teacher_ratio >= config.gate.minimum_nearest_teacher_ratio
        ),
    }
    summary = {
        "verdict": (
            "HISTORY-NECESSARY BENCHMARK VALIDATED"
            if all(criteria.values()) else "HISTORY-NECESSITY GATE FAILED"
        ),
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds_locked_before_run": {
            "minimum_incremental_history_r2": config.gate.minimum_incremental_history_r2,
            "minimum_alias_prevalence": config.gate.minimum_alias_prevalence,
            "minimum_nearest_teacher_ratio": config.gate.minimum_nearest_teacher_ratio,
        },
        "states": len(states),
        "random_pairs": len(pair_rows),
        "nearest_pairs": len(nearest_rows),
        "residual_distance": _quantiles(residual),
        "history_distance": _quantiles(history),
        "teacher_update_distance": _quantiles(teacher),
        "nearest_residual_distance": _quantiles(nearest_residual),
        "nearest_teacher_update_distance": _quantiles(nearest_teacher),
        "r2_current_residual_only": residual_r2,
        "r2_current_residual_plus_history": combined_r2,
        "incremental_history_r2": incremental,
        "alias_prevalence": alias_prevalence,
        "alias_count": int(aliases.sum()),
        "nearest_teacher_to_random_ratio": teacher_ratio,
        "design": {
            "split": "train_latents",
            "finite_hypotheses": 32,
            "ideal_current_function": "pre-observation posterior predictive",
            "correct_update": "posterior_after - posterior_before on fixed probes",
            "pairing": "same K, iteration, and current sensor",
            "residual_tokens": "[x,y,F,e,abs(e),e^2]",
        },
    }
    return summary, pair_rows, nearest_rows


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IFI v3.0 history-necessity gate")
    parser.add_argument("--config", default="configs/v3_0_bayesian.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = V30Config.from_json(args.config)
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("IFI v3.0 gate requires CUDA")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    _seed(config.gate.seed)
    states = collect_gate_states(config, device)
    summary, pairs, nearest = analyse_gate(states, config)
    output = Path(args.output or config.output_dir) / "gate"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.locked.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(output / "random_pairs.csv", pairs)
    _write_csv(output / "nearest_pairs.csv", nearest)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
