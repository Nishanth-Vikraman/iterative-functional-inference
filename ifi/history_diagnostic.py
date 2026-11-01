from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .config import ExperimentConfig
from .functions import ComposedFunction
from .operators import build_function_system, project_update
from .tasks import Episode, SyntheticFunctionBenchmark
from .teachers import RKHSFunctionalGradientTeacher
from .v21_models import DirectIFIPolicy
from .v22_models import PredictiveRecurrentPolicy
from .v23_models import ForwardTrainedPolicy


@dataclass
class StateRecord:
    episode: int
    support_size: int
    iteration: int
    model: str
    residual: np.ndarray
    history: np.ndarray
    teacher_trajectory: np.ndarray
    function_values: np.ndarray
    projection_residual: float


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_models(config: ExperimentConfig, device: torch.device):
    phase_a = torch.load("runs/canonical/phase_a.pt", map_location=device, weights_only=False)
    system, operator = build_function_system(config.operator, device)
    system.load_state_dict(phase_a["system"])
    operator.load_state_dict(phase_a["operator"])
    operator.freeze()

    maximum_support = max(config.tasks.support_sizes)
    v21 = {
        name: DirectIFIPolicy(
            name,
            config.phase_b.latent_dimension,
            config.operator.action_dimension,
            maximum_support,
        ).to(device)
        for name in ("direct_current", "direct_history", "direct_recurrent")
    }
    v21_checkpoint = torch.load(
        "runs/v2_1_direct/phase_b.pt", map_location=device, weights_only=False
    )
    for name, policy in v21.items():
        policy.load_state_dict(v21_checkpoint["policies"][name])

    v22 = PredictiveRecurrentPolicy(
        config.phase_b.latent_dimension,
        config.operator.action_dimension,
        maximum_support,
        config.predictive_state.updater_layers,
    ).to(device)
    v22.load_state_dict(torch.load(
        "runs/v2_2_predictive/phase_b.pt", map_location=device, weights_only=False
    )["policy"])

    v23 = ForwardTrainedPolicy(
        config.phase_b.latent_dimension,
        config.operator.action_dimension,
        maximum_support,
    ).to(device)
    v23.load_state_dict(torch.load(
        "runs/v2_3_forward/phase_b.pt", map_location=device, weights_only=False
    )["policy"])

    models = {**v21, "v2_2_predictive": v22, "v2_3_forward": v23}
    for policy in models.values():
        policy.eval()
    return system, operator, models


def _initial_state(policy, z_data: Tensor):
    return policy.initial_state(z_data)


def _policy_step(policy, raw: Tensor, z_data: Tensor, history, state):
    if isinstance(policy, DirectIFIPolicy):
        result = policy.step(raw, z_data, history, state)
        return result.action, result.next_state
    if isinstance(policy, PredictiveRecurrentPolicy):
        action = policy.predict_action(raw, z_data, state, history)
        return action, policy.update_state(state, raw, action)
    result = policy.step(raw, z_data, state, history)
    return result.action, result.next_state


@torch.no_grad()
def _teacher_trajectory(
    teacher,
    operator,
    current,
    episode: Episode,
    probes: Tensor,
    ridge: float,
    horizon: int,
) -> tuple[Tensor, float]:
    actions = []
    residuals = []
    function = current
    for _ in range(horizon):
        update = teacher.propose_update(
            function, (episode.support_x, episode.support_y), probes
        )
        projection = project_update(operator, function, update, probes, ridge)
        actions.append(projection.phi)
        residuals.append(float(projection.relative_residual))
        function = function.append(projection.phi, detach=True)
    return torch.stack(actions), max(residuals)


@torch.no_grad()
def _collect(
    config: ExperimentConfig,
    system,
    operator,
    models,
    episodes_per_k: int,
    rollout: int,
    continuation: int,
    seed: int,
) -> list[StateRecord]:
    device = next(operator.parameters()).device
    benchmark = SyntheticFunctionBenchmark(config.tasks, seed, device)
    teacher = RKHSFunctionalGradientTeacher(config.teacher)
    probes = torch.linspace(
        config.tasks.x_min,
        config.tasks.x_max,
        config.phase_b.probe_size,
        device=device,
    ).unsqueeze(-1)
    records = []
    episode_key = 0
    for support_size in config.tasks.support_sizes:
        for _ in range(episodes_per_k):
            episode = benchmark.sample_episode(support_size, "heldout_compositions")
            for model_name, policy in models.items():
                z_data = policy.encode_data(episode.support_x, episode.support_y)
                state = _initial_state(policy, z_data)
                history: tuple[Tensor, ...] = ()
                current = ComposedFunction(system, operator)
                for iteration in range(1, rollout + 1):
                    raw_before = policy.raw_residual_tokens(
                        episode.support_x, episode.support_y, current(episode.support_x)
                    )
                    action, next_state = _policy_step(
                        policy, raw_before, z_data, history, state
                    )
                    current = current.append(action, detach=True)
                    history = history + (action.detach(),)
                    state = None if next_state is None else next_state.detach()

                    raw = policy.raw_residual_tokens(
                        episode.support_x, episode.support_y, current(episode.support_x)
                    )
                    targets, projection_residual = _teacher_trajectory(
                        teacher,
                        operator,
                        current,
                        episode,
                        probes,
                        config.operator.projection_ridge,
                        continuation,
                    )
                    order = torch.argsort(raw[:, 0])
                    history_array = torch.zeros(
                        rollout,
                        config.operator.action_dimension,
                        device=probes.device,
                    )
                    history_array[: len(history)] = torch.stack(history)
                    records.append(StateRecord(
                        episode_key,
                        support_size,
                        iteration,
                        model_name,
                        raw[order].cpu().numpy(),
                        history_array.cpu().numpy(),
                        targets.cpu().numpy(),
                        current(probes).cpu().numpy(),
                        projection_residual,
                    ))
            episode_key += 1
    return records


def _r2(y: np.ndarray, columns: list[np.ndarray]) -> float:
    design = np.column_stack([np.ones_like(y), *columns])
    prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = np.square(y - y.mean()).sum()
    return 0.0 if denominator == 0 else 1.0 - np.square(y - prediction).sum() / denominator


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
    }


def _analyse(records: list[StateRecord]):
    scales = {}
    for support_size in sorted({record.support_size for record in records}):
        group = [record for record in records if record.support_size == support_size]
        residual_scale = np.concatenate([record.residual for record in group]).std(axis=0)
        history_scale = np.concatenate([record.history for record in group]).std(axis=0)
        target_scale = np.stack([record.teacher_trajectory for record in group]).std(axis=0)
        function_scale = np.stack([record.function_values for record in group]).std(axis=0)
        scales[support_size] = tuple(
            np.maximum(scale, 1e-5)
            for scale in (residual_scale, history_scale, target_scale, function_scale)
        )

    controlled = []
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.episode, record.support_size, record.iteration)].append(record)
    for group in grouped.values():
        for left, right in itertools.combinations(group, 2):
            r_scale, h_scale, t_scale, f_scale = scales[left.support_size]
            controlled.append({
                "support_size": left.support_size,
                "iteration": left.iteration,
                "model_a": left.model,
                "model_b": right.model,
                "residual_distance": float(np.sqrt(np.mean(np.square(
                    (left.residual - right.residual) / r_scale
                )))),
                "history_distance": float(np.sqrt(np.mean(np.square(
                    (left.history - right.history) / h_scale
                )))),
                "next_action_distance": float(np.sqrt(np.mean(np.square(
                    (left.teacher_trajectory[0] - right.teacher_trajectory[0]) / t_scale[0]
                )))),
                "continuation_distance": float(np.sqrt(np.mean(np.square(
                    (left.teacher_trajectory - right.teacher_trajectory) / t_scale
                )))),
                "function_distance": float(np.sqrt(np.mean(np.square(
                    (left.function_values - right.function_values) / f_scale
                )))),
                "max_projection_residual": max(
                    left.projection_residual, right.projection_residual
                ),
            })

    residual = np.array([row["residual_distance"] for row in controlled])
    history = np.array([row["history_distance"] for row in controlled])
    continuation = np.array([row["continuation_distance"] for row in controlled])
    close_threshold = np.quantile(residual, 0.10)
    history_threshold = np.quantile(history, 0.50)
    divergent_threshold = np.quantile(continuation, 0.75)
    close_history = (residual <= close_threshold) & (history >= history_threshold)
    ambiguous = close_history & (continuation >= divergent_threshold)

    by_pair = {}
    for key, rows in itertools.groupby(
        sorted(controlled, key=lambda row: (row["model_a"], row["model_b"])),
        key=lambda row: (row["model_a"], row["model_b"]),
    ):
        rows = list(rows)
        by_pair[" vs ".join(key)] = {
            "n": len(rows),
            "median_residual_distance": float(np.median([
                row["residual_distance"] for row in rows
            ])),
            "median_continuation_distance": float(np.median([
                row["continuation_distance"] for row in rows
            ])),
        }

    natural = []
    generator = np.random.default_rng(1701)
    for support_size in sorted(scales):
        for iteration in sorted({record.iteration for record in records}):
            group = [
                record for record in records
                if record.support_size == support_size and record.iteration == iteration
            ]
            if len({record.episode for record in group}) < 2:
                continue
            r_scale, h_scale, t_scale, _ = scales[support_size]
            residual_vectors = np.stack([
                (record.residual / r_scale).reshape(-1) for record in group
            ])
            distances = np.sqrt(np.mean(np.square(
                residual_vectors[:, None, :] - residual_vectors[None, :, :]
            ), axis=-1))
            episode_ids = np.array([record.episode for record in group])
            distances[episode_ids[:, None] == episode_ids[None, :]] = np.inf
            nearest = distances.argmin(axis=1)
            for index, neighbor in enumerate(nearest):
                candidates = np.flatnonzero(episode_ids != episode_ids[index])
                random_neighbor = int(generator.choice(candidates))
                for kind, other in (("nearest", int(neighbor)), ("random", random_neighbor)):
                    natural.append({
                        "kind": kind,
                        "support_size": support_size,
                        "iteration": iteration,
                        "residual_distance": float(distances[index, other]),
                        "history_distance": float(np.sqrt(np.mean(np.square(
                            (group[index].history - group[other].history) / h_scale
                        )))),
                        "continuation_distance": float(np.sqrt(np.mean(np.square(
                            (group[index].teacher_trajectory - group[other].teacher_trajectory)
                            / t_scale
                        )))),
                    })

    nearest_rows = [row for row in natural if row["kind"] == "nearest"]
    random_rows = [row for row in natural if row["kind"] == "random"]
    nearest_r = np.array([row["residual_distance"] for row in nearest_rows])
    random_r = np.array([row["residual_distance"] for row in random_rows])
    nearest_t = np.array([row["continuation_distance"] for row in nearest_rows])
    random_t = np.array([row["continuation_distance"] for row in random_rows])
    natural_summary = {
        "n_states": len(nearest_rows),
    }
    if nearest_rows:
        natural_summary.update({
            "nearest_residual_distance": _quantiles(nearest_r),
            "random_residual_distance": _quantiles(random_r),
            "median_nearest_to_random_residual_ratio": float(
                np.median(nearest_r) / np.median(random_r)
            ),
            "nearest_continuation_distance": _quantiles(nearest_t),
            "random_continuation_distance": _quantiles(random_t),
            "median_nearest_to_random_continuation_ratio": float(
                np.median(nearest_t) / np.median(random_t)
            ),
        })
    summary = {
        "controlled_same_episode_cross_model": {
            "n_pairs": len(controlled),
            "residual_distance": _quantiles(residual),
            "history_distance": _quantiles(history),
            "continuation_distance": _quantiles(continuation),
            "correlation_residual_to_continuation": float(np.corrcoef(
                residual, continuation
            )[0, 1]),
            "correlation_history_to_continuation": float(np.corrcoef(
                history, continuation
            )[0, 1]),
            "r2_residual_only": _r2(continuation, [residual]),
            "r2_residual_plus_history": _r2(continuation, [residual, history]),
            "incremental_history_r2": (
                _r2(continuation, [residual, history]) - _r2(continuation, [residual])
            ),
            "close_threshold_q10": float(close_threshold),
            "history_divergent_threshold_median": float(history_threshold),
            "trajectory_divergent_threshold_q75": float(divergent_threshold),
            "close_and_history_divergent_pairs": int(close_history.sum()),
            "ambiguous_pairs": int(ambiguous.sum()),
            "ambiguous_fraction_of_close_history": float(
                ambiguous.sum() / max(1, close_history.sum())
            ),
            "ambiguous_fraction_of_all": float(ambiguous.mean()),
            "median_continuation_all": float(np.median(continuation)),
            "median_continuation_close_history": float(np.median(
                continuation[close_history]
            )),
            "by_model_pair": by_pair,
        },
        "natural_cross_episode_nearest_neighbors": natural_summary,
        "projection": {
            "median_relative_residual": float(np.median([
                record.projection_residual for record in records
            ])),
            "q90_relative_residual": float(np.quantile([
                record.projection_residual for record in records
            ], 0.90)),
        },
    }
    return controlled, natural, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether IFI residual states alias history")
    parser.add_argument("--episodes-per-k", type=int, default=12)
    parser.add_argument("--rollout", type=int, default=5)
    parser.add_argument("--continuation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", default="runs/history_necessity_diagnostic")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json("configs/v2_3_forward.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed(args.seed)
    system, operator, models = _load_models(config, device)
    records = _collect(
        config,
        system,
        operator,
        models,
        args.episodes_per_k,
        args.rollout,
        args.continuation,
        args.seed,
    )
    controlled, natural, summary = _analyse(records)
    summary["design"] = {
        "split": "heldout_compositions",
        "support_sizes": list(config.tasks.support_sizes),
        "episodes_per_k": args.episodes_per_k,
        "rollout_states": list(range(1, args.rollout + 1)),
        "teacher_continuation_steps": args.continuation,
        "models": list(models),
        "device": str(device),
        "residual_distance": "feature-whitened RMS over sorted raw residual tokens",
        "history_distance": "feature-whitened RMS over padded predicted actions",
        "teacher_trajectory": "projected RKHS actions from the exact current function",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "controlled_pairs.csv", controlled)
    if natural:
        _write_csv(output / "natural_neighbors.csv", natural)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
