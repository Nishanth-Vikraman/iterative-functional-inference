from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch
from .v40_training import build_models
from .v40_worlds import transition
from .v41_history_policy_eval import HistoryPolicy, append_with_policy


POLICIES = (
    HistoryPolicy("all_available_history", 0, True, True),
    HistoryPolicy("fifo_16", 16, False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_state(
    config: V40Config,
    population: dict[str, object],
    indices: torch.Tensor,
    capacity: int,
    device: torch.device,
) -> RolloutState:
    count = indices.numel()
    before = torch.zeros(count, capacity, config.residual_dimension)
    action = torch.zeros(count, capacity, config.action_dimension)
    after = torch.zeros(count, capacity, config.residual_dimension)
    saved_capacity = config.history_capacity
    before[:, :saved_capacity] = population["history_before"][indices]
    action[:, :saved_capacity] = population["history_action"][indices]
    after[:, :saved_capacity] = population["history_after"][indices]
    return RolloutState(
        matrices=population["matrices"][indices].to(device),
        singular_values=population["singular_values"][indices].to(device),
        world_seeds=population["world_seeds"][indices].to(device),
        current=population["current"][indices].to(device),
        previous_action=population["previous_action"][indices].to(device),
        history_before=before.to(device),
        history_action=action.to(device),
        history_after=after.to(device),
        history_lengths=population["history_lengths"][indices].to(device),
    )


@torch.inference_mode()
def evaluate(
    config: V40Config,
    model: torch.nn.Module,
    population: dict[str, object],
    indices: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    capacity = config.history_capacity + horizon
    for policy in POLICIES:
        state = _training_state(config, population, indices, capacity, device)
        initial = state.current.norm(dim=-1)
        previous = initial
        ages = population["ages"][indices].tolist()
        world_seeds = state.world_seeds.cpu().tolist()
        for episode, norm in enumerate(initial.cpu().tolist()):
            records.append({
                "history_policy": policy.name,
                "episode": episode,
                "pool_slot": int(indices[episode]),
                "world_seed": int(world_seeds[episode]),
                "checkpoint_age": int(ages[episode]),
                "additional_iteration": 0,
                "absolute_world_age": int(ages[episode]),
                "history_tokens": int(state.history_lengths[episode]),
                "residual_norm": norm,
                "relative_to_checkpoint": 1.0,
                "q": "",
                "action_norm": 0.0,
                "converged": norm < config.convergence_threshold,
                "diverged": norm > config.divergence_threshold,
            })
        for iteration in range(1, horizon + 1):
            before = state.current
            action = model(controller_batch(config, state))
            after = transition(before, state.matrices, action)
            norm = after.norm(dim=-1)
            q = norm / previous.clamp_min(1e-12)
            for episode, values in enumerate(
                zip(
                    norm.cpu().tolist(),
                    q.cpu().tolist(),
                    action.norm(dim=-1).cpu().tolist(),
                    state.history_lengths.cpu().tolist(),
                )
            ):
                records.append({
                    "history_policy": policy.name,
                    "episode": episode,
                    "pool_slot": int(indices[episode]),
                    "world_seed": int(world_seeds[episode]),
                    "checkpoint_age": int(ages[episode]),
                    "additional_iteration": iteration,
                    "absolute_world_age": int(ages[episode]) + iteration,
                    "history_tokens": int(values[3]),
                    "residual_norm": values[0],
                    "relative_to_checkpoint": values[0] / max(float(initial[episode]), 1e-12),
                    "q": values[1],
                    "action_norm": values[2],
                    "converged": values[0] < config.convergence_threshold,
                    "diverged": values[0] > config.divergence_threshold,
                })
            append_with_policy(
                state,
                before,
                action,
                after,
                policy,
                config.bootstrap_actions,
                capacity,
            )
            previous = norm
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in records:
        key = (str(row["history_policy"]), int(row["additional_iteration"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (policy, iteration), rows in sorted(groups.items()):
        norms = [float(row["residual_norm"]) for row in rows]
        ratios = [float(row["relative_to_checkpoint"]) for row in rows]
        q = [float(row["q"]) for row in rows if row["q"] != ""]
        summary.append({
            "history_policy": policy,
            "additional_iteration": iteration,
            "residual_norm_median": statistics.median(norms),
            "residual_norm_mean": statistics.fmean(norms),
            "checkpoint_ratio_median": statistics.median(ratios),
            "q_median": statistics.median(q) if q else "",
            "q_fraction_below_one": sum(value < 1 for value in q) / len(q) if q else "",
            "action_norm_median": statistics.median(float(row["action_norm"]) for row in rows),
            "converged_fraction": sum(bool(row["converged"]) for row in rows) / len(rows),
            "diverged_fraction": sum(bool(row["diverged"]) for row in rows) / len(rows),
        })
    return summary


def _row(summary: list[dict[str, object]], policy: str, iteration: int) -> dict[str, object]:
    return next(
        row for row in summary
        if row["history_policy"] == policy and row["additional_iteration"] == iteration
    )


def report(output: Path, summary: list[dict[str, object]], integrity: dict[str, object]) -> None:
    lines = [
        "# IFI v4.1 — Actual Training-World Long Rollout",
        "",
        "The frozen completed v4.1 query-history controller was restarted from 512 actual query-history population slots saved in its training checkpoint. No gradients or newly sampled/OOD worlds were used.",
        "",
        "| Policy | F0 | F16 | F32 | F64 | F96 | F128 | Best | Best step | F128 / best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    verdicts = {}
    for policy in (item.name for item in POLICIES):
        curve = [row for row in summary if row["history_policy"] == policy]
        best = min(curve, key=lambda row: float(row["residual_norm_median"]))
        values = [
            float(_row(summary, policy, iteration)["residual_norm_median"])
            for iteration in (0, 16, 32, 64, 96, 128)
        ]
        endpoint_best = values[-1] / max(float(best["residual_norm_median"]), 1e-12)
        late_q = statistics.median(
            float(row["q_median"])
            for row in curve
            if 65 <= int(row["additional_iteration"]) <= 128
        )
        keeps_converging = (
            int(best["additional_iteration"]) > 64
            and values[-1] < values[0]
            and late_q < 1
        )
        verdicts[policy] = {
            "keeps_converging": keeps_converging,
            "best_step": int(best["additional_iteration"]),
            "best_residual": float(best["residual_norm_median"]),
            "endpoint_over_best": endpoint_best,
            "late_median_q_f65_f128": late_q,
        }
        lines.append(
            f"| {policy} | "
            + " | ".join(f"{value:.6g}" for value in values)
            + f" | {float(best['residual_norm_median']):.6g} | {best['additional_iteration']} | {endpoint_best:.6g} |"
        )
    primary = verdicts["all_available_history"]
    verdict = (
        "KEEPS CONVERGING BEYOND TRAINING AGE"
        if primary["keeps_converging"]
        else "LATE-ROLLOUT DIVERGENCE PERSISTS ON TRAINING WORLDS"
    )
    lines.extend([
        "",
        f"## Verdict: {verdict}",
        "",
        f"- All-history best step: **{primary['best_step']}**",
        f"- All-history F128/best: **{primary['endpoint_over_best']:.6g}**",
        f"- All-history median q over F65–F128: **{primary['late_median_q_f65_f128']:.6g}**",
        f"- Frozen parameters: **{integrity['all_parameters_frozen']}**",
        f"- Actual stored training worlds: **{integrity['actual_saved_training_worlds']}**",
        "",
        "The checkpoint retained only each slot's latest 16 genuine transitions; earlier evicted transitions cannot be reconstructed. The all-history policy preserves all newly generated transitions after the checkpoint, while FIFO-16 exactly retains the trained memory policy.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "policies": verdicts}, indent=2),
        encoding="utf-8",
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    for policy in (item.name for item in POLICIES):
        curve = [row for row in summary if row["history_policy"] == policy]
        axis.plot(
            [int(row["additional_iteration"]) for row in curve],
            [float(row["residual_norm_median"]) for row in curve],
            label=policy,
        )
    axis.axvline(64, color="black", linestyle="--", linewidth=1, label="max training age")
    axis.set_yscale("log")
    axis.set_xlabel("Additional frozen update iterations")
    axis.set_ylabel("Median residual norm")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "trained_world_long_rollout.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="artifacts/v4_1/config.json")
    parser.add_argument("--checkpoint", default="artifacts/v4_1/checkpoint_final.pt")
    parser.add_argument("--output", default="artifacts/v4_1/trained_world_long_rollout")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=128)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = V40Config.from_json(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("complete") or int(checkpoint["step"]) != config.training_steps:
        raise RuntimeError("The v4.1 checkpoint is not the completed canonical checkpoint")
    population = checkpoint["populations"]["query_history"]
    generator = torch.Generator().manual_seed(41_128)
    indices = torch.randperm(config.pool_size, generator=generator)[: args.episodes]
    device = torch.device("cuda")
    models = build_models(config, device)
    model = models["query_history"]
    model.load_state_dict(checkpoint["models"]["query_history"])
    model.eval()
    model.requires_grad_(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = evaluate(config, model, population, indices, args.horizon, device)
    summary = summarize(records)
    integrity = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "torch_grad_enabled_during_rollout": False,
        "actual_saved_training_worlds": True,
        "selected_population_slots": indices.tolist(),
        "checkpoint_age_min": int(population["ages"][indices].min()),
        "checkpoint_age_median": float(population["ages"][indices].float().median()),
        "checkpoint_age_max": int(population["ages"][indices].max()),
        "training_max_age": config.max_age,
        "additional_rollout_horizon": args.horizon,
    }
    _write_csv(output / "rollouts.csv", records)
    _write_csv(output / "summary.csv", summary)
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    report(output, summary, integrity)
    print(json.dumps({"complete": True, "output": str(output), "integrity": integrity}), flush=True)


if __name__ == "__main__":
    main()
