from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .v40_config import V40Config
from .v40_evaluation import RolloutState, controller_batch, make_rollout_state
from .v40_training import build_models
from .v40_worlds import transition


@dataclass(frozen=True)
class HistoryPolicy:
    name: str
    recent_capacity: int
    preserve_bootstrap: bool
    all_history: bool = False

    @property
    def total_capacity(self) -> int:
        if self.all_history:
            raise ValueError("All-history capacity depends on evaluation horizon")
        return self.recent_capacity + (8 if self.preserve_bootstrap else 0)


POLICIES = (
    HistoryPolicy("fifo_16", 16, False),
    HistoryPolicy("bootstrap_8_plus_recent_8", 8, True),
    HistoryPolicy("bootstrap_8_plus_recent_16", 16, True),
    HistoryPolicy("all_history", 0, True, True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_with_policy(
    state: RolloutState,
    before: Tensor,
    action: Tensor,
    after: Tensor,
    policy: HistoryPolicy,
    bootstrap_count: int,
    maximum_capacity: int,
) -> None:
    rows = torch.arange(state.current.shape[0], device=state.current.device)
    if policy.all_history:
        positions = state.history_lengths
        if int(positions.max()) >= maximum_capacity:
            raise RuntimeError("All-history storage capacity exhausted")
        state.history_before[rows, positions] = before
        state.history_action[rows, positions] = action
        state.history_after[rows, positions] = after
        state.history_lengths += 1
    elif policy.preserve_bootstrap:
        recent_length = state.history_lengths - bootstrap_count
        full = recent_length >= policy.recent_capacity
        if full.any():
            state.history_before[
                full, bootstrap_count : bootstrap_count + policy.recent_capacity - 1
            ] = state.history_before[
                full, bootstrap_count + 1 : bootstrap_count + policy.recent_capacity
            ].clone()
            state.history_action[
                full, bootstrap_count : bootstrap_count + policy.recent_capacity - 1
            ] = state.history_action[
                full, bootstrap_count + 1 : bootstrap_count + policy.recent_capacity
            ].clone()
            state.history_after[
                full, bootstrap_count : bootstrap_count + policy.recent_capacity - 1
            ] = state.history_after[
                full, bootstrap_count + 1 : bootstrap_count + policy.recent_capacity
            ].clone()
        positions = bootstrap_count + recent_length.clamp_max(policy.recent_capacity - 1)
        state.history_before[rows, positions] = before
        state.history_action[rows, positions] = action
        state.history_after[rows, positions] = after
        state.history_lengths = bootstrap_count + (recent_length + 1).clamp_max(
            policy.recent_capacity
        )
    else:
        capacity = policy.recent_capacity
        full = state.history_lengths >= capacity
        if full.any():
            state.history_before[full, : capacity - 1] = state.history_before[
                full, 1:capacity
            ].clone()
            state.history_action[full, : capacity - 1] = state.history_action[
                full, 1:capacity
            ].clone()
            state.history_after[full, : capacity - 1] = state.history_after[
                full, 1:capacity
            ].clone()
        positions = state.history_lengths.clamp_max(capacity - 1)
        state.history_before[rows, positions] = before
        state.history_action[rows, positions] = action
        state.history_after[rows, positions] = after
        state.history_lengths = (state.history_lengths + 1).clamp_max(capacity)
    state.current = after
    state.previous_action = action


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(
    config: V40Config,
    model,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    maximum_capacity = config.bootstrap_actions + config.evaluation_rollout
    storage_config = replace(config, history_capacity=maximum_capacity)
    for bucket_index, (bucket, harder) in enumerate(
        (("same_family", False), ("condition_4_6", True))
    ):
        base = make_rollout_state(
            storage_config,
            config.evaluation_episodes,
            config.seed + 600 + bucket_index,
            device,
            harder,
        )
        seed_values = base.world_seeds.cpu().tolist()
        for policy in POLICIES:
            state = base.clone()
            initial = state.current.norm(dim=-1)
            previous_norm = initial
            initial_values = initial.cpu().tolist()
            for episode, value in enumerate(initial_values):
                records.append({
                    "bucket": bucket,
                    "history_policy": policy.name,
                    "episode": episode,
                    "world_seed": int(seed_values[episode]),
                    "iteration": 0,
                    "history_tokens": config.bootstrap_actions,
                    "residual_norm": value,
                    "q": "",
                    "action_norm": 0.0,
                })
            for iteration in range(1, config.evaluation_rollout + 1):
                before = state.current
                action = model(controller_batch(config, state))
                after = transition(before, state.matrices, action)
                after_norm = after.norm(dim=-1)
                q = after_norm / previous_norm.clamp_min(1e-12)
                after_values = after_norm.cpu().tolist()
                q_values = q.cpu().tolist()
                action_values = action.norm(dim=-1).cpu().tolist()
                history_tokens = int(state.history_lengths[0])
                for episode in range(config.evaluation_episodes):
                    records.append({
                        "bucket": bucket,
                        "history_policy": policy.name,
                        "episode": episode,
                        "world_seed": int(seed_values[episode]),
                        "iteration": iteration,
                        "history_tokens": history_tokens,
                        "residual_norm": after_values[episode],
                        "q": q_values[episode],
                        "action_norm": action_values[episode],
                    })
                append_with_policy(
                    state,
                    before,
                    action,
                    after,
                    policy,
                    config.bootstrap_actions,
                    maximum_capacity,
                )
                previous_norm = after_norm
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in records:
        key = (str(row["bucket"]), str(row["history_policy"]), int(row["iteration"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (bucket, policy, iteration), rows in sorted(groups.items()):
        norms = [float(row["residual_norm"]) for row in rows]
        q = [float(row["q"]) for row in rows if row["q"] != ""]
        summary.append({
            "bucket": bucket,
            "history_policy": policy,
            "iteration": iteration,
            "history_tokens": int(rows[0]["history_tokens"]),
            "residual_norm_median": statistics.median(norms),
            "residual_norm_mean": statistics.fmean(norms),
            "q_median": statistics.median(q) if q else "",
            "q_fraction_below_one": sum(value < 1 for value in q) / len(q) if q else "",
            "action_norm_median": statistics.median(
                float(row["action_norm"]) for row in rows
            ),
        })
    return summary


def _selected(
    summary: list[dict[str, object]], bucket: str, policy: str, iteration: int
) -> dict[str, object]:
    return next(
        row
        for row in summary
        if row["bucket"] == bucket
        and row["history_policy"] == policy
        and row["iteration"] == iteration
    )


def write_report(
    output: Path,
    summary: list[dict[str, object]],
    integrity: dict[str, object],
) -> None:
    lines = [
        "# IFI v4.1 — Frozen History-Policy Evaluation",
        "",
        "No gradients, optimizer, world changes, or checkpoint changes were used.",
        "",
        "## Same-family median residual norm",
        "",
        "| History policy | F0 | F8 | F16 | F32 | F64 | Best | Best step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in (item.name for item in POLICIES):
        values = [
            float(_selected(summary, "same_family", policy, iteration)["residual_norm_median"])
            for iteration in (0, 8, 16, 32, 64)
        ]
        rows = [
            row
            for row in summary
            if row["bucket"] == "same_family" and row["history_policy"] == policy
        ]
        best = min(rows, key=lambda row: float(row["residual_norm_median"]))
        lines.append(
            f"| {policy} | "
            + " | ".join(f"{value:.6g}" for value in values)
            + f" | {float(best['residual_norm_median']):.6g} | {best['iteration']} |"
        )
    f64 = {
        policy.name: float(
            _selected(summary, "same_family", policy.name, 64)["residual_norm_median"]
        )
        for policy in POLICIES
    }
    winner = min(f64, key=f64.get)
    lines.extend([
        "",
        "## Integrity",
        "",
        f"- Frozen checkpoint: `{integrity['checkpoint_sha256']}`",
        f"- All parameters require_grad=False: **{integrity['all_parameters_frozen']}**",
        f"- Original FIFO-16 reproduced: **{integrity['fifo_reproduction_passed']}**",
        f"- Same evaluation world seeds: **{integrity['same_world_seeds']}**",
        f"- Best F64 policy: **{winner}**",
        "",
        "Policies C and D exceed the 16-token history size observed during training; their results are frozen context-length extrapolation diagnostics.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, bucket in zip(axes, ("same_family", "condition_4_6")):
        for policy in (item.name for item in POLICIES):
            rows = [
                row
                for row in summary
                if row["bucket"] == bucket and row["history_policy"] == policy
            ]
            axis.plot(
                [int(row["iteration"]) for row in rows],
                [float(row["residual_norm_median"]) for row in rows],
                label=policy,
            )
        axis.set_title(bucket)
        axis.set_xlabel("Frozen iteration")
        axis.set_ylabel("Median residual norm")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "history_policy_rollouts.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Frozen v4.1 history-policy evaluation")
    parser.add_argument("--config", default="artifacts/v4_1/config.json")
    parser.add_argument(
        "--checkpoint", default="artifacts/v4_1/checkpoint_final.pt"
    )
    parser.add_argument(
        "--output", default="artifacts/v4_1/history_policy_eval"
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "evaluation.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": __import__("os").getpid(), "started": time.time()}))
    config = V40Config.from_json(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen v4.1 evaluation requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not checkpoint["complete"] or int(checkpoint["step"]) != config.training_steps:
        raise RuntimeError("Checkpoint is not the completed v4.1 canonical smoke")
    models = build_models(config, device)
    model = models["query_history"]
    model.load_state_dict(checkpoint["models"]["query_history"])
    model.eval()
    model.requires_grad_(False)
    records = evaluate(config, model, device)
    summary = summarize(records)
    original = list(csv.DictReader(Path("artifacts/v4_1/eval_summary.csv").open()))
    original_fifo = {
        (row["bucket"], int(row["iteration"])): float(row["residual_norm_median"])
        for row in original
        if row["model"] == "query_history"
    }
    reproduced_fifo = {
        (str(row["bucket"]), int(row["iteration"])): float(row["residual_norm_median"])
        for row in summary
        if row["history_policy"] == "fifo_16"
    }
    max_difference = max(
        abs(original_fifo[key] - reproduced_fifo[key]) for key in original_fifo
    )
    original_seeds = {
        int(row["world_seed"])
        for row in csv.DictReader(Path("artifacts/v4_1/eval_rollouts.csv").open())
        if row["model"] == "query_history"
    }
    new_seeds = {int(row["world_seed"]) for row in records}
    integrity = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "torch_grad_enabled_during_evaluation": False,
        "same_world_seeds": original_seeds == new_seeds,
        "fifo_max_median_absolute_difference": max_difference,
        "fifo_reproduction_passed": max_difference < 1e-6,
        "attention_accepts_all_history_tokens": True,
        "maximum_all_history_tokens": config.bootstrap_actions
        + config.evaluation_rollout
        - 1,
    }
    _write_csv(output / "rollouts.csv", records)
    _write_csv(output / "summary.csv", summary)
    (output / "integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    write_report(output, summary, integrity)
    manifest = {
        "complete": True,
        "policies": [policy.name for policy in POLICIES],
        "records": len(records),
        "integrity": integrity,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"phase": "v4_1_history_policy_complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
