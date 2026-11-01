from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .v40_cli import hardware
from .v40_config import V40Config
from .v44_config import V44Config
from .v44_evaluation import ALL_HISTORY, FIFO_16, rollout_summary
from .v44_joint_capacity import atomic_json, fixture_states
from .v44_training import ArmRuntime, fixed_convergence_metrics, quick_early_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    runtime: ArmRuntime,
    step: int,
    source_path: Path,
) -> None:
    payload = {
        "experiment": "IFI v4.4 sampler-gated step-count continuation",
        "step": step,
        "complete": True,
        "resumed_from": str(source_path.resolve()),
        "resumed_from_sha256": sha256(source_path),
        **runtime.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate_endpoint(
    experiment: V44Config,
    base: V40Config,
    runtimes: dict[str, ArmRuntime],
    archive: dict[str, torch.Tensor],
    validation_indices: torch.Tensor,
    unseen_state,
    step: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary = []
    trajectories = []
    for arm, runtime in runtimes.items():
        runtime.model.eval()
        early = quick_early_metrics(runtime.model, runtime.config, runtime.device)
        convergence = fixed_convergence_metrics(
            runtime.model,
            runtime.config,
            archive,
            validation_indices,
            runtime.device,
        )
        arm_rows = []
        for policy in (ALL_HISTORY, FIFO_16):
            rows = rollout_summary(
                experiment,
                runtime.config,
                runtime.model,
                f"{arm}_step{step}",
                11,
                "unseen_same_family",
                unseen_state,
                policy,
            )
            trajectories.extend(rows)
            if policy.name == "all_history":
                arm_rows = rows
        by_step = {int(row["iteration"]): row for row in arm_rows}
        best = min(arm_rows, key=lambda row: float(row["residual_norm_median"]))
        summary.append({
            "step": step,
            "arm": arm,
            "early_f8_over_f0": early["f8_over_f0_median"],
            "early_median_q": early["median_q"],
            "convergence_median_q": convergence["median_q"],
            "convergence_regression_fraction": convergence["regression_fraction"],
            "convergence_action_norm": convergence["action_norm_median"],
            "rollout_f0": by_step[0]["residual_norm_median"],
            "rollout_f8": by_step[8]["residual_norm_median"],
            "rollout_f16": by_step[16]["residual_norm_median"],
            "rollout_f32": by_step[32]["residual_norm_median"],
            "rollout_f64": by_step[64]["residual_norm_median"],
            "rollout_f128": by_step[128]["residual_norm_median"],
            "best_residual": best["residual_norm_median"],
            "best_iteration": best["iteration"],
            "f64_over_best": float(by_step[64]["residual_norm_median"])
            / float(best["residual_norm_median"]),
            "f128_over_best": float(by_step[128]["residual_norm_median"])
            / float(best["residual_norm_median"]),
        })
        runtime.model.train()
    return summary, trajectories


def report(output: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# IFI v4.4 — Step-Count Continuation",
        "",
        "Both capacity arms were resumed from the identical saved step-2,000 states. "
        "The original gate was recorded but deliberately did not stop continuation.",
        "",
        "| Total step | Arm | F8/F0 | convergence q | convergence regression | best | best step | F64/best | F128/best |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['arm']} | {float(row['early_f8_over_f0']):.6g}"
            f" | {float(row['convergence_median_q']):.6g}"
            f" | {float(row['convergence_regression_fraction']):.6g}"
            f" | {float(row['best_residual']):.6g} | {row['best_iteration']}"
            f" | {float(row['f64_over_best']):.6g} | {float(row['f128_over_best']):.6g} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue sampler-gated v4.4 to 6k/12k")
    parser.add_argument(
        "--config",
        default="configs/v4_4_joint_capacity_sampler_gated.json",
    )
    parser.add_argument(
        "--source-dir",
        default="artifacts/v4_4_joint_capacity_sampler_gated/seed_11",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/v4_4_joint_capacity_step_count",
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    experiment = V44Config.from_json(args.config)
    base = V40Config.from_json(experiment.base_config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "continuation_run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    except FileExistsError as error:
        raise RuntimeError("Refusing a duplicate v4.4 step-count continuation") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    atomic_json(output / "hardware.json", hardware(device))
    archive = torch.load(
        experiment.convergence_validation_pool,
        map_location="cpu",
        weights_only=False,
    )
    trajectory_count = int(archive["trajectory_id"].max()) + 1
    cutoff = int(0.8 * trajectory_count)
    train_indices = torch.where(archive["trajectory_id"] < cutoff)[0]
    validation_indices = torch.where(archive["trajectory_id"] >= cutoff)[0]
    source_dir = Path(args.source_dir)
    runtimes = {
        arm: ArmRuntime(
            experiment,
            base,
            arm,
            11,
            device,
            archive,
            train_indices,
        )
        for arm in ("small", "large")
    }
    provenance = {}
    for arm, runtime in runtimes.items():
        source = source_dir / f"{arm}_checkpoint_final.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if int(checkpoint["step"]) != 2000:
            raise RuntimeError(f"Expected step-2,000 source checkpoint: {source}")
        restore = runtime.load_checkpoint_state(checkpoint, 2000)
        sampler = runtime.recovery.enforce_sufficiency(runtime.model)
        if not sampler["passed"]:
            raise RuntimeError(f"Restored {arm} sampler failed coverage gate")
        provenance[arm] = {
            "path": str(source.resolve()),
            "sha256": sha256(source),
            "source_step": int(checkpoint["step"]),
            "restore": restore,
            "post_restore_sampler_gate": sampler,
        }
    atomic_json(output / "resume_provenance.json", provenance)
    if args.audit_only:
        atomic_json(output / "status.json", {"phase": "resume_audit_complete"})
        print(json.dumps({"phase": "resume_audit_complete"}), flush=True)
        return
    reference = torch.load(
        experiment.reference_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    fixtures = fixture_states(
        experiment,
        base,
        reference["populations"]["query_history"],
        device,
    )
    unseen = fixtures["unseen_same_family"]
    del fixtures
    train_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    started = time.time()
    for step in range(2001, 12001):
        for arm in ("small", "large"):
            overall, regimes = runtimes[arm].train_step(step)
            if step % experiment.log_every == 0:
                train_rows.append(overall)
                regime_rows.extend(regimes)
            sampler = runtimes[arm].refresh_if_due(step)
            if sampler is not None:
                sampler_rows.append(sampler)
        if step % experiment.log_every == 0:
            elapsed = time.time() - started
            rate = (step - 2000) / elapsed
            status = {
                "phase": "training",
                "step": step,
                "target": 12000,
                "steps_per_second": rate,
                "eta_seconds": (12000 - step) / rate,
            }
            atomic_json(output / "status.json", status)
            write_csv(output / "train_metrics.csv", train_rows)
            write_csv(output / "regime_metrics.csv", regime_rows)
            write_csv(output / "sampler_coverage.csv", sampler_rows)
            print(json.dumps(status), flush=True)
        if step in (6000, 12000):
            for arm, runtime in runtimes.items():
                save_checkpoint(
                    output / f"{arm}_checkpoint_step_{step}.pt",
                    runtime,
                    step,
                    source_dir / f"{arm}_checkpoint_final.pt",
                )
            endpoint, trajectories = evaluate_endpoint(
                experiment,
                base,
                runtimes,
                archive,
                validation_indices,
                unseen,
                step,
            )
            endpoint_rows.extend(endpoint)
            trajectory_rows.extend(trajectories)
            write_csv(output / "endpoint_summary.csv", endpoint_rows)
            write_csv(output / "long_rollouts.csv", trajectory_rows)
            report(output, endpoint_rows)
            atomic_json(output / "status.json", {
                "phase": "endpoint_complete" if step < 12000 else "complete",
                "step": step,
                "endpoints": endpoint_rows,
            })
    atomic_json(output / "experiment_manifest.json", {
        "complete": True,
        "source_step": 2000,
        "evaluated_steps": [6000, 12000],
        "provenance": provenance,
        "endpoint_summary": endpoint_rows,
    })
    print(json.dumps({"phase": "complete", "endpoints": endpoint_rows}), flush=True)


if __name__ == "__main__":
    main()
