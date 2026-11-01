from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from .v40_config import V40Config
from .v40_evaluation import (
    calibrate_fixed_eta,
    evaluate_history_interventions,
    evaluate_rollouts,
    write_rows,
)
from .v40_report import write_report
from .v40_training import (
    PopulationTrainer,
    build_models,
    integrity_audit,
)
from .v40_population import ReplayPopulation


def _device(config: V40Config) -> torch.device:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("IFI v4.0 smoke requires CUDA")
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def _locked_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if json.dumps(existing, sort_keys=True) != json.dumps(payload, sort_keys=True):
            raise RuntimeError(f"Refusing to modify locked v4.0 artifact: {path}")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _physical_memory() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.total_physical)


def hardware(device: torch.device) -> dict[str, object]:
    disk = shutil.disk_usage(Path.cwd())
    gpu: dict[str, object] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        gpu = {
            "name": properties.name,
            "vram_total_bytes": total,
            "vram_free_bytes_at_inspection": free,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
        try:
            line = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            gpu["driver_version"] = line
        except (OSError, subprocess.SubprocessError):
            gpu["driver_version"] = "unavailable"
    return {
        "gpu": gpu,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "pytorch": torch.__version__,
        "cpu": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "ram_total_bytes": _physical_memory(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "platform": platform.platform(),
    }


def validate_config(config: V40Config) -> None:
    if config.residual_dimension != config.action_dimension:
        raise ValueError("v4.0 requires d_R == d_phi")
    if config.bootstrap_actions > config.history_capacity:
        raise ValueError("Bootstrap history exceeds memory capacity")
    if config.bootstrap_orthogonal and config.bootstrap_actions > config.action_dimension:
        raise ValueError("Orthogonal bootstrap count exceeds action dimension")
    if config.loss_kind not in {"contraction_squared", "softplus_progress"}:
        raise ValueError(f"Unknown v4 loss: {config.loss_kind}")
    if not 0 < config.training_residual_norm_min < config.training_residual_norm_max:
        raise ValueError("Training residual-norm range must be positive and ordered")
    if not 0 < config.evaluation_residual_norm_min < config.evaluation_residual_norm_max:
        raise ValueError("Evaluation residual-norm range must be positive and ordered")
    edges = config.magnitude_bin_log10_edges
    if len(edges) < 2 or any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("Magnitude-bin edges must be strictly increasing")
    if config.stratified_magnitude_sampling and config.batch_size < len(edges) - 1:
        raise ValueError("Stratified batch must cover every magnitude bin")
    if config.singular_max / config.singular_min > 4.0 + 1e-12:
        raise ValueError("Primary v4.0 worlds must have condition bound <= 4")
    if config.gate_step >= config.training_steps:
        raise ValueError("Contraction gate must precede the final training step")


def prepare(config: V40Config, device: torch.device) -> dict[str, object]:
    validate_config(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _locked_json(output / "config.json", config.to_dict())
    hardware_path = output / "hardware.json"
    if not hardware_path.exists():
        _locked_json(hardware_path, hardware(device))
    calibration = calibrate_fixed_eta(config, device)
    _locked_json(output / "fixed_linear_calibration.json", calibration)
    small = replace(config, pool_size=128, batch_size=16)
    models = build_models(small, device)
    populations = {
        name: ReplayPopulation(small, name, small.seed + 200)
        for name in models
    }
    audit = integrity_audit(small, models, populations, device)
    _locked_json(output / "preflight_integrity.json", audit)
    if not audit["passed"]:
        raise RuntimeError("v4.0 preflight integrity failed")
    payload = {
        "phase": "v4_prepared",
        "audit_passed": True,
        "fixed_eta": calibration["selected_eta"],
        "parameter_counts": audit["measurements"]["parameter_counts"],
    }
    print(json.dumps(payload), flush=True)
    return payload


def throughput(config: V40Config, device: torch.device) -> dict[str, object]:
    small = replace(config, pool_size=512, batch_size=config.batch_size)
    trainer = PopulationTrainer(small, device, Path(config.output_dir))
    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    steps = 20
    for step in range(1, steps + 1):
        trainer.train_step(step)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    payload = {
        "steps": steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "batch_size": config.batch_size,
    }
    print(json.dumps(payload), flush=True)
    return payload


def _write_combined_world_registry(
    path: Path,
    trainer: PopulationTrainer,
    evaluation_rows: list[dict[str, object]],
) -> None:
    training_rows = [
        row
        for population in trainer.populations.values()
        for row in population.registry
    ]
    rows = training_rows + evaluation_rows
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config: V40Config, device: torch.device) -> None:
    prepare(config, device)
    output = Path(config.output_dir)
    trainer = PopulationTrainer(config, device, output)
    result = trainer.run()
    stopped = bool(result["stopped_by_gate"])
    evaluation_worlds: list[dict[str, object]] = []
    if stopped:
        (output / "eval_rollouts.csv").write_text("", encoding="utf-8")
        (output / "history_interventions.csv").write_text("", encoding="utf-8")
    else:
        for model in trainer.models.values():
            model.eval()
            model.requires_grad_(False)
        fixed = json.loads(
            (output / "fixed_linear_calibration.json").read_text(encoding="utf-8")
        )["selected_eta"]
        rollout_rows, evaluation_worlds = evaluate_rollouts(
            trainer.models, config, device, float(fixed)
        )
        write_rows(output / "eval_rollouts.csv", rollout_rows)
        interventions = evaluate_history_interventions(
            trainer.models["query_history"], config, device
        )
        write_rows(output / "history_interventions.csv", interventions)
    _write_combined_world_registry(
        output / "world_registry.csv", trainer, evaluation_worlds
    )
    verdict = write_report(config, output, stopped)
    manifest = {
        "experiment": f"IFI {config.experiment_version} GTX population replay smoke",
        "complete": True,
        "pid": os.getpid(),
        "seed": config.seed,
        "final_step": result["final_step"],
        "stopped_by_gate": stopped,
        "verdict": verdict,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"phase": "v4_complete", **manifest}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IFI v4.0 population replay smoke")
    parser.add_argument("command", choices=("prepare", "throughput", "run"))
    parser.add_argument("--config", default="configs/v4_0_gtx_smoke.json")
    args = parser.parse_args(argv)
    config = V40Config.from_json(args.config)
    device = _device(config)
    if args.command == "prepare":
        prepare(config, device)
    elif args.command == "throughput":
        throughput(config, device)
    else:
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        lock = output / "smoke_run.lock"
        try:
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
        except FileExistsError as error:
            raise RuntimeError("Refusing a second IFI v4.0 smoke run") from error
        run(config, device)


if __name__ == "__main__":
    main()
