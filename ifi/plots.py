from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_plots(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    records = _read(output / "evaluation_records.csv")
    paths: list[Path] = []

    heldout = [row for row in records if row["split"] == "heldout_compositions"]
    curves: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in heldout:
        if row["method"] in {"ifi", "ar_no_feedback", "rkhs_teacher"}:
            curves[(row["method"], int(row["iteration"]))].append(float(row["query_mse"]))
    plt.figure(figsize=(6.4, 4.2))
    for method in ("ifi", "ar_no_feedback", "rkhs_teacher"):
        iterations = sorted(i for m, i in curves if m == method)
        plt.plot(iterations, [sum(curves[(method, i)]) / len(curves[(method, i)]) for i in iterations],
                 marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("adaptation iteration")
    plt.ylabel("held-out composition query MSE")
    plt.legend()
    plt.tight_layout()
    path = output / "error_vs_iteration.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    final_iteration = max(int(row["iteration"]) for row in records if row["method"] == "ifi")
    novelty_order = ["seen_parameters", "heldout_combinations", "heldout_compositions"]
    novelty: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        if (
            row["method"] in {"ifi", "ar_no_feedback"}
            and int(row["iteration"]) == final_iteration
        ):
            novelty[(row["method"], row["split"])].append(float(row["query_mse"]))
    plt.figure(figsize=(6.4, 4.2))
    for method in ("ifi", "ar_no_feedback"):
        plt.plot(
            novelty_order,
            [sum(novelty[(method, split)]) / len(novelty[(method, split)]) for split in novelty_order],
            marker="o", label=method,
        )
    plt.ylabel("final query MSE")
    plt.xlabel("increasing functional novelty")
    plt.xticks(rotation=12)
    plt.legend()
    plt.tight_layout()
    path = output / "ifi_vs_ar_novelty.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    k_curves: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in heldout:
        if row["method"] in {"ifi", "ar_no_feedback"} and int(row["iteration"]) == final_iteration:
            k_curves[(row["method"], int(row["support_size"]))].append(float(row["query_mse"]))
    plt.figure(figsize=(6.4, 4.2))
    for method in ("ifi", "ar_no_feedback"):
        ks = sorted(k for m, k in k_curves if m == method)
        plt.plot(ks, [sum(k_curves[(method, k)]) / len(k_curves[(method, k)]) for k in ks],
                 marker="o", label=method)
    plt.xlabel("support size K")
    plt.ylabel("final query MSE")
    plt.legend()
    plt.tight_layout()
    path = output / "error_vs_k.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    intervention_path = output / "feedback_interventions.csv"
    if intervention_path.exists():
        interventions = _read(intervention_path)
        groups: dict[str, list[float]] = defaultdict(list)
        for row in interventions:
            groups[row["intervention"]].append(float(row["query_mse_delta_from_correct"]))
        labels = ["correct", "zero", "wrong_task", "shuffled", "stale"]
        plt.figure(figsize=(7.2, 4.2))
        plt.bar(labels, [sum(groups[x]) / len(groups[x]) for x in labels])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("query MSE change vs correct feedback")
        plt.tight_layout()
        path = output / "feedback_interventions.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    trajectory_path = output / "phi_trajectories.csv"
    if trajectory_path.exists():
        trajectories = _read(trajectory_path)
        example = [
            row for row in trajectories
            if row["split"] == "heldout_compositions"
            and row["seed"] == trajectories[-1]["seed"]
            and row["support_size"] == trajectories[-1]["support_size"]
        ]
        coordinates = sorted({int(row["coordinate"]) for row in example})[:4]
        plt.figure(figsize=(7.2, 4.8))
        for coordinate in coordinates:
            rows = sorted(
                (row for row in example if int(row["coordinate"]) == coordinate),
                key=lambda row: int(row["iteration"]),
            )
            iterations = [int(row["iteration"]) for row in rows]
            plt.plot(iterations, [float(row["teacher_phi"]) for row in rows],
                     linestyle="--", label=f"teacher phi[{coordinate}]")
            plt.plot(iterations, [float(row["ifi_phi"]) for row in rows],
                     label=f"IFI phi[{coordinate}]")
        plt.xlabel("adaptation iteration")
        plt.ylabel("action coordinate")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        path = output / "teacher_vs_ifi_phi.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)
    return paths
