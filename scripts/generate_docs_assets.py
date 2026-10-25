"""Generate documentation charts from completed, machine-readable runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    with (
        PROJECT_ROOT
        / "outputs/mot17/tracker_ablation_day16/tracker_ablation_summary.yaml"
    ).open(encoding="utf-8") as stream:
        ablation = yaml.safe_load(stream)
    holdout = load_json(
        PROJECT_ROOT / "outputs/mot17/final_holdout_day17/final_holdout_summary.json"
    )
    latency = load_json(
        PROJECT_ROOT / "outputs/day17_latency_demo/sample2_result.json"
    )["summary"]["latency"]

    cases = list(ablation["cases"])
    labels = [
        case.replace("thresholds", "thresholds ").replace("buffer", "buffer ")
        for case in cases
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("HOTA", "MOTA", "IDF1"),
        ("Day 16 HOTA", "Day 16 MOTA", "Day 16 IDF1"),
        strict=True,
    ):
        values = [ablation["cases"][case]["combined"][metric] for case in cases]
        axis.bar(labels, values, color="#3b82f6")
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.2)
        axis.tick_params(axis="x", rotation=45, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(ASSET_ROOT / "tracker_ablation.png", dpi=180)
    plt.close(fig)

    holdout_labels = list(holdout["cases"])
    metrics = ("HOTA", "MOTA", "IDF1")
    x = list(range(len(holdout_labels)))
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, metric in enumerate(metrics):
        values = [holdout["cases"][case]["combined"][metric] for case in holdout_labels]
        axis.bar(
            [item + (index - 1) * width for item in x], values, width, label=metric
        )
    axis.set_xticks(x, holdout_labels, rotation=20)
    axis.set_ylim(0, 0.6)
    axis.set_ylabel("TrackEval score")
    axis.set_title("Day 17 frozen hold-out comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(ASSET_ROOT / "holdout_results.png", dpi=180)
    plt.close(fig)

    stage_seconds = latency["stage_seconds"]
    stages = list(stage_seconds)
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    values = [stage_seconds[stage] for stage in stages]
    axis.barh(stages, values, color="#14b8a6")
    axis.set_xlabel("Seconds")
    axis.set_title("Day 17 E6: sample2 pipeline stage time")
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(ASSET_ROOT / "latency_breakdown.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
