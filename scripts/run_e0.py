"""Run TrackEval ground-truth self-evaluation (E0) on tuning sequences."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

for import_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from evaluation.trackeval_bridge import (
    evaluate_mot_challenge,
    prepare_ground_truth_tracker,
)
from tracklab.datasets.splits import load_split_plan


def main() -> int:
    data_root = PROJECT_ROOT / "data" / "raw" / "MOT17" / "train"
    split_config = PROJECT_ROOT / "configs" / "mot17_splits.yaml"
    output_root = PROJECT_ROOT / "outputs" / "mot17" / "e0"

    plan = load_split_plan(split_config)
    sequence_names = plan.sequence_names("tuning")

    print("E0: MOT17 ground-truth self-evaluation")
    print(f"Variant: {plan.variant}")
    print("Split: tuning (hold-out is not used)")
    print("Sequences:", ", ".join(sequence_names))

    tracker_root = output_root / "trackers"
    evaluation_root = output_root / "evaluation"

    counts = prepare_ground_truth_tracker(
        data_root,
        tracker_root,
        sequence_names,
        tracker_name="gt_self",
        variant=plan.variant,
    )

    metrics = evaluate_mot_challenge(
        data_root,
        tracker_root,
        evaluation_root,
        sequence_names,
        tracker_name="gt_self",
        variant=plan.variant,
    )

    report = {
        "experiment": "E0_ground_truth_self_evaluation",
        "dataset": "MOT17",
        "variant": plan.variant,
        "split": "tuning",
        "sequences": list(sequence_names),
        "prediction_rows": counts,
        "metrics": metrics,
        "expected_hota": 1.0,
    }

    report_path = output_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)

    print("\n===== E0 RESULTS =====")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f} ({value * 100:.3f}%)")
    print("Report:", report_path)

    if not math.isclose(metrics["HOTA"], 1.0, rel_tol=0.0, abs_tol=1e-6):
        print("E0 failed: expected HOTA to be approximately 1.0.")
        return 1

    print("E0 passed: HOTA is approximately 100%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
