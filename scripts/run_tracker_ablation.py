"""Run the predeclared Day 16 tracker parameter ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tracker_ablation import (
    TUNING_SEQUENCES,
    run_tracker_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "MOT17" / "train",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "mot17" / "tracker_ablation",
    )
    args = parser.parse_args()
    run_tracker_ablation(
        data_root=args.data_root,
        output_root=args.output_root,
        sequence_names=TUNING_SEQUENCES,
    )


if __name__ == "__main__":
    main()
