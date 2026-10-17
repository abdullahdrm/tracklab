"""Run color-space preprocessing ablations on MOT17 tuning sequences only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation import run_ablation

CASE_IDS = (
    "none",
    "gamma",
    "clahe",
    "gamma_clahe",
    "hsv_clahe",
    "ycrcb_clahe",
    "lab_gamma_clahe",
    "hsv_gamma_clahe",
    "ycrcb_gamma_clahe",
    "lab_clahe_gamma",
    "hsv_clahe_gamma",
    "ycrcb_clahe_gamma",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LAB-L, HSV-V, and YCrCb-Y CLAHE/gamma orders with fixed "
            "YOLO + ByteTrack settings on MOT17 tuning sequences only."
        )
    )
    parser.add_argument(
        "--output-root",
        default="outputs/mot17/ablations/colorspace_20261017",
        help="Fresh output directory; it must be absent or empty.",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"output root is not empty; choose a fresh path: {output_root}")

    run_ablation(
        stage="real",
        output_root=output_root,
        resume=False,
        case_ids=CASE_IDS,
    )


if __name__ == "__main__":
    main()
