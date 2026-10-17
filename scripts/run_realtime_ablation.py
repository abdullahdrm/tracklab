"""Run a fresh, focused real-image preprocessing speed/quality benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation import run_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark baseline, CLAHE, and gamma->CLAHE on MOT17 tuning sequences "
            "with a fresh output directory."
        )
    )
    parser.add_argument(
        "--output-root",
        default="outputs/mot17/ablations/realtime_optimized",
        help="Fresh output root (default: outputs/mot17/ablations/realtime_optimized)",
    )
    args = parser.parse_args()
    run_ablation(
        stage="real",
        output_root=args.output_root,
        resume=False,
        case_ids=("none", "clahe", "gamma_clahe"),
    )


if __name__ == "__main__":
    main()
