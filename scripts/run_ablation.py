"""Run real-image and synthetic-degradation MOT17 ablation experiments."""

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
            "Run TrackLab's E2 real-image preprocessing ablation or E2b "
            "synthetic-degradation matrix on MOT17 tuning sequences only."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("real", "synthetic", "all"),
        default="real",
        help="Run E2 real images, E2b synthetic degradations, or both (default: real).",
    )
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="MOT17 evaluation config (default: configs/eval.yaml).",
    )
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipeline.yaml",
        help="Pipeline config for tracker/preprocessor settings.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/mot17/ablations",
        help="Ablation output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Base random seed for reproducible Gaussian noise (default: 1729).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-sequence predictions with matching saved config.",
    )
    args = parser.parse_args()

    run_ablation(
        stage=args.stage,
        eval_config_path=args.config,
        pipeline_config_path=args.pipeline_config,
        output_root=args.output_root,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
