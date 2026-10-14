"""Run the TrackLab video-processing pipeline from the command line."""

import argparse
import sys
from pathlib import Path

from tracklab.config import load_config
from tracklab.pipeline import VideoPipeline

DEFAULT_CONFIG_PATH = Path("configs/pipeline.yaml")
if not DEFAULT_CONFIG_PATH.is_file():
    _repo_default = Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"
    if _repo_default.is_file():
        DEFAULT_CONFIG_PATH = _repo_default


def main() -> int:
    """Parse arguments, run the pipeline, and print a summary."""

    parser = argparse.ArgumentParser(
        description="Detect and track objects in a video with TrackLab."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source video.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Pipeline YAML path (default: configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated results (default: outputs).",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run = VideoPipeline(config).run(
            input_path=args.input,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"TrackLab error: {exc}", file=sys.stderr)
        return 1

    summary = run.video_result.summary

    print(f"Input video: {run.video_result.meta.name}")
    print(f"Frames processed: {summary['processed_frames']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Unique tracks: {summary['unique_tracks']}")

    if run.video_path is not None:
        print(f"Tracked video: {run.video_path}")
    if run.json_path is not None:
        print(f"JSON results: {run.json_path}")
    if run.mot_path is not None:
        print(f"MOT results: {run.mot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
