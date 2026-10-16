"""Run the configured MOT17 detector baseline and TrackEval metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.mot17_detector_comparison import run_sequence_tracker
from evaluation.trackeval_bridge import evaluate_mot_challenge
from tracklab.config import load_config, resolve_device
from tracklab.datasets.mot17 import read_sequence_info


def _project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root when not absolute."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a progress report so partial runs remain inspectable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_run_info(output_root: Path, sequence: str, detector: str) -> dict:
    """Read counters from a previous per-sequence run when available."""

    previous_report = output_root / f"{sequence}_summary.json"
    if not previous_report.is_file():
        return {"reused_predictions": True}

    try:
        content = json.loads(previous_report.read_text(encoding="utf-8"))
        previous = content["comparisons"][detector]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"reused_predictions": True}

    keys = ("frames", "detections", "tracked_outputs", "elapsed_seconds")
    information = {key: previous[key] for key in keys if key in previous}
    information["reused_predictions"] = True
    return information


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the TrackLab YOLO/ByteTrack baseline on MOT17."
    )
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Evaluation YAML path (default: configs/eval.yaml)",
    )
    parser.add_argument(
        "--split",
        choices=("tuning", "holdout"),
        help="Dataset split; defaults to the value in the evaluation config.",
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        help="Optional explicit sequence selection, overriding the split list.",
    )
    parser.add_argument(
        "--detectors",
        choices=("both", "frcnn", "yolo"),
        help="Detectors to run; defaults to the evaluation config.",
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse existing MOT result files instead of running detection/tracking.",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Explicitly allow hold-out evaluation; reserved for the final evaluation.",
    )
    args = parser.parse_args()

    config_path = _project_path(args.config)
    if not config_path.is_file():
        parser.error(f"Evaluation config not found: {config_path}")

    try:
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        parser.error(f"Invalid YAML in {config_path}: {exc}")

    if not isinstance(settings, dict):
        parser.error("Evaluation configuration must be a YAML mapping.")

    split_name = args.split or settings.get("default_split", "tuning")
    splits = settings.get("splits", {})
    if not isinstance(splits, dict) or split_name not in splits:
        parser.error(f"No sequence list configured for split {split_name!r}.")

    sequence_names = list(args.sequence or splits[split_name])
    if not sequence_names or len(sequence_names) != len(set(sequence_names)):
        parser.error("Sequence selection must be non-empty and contain no duplicates.")

    holdout_sequences = set(splits.get("holdout", []))
    requested_holdout = holdout_sequences.intersection(sequence_names)
    if requested_holdout and not args.allow_holdout:
        parser.error(
            "Hold-out evaluation is reserved for Day 17. "
            "Requested: " + ", ".join(sorted(requested_holdout))
        )

    if args.detectors == "both":
        detector_names = ["frcnn", "yolo"]
    elif args.detectors:
        detector_names = [args.detectors]
    else:
        detector_names = list(settings.get("detectors", ["yolo"]))

    if (
        not detector_names
        or len(detector_names) != len(set(detector_names))
        or not set(detector_names).issubset({"frcnn", "yolo"})
    ):
        parser.error("Configured detectors must be unique values from frcnn/yolo.")

    variant = settings.get("variant", "FRCNN")
    if variant != "FRCNN":
        parser.error("This evaluation config currently supports the FRCNN variant.")

    if any(not name.endswith("-FRCNN") for name in sequence_names):
        parser.error("Every selected sequence must use the configured FRCNN variant.")

    data_root = _project_path(settings.get("data_root", "data/raw/MOT17/train"))
    output_root = _project_path(settings.get("output_root", "outputs/mot17/comparison"))
    if not data_root.is_dir():
        parser.error(f"MOT17 data directory not found: {data_root}")

    yolo_settings = settings.get("yolo", {})
    if not isinstance(yolo_settings, dict):
        parser.error("'yolo' must be a YAML mapping.")

    imgsz = int(yolo_settings.get("imgsz", 960))
    confidence = float(yolo_settings.get("confidence", 0.1))
    iou = float(yolo_settings.get("iou", 0.7))
    if imgsz <= 0 or not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        parser.error(
            "YOLO imgsz must be positive; confidence and iou must be in [0, 1]."
        )

    pipeline_config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    device = resolve_device(str(yolo_settings.get("device", "auto")))
    model_path = str(yolo_settings.get("model_path", "models/yolo11n.pt"))

    if (
        "yolo" in detector_names
        and not args.reuse_predictions
        and not _project_path(model_path).is_file()
    ):
        parser.error(f"YOLO model file not found: {_project_path(model_path)}")

    # Detector inference settings and tracker association settings stay separate.
    evaluation_config = SimpleNamespace(
        detector=SimpleNamespace(
            model_path=model_path,
            imgsz=imgsz,
            inference_conf=confidence,
            iou=iou,
            device=device,
        ),
        tracker=pipeline_config.tracker,
    )

    trackers_root = output_root / "trackers"
    report_name = (
        "selected_sequences_summary.json"
        if args.sequence
        else f"{split_name}_summary.json"
    )
    report_path = output_root / report_name

    report: dict[str, Any] = {
        "status": "in_progress",
        "split": split_name,
        "variant": variant,
        "sequences": sequence_names,
        "protocol": "MOTChallenge pedestrian; TrackEval HOTA, CLEAR, Identity",
        "configuration": {
            "yolo": {
                "model_path": model_path,
                "imgsz": imgsz,
                "confidence": confidence,
                "iou": iou,
                "device": device,
                "class": "person (COCO class 0)",
            },
            "bytetrack": pipeline_config.tracker.model_dump(),
        },
        "results": {
            detector: {"per_sequence": {}, "combined": None}
            for detector in detector_names
        },
    }
    _write_json(report_path, report)

    for sequence_name in sequence_names:
        sequence = read_sequence_info(data_root / sequence_name)

        for detector in detector_names:
            tracker_name = f"{detector}_bytetrack"
            prediction_file = (
                trackers_root / tracker_name / "data" / f"{sequence.name}.txt"
            )

            if args.reuse_predictions:
                if not prediction_file.is_file():
                    raise FileNotFoundError(
                        f"Cannot reuse missing prediction file: {prediction_file}"
                    )
                run_info = _load_run_info(output_root, sequence.name, detector)
            else:
                run_info = run_sequence_tracker(
                    sequence,
                    detector,
                    evaluation_config,
                    prediction_file,
                )

            metrics = evaluate_mot_challenge(
                data_root,
                trackers_root,
                output_root / "trackeval" / "per_sequence" / sequence.name / detector,
                [sequence.name],
                tracker_name=tracker_name,
                variant=variant,
            )
            report["results"][detector]["per_sequence"][sequence.name] = {
                **run_info,
                "metrics": metrics,
                "prediction_file": str(prediction_file),
            }
            _write_json(report_path, report)

            print(f"\n{sequence.name} — {detector.upper()} + ByteTrack")
            for name, value in metrics.items():
                print(
                    f"  {name}: {value:.4f}"
                    if name != "IDSW"
                    else f"  {name}: {value:.0f}"
                )

    for detector in detector_names:
        tracker_name = f"{detector}_bytetrack"
        metrics = evaluate_mot_challenge(
            data_root,
            trackers_root,
            output_root / "trackeval" / "combined" / split_name / detector,
            sequence_names,
            tracker_name=tracker_name,
            variant=variant,
        )
        report["results"][detector]["combined"] = metrics

        print(f"\n{split_name.upper()} COMBINED — {detector.upper()} + ByteTrack")
        for name, value in metrics.items():
            print(
                f"  {name}: {value:.4f}" if name != "IDSW" else f"  {name}: {value:.0f}"
            )

    report["status"] = "complete"
    _write_json(report_path, report)
    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
