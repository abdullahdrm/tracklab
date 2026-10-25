"""Reproducible Day 16 ByteTrack parameter ablation on MOT17 tuning sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from evaluation.mot17_detector_comparison import run_sequence_tracker
from evaluation.trackeval_bridge import evaluate_mot_challenge
from tracklab.config import load_config, resolve_device
from tracklab.datasets.mot17 import read_sequence_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_SEQUENCES = (
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-11-FRCNN",
)


@dataclass(frozen=True)
class TrackerAblationCase:
    """One isolated tracker configuration."""

    case_id: str
    description: str
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "fuse_score": self.fuse_score,
        }


TRACKER_ABLATION_CASES = (
    TrackerAblationCase("baseline", "Current repository baseline"),
    TrackerAblationCase("buffer15", "Shorter lost-track buffer", track_buffer=15),
    TrackerAblationCase("buffer60", "Longer lost-track buffer", track_buffer=60),
    TrackerAblationCase("match07", "Stricter association matching", match_thresh=0.7),
    TrackerAblationCase("match09", "Looser association matching", match_thresh=0.9),
    TrackerAblationCase(
        "thresholds015",
        "Lower high/new-track thresholds",
        track_high_thresh=0.15,
        new_track_thresh=0.15,
    ),
)


def _validate_sequences(sequence_names: tuple[str, ...]) -> None:
    requested = set(sequence_names)
    if not requested or not requested.issubset(TUNING_SEQUENCES):
        raise ValueError(
            "Day 16 tracker ablation accepts only MOT17 tuning sequences: "
            + ", ".join(TUNING_SEQUENCES)
        )


def run_tracker_ablation(
    *,
    data_root: str | Path = PROJECT_ROOT / "data" / "raw" / "MOT17" / "train",
    output_root: str | Path = PROJECT_ROOT / "outputs" / "mot17" / "tracker_ablation",
    sequence_names: tuple[str, ...] = TUNING_SEQUENCES,
) -> Path:
    """Run all declared cases and save per-case TrackEval summaries."""

    _validate_sequences(sequence_names)
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    detector = SimpleNamespace(
        model_path="models/yolo11n.pt",
        imgsz=960,
        inference_conf=0.1,
        iou=0.7,
        device=resolve_device("auto"),
    )
    search_space = {
        "protocol": {
            "split": "tuning",
            "sequences": list(sequence_names),
            "detector": detector.model_path,
            "imgsz": detector.imgsz,
            "confidence": detector.inference_conf,
            "iou": detector.iou,
            "preprocessing": "none",
            "metric_protocol": "MOTChallenge pedestrian; TrackEval HOTA, CLEAR, Identity",
        },
        "cases": [
            {"case_id": case.case_id, "description": case.description, **case.as_dict()}
            for case in TRACKER_ABLATION_CASES
        ],
    }
    (output_root / "search_space.yaml").write_text(
        yaml.safe_dump(search_space, sort_keys=False), encoding="utf-8"
    )

    all_results: dict[str, Any] = {
        "status": "in_progress",
        "protocol": search_space["protocol"],
        "cases": {},
    }
    for case in TRACKER_ABLATION_CASES:
        case_root = output_root / case.case_id
        tracker_config = pipeline.tracker.model_copy(update=case.as_dict())
        config = SimpleNamespace(detector=detector, tracker=tracker_config)
        trackers_root = case_root / "trackers"
        case_result: dict[str, Any] = {
            "description": case.description,
            "tracker": case.as_dict(),
            "per_sequence": {},
        }

        for sequence_name in sequence_names:
            sequence = read_sequence_info(data_root / sequence_name)
            tracker_name = "yolo_bytetrack"
            prediction_file = (
                trackers_root / tracker_name / "data" / f"{sequence.name}.txt"
            )
            run_info = run_sequence_tracker(
                sequence,
                "yolo",
                config,
                prediction_file,
                verbose=False,
            )
            metrics = evaluate_mot_challenge(
                data_root,
                trackers_root,
                case_root / "trackeval" / "per_sequence" / sequence.name,
                [sequence.name],
                tracker_name=tracker_name,
                variant="FRCNN",
            )
            case_result["per_sequence"][sequence.name] = {
                **run_info,
                "metrics": metrics,
            }
            print(
                f"{case.case_id} {sequence.name}: "
                f"HOTA={metrics['HOTA']:.4f}, MOTA={metrics['MOTA']:.4f}, "
                f"IDF1={metrics['IDF1']:.4f}, IDSW={metrics['IDSW']:.0f}"
            )

        case_result["combined"] = evaluate_mot_challenge(
            data_root,
            trackers_root,
            case_root / "trackeval" / "combined",
            list(sequence_names),
            tracker_name="yolo_bytetrack",
            variant="FRCNN",
        )
        all_results["cases"][case.case_id] = case_result
        (output_root / "tracker_ablation_summary.json").write_text(
            json.dumps(all_results, indent=2) + "\n", encoding="utf-8"
        )

    all_results["status"] = "complete"
    summary_path = output_root / "tracker_ablation_summary.yaml"
    summary_path.write_text(
        yaml.safe_dump(all_results, sort_keys=False), encoding="utf-8"
    )
    print(f"Tracker ablation summary: {summary_path}")
    return summary_path
