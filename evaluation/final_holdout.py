"""Run the frozen Day 17 MOT17 hold-out evaluation exactly once."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from evaluation.mot17_detector_comparison import run_sequence_tracker
from evaluation.trackeval_bridge import evaluate_mot_challenge
from tracklab.config import (
    PreprocessingConfig,
    load_config,
    resolve_device,
)
from tracklab.datasets.mot17 import read_sequence_info
from tracklab.preprocessing.factory import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_SEQUENCES = (
    "MOT17-02-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-13-FRCNN",
)
ALLOWED_PREPROCESSING = {"none", "clahe", "ycrcb_clahe_gamma"}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = _resolve_path(path)
    with protocol_path.open("r", encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    if not isinstance(protocol, dict):
        raise TypeError("Day 17 protocol must be a YAML mapping")
    sequences = tuple(protocol.get("sequences", ()))
    if sequences != HOLDOUT_SEQUENCES:
        raise ValueError(
            "Day 17 protocol must contain exactly MOT17-02/10/13-FRCNN in that order"
        )
    if protocol.get("split") != "holdout":
        raise ValueError("Day 17 protocol split must be holdout")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Day 17 protocol must define at least one case")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("Every Day 17 case must be a mapping")
        case_id = case.get("id")
        preprocessing = case.get("preprocessing")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("Day 17 case IDs must be unique non-empty strings")
        if preprocessing not in ALLOWED_PREPROCESSING:
            raise ValueError(f"Unsupported frozen preprocessing: {preprocessing}")
        ids.add(case_id)
    return protocol


def _preprocessing_config(config: Any, name: str) -> PreprocessingConfig:
    data = config.preprocessing.model_dump(mode="python")
    if name == "none":
        data["mode"] = "none"
    elif name == "clahe":
        data["mode"] = "clahe"
    elif name == "ycrcb_clahe_gamma":
        data["mode"] = "colorspace"
        data["color_space"] = "ycrcb"
        data["color_space_order"] = "clahe_gamma"
    else:
        raise ValueError(f"Unsupported preprocessing: {name}")
    return PreprocessingConfig.model_validate(data)


def _validate_tracker(config: Any, protocol: dict[str, Any]) -> None:
    expected = protocol.get("tracker")
    if not isinstance(expected, dict):
        raise TypeError("Day 17 protocol tracker settings are missing")
    actual = config.tracker.model_dump(mode="json")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"Tracker mismatch for {key}: config={actual.get(key)!r}, "
                f"protocol={expected_value!r}"
            )


def run_final_holdout(
    protocol_path: str | Path = PROJECT_ROOT / "configs" / "day17_final_protocol.yaml",
    *,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Evaluate only the frozen hold-out cases and return the full summary."""

    protocol = _load_protocol(protocol_path)
    output_root = _resolve_path(protocol["output_root"])
    summary_path = output_root / "final_holdout_summary.json"
    if summary_path.is_file() and not allow_existing:
        raise FileExistsError(
            f"Hold-out summary already exists: {summary_path}. "
            "Use --allow-existing only to inspect/reuse the same run."
        )

    data_root = _resolve_path(protocol["data_root"])
    sequences = tuple(protocol["sequences"])
    detector_settings = protocol["detector"]
    confidence = float(detector_settings["confidence"])
    iou = float(detector_settings["iou"])
    imgsz = int(detector_settings["imgsz"])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Protocol detector confidence must be in [0, 1]")
    if not math.isfinite(iou) or not 0 <= iou <= 1 or imgsz <= 0:
        raise ValueError("Protocol detector iou/imgsz is invalid")
    device = resolve_device(str(detector_settings["device"]))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "protocol": protocol,
        "protocol_path": str(_resolve_path(protocol_path)),
        "data_root": str(data_root),
        "cases": {},
    }
    _write_json(summary_path, summary)

    for case in protocol["cases"]:
        case_id = case["id"]
        pipeline_path = _resolve_path(case["pipeline_config"])
        pipeline_config = load_config(pipeline_path)
        _validate_tracker(pipeline_config, protocol)
        preprocessor = build_preprocessor(
            _preprocessing_config(pipeline_config, case["preprocessing"])
        )
        run_config = SimpleNamespace(
            detector=SimpleNamespace(
                model_path=pipeline_config.detector.model_path,
                imgsz=imgsz,
                inference_conf=confidence,
                iou=iou,
                device=device,
            ),
            tracker=pipeline_config.tracker,
        )
        case_root = output_root / case_id
        trackers_root = case_root / "trackers"
        per_sequence: dict[str, Any] = {}

        for sequence_name in sequences:
            sequence = read_sequence_info(data_root / sequence_name)
            prediction_file = (
                trackers_root / "yolo_bytetrack" / "data" / f"{sequence.name}.txt"
            )
            print(f"[{case_id}] {sequence_name}: running frozen hold-out case")
            run_info = run_sequence_tracker(
                sequence,
                "yolo",
                run_config,
                prediction_file,
                frame_transform=preprocessor,
                verbose=False,
            )
            metrics = evaluate_mot_challenge(
                data_root,
                trackers_root,
                case_root / "trackeval" / "per_sequence" / sequence.name,
                [sequence.name],
                tracker_name="yolo_bytetrack",
                variant="FRCNN",
            )
            per_sequence[sequence_name] = {
                "run": run_info,
                "metrics": metrics,
                "prediction_file": str(prediction_file),
            }
            print(
                f"  HOTA={metrics['HOTA']:.4f} MOTA={metrics['MOTA']:.4f} "
                f"IDF1={metrics['IDF1']:.4f} IDSW={metrics['IDSW']:.0f}"
            )

        combined = evaluate_mot_challenge(
            data_root,
            trackers_root,
            case_root / "trackeval" / "combined",
            list(sequences),
            tracker_name="yolo_bytetrack",
            variant="FRCNN",
        )
        summary["cases"][case_id] = {
            "description": case["description"],
            "pipeline_config": str(pipeline_path),
            "preprocessing": case["preprocessing"],
            "per_sequence": per_sequence,
            "combined": combined,
        }
        _write_json(summary_path, summary)

    summary["status"] = "complete"
    _write_json(summary_path, summary)
    return summary
