"""Compare precomputed MOT17 detections and YOLO using the same ByteTrack."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from evaluation.trackeval_bridge import evaluate_mot_challenge
from tracklab.config import load_config, resolve_device
from tracklab.datasets.mot17 import MOT17Sequence, read_sequence_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_frcnn_detections(
    detection_file: str | Path,
    seq_length: int,
) -> dict[int, np.ndarray]:
    """Load unsorted MOTChallenge det.txt rows, grouped by 1-based frame.

    Each returned row uses [x1, y1, x2, y2, confidence, class_id].
    The detector file supplies top-left x/y and width/height.
    """

    path = Path(detection_file)
    if not path.is_file():
        raise FileNotFoundError(f"Missing FRCNN detections: {path}")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    grouped: dict[int, list[list[float]]] = defaultdict(list)

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)

        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 7:
                raise ValueError(
                    f"{path}:{line_number}: expected 7 columns, found {len(row)}"
                )

            try:
                frame_value = float(row[0])
                x, y, width, height, score = map(float, row[2:7])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid numeric detection"
                ) from exc

            values = (frame_value, x, y, width, height, score)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path}:{line_number}: detection contains non-finite values"
                )
            if not frame_value.is_integer():
                raise ValueError(
                    f"{path}:{line_number}: frame number must be an integer"
                )

            frame = int(frame_value)
            if not 1 <= frame <= seq_length:
                raise ValueError(
                    f"{path}:{line_number}: frame {frame} is outside 1..{seq_length}"
                )
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{path}:{line_number}: box width and height must be positive"
                )
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{path}:{line_number}: confidence must be in [0, 1]")

            grouped[frame].append([x, y, x + width, y + height, score, 0.0])

    return {
        frame: np.asarray(grouped[frame], dtype=np.float32).reshape(-1, 6)
        for frame in range(1, seq_length + 1)
    }


def detections_to_boxes(
    detections: np.ndarray,
    frame_shape: tuple[int, int],
) -> Boxes:
    """Wrap Nx6 xyxy/conf/class detections in Ultralytics' Results-like type."""

    array = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive")

    return Boxes(array, orig_shape=(height, width))


def tracker_rows_to_mot(
    frame_number: int,
    tracked: np.ndarray,
) -> list[list[int | float]]:
    """Convert ByteTrack Nx8 output into 10-column MOTChallenge tracker rows."""

    if frame_number < 1:
        raise ValueError("MOTChallenge frame numbers are 1-based")

    array = np.asarray(tracked, dtype=np.float32)
    if array.size == 0:
        return []
    if array.ndim != 2 or array.shape[1] != 8:
        raise ValueError(
            f"Expected ByteTrack output with shape (N, 8), got {array.shape}"
        )

    rows: list[list[int | float]] = []
    for item in array:
        x1, y1, x2, y2, track_id, score, _class_id, _det_index = item.tolist()
        if not np.isfinite(item).all():
            raise ValueError("ByteTrack output contains non-finite values")
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            raise ValueError("ByteTrack returned a non-positive bounding box")

        rows.append(
            [
                frame_number,
                int(track_id),
                float(x1),
                float(y1),
                float(width),
                float(height),
                float(score),
                -1,
                -1,
                -1,
            ]
        )

    return rows


def _get_frame_paths(sequence: MOT17Sequence) -> dict[int, Path]:
    """Map actual image indices to files and ensure no frame is missing."""

    paths: dict[int, Path] = {}
    extension = sequence.image_extension.lower()

    for path in sequence.frame_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != extension:
            continue
        try:
            frame_number = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"Unexpected image filename: {path}") from exc
        if frame_number in paths:
            raise ValueError(f"Duplicate frame number {frame_number}")
        paths[frame_number] = path

    expected = set(range(1, sequence.seq_length + 1))
    actual = set(paths)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise ValueError(f"Frame-index mismatch; missing={missing}, extra={extra}")

    return paths


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array to NumPy."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def run_sequence_tracker(
    sequence: MOT17Sequence,
    detector_kind: str,
    config: Any,
    output_file: str | Path,
    *,
    frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    verbose: bool = True,
) -> dict[str, float | int]:
    """Generate MOTChallenge predictions for one detector and one sequence."""

    if detector_kind not in {"frcnn", "yolo"}:
        raise ValueError("detector_kind must be 'frcnn' or 'yolo'")

    frame_paths = _get_frame_paths(sequence)
    frcnn_by_frame: dict[int, np.ndarray] = {}
    model = None

    if detector_kind == "frcnn":
        frcnn_by_frame = load_frcnn_detections(
            sequence.path / "det" / "det.txt",
            sequence.seq_length,
        )
    else:
        model_path = Path(config.detector.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        model = YOLO(str(model_path))

    # A fresh tracker is essential: track IDs must not leak between runs.
    tracker = BYTETracker(SimpleNamespace(**config.tracker.model_dump()))
    device = resolve_device(config.detector.device)
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")

    frame_count = 0
    detection_count = 0
    tracked_count = 0
    start_time = time.perf_counter()

    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")

            for frame_number in range(1, sequence.seq_length + 1):
                image_path = frame_paths[frame_number]
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise ValueError(f"OpenCV could not read {image_path}")

                if detector_kind == "frcnn":
                    detections = frcnn_by_frame[frame_number]
                else:
                    inference_frame = frame
                    if frame_transform is not None:
                        inference_frame = frame_transform(frame)
                        if not isinstance(inference_frame, np.ndarray):
                            raise TypeError("frame_transform must return a NumPy array")
                        if (
                            inference_frame.shape != frame.shape
                            or inference_frame.dtype != frame.dtype
                        ):
                            raise ValueError(
                                "frame_transform must preserve frame shape and dtype"
                            )

                    # Use the ByteTrack low threshold so low-confidence
                    # detections remain available for its second association.
                    predictions = model.predict(
                        source=inference_frame,
                        imgsz=config.detector.imgsz,
                        conf=getattr(
                            config.detector,
                            "inference_conf",
                            config.tracker.track_low_thresh,
                        ),
                        iou=config.detector.iou,
                        classes=[0],  # COCO person class only
                        device=device,
                        verbose=False,
                    )
                    result = predictions[0] if predictions else None
                    boxes = None if result is None else result.boxes
                    if boxes is None or len(boxes) == 0:
                        detections = np.empty((0, 6), dtype=np.float32)
                    else:
                        detections = _to_numpy(boxes.data).astype(
                            np.float32, copy=False
                        )

                boxes_for_tracker = detections_to_boxes(
                    detections,
                    frame.shape[:2],
                )
                tracked = tracker.update(boxes_for_tracker, img=frame)

                for row in tracker_rows_to_mot(frame_number, tracked):
                    writer.writerow(row)

                frame_count += 1
                detection_count += len(detections)
                tracked_count += len(tracked)

                if verbose and (
                    frame_number % 100 == 0 or frame_number == sequence.seq_length
                ):
                    print(
                        f"{detector_kind}: frame {frame_number}/"
                        f"{sequence.seq_length}, "
                        f"detections={detection_count}, "
                        f"tracked outputs={tracked_count}"
                    )

        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - start_time
    return {
        "frames": frame_count,
        "detections": detection_count,
        "tracked_outputs": tracked_count,
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FRCNN and YOLO detections with the same ByteTrack."
    )
    parser.add_argument(
        "--sequence",
        default="MOT17-04-FRCNN",
        help="MOT17 FRCNN sequence folder name (default: MOT17-04-FRCNN)",
    )
    parser.add_argument(
        "--detectors",
        choices=("both", "frcnn", "yolo"),
        default="both",
        help="Which detector paths to run (default: both)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "MOT17" / "train",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "mot17" / "comparison",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not args.sequence.endswith("-FRCNN"):
        parser.error("--sequence must name an MOT17 FRCNN sequence")

    sequence = read_sequence_info(data_root / args.sequence)
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    trackers_root = output_root / "trackers"
    kinds = ("frcnn", "yolo") if args.detectors == "both" else (args.detectors,)

    summary: dict[str, Any] = {
        "sequence": sequence.name,
        "protocol": "MOTChallenge pedestrian; TrackEval HOTA",
        "tracker_config": config.tracker.model_dump(),
        "yolo": {
            "model": config.detector.model_path,
            "imgsz": config.detector.imgsz,
            "confidence_floor": config.tracker.track_low_thresh,
            "classes": [0],
            "device": resolve_device(config.detector.device),
        },
        "comparisons": {},
    }

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"{sequence.name}_summary.json"

    for kind in kinds:
        tracker_name = f"{kind}_bytetrack"
        prediction_file = trackers_root / tracker_name / "data" / f"{sequence.name}.txt"

        run_info = run_sequence_tracker(
            sequence,
            kind,
            config,
            prediction_file,
        )
        metrics = evaluate_mot_challenge(
            data_root,
            trackers_root,
            output_root / "trackeval" / kind,
            [sequence.name],
            tracker_name=tracker_name,
            variant="FRCNN",
        )

        summary["comparisons"][kind] = {
            "tracker_name": tracker_name,
            "prediction_file": str(prediction_file),
            **run_info,
            "metrics": metrics,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"\n{kind.upper()} + ByteTrack metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
