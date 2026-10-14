"""Tests for the YOLO + ByteTrack adapter."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tracklab.config import load_config, resolve_device
from tracklab.schemas import TrackedObject
from tracklab.tracking.yolo_bytetrack import YoloByteTrack


class FakeBoxes:
    """Small stand-in for Ultralytics' Boxes object."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        ids: np.ndarray | None,
    ) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = ids

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeModel:
    """Return predetermined results and record inference parameters."""

    def __init__(self, results: Any) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def track(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.results


def make_config():
    """Load the project's real configuration without loading model weights."""

    return load_config(Path("configs/pipeline.yaml"))


def make_boxes(
    *,
    ids: np.ndarray | None = None,
) -> FakeBoxes:
    """Build a one-detection fake result."""

    if ids is None:
        ids = np.array([42])

    return FakeBoxes(
        xyxy=np.array([[10.2, 20.1, 50.9, 80.8]], dtype=np.float32),
        conf=np.array([0.87], dtype=np.float32),
        cls=np.array([2.0], dtype=np.float32),
        ids=ids,
    )


def test_converts_model_result_to_tracked_object() -> None:
    config = make_config()
    model = FakeModel([SimpleNamespace(boxes=make_boxes())])
    tracker = YoloByteTrack(config, model=model)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    objects = tracker.track_frame(frame)

    assert objects == [
        TrackedObject(
            track_id=42,
            x1=pytest.approx(10.2),
            y1=pytest.approx(20.1),
            x2=pytest.approx(50.9),
            y2=pytest.approx(80.8),
            score=pytest.approx(0.87),
            class_id=2,
        )
    ]


def test_uses_persistent_tracking_and_configured_parameters() -> None:
    config = make_config()
    model = FakeModel([SimpleNamespace(boxes=make_boxes())])
    tracker = YoloByteTrack(config, model=model)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    tracker.track_frame(frame)

    call = model.calls[0]
    assert call["persist"] is True
    assert call["tracker"] == str(config.tracker_yaml)
    assert call["imgsz"] == config.detector.imgsz
    assert call["conf"] == config.detector.conf
    assert call["iou"] == config.detector.iou
    assert call["classes"] == config.detector.classes
    assert call["device"] == resolve_device(config.detector.device)
    assert call["verbose"] is False


def test_returns_empty_list_when_no_results() -> None:
    tracker = YoloByteTrack(make_config(), model=FakeModel([]))
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    assert tracker.track_frame(frame) == []


def test_returns_empty_list_when_boxes_are_empty() -> None:
    empty_boxes = FakeBoxes(
        xyxy=np.empty((0, 4), dtype=np.float32),
        conf=np.empty((0,), dtype=np.float32),
        cls=np.empty((0,), dtype=np.float32),
        ids=np.empty((0,), dtype=np.float32),
    )
    tracker = YoloByteTrack(
        make_config(),
        model=FakeModel([SimpleNamespace(boxes=empty_boxes)]),
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    assert tracker.track_frame(frame) == []


def test_ignores_detections_without_track_ids() -> None:
    boxes = make_boxes()
    boxes.id = None

    tracker = YoloByteTrack(
        make_config(),
        model=FakeModel([SimpleNamespace(boxes=boxes)]),
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    assert tracker.track_frame(frame) == []


def test_rejects_invalid_frame_shape() -> None:
    tracker = YoloByteTrack(make_config(), model=FakeModel([]))

    with pytest.raises(ValueError, match="HxWx3"):
        tracker.track_frame(np.zeros((100, 120), dtype=np.uint8))
