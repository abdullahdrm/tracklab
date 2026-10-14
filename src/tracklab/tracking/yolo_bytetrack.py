"""Ultralytics YOLO detection with persistent ByteTrack tracking."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tracklab.config import PipelineConfig, resolve_device
from tracklab.schemas import TrackedObject

Frame = NDArray[np.uint8]


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a NumPy array or tensor-like value into a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


class YoloByteTrack:
    """Detect objects and maintain track IDs across consecutive frames.

    Create one instance per video. Reuse that instance for every frame of
    that video so ByteTrack can maintain its temporal state.
    """

    def __init__(
        self,
        config: PipelineConfig,
        model: Any | None = None,
    ) -> None:
        """Initialize the tracker, optionally using an injected model.

        Args:
            config: Validated TrackLab pipeline configuration.
            model: Optional YOLO-compatible model, useful for testing.
                If omitted, the configured Ultralytics model is loaded.
        """

        self.config = config
        self.device = resolve_device(config.detector.device)

        if model is None:
            from ultralytics import YOLO

            model = YOLO(config.detector.model_path)

        self.model = model

    def track_frame(self, frame: Frame) -> list[TrackedObject]:
        """Return tracked objects detected in one BGR video frame.

        Frames must be passed in chronological order. Do not reuse this
        tracker instance for an unrelated video.

        Args:
            frame: A non-empty OpenCV image with shape (height, width, 3).

        Returns:
            Tracked objects with bounding boxes in pixel coordinates.
            An empty list is returned when no active track IDs are available.

        Raises:
            ValueError: If the frame has an invalid shape or the model
                returns inconsistent detection arrays.
        """

        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
            or frame.shape[2] != 3
        ):
            raise ValueError("frame must be a non-empty HxWx3 NumPy array")

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=str(self.config.tracker_yaml),
            imgsz=self.config.detector.imgsz,
            conf=self.config.detector.conf,
            iou=self.config.detector.iou,
            classes=self.config.detector.classes,
            device=self.device,
            verbose=False,
        )

        if results is None:
            return []

        if isinstance(results, (list, tuple)):
            if not results:
                return []
            result = results[0]
        else:
            result = results

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        track_ids = getattr(boxes, "id", None)
        if track_ids is None:
            # Detections without tracker IDs are not tracked objects.
            return []

        xyxy = _to_numpy(boxes.xyxy)
        scores = _to_numpy(boxes.conf).reshape(-1)
        class_ids = _to_numpy(boxes.cls).reshape(-1)
        track_ids = _to_numpy(track_ids).reshape(-1)

        count = len(track_ids)
        if xyxy.shape != (count, 4) or len(scores) != count or len(class_ids) != count:
            raise ValueError(
                "YOLO returned inconsistent arrays for boxes, scores, "
                "classes, and track IDs"
            )

        objects: list[TrackedObject] = []

        for coordinates, score, class_id, track_id in zip(
            xyxy, scores, class_ids, track_ids, strict=True
        ):
            if not np.isfinite(track_id):
                continue

            x1, y1, x2, y2 = coordinates

            objects.append(
                TrackedObject(
                    track_id=int(track_id),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(score),
                    class_id=int(class_id),
                )
            )

        return objects
