"""Draw tracked objects on video frames."""

from collections.abc import Mapping, Sequence
from math import isfinite

import cv2
import numpy as np
from numpy.typing import NDArray

from tracklab.config import LineConfig
from tracklab.schemas import TrackedObject

Frame = NDArray[np.uint8]
ClassNames = Mapping[int, str] | Sequence[str] | None


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track ID."""

    return (
        (track_id * 37 + 91) % 156 + 100,
        (track_id * 17 + 53) % 156 + 100,
        (track_id * 29 + 71) % 156 + 100,
    )


def _class_name(class_id: int, class_names: ClassNames) -> str:
    """Resolve a class ID to a name, with a safe fallback."""

    if isinstance(class_names, Mapping):
        name = class_names.get(class_id)
        return str(name) if name is not None else f"class_{class_id}"

    if class_names is not None and 0 <= class_id < len(class_names):
        return str(class_names[class_id])

    return f"class_{class_id}"


def draw_tracks(
    frame: Frame,
    objects: Sequence[TrackedObject],
    class_names: ClassNames = None,
    show_confidence: bool = True,
    thickness: int = 2,
) -> Frame:
    """Return a copy of a frame annotated with tracked-object boxes and labels.

    Bounding boxes use pixel coordinates in the format
    (x1, y1, x2, y2), where the two points are opposite corners.

    Args:
        frame: Source image in OpenCV's BGR channel order.
        objects: Tracked objects detected in this frame.
        class_names: Optional mapping or sequence of class names.
        show_confidence: Whether to display each detection's confidence.
        thickness: Bounding-box line thickness in pixels.

    Returns:
        A newly allocated image with bounding boxes and labels drawn.

    Raises:
        ValueError: If the frame is not a non-empty three-channel image
            or the requested line thickness is less than one.
    """

    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
    ):
        raise ValueError("frame must be a non-empty HxWx3 NumPy array")

    if thickness < 1:
        raise ValueError("thickness must be at least 1")

    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for obj in objects:
        coordinates = (obj.x1, obj.y1, obj.x2, obj.y2)

        if not all(isfinite(value) for value in coordinates):
            continue

        x1, y1, x2, y2 = (round(value) for value in coordinates)

        # Clip coordinates to the image boundaries.
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        # Skip boxes that are empty or invalid after clipping.
        if x2 <= x1 or y2 <= y1:
            continue

        color = _track_color(obj.track_id)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        label = f"{_class_name(obj.class_id, class_names)} ID:{obj.track_id}"
        if show_confidence:
            label += f" {obj.score:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.65, min(1.35, height / 720.0 * 0.9))
        text_thickness = max(2, round(font_scale * 1.5))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )

        padding = max(5, round(font_scale * 6))
        panel_top = max(0, y1 - text_height - baseline - 2 * padding)
        panel_width = text_width + 2 * padding
        panel_left = min(x1, max(0, width - 1 - panel_width))
        panel_right = min(width - 1, panel_left + panel_width)
        panel_bottom = min(height - 1, panel_top + text_height + baseline + 2 * padding)

        cv2.rectangle(
            annotated,
            (panel_left, panel_top),
            (panel_right, panel_bottom),
            color,
            thickness=-1,
        )

        text_origin = (
            panel_left + padding,
            min(height - 1, panel_top + padding + text_height),
        )
        cv2.putText(
            annotated,
            label,
            text_origin,
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def draw_analytics(
    frame: Frame,
    *,
    lines: Sequence[LineConfig] = (),
    line_counts: Mapping[str, Mapping[str, int]] | None = None,
    trajectories: Mapping[int, Sequence[tuple[float, float]]] | None = None,
) -> Frame:
    """Draw configured counting lines, counts, and trajectory trails."""

    annotated = frame.copy()
    line_counts = line_counts or {}
    trajectories = trajectories or {}

    for track_id, points in trajectories.items():
        if len(points) < 2:
            continue
        polyline = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            annotated,
            [polyline],
            isClosed=False,
            color=_track_color(int(track_id)),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    for index, line in enumerate(lines):
        p1 = tuple(round(value) for value in line.p1)
        p2 = tuple(round(value) for value in line.p2)
        cv2.line(annotated, p1, p2, (0, 220, 255), 3, cv2.LINE_AA)
        counts = line_counts.get(line.id, {})
        label = (
            f"{line.id}: +→− {counts.get('positive_to_negative', 0)}  "
            f"−→+ {counts.get('negative_to_positive', 0)}"
        )
        origin = (20, 35 + index * 30)
        cv2.putText(
            annotated,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated
