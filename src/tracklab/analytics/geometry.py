"""Geometry helpers used by line crossing and trajectory analytics."""

from __future__ import annotations

from typing import Literal

from tracklab.config import LineConfig

CrossingDirection = Literal["positive_to_negative", "negative_to_positive"]


def resolve_line(line: LineConfig, *, width: int, height: int) -> LineConfig:
    """Resolve one line into source-frame pixel coordinates.

    Pixel-space lines are returned unchanged. Normalized coordinates use the
    inclusive image range, so 0.0 maps to pixel 0 and 1.0 maps to pixel
    ``width - 1`` or ``height - 1``. The returned object is a copy, leaving
    the validated shared configuration safe to reuse for another video.
    """

    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if line.coordinate_space == "pixels":
        return line

    max_x = width - 1
    max_y = height - 1
    return line.model_copy(
        update={
            "p1": (line.p1[0] * max_x, line.p1[1] * max_y),
            "p2": (line.p2[0] * max_x, line.p2[1] * max_y),
            "coordinate_space": "pixels",
        }
    )


def resolve_lines(
    lines: tuple[LineConfig, ...] | list[LineConfig],
    *,
    width: int,
    height: int,
) -> tuple[LineConfig, ...]:
    """Resolve all configured lines once for one source-frame size."""

    return tuple(resolve_line(line, width=width, height=height) for line in lines)


def oriented_side(point: tuple[float, float], line: LineConfig) -> float:
    """Return the signed cross product of the line and point vectors."""

    x, y = point
    x1, y1 = line.p1
    x2, y2 = line.p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crossing_direction(
    previous_side: float,
    current_side: float,
    *,
    epsilon: float = 1e-9,
) -> CrossingDirection | None:
    """Return a direction only when a point crosses from one side to the other."""

    if abs(previous_side) <= epsilon or abs(current_side) <= epsilon:
        return None
    if previous_side > 0 > current_side:
        return "positive_to_negative"
    if previous_side < 0 < current_side:
        return "negative_to_positive"
    return None
