"""Geometry helpers used by line crossing and trajectory analytics."""

from __future__ import annotations

from typing import Literal

from tracklab.config import LineConfig

CrossingDirection = Literal["positive_to_negative", "negative_to_positive"]


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
