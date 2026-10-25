"""Time-aware trajectory history for tracked objects."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from tracklab.config import TrajectoryConfig
from tracklab.schemas import TrackedObject


@dataclass(frozen=True, slots=True)
class _TrajectoryPoint:
    """A trajectory point together with its source-video timestamp."""

    timestamp_s: float
    point: tuple[float, float]


class TrajectoryStore:
    """Store complete object histories and expose time-bounded draw trails."""

    def __init__(self, config: TrajectoryConfig) -> None:
        self.config = config
        self._points: dict[int, list[_TrajectoryPoint]] = defaultdict(list)
        self._implicit_timestamp_s = 0.0

    def update(
        self,
        objects: list[TrackedObject] | tuple[TrackedObject, ...],
        timestamp_s: float | None = None,
    ) -> None:
        """Append points for visible objects at the current video timestamp.

        The pipeline supplies a timestamp derived from the source frame index
        and FPS. Direct callers may omit it for backwards compatibility; those
        calls use one-second sample intervals.
        """

        if timestamp_s is None:
            timestamp_s = self._implicit_timestamp_s
            self._implicit_timestamp_s += 1.0
        elif not isfinite(timestamp_s) or timestamp_s < 0.0:
            raise ValueError("timestamp_s must be a finite, non-negative number")

        for obj in objects:
            self._points[obj.track_id].append(
                _TrajectoryPoint(timestamp_s=timestamp_s, point=obj.foot_point)
            )

    def _smooth(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Apply a trailing moving average without changing trajectory length."""

        window = self.config.smooth_window
        smoothed: list[tuple[float, float]] = []
        for index in range(len(points)):
            chunk = points[max(0, index - window + 1) : index + 1]
            smoothed.append(
                (
                    sum(point[0] for point in chunk) / len(chunk),
                    sum(point[1] for point in chunk) / len(chunk),
                )
            )
        return smoothed

    def _visible_points(
        self,
        points: list[_TrajectoryPoint],
        timestamp_s: float | None,
    ) -> list[tuple[float, float]]:
        """Return points inside the configured inclusive time window."""

        if timestamp_s is None:
            timestamp_s = max(point.timestamp_s for point in points)

        cutoff_s = timestamp_s - self.config.trail_duration_seconds
        return [
            item.point for item in points if cutoff_s <= item.timestamp_s <= timestamp_s
        ]

    def visible_trajectories(
        self, timestamp_s: float | None = None
    ) -> dict[int, list[tuple[float, float]]]:
        """Return smoothed trails that fit the current visualization window."""

        return {
            track_id: self._smooth(self._visible_points(points, timestamp_s))
            for track_id, points in self._points.items()
            if len(points) >= self.config.min_len
            and self._visible_points(points, timestamp_s)
        }

    def summary(self) -> dict[str, list[list[float]]]:
        """Return complete smoothed histories in a JSON-safe representation."""

        return {
            str(track_id): [
                [float(x), float(y)]
                for x, y in self._smooth([item.point for item in points])
            ]
            for track_id, points in self._points.items()
            if len(points) >= self.config.min_len
        }
