"""Bounded in-memory trajectory history for tracked objects."""

from __future__ import annotations

from collections import defaultdict

from tracklab.config import TrajectoryConfig
from tracklab.schemas import TrackedObject


class TrajectoryStore:
    """Store object points and expose smoothed trajectories for drawing/export."""

    def __init__(self, config: TrajectoryConfig) -> None:
        self.config = config
        self._points: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def update(self, objects: list[TrackedObject] | tuple[TrackedObject, ...]) -> None:
        """Append the configured point for each visible tracked object."""

        for obj in objects:
            point = obj.foot_point
            self._points[obj.track_id].append(point)

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

    def visible_trajectories(self) -> dict[int, list[tuple[float, float]]]:
        """Return smoothed trajectories that reached the configured minimum length."""

        return {
            track_id: self._smooth(points)
            for track_id, points in self._points.items()
            if len(points) >= self.config.min_len
        }

    def summary(self) -> dict[str, list[list[float]]]:
        """Return trajectories in a JSON-safe representation."""

        return {
            str(track_id): [[float(x), float(y)] for x, y in points]
            for track_id, points in self.visible_trajectories().items()
        }
