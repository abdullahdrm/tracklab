"""Directional line-crossing counter for tracked objects."""

from __future__ import annotations

from dataclasses import dataclass

from tracklab.analytics.geometry import (
    CrossingDirection,
    crossing_direction,
    oriented_side,
)
from tracklab.config import AnalyticsConfig, LineConfig
from tracklab.schemas import TrackedObject


@dataclass
class _TrackLineState:
    last_side: float
    first_frame: int
    last_crossing_frame: int | None = None


class LineCounter:
    """Count directional crossings while suppressing short-lived track noise."""

    def __init__(self, config: AnalyticsConfig) -> None:
        self.config = config
        self.lines: tuple[LineConfig, ...] = tuple(config.lines)
        self._states: dict[tuple[str, int], _TrackLineState] = {}
        self._counts: dict[str, dict[str, int]] = {
            line.id: {"positive_to_negative": 0, "negative_to_positive": 0}
            for line in self.lines
        }

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """Return a JSON-safe copy of the current line counts."""

        return {line_id: values.copy() for line_id, values in self._counts.items()}

    def update(
        self,
        frame_index: int,
        objects: list[TrackedObject] | tuple[TrackedObject, ...],
    ) -> list[tuple[str, int, CrossingDirection]]:
        """Update state and return crossings detected in the current frame."""

        events: list[tuple[str, int, CrossingDirection]] = []
        seen: set[tuple[str, int]] = set()
        for obj in objects:
            point = obj.foot_point if self.config.point == "foot" else obj.center
            for line in self.lines:
                key = (line.id, obj.track_id)
                seen.add(key)
                side = oriented_side(point, line)
                state = self._states.get(key)
                if state is None:
                    self._states[key] = _TrackLineState(side, frame_index)
                    continue

                direction = crossing_direction(state.last_side, side)
                age = frame_index - state.first_frame + 1
                cooldown_ok = state.last_crossing_frame is None or (
                    frame_index - state.last_crossing_frame
                    >= self.config.cooldown_frames
                )
                if (
                    direction is not None
                    and age >= self.config.min_track_age
                    and cooldown_ok
                ):
                    self._counts[line.id][direction] += 1
                    state.last_crossing_frame = frame_index
                    events.append((line.id, obj.track_id, direction))
                # Keep the last non-zero side while the point lies on the line;
                # otherwise a positive -> line -> negative path would be lost.
                if abs(side) > 1e-9:
                    state.last_side = side

        # A track that was not visible this frame cannot be considered to have crossed.
        # Its previous side is retained so a later reappearance still requires an
        # actual sign change in the observed points.
        del seen
        return events

    def summary(self) -> dict[str, dict[str, int | dict[str, int]]]:
        """Return counts with a convenient total for JSON/API consumers."""

        return {
            line_id: {
                **values,
                "total": sum(values.values()),
            }
            for line_id, values in self._counts.items()
        }
