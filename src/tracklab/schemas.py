"""Shared data structures used throughout the TrackLab pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackedObject:
    """A detected object associated with a tracker ID.

    Bounding boxes use the original frame's pixel coordinates:
    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right.
    """

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    @property
    def foot_point(self) -> tuple[float, float]:
        """Return the midpoint of the bottom edge of the bounding box."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Return the box as (top-left x, top-left y, width, height)."""
        return (
            self.x1,
            self.y1,
            self.x2 - self.x1,
            self.y2 - self.y1,
        )


@dataclass
class FrameResult:
    """Tracking results for one video frame.

    Frame indices are zero-based.
    """

    frame_index: int
    timestamp_s: float
    objects: list[TrackedObject]


@dataclass
class VideoMeta:
    """Basic metadata describing a source video."""

    name: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class VideoResult:
    """Complete result produced by processing a video."""

    meta: VideoMeta
    frames: list[FrameResult]
    summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
