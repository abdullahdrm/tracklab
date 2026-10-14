"""Unit tests for TrackLab data schemas."""

from tracklab.schemas import (
    FrameResult,
    TrackedObject,
    VideoMeta,
    VideoResult,
)


def test_tracked_object_geometry() -> None:
    obj = TrackedObject(
        track_id=1,
        x1=10,
        y1=20,
        x2=30,
        y2=60,
        score=0.95,
        class_id=0,
    )

    assert obj.foot_point == (20.0, 60)
    assert obj.center == (20.0, 40.0)
    assert obj.xywh == (10, 20, 20, 40)


def test_frame_result_stores_objects() -> None:
    obj = TrackedObject(
        track_id=7,
        x1=1,
        y1=2,
        x2=11,
        y2=22,
        score=0.8,
        class_id=0,
    )

    frame = FrameResult(
        frame_index=0,
        timestamp_s=0.0,
        objects=[obj],
    )

    assert frame.frame_index == 0
    assert frame.timestamp_s == 0.0
    assert frame.objects == [obj]


def test_video_result_stores_metadata_and_frames() -> None:
    meta = VideoMeta(
        name="sample.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=450,
    )
    frame = FrameResult(
        frame_index=0,
        timestamp_s=0.0,
        objects=[],
    )

    result = VideoResult(meta=meta, frames=[frame])

    assert result.meta.name == "sample.mp4"
    assert result.meta.width == 1920
    assert result.meta.height == 1080
    assert result.meta.fps == 30.0
    assert result.meta.frame_count == 450
    assert len(result.frames) == 1
    assert result.summary == {}
    assert result.config == {}


def test_video_result_default_dictionaries_are_independent() -> None:
    meta = VideoMeta(
        name="sample.mp4",
        width=640,
        height=480,
        fps=25.0,
        frame_count=100,
    )

    first = VideoResult(meta=meta, frames=[])
    second = VideoResult(meta=meta, frames=[])

    first.summary["unique_tracks"] = 3

    assert second.summary == {}
