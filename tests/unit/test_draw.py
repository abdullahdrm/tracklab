"""Tests for tracked-object visualization."""

import numpy as np
import pytest

from tracklab.schemas import TrackedObject
from tracklab.viz.draw import draw_tracks


def make_object(
    *,
    track_id: int = 7,
    x1: float = 10,
    y1: float = 10,
    x2: float = 40,
    y2: float = 40,
    score: float = 0.91,
    class_id: int = 2,
) -> TrackedObject:
    """Create a tracked object for visualization tests."""

    return TrackedObject(
        track_id=track_id,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=score,
        class_id=class_id,
    )


def test_empty_objects_return_an_unchanged_copy() -> None:
    frame = np.zeros((64, 80, 3), dtype=np.uint8)

    result = draw_tracks(frame, [])

    assert result is not frame
    assert np.array_equal(result, frame)


def test_draws_box_without_mutating_source_frame() -> None:
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    original = frame.copy()

    result = draw_tracks(
        frame,
        [make_object()],
        class_names={2: "car"},
    )

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
    assert np.any(result != original)
    assert np.array_equal(frame, original)
    # The bottom box edge should be visible away from the text label.
    assert np.any(result[40, 20] != original[40, 20])


def test_clips_boxes_to_frame_boundaries() -> None:
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    obj = make_object(x1=-10, y1=-5, x2=20, y2=20)

    result = draw_tracks(frame, [obj])

    assert np.any(result != frame)
    assert np.any(result[20, 10] != frame[20, 10])


def test_skips_invalid_box() -> None:
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    obj = make_object(x1=30, y1=30, x2=10, y2=10)

    result = draw_tracks(frame, [obj])

    assert np.array_equal(result, frame)


def test_skips_non_finite_box_coordinates() -> None:
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    obj = make_object(x1=float("nan"))

    result = draw_tracks(frame, [obj])

    assert np.array_equal(result, frame)


def test_rejects_invalid_frame_shape() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        draw_tracks(np.zeros((64, 80), dtype=np.uint8), [])


def test_rejects_non_positive_thickness() -> None:
    frame = np.zeros((64, 80, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="thickness"):
        draw_tracks(frame, [], thickness=0)


def test_draws_box_at_right_boundary_without_overflow() -> None:
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    obj = make_object(x1=70, y1=10, x2=79, y2=40)

    result = draw_tracks(frame, [obj], class_names={2: "car"})
    assert result.shape == frame.shape
    assert np.any(result != frame)
