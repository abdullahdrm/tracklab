"""Tests for line crossing and trajectory analytics."""

import pytest

from tracklab.analytics.geometry import (
    crossing_direction,
    oriented_side,
    resolve_line,
    resolve_lines,
)
from tracklab.analytics.line_counter import LineCounter
from tracklab.analytics.trajectories import TrajectoryStore
from tracklab.config import AnalyticsConfig, LineConfig, TrajectoryConfig
from tracklab.schemas import TrackedObject


def _object(track_id: int, x: float, y: float) -> TrackedObject:
    return TrackedObject(track_id, x - 1, y - 1, x + 1, y + 1, 0.9, 0)


def test_oriented_side_and_crossing_direction() -> None:
    line = LineConfig(id="gate", p1=(0, 0), p2=(10, 0))
    assert oriented_side((2, 3), line) > 0
    assert crossing_direction(2, -1) == "positive_to_negative"
    assert crossing_direction(-1, 2) == "negative_to_positive"
    assert crossing_direction(0, -1) is None


def test_legacy_pixel_lines_are_not_rescaled() -> None:
    line = LineConfig(id="legacy", p1=(100, 540), p2=(1800, 540))

    resolved = resolve_line(line, width=768, height=432)

    assert resolved is line
    assert resolved.coordinate_space == "pixels"
    assert resolved.p1 == (100, 540)
    assert resolved.p2 == (1800, 540)


def test_normalized_lines_resolve_using_inclusive_pixel_boundaries() -> None:
    line = LineConfig(
        id="normalized",
        coordinate_space="normalized",
        p1=(0.0, 0.5),
        p2=(1.0, 1.0),
    )

    full_hd = resolve_line(line, width=1920, height=1080)
    small = resolve_line(line, width=768, height=432)

    assert full_hd.coordinate_space == "pixels"
    assert full_hd.p1 == (0.0, 539.5)
    assert full_hd.p2 == (1919.0, 1079.0)
    assert small.p1 == (0.0, 215.5)
    assert small.p2 == (767.0, 431.0)
    assert line.coordinate_space == "normalized"
    assert line.p1 == (0.0, 0.5)


def test_normalized_coordinates_require_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"within \[0\.0, 1\.0\]"):
        LineConfig(
            id="invalid",
            coordinate_space="normalized",
            p1=(-0.01, 0.5),
            p2=(1.0, 0.5),
        )


def test_line_coordinates_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        LineConfig(
            id="non-finite",
            coordinate_space="normalized",
            p1=(float("nan"), 0.5),
            p2=(1.0, 0.5),
        )


def test_resolving_lines_for_two_videos_does_not_mutate_shared_config() -> None:
    line = LineConfig(
        id="shared",
        coordinate_space="normalized",
        p1=(0.1, 0.5),
        p2=(0.9, 0.5),
    )

    first = resolve_lines([line], width=1920, height=1080)[0]
    second = resolve_lines([line], width=768, height=432)[0]

    assert first.p1 != second.p1
    assert first.p2 != second.p2
    assert line.coordinate_space == "normalized"
    assert line.p1 == (0.1, 0.5)


def test_normalized_line_counter_counts_a_directional_crossing() -> None:
    config = AnalyticsConfig(
        lines=[
            LineConfig(
                id="gate",
                coordinate_space="normalized",
                p1=(0.0, 0.5),
                p2=(1.0, 0.5),
            )
        ],
        point="center",
        min_track_age=2,
    )
    resolved = resolve_lines(config.lines, width=100, height=100)
    counter = LineCounter(config, lines=resolved)

    counter.update(0, [_object(4, 50, 60)])
    events = counter.update(1, [_object(4, 50, 40)])

    assert events == [("gate", 4, "positive_to_negative")]
    assert counter.summary()["gate"]["total"] == 1


def test_line_counter_requires_track_age_and_counts_direction() -> None:
    config = AnalyticsConfig(
        lines=[LineConfig(id="gate", p1=(0, 0), p2=(10, 0))],
        point="center",
        min_track_age=2,
        cooldown_frames=3,
    )
    counter = LineCounter(config)
    counter.update(0, [_object(4, 5, -2)])
    counter.update(1, [_object(4, 5, 2)])
    counter.update(2, [_object(4, 5, -2)])
    assert counter.summary()["gate"]["negative_to_positive"] == 1
    assert counter.summary()["gate"]["positive_to_negative"] == 0


def test_line_counter_cooldown_suppresses_immediate_recrossing() -> None:
    config = AnalyticsConfig(
        lines=[LineConfig(id="gate", p1=(0, 0), p2=(10, 0))],
        point="center",
        min_track_age=1,
        cooldown_frames=4,
    )
    counter = LineCounter(config)
    for frame, y in enumerate((-2, 2, -2, 2, -2, 2)):
        counter.update(frame, [_object(4, 5, y)])
    assert counter.summary()["gate"]["total"] == 2


def test_line_counter_handles_a_point_that_lands_on_the_line() -> None:
    config = AnalyticsConfig(
        lines=[LineConfig(id="gate", p1=(0, 0), p2=(10, 0))],
        point="center",
        min_track_age=1,
    )
    counter = LineCounter(config)
    for frame, y in enumerate((-2, 0, 2)):
        counter.update(frame, [_object(4, 5, y)])
    assert counter.summary()["gate"]["negative_to_positive"] == 1


def test_trajectory_store_smooths_and_filters_short_tracks() -> None:
    store = TrajectoryStore(TrajectoryConfig(min_len=3, smooth_window=2))
    for x in range(3):
        store.update([_object(4, float(x), 10)])
    trajectories = store.visible_trajectories()
    assert list(trajectories) == [4]
    assert len(trajectories[4]) == 3
    assert trajectories[4][1][0] == 0.5
    assert store.summary()["4"][-1] == [1.5, 11.0]
