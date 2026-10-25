"""Tests for line crossing and trajectory analytics."""

from tracklab.analytics.geometry import crossing_direction, oriented_side
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
