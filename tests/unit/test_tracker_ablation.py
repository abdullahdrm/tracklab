"""Tests for the predeclared Day 16 tracker ablation protocol."""

import pytest

from evaluation.tracker_ablation import (
    TRACKER_ABLATION_CASES,
    TUNING_SEQUENCES,
    _validate_sequences,
)


def test_tracker_ablation_cases_are_unique_and_include_required_groups() -> None:
    ids = [case.case_id for case in TRACKER_ABLATION_CASES]
    assert len(ids) == len(set(ids))
    assert {case.track_buffer for case in TRACKER_ABLATION_CASES} == {15, 30, 60}
    assert {case.match_thresh for case in TRACKER_ABLATION_CASES} == {0.7, 0.8, 0.9}
    assert any(case.case_id == "thresholds015" for case in TRACKER_ABLATION_CASES)


def test_tracker_ablation_rejects_holdout_sequences() -> None:
    _validate_sequences(TUNING_SEQUENCES)
    with pytest.raises(ValueError, match="only MOT17 tuning sequences"):
        _validate_sequences(("MOT17-02-FRCNN",))
