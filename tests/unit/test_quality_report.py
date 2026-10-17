"""Unit tests for quality metric aggregation."""

import pytest

from tracklab.quality.report import (
    QUALITY_METRICS,
    summarize_quality_samples,
)


def make_sample(offset: float = 0.0) -> dict[str, float]:
    """Create a complete and deterministic quality sample."""
    return {
        "brightness": 10.0 + offset,
        "contrast": 20.0 + offset,
        "sharpness": 30.0 + offset,
        "noise_estimate": 2.0 + offset,
    }


def test_summarizes_mean_min_and_max() -> None:
    result = summarize_quality_samples(
        [make_sample(), make_sample(20.0)],
    )

    assert result["sample_count"] == 2

    metrics = result["metrics"]
    assert isinstance(metrics, dict)

    assert metrics["brightness"] == {
        "mean": 20.0,
        "min": 10.0,
        "max": 30.0,
    }
    assert metrics["contrast"] == {
        "mean": 30.0,
        "min": 20.0,
        "max": 40.0,
    }
    assert metrics["sharpness"] == {
        "mean": 40.0,
        "min": 30.0,
        "max": 50.0,
    }
    assert metrics["noise_estimate"] == {
        "mean": 12.0,
        "min": 2.0,
        "max": 22.0,
    }


def test_single_sample_has_identical_mean_min_and_max() -> None:
    sample = make_sample(5.0)

    result = summarize_quality_samples([sample])

    assert result["sample_count"] == 1
    metrics = result["metrics"]
    assert isinstance(metrics, dict)

    for name in QUALITY_METRICS:
        assert metrics[name] == {
            "mean": sample[name],
            "min": sample[name],
            "max": sample[name],
        }


def test_empty_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_quality_samples([])


def test_missing_metric_is_rejected() -> None:
    sample = make_sample()
    del sample["sharpness"]

    with pytest.raises(ValueError, match="sharpness"):
        summarize_quality_samples([sample])


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_metric_values_are_rejected(value: float) -> None:
    sample = make_sample()
    sample["brightness"] = value

    with pytest.raises(ValueError):
        summarize_quality_samples([sample])


def test_non_numeric_metric_is_rejected() -> None:
    sample = make_sample()
    sample["brightness"] = "too bright"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="numeric"):
        summarize_quality_samples([sample])


def test_non_mapping_sample_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping"):
        summarize_quality_samples([None])  # type: ignore[list-item]


def test_build_quality_report_samples_zero_based_frames() -> None:
    import numpy as np

    from tracklab.quality.report import build_quality_report

    frames = [
        (0, np.zeros((16, 16, 3), dtype=np.uint8)),
        (1, np.full((16, 16, 3), 128, dtype=np.uint8)),
        (2, np.full((16, 16, 3), 255, dtype=np.uint8)),
        (3, np.full((16, 16, 3), 64, dtype=np.uint8)),
        (4, np.full((16, 16, 3), 32, dtype=np.uint8)),
    ]

    report = build_quality_report(frames, every_n_frames=2)

    assert report["sampling"] == {
        "every_n_frames": 2,
        "frame_index_base": 0,
    }
    assert report["sample_count"] == 3
    assert report["sampled_frame_indices"] == [0, 2, 4]

    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["brightness"] == {
        "mean": (0.0 + 255.0 + 32.0) / 3,
        "min": 0.0,
        "max": 255.0,
    }


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_build_quality_report_rejects_invalid_interval(interval) -> None:
    import numpy as np

    from tracklab.quality.report import build_quality_report

    frames = [(0, np.zeros((8, 8, 3), dtype=np.uint8))]

    with pytest.raises(ValueError, match="positive integer"):
        build_quality_report(frames, every_n_frames=interval)


def test_build_quality_report_rejects_empty_sampling_result() -> None:
    import numpy as np

    from tracklab.quality.report import build_quality_report

    frames = [(1, np.zeros((8, 8, 3), dtype=np.uint8))]

    with pytest.raises(ValueError, match="No frames"):
        build_quality_report(frames, every_n_frames=15)


def test_build_quality_report_rejects_out_of_order_frames() -> None:
    import numpy as np

    from tracklab.quality.report import build_quality_report

    frames = [
        (2, np.zeros((8, 8, 3), dtype=np.uint8)),
        (1, np.zeros((8, 8, 3), dtype=np.uint8)),
    ]

    with pytest.raises(ValueError, match="strictly increasing"):
        build_quality_report(frames, every_n_frames=1)
