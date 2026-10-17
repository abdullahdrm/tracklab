"""Unit tests for frame-level image quality metrics."""

import numpy as np
import pytest

from tracklab.quality.metrics import compute_frame_quality


def test_black_frame_has_zero_metrics() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    result = compute_frame_quality(frame)

    assert result == {
        "brightness": 0.0,
        "contrast": 0.0,
        "sharpness": 0.0,
        "noise_estimate": 0.0,
    }


def test_white_frame_has_zero_contrast_and_sharpness() -> None:
    frame = np.full((48, 64, 3), 255, dtype=np.uint8)

    result = compute_frame_quality(frame)

    assert result["brightness"] == 255.0
    assert result["contrast"] == 0.0
    assert result["sharpness"] == 0.0
    assert result["noise_estimate"] == 0.0


def test_contrast_and_sharpness_detect_image_variation() -> None:
    pattern = np.indices((64, 64)).sum(axis=0) % 2
    gray = (pattern * 255).astype(np.uint8)
    frame = np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    result = compute_frame_quality(frame)

    assert result["contrast"] > 0
    assert result["sharpness"] > 0


def test_metrics_are_finite_and_non_negative() -> None:
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)

    result = compute_frame_quality(frame)

    assert set(result) == {
        "brightness",
        "contrast",
        "sharpness",
        "noise_estimate",
    }

    for value in result.values():
        assert np.isfinite(value)
        assert value >= 0


def test_input_frame_is_not_modified() -> None:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    original = frame.copy()

    compute_frame_quality(frame)

    np.testing.assert_array_equal(frame, original)


def test_non_numpy_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        compute_frame_quality([[0, 0, 0]])


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32, 4), dtype=np.uint8),
        np.zeros((0, 32, 3), dtype=np.uint8),
        np.zeros((32, 0, 3), dtype=np.uint8),
    ],
)
def test_invalid_frame_shapes_are_rejected(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        compute_frame_quality(frame)


def test_non_uint8_frame_is_rejected() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="np.uint8"):
        compute_frame_quality(frame)
