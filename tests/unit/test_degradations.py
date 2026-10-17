"""Unit tests for reproducible synthetic image degradations."""

import numpy as np
import pytest

from evaluation.degradations import DEGRADATION_NAMES, apply_degradation


def make_test_frame() -> np.ndarray:
    """Return a spatially varying BGR image with edges and texture."""
    yy, xx = np.indices((64, 64))
    pattern = ((xx // 4 + yy // 4) % 2) * 180 + 35
    frame = np.stack(
        (pattern, np.roll(pattern, 3, axis=0), np.roll(pattern, 5, axis=1)),
        axis=2,
    )
    return frame.astype(np.uint8)


def test_none_returns_the_original_frame_unchanged() -> None:
    frame = make_test_frame()
    before = frame.copy()

    output = apply_degradation(frame, "none")

    assert output is frame
    np.testing.assert_array_equal(frame, before)


@pytest.mark.parametrize(
    "kind",
    ["low_light", "low_contrast", "gaussian_noise", "gaussian_blur"],
)
def test_degradations_preserve_shape_dtype_and_input(kind: str) -> None:
    frame = make_test_frame()
    before = frame.copy()

    output = apply_degradation(frame, kind)  # type: ignore[arg-type]

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(frame, before)
    assert not np.array_equal(output, frame)


def test_low_light_reduces_mean_brightness() -> None:
    frame = np.full((32, 32, 3), 180, dtype=np.uint8)

    output = apply_degradation(frame, "low_light")

    assert float(output.mean()) < float(frame.mean())


def test_low_contrast_reduces_standard_deviation() -> None:
    frame = make_test_frame()

    output = apply_degradation(frame, "low_contrast")

    assert float(output.std()) < float(frame.std())


def test_gaussian_noise_is_reproducible_for_a_fixed_seed() -> None:
    frame = make_test_frame()

    first = apply_degradation(frame, "gaussian_noise", seed=123)
    second = apply_degradation(frame, "gaussian_noise", seed=123)
    different_seed = apply_degradation(frame, "gaussian_noise", seed=124)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different_seed)


def test_supported_degradation_names_are_unique() -> None:
    assert len(DEGRADATION_NAMES) == len(set(DEGRADATION_NAMES))
    assert DEGRADATION_NAMES[0] == "none"


@pytest.mark.parametrize("kind", ["jpeg_artifact", "unknown", ""])
def test_unknown_degradation_is_rejected(kind: str) -> None:
    with pytest.raises(ValueError, match="Unsupported degradation"):
        apply_degradation(make_test_frame(), kind)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        apply_degradation(make_test_frame(), "gaussian_noise", seed=seed)  # type: ignore[arg-type]


def test_non_uint8_input_is_rejected() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="np.uint8"):
        apply_degradation(frame)  # type: ignore[arg-type]
