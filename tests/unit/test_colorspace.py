"""Tests for detector-oriented color-space preprocessing."""

import numpy as np
import pytest

from tracklab.config import PreprocessingConfig
from tracklab.preprocessing.colorspace import build_color_space_preprocessor
from tracklab.preprocessing.factory import build_preprocessor


@pytest.mark.parametrize("color_space", ["lab", "hsv", "ycrcb"])
@pytest.mark.parametrize("order", ["clahe", "gamma_clahe", "clahe_gamma"])
def test_color_space_preprocessor_preserves_frame_contract(color_space, order) -> None:
    rng = np.random.default_rng(20261017)
    frame = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    original = frame.copy()

    transform = build_color_space_preprocessor(
        color_space=color_space,
        order=order,
        gamma=0.8,
        clip_limit=2.0,
        tile_grid=(8, 8),
    )
    output = transform(frame)

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not np.shares_memory(output, frame)
    assert not np.array_equal(output, frame)
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("color_space", ["lab", "hsv", "ycrcb"])
@pytest.mark.parametrize("order", ["clahe", "gamma_clahe", "clahe_gamma"])
def test_factory_builds_color_space_pipeline(color_space, order) -> None:
    config = PreprocessingConfig(
        mode="colorspace",
        color_space=color_space,
        color_space_order=order,
    )
    frame = np.random.default_rng(17).integers(
        0, 256, size=(96, 128, 3), dtype=np.uint8
    )

    output = build_preprocessor(config)(frame)

    assert output.shape == frame.shape
    assert output.dtype == np.uint8


def test_color_space_preprocessor_rejects_unknown_space_or_order() -> None:
    with pytest.raises(ValueError, match="Unsupported color space"):
        build_color_space_preprocessor(color_space="rgb")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported color-space order"):
        build_color_space_preprocessor(order="gamma_only")  # type: ignore[arg-type]
