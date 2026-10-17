"""Tests for standalone classical preprocessing transforms."""

from pathlib import Path

import numpy as np
import pytest

from tracklab.config import DenoiseConfig, load_config
from tracklab.preprocessing.clahe import apply_clahe
from tracklab.preprocessing.denoise import apply_denoise
from tracklab.preprocessing.factory import build_preprocessor
from tracklab.preprocessing.gamma import apply_gamma

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sample_frame() -> np.ndarray:
    y, x = np.indices((48, 64))
    gray = ((x * 3 + y * 5) % 256).astype(np.uint8)
    return np.stack((gray, np.roll(gray, 3, axis=1), np.roll(gray, 5, axis=0)), axis=2)


def make_config():
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml").preprocessing


@pytest.mark.parametrize("transform", [apply_gamma, apply_clahe])
def test_enhancement_preserves_shape_dtype_and_input(transform) -> None:
    frame = sample_frame()
    original = frame.copy()

    output = transform(frame)

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(frame, original)
    assert not np.shares_memory(output, frame)


def test_gamma_below_one_brightens_midtones() -> None:
    frame = np.full((16, 16, 3), 64, dtype=np.uint8)

    output = apply_gamma(frame, gamma=0.8)

    assert float(output.mean()) > float(frame.mean())


def test_clahe_rejects_invalid_parameters() -> None:
    frame = sample_frame()
    with pytest.raises(ValueError, match="clip_limit"):
        apply_clahe(frame, clip_limit=0)
    with pytest.raises(ValueError, match="tile_grid"):
        apply_clahe(frame, tile_grid=(8, 0))


@pytest.mark.parametrize("method", ["bilateral", "fastNlMeans"])
def test_denoise_methods_preserve_geometry_and_input(method: str) -> None:
    frame = sample_frame()
    original = frame.copy()
    config = DenoiseConfig(method=method)

    output = apply_denoise(frame, config)

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(frame, original)
    assert not np.shares_memory(output, frame)


def test_denoise_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        apply_denoise(np.zeros((10, 10), dtype=np.uint8), DenoiseConfig())


def test_none_mode_returns_original_frame_unchanged() -> None:
    frame = sample_frame()
    original = frame.copy()

    preprocess = build_preprocessor(make_config())
    output = preprocess(frame)

    assert output is frame
    np.testing.assert_array_equal(frame, original)


def test_factory_gamma_mode_preserves_geometry() -> None:
    config = make_config()
    config.mode = "gamma"
    output = build_preprocessor(config)(sample_frame())

    assert output.shape == (48, 64, 3)
    assert output.dtype == np.uint8


def test_factory_chain_applies_operations_in_order() -> None:
    config = make_config()
    config.mode = "chain"
    config.chain = ["gamma", "clahe"]

    output = build_preprocessor(config)(sample_frame())

    assert output.shape == (48, 64, 3)
    assert output.dtype == np.uint8


def test_enabled_router_is_explicitly_not_implemented() -> None:
    config = make_config()
    config.router.enabled = True

    with pytest.raises(NotImplementedError, match="router"):
        build_preprocessor(config)


def test_precompiled_factory_transform_matches_direct_transforms() -> None:
    """Compiled reusable chain should keep the public transforms' pixel results."""
    from tracklab.config import PreprocessingConfig
    from tracklab.preprocessing.factory import build_preprocessor

    config = PreprocessingConfig(mode="chain", chain=["gamma", "clahe"])
    transform = build_preprocessor(config)
    frame = np.random.default_rng(2026).integers(
        0, 256, size=(96, 128, 3), dtype=np.uint8
    )
    original = frame.copy()

    actual = transform(frame)
    expected = apply_clahe(
        apply_gamma(frame, config.gamma.value),
        config.clahe.clip_limit,
        config.clahe.tile_grid,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frame, original)
