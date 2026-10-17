"""CLAHE contrast enhancement on the luminance channel."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import cv2

from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame


def validate_clahe_parameters(
    clip_limit: float,
    tile_grid: tuple[int, int],
) -> tuple[float, tuple[int, int]]:
    if isinstance(clip_limit, bool) or not isinstance(clip_limit, Real):
        raise TypeError("clip_limit must be a real number")
    clip_value = float(clip_limit)
    if not math.isfinite(clip_value) or clip_value <= 0:
        raise ValueError("clip_limit must be finite and greater than zero")
    if (
        not isinstance(tile_grid, (tuple, list))
        or len(tile_grid) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in tile_grid
        )
    ):
        raise ValueError("tile_grid must contain two positive integers")
    return clip_value, (int(tile_grid[0]), int(tile_grid[1]))


def build_clahe_preprocessor(
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Callable[[Frame], Frame]:
    """Build a reusable CLAHE transform; configure OpenCV only once per pipeline."""
    clip_value, grid = validate_clahe_parameters(clip_limit, tile_grid)
    clahe = cv2.createCLAHE(clipLimit=clip_value, tileGridSize=grid)

    def process(frame: Frame) -> Frame:
        validate_bgr_frame(frame)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # Modify only L in the converted LAB buffer, avoiding split + merge
        # allocations for the two chroma channels on every frame.
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return validate_output_frame(frame, output)

    return process


def apply_clahe(
    frame: Frame,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Frame:
    """Apply CLAHE to the L channel in LAB color space.

    Processing only luminance limits unnecessary color shifts. The input frame
    is not modified and output geometry/dtype match the input.
    """
    return build_clahe_preprocessor(clip_limit, tile_grid)(frame)
