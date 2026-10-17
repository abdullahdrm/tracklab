"""CLAHE contrast enhancement on the luminance channel."""

from __future__ import annotations

import math
from numbers import Real

import cv2

from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame


def apply_clahe(
    frame: Frame,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Frame:
    """Apply CLAHE to the L channel in LAB color space.

    Processing only luminance limits unnecessary color shifts. The input
    frame is not modified and output geometry/dtype match the input.
    """
    validate_bgr_frame(frame)
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

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_value,
        tileGridSize=(int(tile_grid[0]), int(tile_grid[1])),
    )
    enhanced = clahe.apply(lightness)
    output_lab = cv2.merge((enhanced, channel_a, channel_b))
    output = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return validate_output_frame(frame, output)
