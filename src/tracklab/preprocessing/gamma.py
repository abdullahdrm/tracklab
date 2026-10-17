"""Gamma correction for uint8 BGR frames."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import cv2
import numpy as np

from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame


def build_gamma_lookup(gamma: float) -> np.ndarray:
    """Validate gamma and create its 256-entry lookup table once."""
    if isinstance(gamma, bool) or not isinstance(gamma, Real):
        raise TypeError("gamma must be a real number")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0:
        raise ValueError("gamma must be finite and greater than zero")
    return np.clip(
        np.round(((np.arange(256, dtype=np.float32) / 255.0) ** gamma_value) * 255.0),
        0,
        255,
    ).astype(np.uint8)


def build_gamma_preprocessor(gamma: float = 0.8) -> Callable[[Frame], Frame]:
    """Build a reusable gamma transform with its lookup table precomputed."""
    lookup = build_gamma_lookup(gamma)

    def process(frame: Frame) -> Frame:
        validate_bgr_frame(frame)
        output = cv2.LUT(frame, lookup)
        return validate_output_frame(frame, output)

    return process


def apply_gamma(frame: Frame, gamma: float = 0.8) -> Frame:
    """Apply gamma correction without modifying the input frame."""
    return build_gamma_preprocessor(gamma)(frame)
