"""Gamma correction for uint8 BGR frames."""

from __future__ import annotations

import math
from numbers import Real

import cv2
import numpy as np

from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame


def apply_gamma(frame: Frame, gamma: float = 0.8) -> Frame:
    """Apply gamma correction using a lookup table.

    Values below 1 brighten the image; values above 1 darken it.
    The input frame is not modified.
    """
    validate_bgr_frame(frame)
    if isinstance(gamma, bool) or not isinstance(gamma, Real):
        raise TypeError("gamma must be a real number")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0:
        raise ValueError("gamma must be finite and greater than zero")

    lookup = np.clip(
        np.round(((np.arange(256, dtype=np.float32) / 255.0) ** gamma_value) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    output = cv2.LUT(frame, lookup)
    return validate_output_frame(frame, output)
