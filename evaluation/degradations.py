"""Deterministic synthetic image degradations for MOT ablation experiments."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import cv2
import numpy as np

from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame

DegradationName = Literal[
    "none",
    "low_light",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
]

DEGRADATION_NAMES: tuple[DegradationName, ...] = (
    "none",
    "low_light",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
)


def apply_degradation(
    frame: Frame,
    kind: DegradationName = "none",
    *,
    seed: int = 0,
) -> Frame:
    """Apply one fixed-severity synthetic degradation to a BGR frame.

    The input is never modified. The ``none`` condition returns the original
    frame unchanged. Random noise is reproducible for a given seed.

    Degradation severities are intentionally fixed for the first ablation:
    gamma=1.8 for low light, contrast factor=0.55, Gaussian noise sigma=15,
    and Gaussian blur kernel=7 with sigma=1.5.
    """
    validate_bgr_frame(frame)

    if kind not in DEGRADATION_NAMES:
        raise ValueError(
            f"Unsupported degradation {kind!r}; choose from {DEGRADATION_NAMES}."
        )
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    if kind == "none":
        return frame

    if kind == "low_light":
        values = np.arange(256, dtype=np.float32) / 255.0
        lookup = np.rint(np.power(values, 1.8) * 255.0).clip(0, 255).astype(np.uint8)
        output = cv2.LUT(frame, lookup)
    elif kind == "low_contrast":
        output = (
            np.rint(frame.astype(np.float32) * 0.55 + 127.5 * 0.45)
            .clip(0, 255)
            .astype(np.uint8)
        )
    elif kind == "gaussian_noise":
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(0.0, 15.0, size=frame.shape).astype(np.float32)
        output = np.rint(frame.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    elif kind == "gaussian_blur":
        output = cv2.GaussianBlur(frame, (7, 7), sigmaX=1.5)
    else:  # Keeps the validation exhaustive if names are expanded later.
        raise ValueError(f"Unsupported degradation: {kind}")

    return validate_output_frame(frame, output)
