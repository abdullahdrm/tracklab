"""Optional classical denoising filters for uint8 BGR frames."""

from __future__ import annotations

import math
from numbers import Real

import cv2

from tracklab.config import DenoiseConfig
from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame


def apply_denoise(frame: Frame, config: DenoiseConfig) -> Frame:
    """Apply the configured denoising method without modifying the input."""
    validate_bgr_frame(frame)

    if config.method == "bilateral":
        output = cv2.bilateralFilter(
            frame,
            d=config.d,
            sigmaColor=float(config.sigma_color),
            sigmaSpace=float(config.sigma_space),
        )
    elif config.method == "fastNlMeans":
        for name, value in (("h", config.h), ("h_color", config.h_color)):
            if (
                not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        output = cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=float(config.h),
            hColor=float(config.h_color),
            templateWindowSize=7,
            searchWindowSize=21,
        )
    else:  # Defensive guard for non-validated config objects.
        raise ValueError(f"Unsupported denoise method: {config.method}")

    return validate_output_frame(frame, output)
