"""Frame-level image quality metrics for TrackLab."""

from __future__ import annotations

import cv2
import numpy as np


def compute_frame_quality(frame: np.ndarray) -> dict[str, float]:
    """Compute basic quality metrics for a BGR video frame.

    Args:
        frame: A uint8 BGR image with shape (height, width, 3).

    Returns:
        A dictionary containing brightness, contrast, sharpness,
        and a robust high-frequency noise estimate.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the frame has an invalid shape or dtype.

    Notes:
        The noise estimate is a proxy, not a direct measurement of
        sensor noise. Fine textures and edges can also increase it.
        The input frame is not modified.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")

    if frame.dtype != np.uint8:
        raise ValueError("frame dtype must be np.uint8")

    if (
        frame.ndim != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
        or frame.shape[2] != 3
    ):
        raise ValueError("frame must be a non-empty HxWx3 BGR image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian))

    # Convert before subtraction to avoid uint8 underflow/wraparound.
    gray_float = gray.astype(np.float32)
    median_filtered = cv2.medianBlur(gray, 3).astype(np.float32)
    residual = gray_float - median_filtered

    residual_median = float(np.median(residual))
    absolute_deviation = np.abs(residual - residual_median)
    mad = float(np.median(absolute_deviation))

    # Consistency factor for a robust Gaussian-noise scale estimate.
    noise_estimate = mad / 0.67448975

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "noise_estimate": float(noise_estimate),
    }
