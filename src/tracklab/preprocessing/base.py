"""Shared contracts and validation for frame preprocessors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


def validate_bgr_frame(frame: np.ndarray) -> None:
    """Require a non-empty uint8 BGR image without modifying it."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.dtype != np.uint8:
        raise ValueError("frame dtype must be np.uint8")
    if (
        frame.ndim != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
        or frame.shape[2] != 3
    ):
        raise ValueError("frame must be a non-empty HxWx3 BGR image")


def validate_output_frame(source: Frame, output: Frame) -> Frame:
    """Validate that a preprocessor kept image geometry and dtype intact."""
    validate_bgr_frame(output)
    if output.shape != source.shape:
        raise ValueError(
            f"Preprocessing must preserve frame shape {source.shape}; got {output.shape}."
        )
    if output.dtype != source.dtype:
        raise ValueError(
            f"Preprocessing must preserve frame dtype {source.dtype}; got {output.dtype}."
        )
    return output
