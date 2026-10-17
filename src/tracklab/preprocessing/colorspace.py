"""Detection-oriented gamma/CLAHE enhancement in LAB, HSV, or YCrCb."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import cv2

from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame
from tracklab.preprocessing.clahe import validate_clahe_parameters
from tracklab.preprocessing.gamma import build_gamma_lookup

ColorSpaceName = Literal["lab", "hsv", "ycrcb"]
ColorSpaceOrder = Literal["clahe", "gamma_clahe", "clahe_gamma"]

_COLOR_SPACE_CONVERSIONS: dict[str, tuple[int, int, int]] = {
    # Each tuple is (BGR -> space, space -> BGR, brightness/value channel index).
    "lab": (cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, 0),
    "hsv": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, 2),
    "ycrcb": (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR, 0),
}
_VALID_ORDERS = {"clahe", "gamma_clahe", "clahe_gamma"}


def build_color_space_preprocessor(
    color_space: ColorSpaceName = "lab",
    order: ColorSpaceOrder = "gamma_clahe",
    gamma: float = 0.8,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Callable[[Frame], Frame]:
    """Compile a reusable brightness-channel transform for detector inputs.

    LAB operates on L, HSV on V, and YCrCb on Y. The frame is converted to the
    selected space once, only the selected channel is changed, and the result is
    converted back to BGR once. Gamma/CLAHE objects are prepared before frames
    arrive so their setup cost is not paid on every frame.
    """
    if color_space not in _COLOR_SPACE_CONVERSIONS:
        raise ValueError(f"Unsupported color space: {color_space}")
    if order not in _VALID_ORDERS:
        raise ValueError(f"Unsupported color-space order: {order}")

    clip_value, grid = validate_clahe_parameters(clip_limit, tile_grid)
    clahe = cv2.createCLAHE(clipLimit=clip_value, tileGridSize=grid)
    gamma_lut = build_gamma_lookup(gamma) if order != "clahe" else None
    to_space, to_bgr, channel_index = _COLOR_SPACE_CONVERSIONS[color_space]

    def process(frame: Frame) -> Frame:
        validate_bgr_frame(frame)
        converted = cv2.cvtColor(frame, to_space)
        channel = converted[:, :, channel_index]

        if order == "clahe":
            enhanced = clahe.apply(channel)
        elif order == "gamma_clahe":
            assert gamma_lut is not None
            enhanced = clahe.apply(cv2.LUT(channel, gamma_lut))
        else:
            assert gamma_lut is not None
            enhanced = cv2.LUT(clahe.apply(channel), gamma_lut)

        converted[:, :, channel_index] = enhanced
        output = cv2.cvtColor(converted, to_bgr)
        return validate_output_frame(frame, output)

    return process
