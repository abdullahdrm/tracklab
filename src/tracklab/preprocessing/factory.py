"""Build a frame preprocessor from validated TrackLab settings."""

from __future__ import annotations

from collections.abc import Callable

from tracklab.config import PreprocessingConfig
from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame
from tracklab.preprocessing.clahe import apply_clahe
from tracklab.preprocessing.denoise import apply_denoise
from tracklab.preprocessing.gamma import apply_gamma

Preprocessor = Callable[[Frame], Frame]


def build_preprocessor(config: PreprocessingConfig) -> Preprocessor:
    """Return a frame transform for none, gamma, CLAHE, denoise, or chain.

    ``none`` returns the original frame object unchanged. Other modes create
    a transformed frame and preserve its shape and uint8 BGR representation.
    The optional router is currently reserved and raises when enabled.
    """
    if config.router.enabled:
        raise NotImplementedError("The preprocessing router is not implemented yet.")

    def process(frame: Frame) -> Frame:
        validate_bgr_frame(frame)
        if config.mode == "none":
            return frame

        output = frame
        operations = {
            "gamma": lambda image: apply_gamma(image, config.gamma.value),
            "clahe": lambda image: apply_clahe(
                image,
                clip_limit=config.clahe.clip_limit,
                tile_grid=config.clahe.tile_grid,
            ),
            "denoise": lambda image: apply_denoise(image, config.denoise),
        }

        if config.mode == "chain":
            selected = config.chain
        else:
            selected = [config.mode]

        for operation in selected:
            try:
                output = operations[operation](output)
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported preprocessing operation: {operation}"
                ) from exc
            validate_output_frame(frame, output)

        return output

    return process
