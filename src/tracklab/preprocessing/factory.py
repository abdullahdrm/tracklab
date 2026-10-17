"""Build a frame preprocessor from validated TrackLab settings."""

from __future__ import annotations

from collections.abc import Callable

from tracklab.config import PreprocessingConfig
from tracklab.preprocessing.base import Frame, validate_bgr_frame, validate_output_frame
from tracklab.preprocessing.clahe import build_clahe_preprocessor
from tracklab.preprocessing.colorspace import build_color_space_preprocessor
from tracklab.preprocessing.denoise import apply_denoise
from tracklab.preprocessing.gamma import build_gamma_preprocessor

Preprocessor = Callable[[Frame], Frame]


def build_preprocessor(config: PreprocessingConfig) -> Preprocessor:
    """Compile a reusable transform for none, one method, or a fixed chain.

    Parameter-dependent work (gamma LUT and CLAHE object construction) is done
    once when a pipeline is built, rather than repeating it for every frame.
    The optional router is reserved and raises when enabled.
    """
    if config.router.enabled:
        raise NotImplementedError("The preprocessing router is not implemented yet.")

    if config.mode == "colorspace":
        return build_color_space_preprocessor(
            color_space=config.color_space,
            order=config.color_space_order,
            gamma=config.gamma.value,
            clip_limit=config.clahe.clip_limit,
            tile_grid=config.clahe.tile_grid,
        )

    if config.mode == "none":

        def identity(frame: Frame) -> Frame:
            validate_bgr_frame(frame)
            return frame

        return identity

    operations: dict[str, Preprocessor] = {
        "gamma": build_gamma_preprocessor(config.gamma.value),
        "clahe": build_clahe_preprocessor(
            clip_limit=config.clahe.clip_limit,
            tile_grid=config.clahe.tile_grid,
        ),
        "denoise": lambda image: apply_denoise(image, config.denoise),
    }
    selected = config.chain if config.mode == "chain" else [config.mode]
    try:
        transforms = tuple(operations[name] for name in selected)
    except KeyError as exc:
        raise ValueError(f"Unsupported preprocessing operation: {exc.args[0]}") from exc

    def process(frame: Frame) -> Frame:
        validate_bgr_frame(frame)
        output = frame
        for transform in transforms:
            output = transform(output)
            validate_output_frame(frame, output)
        return output

    return process
