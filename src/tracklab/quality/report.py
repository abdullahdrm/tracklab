"""Aggregation utilities for sampled frame quality metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real

import numpy as np

from tracklab.quality.metrics import compute_frame_quality

QUALITY_METRICS = (
    "brightness",
    "contrast",
    "sharpness",
    "noise_estimate",
)


def summarize_quality_samples(
    samples: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    """Summarize previously computed quality metrics across sampled frames.

    Each sample must contain brightness, contrast, sharpness, and
    noise_estimate values. Additional keys are ignored.

    Returns:
        A JSON-serializable dictionary with the sample count and each
        metric's mean, minimum, and maximum.

    Raises:
        ValueError: If no samples are supplied, a required metric is missing,
            or a metric value is negative or non-finite.
        TypeError: If a sample is not a mapping or a metric is not numeric.
    """
    if not samples:
        raise ValueError("At least one quality sample is required.")

    values_by_metric: dict[str, list[float]] = {name: [] for name in QUALITY_METRICS}

    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"Sample {sample_index} must be a mapping.")

        missing = [name for name in QUALITY_METRICS if name not in sample]
        if missing:
            raise ValueError(
                f"Sample {sample_index} is missing metrics: {', '.join(missing)}"
            )

        for name in QUALITY_METRICS:
            raw_value = sample[name]

            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError(
                    f"Metric '{name}' in sample {sample_index} must be numeric."
                )

            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"Metric '{name}' in sample {sample_index} must be finite."
                )
            if value < 0:
                raise ValueError(
                    f"Metric '{name}' in sample {sample_index} cannot be negative."
                )

            values_by_metric[name].append(value)

    metric_summary = {
        name: {
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
        for name, values in values_by_metric.items()
    }

    return {
        "sample_count": len(samples),
        "metrics": metric_summary,
    }


def build_quality_report(
    frames: Iterable[tuple[int, np.ndarray]],
    every_n_frames: int = 15,
) -> dict[str, object]:
    """Measure frame quality at a fixed interval in a frame stream.

    The input is an iterable of (zero-based frame index, BGR image) pairs,
    such as those yielded by tracklab.io.video.read_frames.

    Frame zero is included. For every_n_frames=15, the selected indices
    are 0, 15, 30, and so on.

    Returns:
        A JSON-serializable report containing sampling settings, sampled
        frame indices, per-frame metrics, and aggregate metric summaries.

    Raises:
        ValueError: If the interval is invalid, frame indices are not
            strictly increasing and non-negative, or no frame is sampled.
    """
    if (
        isinstance(every_n_frames, bool)
        or not isinstance(every_n_frames, Integral)
        or every_n_frames <= 0
    ):
        raise ValueError("every_n_frames must be a positive integer.")

    interval = int(every_n_frames)
    samples: list[dict[str, float | int]] = []
    previous_index = -1

    for frame_index, frame in frames:
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, Integral)
            or frame_index < 0
        ):
            raise ValueError("Frame indices must be non-negative integers.")

        index = int(frame_index)
        if index <= previous_index:
            raise ValueError("Frame indices must be strictly increasing.")
        previous_index = index

        if index % interval != 0:
            continue

        metrics = compute_frame_quality(frame)
        samples.append({"frame_index": index, **metrics})

    if not samples:
        raise ValueError("No frames were selected for quality sampling.")

    metric_summary = summarize_quality_samples(
        [{name: sample[name] for name in QUALITY_METRICS} for sample in samples]
    )

    return {
        "sampling": {
            "every_n_frames": interval,
            "frame_index_base": 0,
        },
        "sample_count": len(samples),
        "sampled_frame_indices": [int(sample["frame_index"]) for sample in samples],
        "samples": samples,
        "metrics": metric_summary["metrics"],
    }
