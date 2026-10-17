"""Unit tests for the controlled ablation matrix and frame transforms."""

from pathlib import Path

import numpy as np
import pytest

from evaluation.ablation import (
    PREPROCESSING_MODES,
    REAL_COLORSPACE_CASES,
    REAL_PREPROCESSING_CHAINS,
    SYNTHETIC_DEGRADATIONS,
    TUNING_SEQUENCES,
    build_ablation_cases,
    build_frame_transform,
    select_ablation_cases,
    validate_tuning_split,
)
from tracklab.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_real_ablation_matrix_includes_single_methods_chains_and_color_spaces() -> None:
    cases = build_ablation_cases("real")

    expected_count = (
        len(PREPROCESSING_MODES)
        + len(REAL_PREPROCESSING_CHAINS)
        + len(REAL_COLORSPACE_CASES)
    )
    assert len(cases) == expected_count

    single_end = len(PREPROCESSING_MODES)
    chain_end = single_end + len(REAL_PREPROCESSING_CHAINS)
    single_cases = cases[:single_end]
    chain_cases = cases[single_end:chain_end]
    color_cases = cases[chain_end:]

    assert tuple(case.preprocessing for case in single_cases) == PREPROCESSING_MODES
    assert tuple(case.chain for case in chain_cases) == REAL_PREPROCESSING_CHAINS
    assert all(case.preprocessing == "chain" for case in chain_cases)
    assert tuple(case.case_id for case in chain_cases) == tuple(
        "_".join(chain) for chain in REAL_PREPROCESSING_CHAINS
    )
    assert (
        tuple(
            (case.case_id, case.color_space, case.color_space_order)
            for case in color_cases
        )
        == REAL_COLORSPACE_CASES
    )
    assert all(case.preprocessing == "colorspace" for case in color_cases)
    assert {case.degradation for case in cases} == {"none"}
    assert {case.stage for case in cases} == {"real"}


def test_synthetic_matrix_covers_each_degradation_and_preprocessor() -> None:
    cases = build_ablation_cases("synthetic")

    assert len(cases) == len(SYNTHETIC_DEGRADATIONS) * len(PREPROCESSING_MODES)
    assert {case.degradation for case in cases} == set(SYNTHETIC_DEGRADATIONS)
    assert {case.preprocessing for case in cases} == set(PREPROCESSING_MODES)
    assert len({case.case_id for case in cases}) == len(cases)


def test_all_stage_has_thirty_four_unique_cases() -> None:
    cases = build_ablation_cases("all")

    assert len(cases) == 34
    assert len({(case.stage, case.case_id) for case in cases}) == 34


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="stage must be"):
        build_ablation_cases("holdout")  # type: ignore[arg-type]


def test_tuning_split_is_allowlisted_and_holdout_is_separate() -> None:
    result = validate_tuning_split(
        list(TUNING_SEQUENCES),
        ["MOT17-02-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"],
    )
    assert result == TUNING_SEQUENCES

    with pytest.raises(ValueError, match="only permits"):
        validate_tuning_split(
            [*TUNING_SEQUENCES, "MOT17-02-FRCNN"],
            ["MOT17-02-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"],
        )


def _preprocessing_config():
    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml").preprocessing


def test_none_transform_returns_the_original_frame() -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    transform = build_frame_transform(_preprocessing_config(), "none")

    result = transform(frame)

    assert result is frame


def test_chained_preprocessor_applies_multiple_methods_and_preserves_geometry() -> None:
    config = _preprocessing_config().model_copy(
        update={"mode": "chain", "chain": ["gamma", "clahe"]}
    )
    frame = np.full((24, 32, 3), 80, dtype=np.uint8)
    before = frame.copy()

    result = build_frame_transform(config)(frame)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
    assert not np.array_equal(result, frame)
    np.testing.assert_array_equal(frame, before)


def test_gamma_transform_preserves_geometry_and_does_not_mutate_input() -> None:
    config = _preprocessing_config().model_copy(update={"mode": "gamma"})
    frame = np.full((24, 32, 3), 80, dtype=np.uint8)
    before = frame.copy()
    transform = build_frame_transform(config)

    result = transform(frame)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
    assert not np.array_equal(result, frame)
    np.testing.assert_array_equal(frame, before)


def test_synthetic_noise_is_reproducible_per_sequence() -> None:
    config = _preprocessing_config()
    frames = [np.full((24, 32, 3), 100, dtype=np.uint8) for _ in range(2)]
    first_transform = build_frame_transform(config, "gaussian_noise", seed=99)
    second_transform = build_frame_transform(config, "gaussian_noise", seed=99)

    first_outputs = [first_transform(frame.copy()) for frame in frames]
    second_outputs = [second_transform(frame.copy()) for frame in frames]

    for first, second in zip(first_outputs, second_outputs, strict=True):
        np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first_outputs[0], first_outputs[1])


def test_transform_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be"):
        build_frame_transform(_preprocessing_config(), seed=-1)


def test_case_selector_allows_focused_realtime_benchmark() -> None:
    cases = select_ablation_cases("real", ("none", "clahe", "gamma_clahe"))
    assert tuple(case.case_id for case in cases) == ("none", "clahe", "gamma_clahe")


def test_case_selector_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="Unknown case ID"):
        select_ablation_cases("real", ("not_a_real_case",))
