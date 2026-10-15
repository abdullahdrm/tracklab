"""Tests for the reproducible MOT17 tuning/hold-out split."""

from pathlib import Path

import pytest
import yaml

from tracklab.datasets.splits import load_split_plan, validate_split_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_split_matches_planned_sequence_ids() -> None:
    plan = load_split_plan(PROJECT_ROOT / "configs" / "mot17_splits.yaml")

    assert plan.variant == "FRCNN"
    assert plan.sequence_names("tuning") == (
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-11-FRCNN",
    )
    assert plan.sequence_names("holdout") == (
        "MOT17-02-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-13-FRCNN",
    )


def test_split_validation_rejects_missing_sequence() -> None:
    plan = load_split_plan(PROJECT_ROOT / "configs" / "mot17_splits.yaml")

    with pytest.raises(ValueError, match="missing from the dataset"):
        validate_split_plan(plan, {"MOT17-04-FRCNN"})


def test_loader_rejects_overlapping_splits(tmp_path: Path) -> None:
    config_path = tmp_path / "splits.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "MOT17",
                "variant": "FRCNN",
                "splits": {
                    "tuning": ["02"],
                    "holdout": ["02"],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both tuning and holdout"):
        load_split_plan(config_path)
