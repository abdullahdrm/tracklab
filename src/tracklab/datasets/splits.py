"""Reproducible train-sequence split definitions for MOT17 experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPECTED_SPLITS = {"tuning", "holdout"}
SUPPORTED_VARIANTS = {"DPM", "FRCNN", "SDP"}


@dataclass(frozen=True, slots=True)
class MOT17SplitPlan:
    """Sequence IDs allocated to tuning and hold-out."""

    variant: str
    splits: dict[str, tuple[str, ...]]

    def sequence_names(self, split_name: str) -> tuple[str, ...]:
        """Return full MOT17 sequence names for one split."""

        if split_name not in self.splits:
            raise KeyError(f"Unknown split: {split_name}")

        return tuple(
            f"MOT17-{sequence_id}-{self.variant}"
            for sequence_id in self.splits[split_name]
        )


def validate_split_plan(
    plan: MOT17SplitPlan,
    available_sequence_names: set[str],
) -> None:
    """Reject overlap and sequence names absent from the dataset."""

    requested: list[str] = []

    for split_name in EXPECTED_SPLITS:
        if split_name not in plan.splits or not plan.splits[split_name]:
            raise ValueError(f"Split {split_name!r} must not be empty")

        requested.extend(plan.sequence_names(split_name))

    if len(requested) != len(set(requested)):
        raise ValueError("A sequence occurs more than once across splits")

    missing = sorted(set(requested) - available_sequence_names)
    if missing:
        raise ValueError(
            "Configured sequences are missing from the dataset: " + ", ".join(missing)
        )


def load_split_plan(path: str | Path) -> MOT17SplitPlan:
    """Load and validate the project's YAML split definition."""

    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Split configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise TypeError("Split configuration must be a YAML mapping")

    if raw.get("dataset") != "MOT17":
        raise ValueError("Split configuration must declare dataset: MOT17")

    variant = raw.get("variant")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported MOT17 variant: {variant!r}")

    raw_splits = raw.get("splits")
    if not isinstance(raw_splits, dict) or set(raw_splits) != EXPECTED_SPLITS:
        raise ValueError(f"'splits' must contain exactly: {sorted(EXPECTED_SPLITS)}")

    parsed: dict[str, tuple[str, ...]] = {}

    for split_name in ("tuning", "holdout"):
        values = raw_splits[split_name]

        if not isinstance(values, list) or not values:
            raise ValueError(f"Split {split_name!r} must be a non-empty list")

        sequence_ids: list[str] = []
        for value in values:
            sequence_id = str(value)

            if not re.fullmatch(r"\d{2}", sequence_id):
                raise ValueError(
                    f"Invalid sequence ID {value!r}; use two-digit quoted IDs "
                    "such as '02'"
                )

            sequence_ids.append(sequence_id)

        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError(f"Duplicate sequence ID inside {split_name!r}")

        parsed[split_name] = tuple(sequence_ids)

    plan = MOT17SplitPlan(variant=variant, splits=parsed)

    all_ids = [
        sequence_id
        for sequence_ids in plan.splits.values()
        for sequence_id in sequence_ids
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A sequence ID occurs in both tuning and holdout")

    return plan
