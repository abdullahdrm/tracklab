"""Convert MOT17 pedestrian annotations to a sequence-level YOLO dataset."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from tracklab.datasets.mot17 import (
    MOT17GroundTruth,
    MOT17Sequence,
    iter_ground_truth_rows,
    read_sequence_info,
)
from tracklab.datasets.splits import load_split_plan


def mot17_row_to_yolo(
    row: MOT17GroundTruth,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Return normalized YOLO (x_center, y_center, width, height) for a person.

    MOT17 uses one-based pixel coordinates and xywh annotations. The output
    class is intentionally not included here; this project maps pedestrians
    to YOLO class 0 when writing the label file.
    """
    if (
        not math.isfinite(float(image_width))
        or not math.isfinite(float(image_height))
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("image dimensions must be finite and positive")

    # MOT17 class 1 is pedestrian. Rows with mark <= 0 are ignored regions.
    if row.class_id != 1 or row.mark <= 0:
        return None

    x1, y1 = float(row.left), float(row.top)
    box_width, box_height = float(row.width), float(row.height)
    if not all(math.isfinite(value) for value in (x1, y1, box_width, box_height)):
        return None
    if box_width <= 0 or box_height <= 0:
        return None

    x2, y2 = x1 + box_width, y1 + box_height

    # Clip both corners to the image, then calculate a new positive-area box.
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))

    clipped_width, clipped_height = x2 - x1, y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    x_center = (x1 + x2) / (2.0 * image_width)
    y_center = (y1 + y2) / (2.0 * image_height)
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height

    return (x_center, y_center, normalized_width, normalized_height)


def _normalize_sequence_id(value: str | int, variant: str) -> str:
    """Accept a two-digit sequence ID or its full MOT17 sequence name."""
    text = str(value).strip()
    full_name = re.fullmatch(rf"MOT17-(\d{{2}})-{re.escape(variant)}", text)
    if full_name:
        return full_name.group(1)
    if re.fullmatch(r"\d{1,2}", text):
        return text.zfill(2)
    raise ValueError(
        f"Invalid validation sequence {value!r}; provide a two-digit ID "
        f"or MOT17-XX-{variant}."
    )


def _load_sequence_records(
    sequence: MOT17Sequence,
) -> tuple[list[tuple[Path, list[tuple[float, float, float, float]]]], int]:
    """Pair each sequence image with its valid normalized pedestrian boxes."""
    image_paths = sorted(
        (
            path
            for path in sequence.frame_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == sequence.image_extension.lower()
        ),
        key=lambda path: path.name,
    )
    if len(image_paths) != sequence.seq_length:
        raise ValueError(
            f"Frame-count mismatch for {sequence.name}: "
            f"metadata says {sequence.seq_length}, found {len(image_paths)}"
        )

    rows_by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(
        list
    )
    for row in iter_ground_truth_rows(sequence.gt_file, seq_length=sequence.seq_length):
        box = mot17_row_to_yolo(
            row,
            image_width=sequence.width,
            image_height=sequence.height,
        )
        if box is not None:
            rows_by_frame[row.frame].append(box)

    # MOTChallenge frame numbers are one-based. The sorted image order matches
    # the sequence frame order for the zero-padded image names in MOT17.
    paired = [
        (image_path, rows_by_frame.get(frame_index, []))
        for frame_index, image_path in enumerate(image_paths, start=1)
    ]
    return paired, sum(len(boxes) for _, boxes in paired)


def _symlink_source(source: Path, destination: Path) -> None:
    """Create an image symlink, accepting an identical one on repeat runs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(
            f"Conflicting image symlink already exists: {destination}"
        )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing image: {destination}")
    destination.symlink_to(source.resolve())


def convert_mot17_to_yolo(
    data_root: str | Path,
    output_root: str | Path,
    *,
    split_config: str | Path,
    validation_sequence_id: str | int = "11",
) -> dict[str, Any]:
    """Build train/val image symlinks and YOLO labels from the tuning split only.

    The configured tuning sequences are divided by sequence, not by frame:
    the chosen validation sequence is held out from training and all other
    tuning sequences are used for training. Configured hold-out sequences are
    never opened or copied. The output directory should be dedicated to this
    generated dataset.
    """
    data_path = Path(data_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    split_path = Path(split_config).expanduser().resolve()

    if not data_path.is_dir():
        raise FileNotFoundError(f"MOT17 data directory not found: {data_path}")
    if output_path == data_path or data_path in output_path.parents:
        raise ValueError("output_root must not be inside the source MOT17 data_root")
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(
            f"Dataset output root is not a directory: {output_path}"
        )

    plan = load_split_plan(split_path)
    variant = plan.variant
    validation_id = _normalize_sequence_id(validation_sequence_id, variant)
    tuning_ids = plan.splits["tuning"]
    if validation_id not in tuning_ids:
        raise ValueError(
            f"Validation sequence {validation_id} must belong to the tuning split; "
            "hold-out sequences are reserved for final evaluation."
        )

    train_ids = tuple(
        sequence_id for sequence_id in tuning_ids if sequence_id != validation_id
    )
    if not train_ids:
        raise ValueError("At least one tuning sequence must remain for training.")

    selected = {
        "train": tuple(f"MOT17-{sequence_id}-{variant}" for sequence_id in train_ids),
        "val": (f"MOT17-{validation_id}-{variant}",),
    }

    # Read only selected tuning sequences. Do not discover all folders under
    # data_root: a hold-out directory may be intentionally inaccessible here.
    loaded: dict[
        str,
        list[
            tuple[
                MOT17Sequence,
                list[tuple[Path, list[tuple[float, float, float, float]]]],
            ]
        ],
    ] = {}
    label_counts: dict[str, int] = {}
    for split_name, sequence_names in selected.items():
        loaded[split_name] = []
        total_labels = 0
        for sequence_name in sequence_names:
            sequence = read_sequence_info(data_path / sequence_name)
            frame_records, count = _load_sequence_records(sequence)
            loaded[split_name].append((sequence, frame_records))
            total_labels += count
        label_counts[split_name] = total_labels

    # Ensure an existing output root cannot silently mix in hold-out folders.
    holdout_names = {
        f"MOT17-{sequence_id}-{variant}" for sequence_id in plan.splits["holdout"]
    }
    if output_path.exists():
        contaminated = [
            path
            for path in output_path.rglob("*")
            if any(part in holdout_names for part in path.parts)
        ]
        if contaminated:
            raise ValueError(
                "output_root already contains hold-out sequence paths; "
                "choose a clean dataset output directory."
            )

    image_counts: dict[str, int] = {}
    for split_name, sequences in loaded.items():
        image_count = 0
        for sequence, frame_records in sequences:
            for source_image, boxes in frame_records:
                relative_image = source_image.relative_to(sequence.path)
                image_destination = (
                    output_path / "images" / split_name / sequence.name / relative_image
                )
                label_destination = (
                    output_path
                    / "labels"
                    / split_name
                    / sequence.name
                    / relative_image.with_suffix(".txt")
                )
                _symlink_source(source_image, image_destination)
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                label_text = "".join(
                    "0 " + " ".join(f"{value:.6f}" for value in box) + "\n"
                    for box in boxes
                )
                label_destination.write_text(label_text, encoding="utf-8")
                image_count += 1
        image_counts[split_name] = image_count

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_yaml = output_path / "dataset.yaml"
    config = {
        "path": str(output_path),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }
    dataset_yaml.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return {
        "dataset_yaml": str(dataset_yaml),
        "train_sequences": list(selected["train"]),
        "validation_sequences": list(selected["val"]),
        "image_counts": image_counts,
        "label_counts": label_counts,
    }
