"""Readers and structural validation for the MOT17 dataset."""

from __future__ import annotations

import configparser
import csv
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_VARIANTS = {"DPM", "FRCNN", "SDP"}


@dataclass(frozen=True, slots=True)
class MOT17Sequence:
    """Metadata and paths for one MOT17 sequence."""

    name: str
    path: Path
    frame_dir: Path
    gt_file: Path
    frame_rate: int
    seq_length: int
    width: int
    height: int
    image_extension: str


@dataclass(frozen=True, slots=True)
class MOT17GroundTruth:
    """One raw nine-column MOT17 ground-truth annotation.

    The original columns are:
    frame, id, left, top, width, height, mark, class_id, visibility.

    This reader preserves the original mark and class fields. It deliberately
    does not decide which annotations are valid for pedestrian evaluation.
    """

    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    mark: float
    class_id: int
    visibility: float


def read_sequence_info(
    sequence_dir: str | Path,
    *,
    validate_files: bool = True,
) -> MOT17Sequence:
    """Read seqinfo.ini and optionally validate images and ground truth."""

    sequence_path = Path(sequence_dir).expanduser().resolve()
    info_path = sequence_path / "seqinfo.ini"

    if not info_path.is_file():
        raise FileNotFoundError(f"Missing sequence metadata: {info_path}")

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(info_path, encoding="utf-8")

    if "Sequence" not in parser:
        raise ValueError(f"Missing [Sequence] section in {info_path}")

    section = parser["Sequence"]

    try:
        name = section["name"].strip()
        image_dir_name = section["imDir"].strip()
        frame_rate = int(section["frameRate"])
        seq_length = int(section["seqLength"])
        width = int(section["imWidth"])
        height = int(section["imHeight"])
        image_extension = section["imExt"].strip()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid sequence metadata in {info_path}") from exc

    if name != sequence_path.name:
        raise ValueError(
            f"Sequence name mismatch: directory is {sequence_path.name!r}, "
            f"metadata says {name!r}"
        )

    if min(frame_rate, seq_length, width, height) <= 0:
        raise ValueError(f"Non-positive sequence dimensions or rate in {info_path}")

    if not image_extension.startswith("."):
        raise ValueError(f"Invalid image extension in {info_path}")

    frame_dir = (sequence_path / image_dir_name).resolve()
    gt_file = sequence_path / "gt" / "gt.txt"

    if not frame_dir.is_relative_to(sequence_path):
        raise ValueError(f"Image directory escapes sequence folder: {frame_dir}")

    if validate_files:
        if not frame_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {frame_dir}")

        if not gt_file.is_file():
            raise FileNotFoundError(f"Missing ground-truth file: {gt_file}")

        actual_frames = sum(
            1
            for file_path in frame_dir.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() == image_extension.lower()
        )

        if actual_frames != seq_length:
            raise ValueError(
                f"Frame-count mismatch for {name}: "
                f"seqinfo.ini says {seq_length}, found {actual_frames}"
            )

    return MOT17Sequence(
        name=name,
        path=sequence_path,
        frame_dir=frame_dir,
        gt_file=gt_file,
        frame_rate=frame_rate,
        seq_length=seq_length,
        width=width,
        height=height,
        image_extension=image_extension,
    )


def discover_sequences(
    root: str | Path,
    *,
    variant: str = "FRCNN",
) -> list[MOT17Sequence]:
    """Discover and validate sequence folders for one detector variant."""

    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant {variant!r}; expected one of "
            f"{sorted(SUPPORTED_VARIANTS)}"
        )

    root_path = Path(root).expanduser().resolve()

    if not root_path.is_dir():
        raise FileNotFoundError(f"MOT17 training directory not found: {root_path}")

    directories = sorted(
        path for path in root_path.glob(f"MOT17-*-{variant}") if path.is_dir()
    )

    if not directories:
        raise FileNotFoundError(
            f"No MOT17 sequence folders with variant {variant!r} found in {root_path}"
        )

    return [read_sequence_info(path) for path in directories]


def iter_ground_truth_rows(
    gt_file: str | Path,
    *,
    seq_length: int | None = None,
) -> Iterator[MOT17GroundTruth]:
    """Iterate over raw MOT17 ground-truth rows without class filtering."""

    path = Path(gt_file)

    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 columns, found {len(row)}"
                )

            try:
                frame = int(row[0].strip())
                track_id = int(row[1].strip())
                left, top, width, height, mark = (
                    float(value.strip()) for value in row[2:7]
                )
                class_id = int(row[7].strip())
                visibility = float(row[8].strip())
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid numeric annotation"
                ) from exc

            numeric_values = (
                left,
                top,
                width,
                height,
                mark,
                visibility,
            )

            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(
                    f"{path}:{line_number}: annotation contains non-finite values"
                )

            if frame < 1:
                raise ValueError(f"{path}:{line_number}: frame must be >= 1")

            if seq_length is not None and frame > seq_length:
                raise ValueError(
                    f"{path}:{line_number}: frame {frame} exceeds "
                    f"sequence length {seq_length}"
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{path}:{line_number}: bounding-box dimensions must be positive"
                )

            yield MOT17GroundTruth(
                frame=frame,
                track_id=track_id,
                left=left,
                top=top,
                width=width,
                height=height,
                mark=mark,
                class_id=class_id,
                visibility=visibility,
            )
