"""Convert BDD100K detection annotations to a six-class YOLO dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

BDD_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)

# The requested COCO categories are remapped to contiguous target IDs 0..5.
_CATEGORY_TO_CLASS = {
    "pedestrian": 0,
    "rider": 0,
    "person": 0,
    "bicycle": 1,
    "bike": 1,
    "car": 2,
    "motorcycle": 3,
    "motor": 3,
    "bus": 4,
    "truck": 5,
}

# BDD100K detection images use 1280x720 pixels. Keep these defaults explicit;
# the converter does not decode images, so it also works with image symlinks.
BDD_IMAGE_WIDTH = 1280
BDD_IMAGE_HEIGHT = 720


def bdd_object_to_yolo(
    obj: dict[str, Any],
    *,
    image_width: int = BDD_IMAGE_WIDTH,
    image_height: int = BDD_IMAGE_HEIGHT,
) -> tuple[int, float, float, float, float] | None:
    """Convert one BDD100K box to ``class, x_center, y_center, w, h``.

    Unselected categories and degenerate/out-of-image boxes return ``None``.
    BDD100K uses the Scalabel pixel-coordinate convention: box width and height
    are ``x2 - x1 + 1`` and ``y2 - y1 + 1``. Coordinates are clipped to the
    valid pixel range before normalization.
    """
    if (
        not isinstance(image_width, int)
        or isinstance(image_width, bool)
        or not isinstance(image_height, int)
        or isinstance(image_height, bool)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("image dimensions must be positive integers")

    if not isinstance(obj, dict):
        raise TypeError("BDD object must be a mapping")

    category = obj.get("category")
    if not isinstance(category, str):
        raise TypeError("BDD object category must be a string")
    class_id = _CATEGORY_TO_CLASS.get(category.strip().lower())
    if class_id is None:
        return None

    box = obj.get("box2d")
    if not isinstance(box, dict):
        raise TypeError("selected BDD object is missing a box2d mapping")

    try:
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BDD box2d must contain numeric x1, y1, x2, and y2") from exc

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("BDD box coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        return None

    # Coordinates refer to pixels, whose valid indices end at width-1/height-1.
    x1 = min(max(x1, 0.0), float(image_width - 1))
    x2 = min(max(x2, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    y2 = min(max(y2, 0.0), float(image_height - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    width = (x2 - x1 + 1.0) / image_width
    height = (y2 - y1 + 1.0) / image_height
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height

    # Floating-point clipping protects the serialized YOLO coordinates.
    values = tuple(min(max(v, 0.0), 1.0) for v in (x_center, y_center, width, height))
    return (class_id, *values)


def _read_label_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"BDD labels JSON not found: {path}")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BDD labels JSON: {path}") from exc
    if not isinstance(records, list):
        raise TypeError(f"BDD labels JSON must contain a list: {path}")
    if any(not isinstance(record, dict) for record in records):
        raise TypeError(f"Every BDD labels record must be a mapping: {path}")
    return records


def _convert_split(
    *,
    split_name: str,
    images_dir: Path,
    labels_json: Path,
    output_root: Path,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"BDD image directory not found: {images_dir}")

    records = _read_label_records(labels_json)
    record_names: set[str] = set()
    image_count = 0
    box_count = 0

    for record in records:
        raw_name = record.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"BDD labels record in {labels_json} has no image name")
        name = raw_name.strip()
        # Accept a plain filename only, avoiding path traversal and split leakage.
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"BDD image name must be a filename, got {name!r}")
        if name in record_names:
            raise ValueError(f"Duplicate image record {name!r} in {labels_json}")
        record_names.add(name)

        source = images_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Image referenced by labels is missing: {source}")
        destination = output_root / "images" / split_name / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())

        labels = record.get("labels")
        if labels is None:
            labels = []
        if not isinstance(labels, list):
            raise TypeError(f"Labels for {name!r} must be a list or null")

        lines: list[str] = []
        for obj in labels:
            if not isinstance(obj, dict):
                raise TypeError(f"An object in {name!r} is not a mapping")
            converted = bdd_object_to_yolo(
                obj, image_width=image_width, image_height=image_height
            )
            if converted is None:
                continue
            class_id, x_center, y_center, width, height = converted
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            box_count += 1

        label_path = output_root / "labels" / split_name / f"{Path(name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        image_count += 1

    return image_count, box_count


def convert_bdd100k_to_yolo(
    *,
    train_images_dir: str | Path,
    val_images_dir: str | Path,
    train_labels_json: str | Path,
    val_labels_json: str | Path,
    output_root: str | Path,
    image_width: int = BDD_IMAGE_WIDTH,
    image_height: int = BDD_IMAGE_HEIGHT,
) -> dict[str, Any]:
    """Create a six-class YOLO dataset with separate train/validation sources.

    Images are linked rather than copied. The destination must not already
    exist: refusing it avoids mixing new labels with stale or mismatched files.
    """
    output = Path(output_root).expanduser().resolve()
    train_images = Path(train_images_dir).expanduser().resolve()
    val_images = Path(val_images_dir).expanduser().resolve()
    train_json = Path(train_labels_json).expanduser().resolve()
    val_json = Path(val_labels_json).expanduser().resolve()

    if output.exists():
        raise FileExistsError(
            f"Dataset output already exists; refusing to overwrite or merge: {output}"
        )
    if output in train_images.parents or output in val_images.parents:
        raise ValueError(
            "output_root must not be an ancestor of source image directories"
        )
    if train_images == val_images:
        raise ValueError("train_images_dir and val_images_dir must be separate")
    if not isinstance(image_width, int) or not isinstance(image_height, int):
        raise TypeError("image dimensions must be integers")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    # Validate source files and records before creating the destination.
    if not train_images.is_dir():
        raise FileNotFoundError(f"BDD train image directory not found: {train_images}")
    if not val_images.is_dir():
        raise FileNotFoundError(
            f"BDD validation image directory not found: {val_images}"
        )
    _read_label_records(train_json)
    _read_label_records(val_json)

    output.mkdir(parents=True, exist_ok=False)
    try:
        train_image_count, train_box_count = _convert_split(
            split_name="train",
            images_dir=train_images,
            labels_json=train_json,
            output_root=output,
            image_width=image_width,
            image_height=image_height,
        )
        val_image_count, val_box_count = _convert_split(
            split_name="val",
            images_dir=val_images,
            labels_json=val_json,
            output_root=output,
            image_width=image_width,
            image_height=image_height,
        )

        config = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(BDD_CLASS_NAMES)},
        }
        (output / "dataset.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        # Remove a partial generated dataset so a rerun cannot mistake it for complete.
        import shutil

        shutil.rmtree(output, ignore_errors=True)
        raise

    return {
        "dataset_yaml": str(output / "dataset.yaml"),
        "class_names": list(BDD_CLASS_NAMES),
        "image_counts": {"train": train_image_count, "val": val_image_count},
        "box_counts": {"train": train_box_count, "val": val_box_count},
    }
