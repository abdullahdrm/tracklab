"""Test-first contract for six-class BDD100K-to-YOLO conversion."""

import json
from pathlib import Path

import pytest
import yaml

from tracklab.datasets.bdd100k_to_yolo import (
    BDD_CLASS_NAMES,
    bdd_object_to_yolo,
    convert_bdd100k_to_yolo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_bdd_record(name: str, labels: list[dict]) -> dict:
    return {"name": name, "labels": labels}


def make_box(category: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"category": category, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def test_class_order_matches_selected_coco_categories_after_dense_remap() -> None:
    assert BDD_CLASS_NAMES == (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    )


@pytest.mark.parametrize("category", ["pedestrian", "rider"])
def test_pedestrians_and_riders_map_to_person_class_zero(category: str) -> None:
    result = bdd_object_to_yolo(
        make_box(category, 10, 20, 29, 39), image_width=100, image_height=100
    )
    assert result is not None
    class_id, x_center, y_center, width, height = result
    assert class_id == 0
    assert x_center == pytest.approx(0.195)
    assert y_center == pytest.approx(0.295)
    assert width == pytest.approx(0.20)
    assert height == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("category", "expected_class"),
    [
        ("pedestrian", 0),
        ("rider", 0),
        ("bicycle", 1),
        ("bike", 1),
        ("car", 2),
        ("motorcycle", 3),
        ("motor", 3),
        ("bus", 4),
        ("truck", 5),
    ],
)
def test_selected_categories_map_to_contiguous_yolo_classes(
    category: str, expected_class: int
) -> None:
    result = bdd_object_to_yolo(
        make_box(category, 10, 10, 30, 30), image_width=100, image_height=100
    )
    assert result is not None
    assert result[0] == expected_class


@pytest.mark.parametrize(
    "category", ["train", "traffic light", "traffic sign", "unknown"]
)
def test_unselected_categories_are_ignored(category: str) -> None:
    result = bdd_object_to_yolo(
        make_box(category, 10, 10, 30, 30), image_width=100, image_height=100
    )
    assert result is None


def test_box_is_clipped_and_normalized_after_clipping() -> None:
    result = bdd_object_to_yolo(
        make_box("car", -10, 10, 19, 120), image_width=100, image_height=100
    )
    assert result is not None
    class_id, x_center, y_center, width, height = result
    assert class_id == 2
    assert all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height))
    assert width > 0.0
    assert height > 0.0


def test_malformed_or_non_finite_box_is_rejected() -> None:
    bad_box = make_box("pedestrian", 1, 2, float("nan"), 10)
    with pytest.raises(ValueError, match="finite"):
        bdd_object_to_yolo(bad_box, image_width=100, image_height=100)


def test_zero_area_box_is_ignored() -> None:
    result = bdd_object_to_yolo(
        make_box("car", 10, 10, 10, 10), image_width=100, image_height=100
    )
    assert result is None


def test_dataset_converter_writes_six_class_yaml_and_keeps_splits_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bdd100k"
    train_images = source / "images" / "train"
    val_images = source / "images" / "val"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "train_001.jpg").write_bytes(b"image-placeholder")
    (val_images / "val_001.jpg").write_bytes(b"image-placeholder")

    train_json = source / "labels" / "det_train.json"
    val_json = source / "labels" / "det_val.json"
    train_json.parent.mkdir(parents=True)
    train_json.write_text(
        json.dumps(
            [
                make_bdd_record(
                    "train_001.jpg",
                    [
                        make_box("pedestrian", 10, 10, 29, 39),
                        make_box("rider", 40, 10, 59, 39),
                        make_box("car", 30, 20, 79, 69),
                        make_box("traffic sign", 1, 1, 5, 5),
                    ],
                )
            ]
        ),
        encoding="utf-8",
    )
    val_json.write_text(
        json.dumps(
            [make_bdd_record("val_001.jpg", [make_box("truck", 10, 10, 59, 49)])]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "converted"
    report = convert_bdd100k_to_yolo(
        train_images_dir=train_images,
        val_images_dir=val_images,
        train_labels_json=train_json,
        val_labels_json=val_json,
        output_root=output,
    )

    dataset_yaml = output / "dataset.yaml"
    assert dataset_yaml.is_file()
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["names"] == {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "bus",
        5: "truck",
    }

    train_label = output / "labels" / "train" / "train_001.txt"
    val_label = output / "labels" / "val" / "val_001.txt"
    train_lines = train_label.read_text(encoding="utf-8").splitlines()
    val_lines = val_label.read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in train_lines] == ["0", "0", "2"]
    assert [line.split()[0] for line in val_lines] == ["5"]
    assert (output / "images" / "train" / "train_001.jpg").is_symlink()
    assert (output / "images" / "val" / "val_001.jpg").is_symlink()
    assert report["image_counts"] == {"train": 1, "val": 1}
    assert report["box_counts"] == {"train": 3, "val": 1}


def test_converter_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    train_images = source / "images" / "train"
    val_images = source / "images" / "val"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    train_json = source / "labels_train.json"
    val_json = source / "labels_val.json"
    train_json.write_text("[]", encoding="utf-8")
    val_json.write_text("[]", encoding="utf-8")
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        convert_bdd100k_to_yolo(
            train_images_dir=train_images,
            val_images_dir=val_images,
            train_labels_json=train_json,
            val_labels_json=val_json,
            output_root=output,
        )
