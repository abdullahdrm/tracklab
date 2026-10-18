"""Test-first contract for converting MOT17 pedestrian GT to YOLO labels."""

from pathlib import Path

import pytest
import yaml

from tracklab.datasets.mot17 import MOT17GroundTruth
from tracklab.datasets.to_yolo import convert_mot17_to_yolo, mot17_row_to_yolo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CONFIG = PROJECT_ROOT / "configs" / "mot17_splits.yaml"


def make_row(
    *,
    frame: int = 1,
    left: float = 10,
    top: float = 20,
    width: float = 30,
    height: float = 40,
    mark: float = 1,
    class_id: int = 1,
    visibility: float = 0.9,
) -> MOT17GroundTruth:
    return MOT17GroundTruth(
        frame=frame,
        track_id=1,
        left=left,
        top=top,
        width=width,
        height=height,
        mark=mark,
        class_id=class_id,
        visibility=visibility,
    )


def make_sequence(root: Path, sequence_id: str) -> Path:
    name = f"MOT17-{sequence_id}-FRCNN"
    sequence_dir = root / name
    image_dir = sequence_dir / "img1"
    gt_dir = sequence_dir / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    # The converter should link the source images rather than duplicate the data.
    (image_dir / "000001.jpg").write_bytes(b"frame-one")
    (image_dir / "000002.jpg").write_bytes(b"frame-two")
    (sequence_dir / "seqinfo.ini").write_text(
        f"[Sequence]\n"
        f"name={name}\n"
        f"imDir=img1\n"
        f"frameRate=30\n"
        f"seqLength=2\n"
        f"imWidth=100\n"
        f"imHeight=100\n"
        f"imExt=.jpg\n",
        encoding="utf-8",
    )
    (gt_dir / "gt.txt").write_text(
        "1,1,10,20,30,40,1,1,0.9\n"  # valid pedestrian
        "1,2,12,22,20,20,1,7,1.0\n"  # non-pedestrian class
        "1,3,20,20,20,20,0,1,1.0\n",  # ignored annotation
        encoding="utf-8",
    )
    return sequence_dir


def test_mot17_box_is_converted_to_normalized_yolo_xywh() -> None:
    result = mot17_row_to_yolo(make_row(), image_width=100, image_height=200)

    assert result == pytest.approx((0.25, 0.20, 0.30, 0.20))


def test_non_pedestrian_and_marked_ignore_rows_are_not_training_targets() -> None:
    non_pedestrian = make_row(class_id=7)
    ignored = make_row(mark=0)

    assert mot17_row_to_yolo(non_pedestrian, image_width=100, image_height=100) is None
    assert mot17_row_to_yolo(ignored, image_width=100, image_height=100) is None


def test_box_is_clipped_to_image_boundaries_before_normalization() -> None:
    row = make_row(left=-10, top=-5, width=30, height=25)

    result = mot17_row_to_yolo(row, image_width=100, image_height=100)

    assert result == pytest.approx((0.10, 0.10, 0.20, 0.20))


def test_box_fully_outside_image_is_skipped() -> None:
    row = make_row(left=120, top=10, width=20, height=20)

    assert mot17_row_to_yolo(row, image_width=100, image_height=100) is None


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_invalid_image_dimensions_are_rejected(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="image dimensions"):
        mot17_row_to_yolo(make_row(), image_width=width, image_height=height)


def test_converter_creates_sequence_level_train_val_dataset_without_holdout(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "MOT17" / "train"
    data_root.mkdir(parents=True)
    for sequence_id in ("04", "05", "09", "11"):
        make_sequence(data_root, sequence_id)

    # Deliberately malformed hold-out folder: conversion must not inspect it.
    (data_root / "MOT17-02-FRCNN").mkdir()

    output_root = tmp_path / "yolo_dataset"
    convert_mot17_to_yolo(
        data_root,
        output_root,
        split_config=SPLIT_CONFIG,
        validation_sequence_id="11",
    )

    dataset_config = yaml.safe_load(
        (output_root / "dataset.yaml").read_text(encoding="utf-8")
    )
    assert Path(dataset_config["path"]) == output_root.resolve()
    assert dataset_config["train"] == "images/train"
    assert dataset_config["val"] == "images/val"
    assert dataset_config["names"] == {0: "person"}

    train_ids = {"04", "05", "09"}
    for sequence_id in train_ids:
        image = (
            output_root
            / "images"
            / "train"
            / f"MOT17-{sequence_id}-FRCNN"
            / "img1"
            / "000001.jpg"
        )
        label = (
            output_root
            / "labels"
            / "train"
            / f"MOT17-{sequence_id}-FRCNN"
            / "img1"
            / "000001.txt"
        )
        assert image.is_symlink()
        assert (
            image.resolve()
            == (
                data_root / f"MOT17-{sequence_id}-FRCNN" / "img1" / "000001.jpg"
            ).resolve()
        )
        assert label.is_file()
        fields = label.read_text(encoding="utf-8").strip().split()
        assert fields[0] == "0"
        assert [float(value) for value in fields[1:]] == pytest.approx(
            (0.25, 0.40, 0.30, 0.40)
        )

        # A frame without an eligible pedestrian must still have an empty label.
        empty_label = label.with_name("000002.txt")
        assert empty_label.is_file()
        assert empty_label.read_text(encoding="utf-8") == ""

    val_image = (
        output_root / "images" / "val" / "MOT17-11-FRCNN" / "img1" / "000001.jpg"
    )
    assert val_image.is_symlink()
    assert (
        output_root / "labels" / "val" / "MOT17-11-FRCNN" / "img1" / "000001.txt"
    ).is_file()

    all_dataset_paths = [p.as_posix() for p in output_root.rglob("*")]
    assert not any("MOT17-02-FRCNN" in path for path in all_dataset_paths)
    assert not any("MOT17-10-FRCNN" in path for path in all_dataset_paths)
    assert not any("MOT17-13-FRCNN" in path for path in all_dataset_paths)


def test_converter_refuses_holdout_sequence_as_validation(tmp_path: Path) -> None:
    data_root = tmp_path / "MOT17" / "train"
    data_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="tuning"):
        convert_mot17_to_yolo(
            data_root,
            tmp_path / "yolo_dataset",
            split_config=SPLIT_CONFIG,
            validation_sequence_id="02",
        )
