"""Unit tests for fine-tuning configuration and dataset preflight validation."""

from pathlib import Path

import pytest
import yaml

from training.finetune import (
    CLASS_NAMES,
    load_finetune_config,
    validate_dataset_yaml,
    validate_prerequisites,
)


def _write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def _valid_config() -> dict:
    return {
        "model_path": "models/yolo11n.pt",
        "dataset_yaml": "data/processed/BDD100K_yolo6/dataset.yaml",
        "output_root": "outputs/training",
        "run_name": "test_run",
        "imgsz": 640,
        "epochs": 30,
        "batch": 4,
        "workers": 0,
        "device": "mps",
        "fraction": 1.0,
        "amp": False,
        "patience": 10,
    }


def _make_dataset(tmp_path: Path, names: list[str] | None = None) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "sample.jpg").write_bytes(b"placeholder")
        (root / "labels" / split / "sample.txt").write_text("", encoding="utf-8")
    _write_yaml(
        root / "dataset.yaml",
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(names or CLASS_NAMES)},
        },
    )
    return root / "dataset.yaml"


def test_load_finetune_config_accepts_six_class_pilot_settings(tmp_path: Path) -> None:
    path = tmp_path / "finetune.yaml"
    _write_yaml(path, _valid_config())
    config = load_finetune_config(path)
    assert config["device"] == "mps"
    assert config["fraction"] == 1.0
    assert config["batch"] == 4


def test_load_finetune_config_rejects_invalid_fraction(tmp_path: Path) -> None:
    config = _valid_config()
    config["fraction"] = 0
    path = tmp_path / "finetune.yaml"
    _write_yaml(path, config)
    with pytest.raises(ValueError, match="fraction"):
        load_finetune_config(path)


def test_load_finetune_config_rejects_unknown_device(tmp_path: Path) -> None:
    config = _valid_config()
    config["device"] = "metal"
    path = tmp_path / "finetune.yaml"
    _write_yaml(path, config)
    with pytest.raises(ValueError, match="Unsupported device"):
        load_finetune_config(path)


def test_validate_dataset_yaml_accepts_exact_class_order(tmp_path: Path) -> None:
    path = _make_dataset(tmp_path)
    report = validate_dataset_yaml(path)
    assert report["names"] == list(CLASS_NAMES)


def test_validate_dataset_yaml_rejects_wrong_class_order(tmp_path: Path) -> None:
    wrong_order = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
    path = _make_dataset(tmp_path, wrong_order)
    with pytest.raises(ValueError, match="classes/order mismatch"):
        validate_dataset_yaml(path)


def test_validate_dataset_yaml_requires_image_and_label_splits(tmp_path: Path) -> None:
    path = _make_dataset(tmp_path)
    (tmp_path / "dataset" / "labels" / "val" / "sample.txt").unlink()
    (tmp_path / "dataset" / "labels" / "val").rmdir()
    with pytest.raises(FileNotFoundError, match="val label path"):
        validate_dataset_yaml(path)


def test_validate_prerequisites_requires_pretrained_weights(tmp_path: Path) -> None:
    dataset_yaml = _make_dataset(tmp_path)
    config = _valid_config()
    config["dataset_yaml"] = str(dataset_yaml)
    with pytest.raises(FileNotFoundError, match="weights not found"):
        validate_prerequisites(config, repo_root=tmp_path)


def test_validate_prerequisites_accepts_model_and_dataset(tmp_path: Path) -> None:
    model = tmp_path / "models" / "yolo11n.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"mock pretrained weights")
    dataset_yaml = _make_dataset(tmp_path)
    config = _valid_config()
    config["model_path"] = str(model)
    config["dataset_yaml"] = str(dataset_yaml)
    report = validate_prerequisites(config, repo_root=tmp_path)
    assert report["class_names"] == list(CLASS_NAMES)
    assert report["device"] == "mps"


def test_load_finetune_config_rejects_path_like_run_name(tmp_path: Path) -> None:
    config = _valid_config()
    config["run_name"] = "../overwrite"
    path = tmp_path / "finetune.yaml"
    _write_yaml(path, config)
    loaded = load_finetune_config(path)
    dataset_yaml = _make_dataset(tmp_path)
    model = tmp_path / "models" / "yolo11n.pt"
    model.parent.mkdir(exist_ok=True)
    model.write_bytes(b"mock weights")
    loaded["model_path"] = str(model)
    loaded["dataset_yaml"] = str(dataset_yaml)
    with pytest.raises(ValueError, match="run_name"):
        validate_prerequisites(loaded, repo_root=tmp_path)
