"""Unit tests for TrackLab configuration loading and validation."""

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tracklab.config import load_config, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_loads_default_configuration() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")

    assert config.detector.model_path == "models/yolo11n.pt"
    assert config.detector.imgsz == 960
    assert config.detector.classes == [0, 1, 2, 3, 5, 7]
    assert config.tracker.tracker_type == "bytetrack"
    assert config.tracker.track_buffer == 30
    assert config.preprocessing.mode == "none"
    assert config.analytics.point == "foot"
    assert config.output.save_video is True
    assert config.output.save_json is True
    assert config.tracker_yaml.is_file()


def test_rejects_invalid_confidence_threshold(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs"
    copied_configs = tmp_path / "configs"
    shutil.copytree(source, copied_configs)

    detector_path = copied_configs / "detector.yaml"
    contents = detector_path.read_text(encoding="utf-8")
    assert "conf: 0.25" in contents

    detector_path.write_text(
        contents.replace("conf: 0.25", "conf: 1.5"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(copied_configs / "pipeline.yaml")


def test_missing_main_config_raises_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_resolve_device() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in {"mps", "cuda", "cpu"}


def test_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("quantum")
