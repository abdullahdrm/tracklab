"""Validate and run a reproducible six-class YOLO fine-tuning experiment.

By default this module only validates the local model and dataset. Training
requires an explicit ``--train`` flag. ``--pilot`` runs one epoch on a 1%
training subset to verify the end-to-end training path before a longer run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "finetune.yaml"
CLASS_NAMES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def load_finetune_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a fine-tuning config and validate its non-path settings."""
    config_path = Path(path).expanduser().resolve()
    config = _read_yaml(config_path)
    required = {
        "model_path",
        "dataset_yaml",
        "output_root",
        "run_name",
        "imgsz",
        "epochs",
        "batch",
        "workers",
        "device",
        "fraction",
        "amp",
        "cls_remap",
        "patience",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Missing fine-tuning config keys: " + ", ".join(missing))

    for key in ("model_path", "dataset_yaml", "output_root", "run_name", "device"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise TypeError(f"{key} must be a non-empty string")

    for key in ("imgsz", "epochs", "batch"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise TypeError(f"{key} must be an integer")
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")

    if isinstance(config["workers"], bool) or not isinstance(config["workers"], int):
        raise TypeError("workers must be an integer")
    if config["workers"] < 0:
        raise ValueError("workers must be non-negative")

    if isinstance(config["patience"], bool) or not isinstance(config["patience"], int):
        raise TypeError("patience must be an integer")
    if config["patience"] < 0:
        raise ValueError("patience must be non-negative")

    fraction = config["fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)):
        raise TypeError("fraction must be a number")
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    config["fraction"] = float(fraction)

    if not isinstance(config["amp"], bool):
        raise TypeError("amp must be a boolean")
    if not isinstance(config["cls_remap"], bool):
        raise TypeError("cls_remap must be a boolean")
    device = config["device"].strip().lower()
    valid_device = device in {"auto", "mps", "cpu", "cuda"} or (
        device.startswith("cuda:") and device[5:].isdigit()
    )
    if not valid_device:
        raise ValueError(f"Unsupported device: {config['device']!r}")
    config["device"] = device
    return config


def _resolve_repo_path(value: str | Path, repo_root: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(repo_root) / candidate
    return candidate.resolve()


def _class_names_from_yaml(raw: Mapping[str, Any]) -> tuple[str, ...]:
    names = raw.get("names")
    if isinstance(names, list):
        return tuple(str(name) for name in names)
    if isinstance(names, dict):
        try:
            indexed = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset class-name keys must be integer IDs") from exc
        if set(indexed) != set(range(len(indexed))):
            raise ValueError("Dataset class IDs must be contiguous and start at zero")
        return tuple(indexed[index] for index in range(len(indexed)))
    raise TypeError("Dataset YAML 'names' must be a list or an ID-to-name mapping")


def validate_dataset_yaml(
    dataset_yaml: str | Path,
    *,
    expected_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Verify exact class order and the on-disk train/validation structure."""
    yaml_path = Path(dataset_yaml).expanduser().resolve()
    raw = _read_yaml(yaml_path)
    actual_names = _class_names_from_yaml(raw)
    if actual_names != expected_names:
        raise ValueError(
            f"Dataset classes/order mismatch: expected {list(expected_names)}, "
            f"got {list(actual_names)}"
        )

    root_value = raw.get("path", str(yaml_path.parent))
    if not isinstance(root_value, str) or not root_value.strip():
        raise TypeError("Dataset YAML 'path' must be a non-empty string")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    for split in ("train", "val"):
        value = raw.get(split)
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"Dataset YAML '{split}' must be a non-empty path")
        images_dir = Path(value).expanduser()
        if not images_dir.is_absolute():
            images_dir = root / images_dir
        images_dir = images_dir.resolve()
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset {split} image path not found: {images_dir}"
            )
        image_count = sum(
            1
            for candidate in images_dir.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )
        if image_count == 0:
            raise ValueError(
                f"Dataset {split} image path contains no images: {images_dir}"
            )

        labels_dir = root / "labels" / split
        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset {split} label path not found: {labels_dir}"
            )
        label_count = sum(
            1 for candidate in labels_dir.rglob("*.txt") if candidate.is_file()
        )
        if label_count == 0:
            raise ValueError(
                f"Dataset {split} label path contains no .txt files: {labels_dir}"
            )
        if label_count != image_count:
            raise ValueError(
                f"Dataset {split} image/label count mismatch: "
                f"{image_count} images versus {label_count} labels"
            )

    return {"yaml_path": str(yaml_path), "root": str(root), "names": list(actual_names)}


def validate_prerequisites(
    config: Mapping[str, Any], *, repo_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Validate model weights and six-class dataset without starting training."""
    root = Path(repo_root).expanduser().resolve()
    model_path = _resolve_repo_path(str(config["model_path"]), root)
    dataset_yaml = _resolve_repo_path(str(config["dataset_yaml"]), root)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pretrained model weights not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt pretrained model file: {model_path}")
    dataset = validate_dataset_yaml(dataset_yaml)
    output_root = _resolve_repo_path(str(config["output_root"]), root)
    run_name = str(config["run_name"])
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    return {
        "model_path": str(model_path),
        "dataset_yaml": str(dataset_yaml),
        "dataset_root": dataset["root"],
        "class_names": dataset["names"],
        "output_root": str(output_root),
        "run_name": run_name,
        "imgsz": int(config["imgsz"]),
        "epochs": int(config["epochs"]),
        "batch": int(config["batch"]),
        "workers": int(config["workers"]),
        "device": str(config["device"]),
        "fraction": float(config["fraction"]),
        "cls_remap": bool(config["cls_remap"]),
    }


def run_finetuning(
    config: Mapping[str, Any],
    *,
    pilot: bool = False,
    repo_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Run a training job, returning its output directory."""
    report = validate_prerequisites(config, repo_root=repo_root)
    from ultralytics import YOLO

    output_root = Path(report["output_root"])
    if pilot:
        epochs = 1
        fraction = 0.01
        run_name = f"{report['run_name']}_pilot"
    else:
        epochs = int(config["epochs"])
        fraction = float(config["fraction"])
        run_name = report["run_name"]
    output_dir = output_root / run_name
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing training run directory: {output_dir}"
        )

    model = YOLO(report["model_path"])
    model.train(
        **_build_training_kwargs(
            config,
            report,
            epochs=epochs,
            fraction=fraction,
            output_root=output_root,
            run_name=run_name,
        )
    )
    actual_output = Path(model.trainer.save_dir).resolve()
    results_path = actual_output / "results.csv"
    best_weights = actual_output / "weights" / "best.pt"
    last_weights = actual_output / "weights" / "last.pt"
    summary = {
        "pilot": pilot,
        "model_path": report["model_path"],
        "dataset_yaml": report["dataset_yaml"],
        "class_names": report["class_names"],
        "imgsz": int(config["imgsz"]),
        "epochs_requested": epochs,
        "fraction": fraction,
        "cls_remap": bool(config["cls_remap"]),
        "device": str(config["device"]),
        "output_dir": str(actual_output),
        "results_csv_exists": results_path.is_file(),
        "best_weights_exists": best_weights.is_file(),
        "last_weights_exists": last_weights.is_file(),
    }
    (actual_output / "tracklab_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if (
        not results_path.is_file()
        or not last_weights.is_file()
        or not best_weights.is_file()
    ):
        raise RuntimeError(
            "Training ended without results.csv, best.pt, or last.pt; inspect logs."
        )
    return actual_output


def _build_training_kwargs(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    epochs: int,
    fraction: float,
    output_root: Path,
    run_name: str,
) -> dict[str, Any]:
    """Build explicit Ultralytics arguments for reproducible training."""

    return {
        "data": report["dataset_yaml"],
        "imgsz": int(config["imgsz"]),
        "epochs": epochs,
        "batch": int(config["batch"]),
        "workers": int(config["workers"]),
        "device": str(config["device"]),
        "fraction": fraction,
        "amp": bool(config["amp"]),
        "cls_remap": bool(config["cls_remap"]),
        "cache": False,
        "val": True,
        "save": True,
        "plots": True,
        "patience": int(config["patience"]),
        "project": str(output_root),
        "name": run_name,
        "exist_ok": False,
        "verbose": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Fine-tuning YAML"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually start training; without this flag only validate.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="One epoch on a small training subset (requires --train).",
    )
    args = parser.parse_args(argv)
    if args.pilot and not args.train:
        parser.error("--pilot requires --train")

    config = load_finetune_config(args.config)
    report = validate_prerequisites(config)
    print("PASS: fine-tuning prerequisites validated.")
    print(json.dumps(report, indent=2))
    if not args.train:
        print("No training started. Add --train --pilot to run the small pilot.")
        return 0
    output_dir = run_finetuning(config, pilot=args.pilot)
    print(f"Training run finished. Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
