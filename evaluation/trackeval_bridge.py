"""Bridge between MOT17 annotations and the pinned TrackEval implementation."""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path
from typing import Any

from tracklab.datasets.mot17 import (
    MOT17Sequence,
    discover_sequences,
    iter_ground_truth_rows,
)


def load_trackeval() -> Any:
    """Load the pinned TrackEval source with process-local NumPy compatibility."""

    import numpy as np

    repository_root = Path(__file__).resolve().parents[1]
    trackeval_root = repository_root / "third_party" / "TrackEval"
    package_init = trackeval_root / "trackeval" / "__init__.py"

    if not package_init.is_file():
        raise FileNotFoundError(
            "TrackEval is missing. Run: git submodule update --init --recursive"
        )

    for alias, replacement in (
        ("float", float),
        ("int", int),
        ("bool", bool),
    ):
        if not hasattr(np, alias):
            setattr(np, alias, replacement)

    trackeval_path = str(trackeval_root.resolve())
    existing = sys.modules.get("trackeval")

    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path and Path(existing_path).resolve().is_relative_to(
            trackeval_root.resolve()
        ):
            return existing
        raise RuntimeError(
            "A different 'trackeval' package was imported earlier in this process."
        )

    if trackeval_path in sys.path:
        sys.path.remove(trackeval_path)
    sys.path.insert(0, trackeval_path)

    return importlib.import_module("trackeval")


def _select_sequences(
    data_root: str | Path,
    sequence_names: tuple[str, ...] | list[str],
    *,
    variant: str,
) -> list[MOT17Sequence]:
    """Select the requested sequence names, rejecting duplicates and missing data."""

    names = tuple(sequence_names)

    if not names:
        raise ValueError("At least one sequence must be selected.")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate sequence names are not allowed.")

    sequences = discover_sequences(data_root, variant=variant)
    by_name = {sequence.name: sequence for sequence in sequences}
    missing = sorted(set(names) - set(by_name))

    if missing:
        raise ValueError("Requested sequences are missing: " + ", ".join(missing))

    return [by_name[name] for name in names]


def write_ground_truth_as_predictions(
    sequence: MOT17Sequence,
    output_file: str | Path,
) -> int:
    """Write marked pedestrian GT rows in TrackEval's 10-column tracker format."""

    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    written = 0

    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")

            for row in iter_ground_truth_rows(
                sequence.gt_file,
                seq_length=sequence.seq_length,
            ):
                # MOTChallenge pedestrian class is 1. Keep official mark
                # and distractor filtering rules in TrackEval itself.
                if row.class_id != 1 or row.mark == 0:
                    continue

                writer.writerow(
                    [
                        row.frame,
                        row.track_id,
                        repr(row.left),
                        repr(row.top),
                        repr(row.width),
                        repr(row.height),
                        "1.0",
                        "1",
                        "-1",
                        "-1",
                    ]
                )
                written += 1

        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return written


def prepare_ground_truth_tracker(
    data_root: str | Path,
    trackers_root: str | Path,
    sequence_names: tuple[str, ...] | list[str],
    *,
    tracker_name: str = "gt_self",
    variant: str = "FRCNN",
) -> dict[str, int]:
    """Prepare per-sequence tracker text files using ground-truth boxes and IDs."""

    if not tracker_name or Path(tracker_name).name != tracker_name:
        raise ValueError("tracker_name must be a simple directory name.")

    sequences = _select_sequences(data_root, sequence_names, variant=variant)
    tracker_dir = Path(trackers_root) / tracker_name / "data"
    counts: dict[str, int] = {}

    for sequence in sequences:
        path = tracker_dir / f"{sequence.name}.txt"
        counts[sequence.name] = write_ground_truth_as_predictions(sequence, path)

    return counts


def evaluate_mot_challenge(
    data_root: str | Path,
    trackers_root: str | Path,
    output_root: str | Path,
    sequence_names: tuple[str, ...] | list[str],
    *,
    tracker_name: str = "gt_self",
    variant: str = "FRCNN",
) -> dict[str, float]:
    """Run TrackEval HOTA on the requested MOT17 sequences."""

    import numpy as np

    trackeval = load_trackeval()
    sequences = _select_sequences(data_root, sequence_names, variant=variant)

    tracker_dir = Path(trackers_root) / tracker_name / "data"
    for sequence in sequences:
        tracker_file = tracker_dir / f"{sequence.name}.txt"
        if not tracker_file.is_file():
            raise FileNotFoundError(f"Missing tracker file: {tracker_file}")

    dataset_config = {
        "GT_FOLDER": str(Path(data_root).resolve()),
        "TRACKERS_FOLDER": str(Path(trackers_root).resolve()),
        "TRACKERS_TO_EVAL": [tracker_name],
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_FOLDER": str(Path(output_root).resolve()),
        "OUTPUT_SUB_FOLDER": "",
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": {s.name: s.seq_length for s in sequences},
        "CLASSES_TO_EVAL": ["pedestrian"],
        "DO_PREPROC": True,
        "PRINT_CONFIG": False,
    }

    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update(
        {
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "RETURN_ON_ERROR": False,
            "LOG_ON_ERROR": None,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "OUTPUT_EMPTY_CLASSES": False,
            "PLOT_CURVES": False,
        }
    )

    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    evaluator = trackeval.Evaluator(evaluator_config)
    results, messages = evaluator.evaluate(
        [dataset],
        [trackeval.metrics.HOTA()],
    )

    dataset_name = dataset.get_name()
    message = messages[dataset_name].get(tracker_name)

    if message != "Success":
        raise RuntimeError(f"TrackEval failed: {message}")

    tracker_results = results[dataset_name][tracker_name]
    if tracker_results is None:
        raise RuntimeError("TrackEval returned no results.")

    combined = tracker_results["COMBINED_SEQ"]["pedestrian"]["HOTA"]
    return {
        name: float(np.mean(combined[name]))
        for name in ("HOTA", "DetA", "AssA", "LocA")
    }
