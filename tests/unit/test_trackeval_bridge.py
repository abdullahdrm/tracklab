"""Tests for MOT17 conversion and TrackEval self-evaluation."""

import csv
from pathlib import Path

import pytest

from evaluation.trackeval_bridge import (
    evaluate_mot_challenge,
    prepare_ground_truth_tracker,
    write_ground_truth_as_predictions,
)
from tracklab.datasets.mot17 import read_sequence_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKEVAL_INIT = (
    PROJECT_ROOT / "third_party" / "TrackEval" / "trackeval" / "__init__.py"
)


def make_synthetic_sequence(root: Path) -> Path:
    """Create a minimal two-frame MOT17 sequence for testing."""

    sequence = root / "MOT17-02-FRCNN"
    image_dir = sequence / "img1"
    gt_dir = sequence / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    # Placeholder files suffice: evaluation reads annotations, not pixels.
    (image_dir / "000001.jpg").write_bytes(b"frame 1")
    (image_dir / "000002.jpg").write_bytes(b"frame 2")

    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\n"
        "name=MOT17-02-FRCNN\n"
        "imDir=img1\n"
        "frameRate=30\n"
        "seqLength=2\n"
        "imWidth=640\n"
        "imHeight=480\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )

    (gt_dir / "gt.txt").write_text(
        # Valid pedestrian with persistent ID 42.
        "1,42,10,20,30,40,1,1,1\n"
        # Zero-marked pedestrian: excluded by the bridge.
        "1,3,90,100,20,50,0,1,1\n"
        # Distractor class 7: not converted to a pedestrian prediction.
        "1,7,200,210,20,50,1,7,1\n"
        # Same pedestrian in the second frame.
        "2,42,11,20,30,40,1,1,1\n",
        encoding="utf-8",
    )

    return sequence


def test_writer_keeps_only_marked_pedestrians(tmp_path: Path) -> None:
    sequence_dir = make_synthetic_sequence(tmp_path / "train")
    sequence = read_sequence_info(sequence_dir)
    tracker_file = tmp_path / "trackers" / "gt_self" / "data" / f"{sequence.name}.txt"

    count = write_ground_truth_as_predictions(sequence, tracker_file)

    with tracker_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert count == 2
    assert len(rows) == 2
    assert all(len(row) == 10 for row in rows)
    assert [row[0] for row in rows] == ["1", "2"]
    assert [row[1] for row in rows] == ["42", "42"]
    assert all(row[7] == "1" for row in rows)


def test_trackeval_gt_self_evaluation_is_perfect(tmp_path: Path) -> None:
    if not TRACKEVAL_INIT.is_file():
        pytest.skip("TrackEval submodule is not initialized.")

    data_root = tmp_path / "train"
    sequence_dir = make_synthetic_sequence(data_root)
    sequence_name = sequence_dir.name
    trackers_root = tmp_path / "trackers"
    evaluation_root = tmp_path / "evaluation"

    counts = prepare_ground_truth_tracker(
        data_root,
        trackers_root,
        [sequence_name],
    )

    metrics = evaluate_mot_challenge(
        data_root,
        trackers_root,
        evaluation_root,
        [sequence_name],
    )

    assert counts[sequence_name] == 2
    assert metrics["HOTA"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["DetA"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["AssA"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["LocA"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["MOTA"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["IDSW"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["IDF1"] == pytest.approx(1.0, abs=1e-6)
