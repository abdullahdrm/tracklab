"""Tests for MOT17 metadata and annotation parsing."""

from pathlib import Path

import pytest

from tracklab.datasets.mot17 import (
    discover_sequences,
    iter_ground_truth_rows,
    read_sequence_info,
)


def make_sequence(root: Path, name: str = "MOT17-02-FRCNN") -> Path:
    sequence = root / name
    frame_dir = sequence / "img1"
    gt_dir = sequence / "gt"
    frame_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    (frame_dir / "000001.jpg").write_bytes(b"test frame placeholder")
    (sequence / "seqinfo.ini").write_text(
        f"[Sequence]\n"
        f"name={name}\n"
        f"imDir=img1\n"
        f"frameRate=30\n"
        f"seqLength=1\n"
        f"imWidth=1920\n"
        f"imHeight=1080\n"
        f"imExt=.jpg\n",
        encoding="utf-8",
    )
    (gt_dir / "gt.txt").write_text(
        "1,1,912,484,97,109,0,7,1\n",
        encoding="utf-8",
    )
    return sequence


def test_reads_sequence_metadata_and_validates_frame_count(
    tmp_path: Path,
) -> None:
    sequence_dir = make_sequence(tmp_path)

    sequence = read_sequence_info(sequence_dir)

    assert sequence.name == "MOT17-02-FRCNN"
    assert sequence.frame_rate == 30
    assert sequence.seq_length == 1
    assert (sequence.width, sequence.height) == (1920, 1080)
    assert sequence.frame_dir.is_dir()
    assert sequence.gt_file.is_file()


def test_discovers_only_requested_detector_variant(tmp_path: Path) -> None:
    make_sequence(tmp_path, "MOT17-02-FRCNN")
    make_sequence(tmp_path, "MOT17-02-SDP")

    sequences = discover_sequences(tmp_path, variant="FRCNN")

    assert [sequence.name for sequence in sequences] == ["MOT17-02-FRCNN"]


def test_preserves_raw_ground_truth_fields(tmp_path: Path) -> None:
    sequence_dir = make_sequence(tmp_path)

    rows = list(iter_ground_truth_rows(sequence_dir / "gt" / "gt.txt", seq_length=1))

    assert len(rows) == 1
    assert rows[0].frame == 1
    assert rows[0].track_id == 1
    assert (rows[0].left, rows[0].top) == (912.0, 484.0)
    assert (rows[0].width, rows[0].height) == (97.0, 109.0)
    assert rows[0].mark == 0.0
    assert rows[0].class_id == 7
    assert rows[0].visibility == 1.0


def test_rejects_ground_truth_with_wrong_column_count(tmp_path: Path) -> None:
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text("1,1,10,20,30,40,1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 9 columns"):
        list(iter_ground_truth_rows(gt_file))


def test_rejects_ground_truth_frame_outside_sequence(tmp_path: Path) -> None:
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text("2,1,10,20,30,40,1,1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds sequence length"):
        list(iter_ground_truth_rows(gt_file, seq_length=1))


def test_rejects_sequence_frame_count_mismatch(tmp_path: Path) -> None:
    sequence_dir = make_sequence(tmp_path)
    info_path = sequence_dir / "seqinfo.ini"
    info_path.write_text(
        info_path.read_text(encoding="utf-8").replace("seqLength=1", "seqLength=2"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Frame-count mismatch"):
        read_sequence_info(sequence_dir)
