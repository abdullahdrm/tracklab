"""Unit tests for the MOT17 detector-comparison helpers."""

import numpy as np
import pytest

from evaluation.mot17_detector_comparison import (
    detections_to_boxes,
    load_frcnn_detections,
    tracker_rows_to_mot,
)


def test_frcnn_detections_are_grouped_by_frame_and_converted_to_xyxy(
    tmp_path,
) -> None:
    path = tmp_path / "det.txt"
    path.write_text(
        "2,-1,20,30,5,10,0.9\n1,-1,3,4,7,8,0.8\n2,-1,40,50,6,12,0.7\n",
        encoding="utf-8",
    )

    detections = load_frcnn_detections(path, seq_length=2)

    assert detections[1].shape == (1, 6)
    assert detections[2].shape == (2, 6)
    np.testing.assert_allclose(
        detections[1][0],
        [3, 4, 10, 12, 0.8, 0],
    )
    np.testing.assert_allclose(
        detections[2][0],
        [20, 30, 25, 40, 0.9, 0],
    )


def test_frcnn_reader_rejects_out_of_range_frame(tmp_path) -> None:
    path = tmp_path / "det.txt"
    path.write_text("3,-1,20,30,5,10,0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        load_frcnn_detections(path, seq_length=2)


def test_empty_detections_can_be_passed_to_byte_tracker() -> None:
    boxes = detections_to_boxes(
        np.empty((0, 6), dtype=np.float32),
        frame_shape=(100, 200),
    )

    assert len(boxes) == 0
    assert boxes.xywh.shape == (0, 4)


def test_tracker_output_converts_to_ten_column_mot_format() -> None:
    tracked = np.array(
        [[10, 20, 40, 80, 7, 0.9, 0, 0]],
        dtype=np.float32,
    )

    rows = tracker_rows_to_mot(1, tracked)

    assert len(rows) == 1
    assert len(rows[0]) == 10
    assert rows[0][:2] == [1, 7]
    np.testing.assert_allclose(rows[0][2:7], [10, 20, 30, 60, 0.9])
    assert rows[0][7:] == [-1, -1, -1]


def test_empty_tracker_output_writes_no_rows() -> None:
    assert tracker_rows_to_mot(1, np.asarray([])) == []
