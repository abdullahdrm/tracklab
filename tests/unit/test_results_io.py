"""Unit tests for TrackLab result serialization."""

import json
from pathlib import Path

import numpy as np

from tracklab.io.results import load_json, save_json, save_mot_txt
from tracklab.schemas import (
    FrameResult,
    TrackedObject,
    VideoMeta,
    VideoResult,
)


def make_result() -> VideoResult:
    obj = TrackedObject(
        track_id=7,
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=60.0,
        score=0.95,
        class_id=0,
    )

    frame = FrameResult(
        frame_index=0,
        timestamp_s=0.0,
        objects=[obj],
    )

    meta = VideoMeta(
        name="sample.mp4",
        width=640,
        height=480,
        fps=25.0,
        frame_count=1,
    )

    return VideoResult(
        meta=meta,
        frames=[frame],
        summary={"unique_tracks": 1},
        config={"classes": [0, 1, 2, 3, 5, 7]},
    )


def test_json_round_trip(tmp_path: Path) -> None:
    original = make_result()
    path = tmp_path / "nested" / "result.json"

    save_json(original, path)
    restored = load_json(path)

    assert restored == original


def test_json_can_omit_frame_details(tmp_path: Path) -> None:
    original = make_result()
    path = tmp_path / "summary.json"

    save_json(original, path, include_frames=False)

    payload = json.loads(path.read_text(encoding="utf-8"))
    restored = load_json(path)

    assert "frames" not in payload
    assert restored.frames == []
    assert restored.summary == original.summary
    assert restored.meta == original.meta


def test_gzip_json_round_trip(tmp_path: Path) -> None:
    original = make_result()
    path = tmp_path / "result.json.gz"

    save_json(original, path, gzip=True)

    assert path.read_bytes().startswith(b"\x1f\x8b")
    assert load_json(path) == original


def test_mot_export_uses_one_based_frame_indices(tmp_path: Path) -> None:
    original = make_result()
    path = tmp_path / "mot" / "result.txt"

    save_mot_txt(original.frames, path)

    assert path.read_text(encoding="utf-8") == (
        "1,7,10.00,20.00,20.00,40.00,0.950000,-1,-1,-1\n"
    )


def test_json_serializes_numpy_arrays_and_sets(tmp_path: Path) -> None:
    original = make_result()
    original.summary["custom_array"] = np.array([1, 2, 3])
    original.summary["custom_set"] = {4, 5, 6}
    path = tmp_path / "numpy_set_result.json"

    save_json(original, path)
    restored = load_json(path)

    assert restored.summary["custom_array"] == [1, 2, 3]
    assert restored.summary["custom_set"] == [4, 5, 6]
