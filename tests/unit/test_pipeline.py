"""Tests for the end-to-end video pipeline."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from tracklab.config import load_config
from tracklab.io.results import load_json
from tracklab.io.video import FrameWriter, probe_video
from tracklab.pipeline import VideoPipeline
from tracklab.schemas import TrackedObject

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeTracker:
    """Return a predictable track without loading a machine-learning model."""

    def __init__(self) -> None:
        self.calls = 0

    def track_frame(self, frame: np.ndarray) -> list[TrackedObject]:
        self.calls += 1
        return [
            TrackedObject(
                track_id=11,
                x1=5.0,
                y1=5.0,
                x2=40.0,
                y2=35.0,
                score=0.9,
                class_id=2,
            )
        ]


def make_config():
    """Load the actual project configuration."""

    return load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")


def create_test_video(path: Path, frame_count: int = 4) -> None:
    """Write a small video to exercise the pipeline without test fixtures."""

    with FrameWriter(path, fps=10.0, size=(64, 48)) as writer:
        for index in range(frame_count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 1] = index * 30
            writer.write(frame)


def _fake_transcode(source: str | Path, destination: str | Path) -> Path:
    """Copy the test video in place of invoking the external FFmpeg binary."""

    output_path = Path(destination)
    shutil.copyfile(source, output_path)
    return output_path


def test_pipeline_writes_video_json_and_mot_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tracklab.pipeline.transcode_video_h264", _fake_transcode)
    source = tmp_path / "source.mp4"
    output_dir = tmp_path / "results"
    create_test_video(source)

    config = make_config()
    config.output.save_mot_txt = True
    tracker = FakeTracker()

    run = VideoPipeline(config, tracker=tracker).run(source, output_dir)

    assert tracker.calls == 4
    assert run.video_path is not None and run.video_path.is_file()
    assert run.json_path is not None and run.json_path.is_file()
    assert run.mot_path is not None and run.mot_path.is_file()

    assert probe_video(run.video_path).frame_count == 4

    saved = load_json(run.json_path)
    assert saved.meta.name == source.name
    assert len(saved.frames) == 4
    assert saved.summary["processed_frames"] == 4
    assert saved.summary["total_detections"] == 4
    assert saved.summary["unique_tracks"] == 1
    assert saved.summary["objects_by_class_id"] == {"2": 4}
    assert all(frame.objects[0].track_id == 11 for frame in saved.frames)

    mot_lines = run.mot_path.read_text(encoding="utf-8").splitlines()
    assert len(mot_lines) == 4
    assert mot_lines[0].startswith("1,11,")
    assert mot_lines[-1].startswith("4,11,")


def test_pipeline_respects_disabled_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output_dir = tmp_path / "results"
    create_test_video(source, frame_count=2)

    config = make_config()
    config.output.save_video = False
    config.output.save_json = False
    config.output.save_mot_txt = False

    run = VideoPipeline(config, tracker=FakeTracker()).run(
        source,
        output_dir,
    )

    assert run.video_path is None
    assert run.json_path is None
    assert run.mot_path is None
    assert len(run.video_result.frames) == 2
    assert list(output_dir.iterdir()) == []
