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
    latency = saved.summary["latency"]
    assert latency["frame_count"] == 4
    assert latency["total_seconds"] > 0
    assert latency["effective_fps"] > 0
    assert set(latency["stage_seconds"]) == {
        "decode",
        "quality",
        "preprocessing",
        "detector_tracker",
        "analytics",
        "draw",
        "output_write",
        "encode_transcode",
    }

    mot_lines = run.mot_path.read_text(encoding="utf-8").splitlines()
    assert len(mot_lines) == 4
    assert mot_lines[0].startswith("1,11,")
    assert mot_lines[-1].startswith("4,11,")


def test_pipeline_passes_one_resolved_line_to_counter_and_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tracklab.pipeline.transcode_video_h264", _fake_transcode)
    source = tmp_path / "normalized_source.mp4"
    create_test_video(source, frame_count=2)

    config = make_config()
    captured_lines = []
    counter_lines = []

    from tracklab.analytics.line_counter import LineCounter as RealLineCounter

    class CapturingLineCounter(RealLineCounter):
        def __init__(self, config, lines=None):
            counter_lines.append(tuple(lines or ()))
            super().__init__(config, lines=lines)

    def capture_analytics(frame, *, lines, line_counts=None, trajectories=None):
        captured_lines.append(tuple(lines))
        return frame

    monkeypatch.setattr("tracklab.pipeline.draw_analytics", capture_analytics)
    monkeypatch.setattr("tracklab.pipeline.LineCounter", CapturingLineCounter)

    run = VideoPipeline(config, tracker=FakeTracker()).run(
        source,
        tmp_path / "results",
    )

    assert len(captured_lines) == 2
    resolved = captured_lines[0]
    assert resolved == captured_lines[1]
    assert counter_lines == [resolved]
    assert resolved[0].coordinate_space == "pixels"
    assert resolved[0].p1 == pytest.approx((63 * 0.052083333333333336, 47 * 0.5))
    assert resolved[0].p2 == pytest.approx((63 * 0.9375, 47 * 0.5))
    assert run.video_result.config["analytics"]["lines"][0]["coordinate_space"] == (
        "normalized"
    )


def test_default_tracker_is_created_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    create_test_video(source, frame_count=4)

    config = make_config()
    config.output.save_video = False
    config.output.save_json = False
    config.output.save_mot_txt = False

    created_with = []

    def fake_tracker_factory(received_config):
        created_with.append(received_config)
        return FakeTracker()

    monkeypatch.setattr(
        "tracklab.pipeline.YoloByteTrack",
        fake_tracker_factory,
    )

    pipeline = VideoPipeline(config)

    # Model çalıştırma başlamadan yüklenmemeli.
    assert created_with == []

    run = pipeline.run(source, tmp_path / "results")

    # Tek video çalıştırması, yalnızca bir tracker/model oluşturmalı.
    assert created_with == [config]
    assert run.video_result.summary["processed_frames"] == 4


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


def test_pipeline_cleans_up_raw_video_on_failure(tmp_path: Path) -> None:
    source = tmp_path / "failing_source.mp4"
    output_dir = tmp_path / "results"
    create_test_video(source, frame_count=3)

    raw_path = output_dir / ".failing_source_tracked_raw.mp4"

    class FailingTracker:
        def __init__(self) -> None:
            self.calls = 0
            self.raw_file_seen = False

        def track_frame(self, frame: np.ndarray) -> list[TrackedObject]:
            self.calls += 1
            if self.calls == 1:
                return []

            self.raw_file_seen = raw_path.is_file()
            raise RuntimeError("Tracker crashed mid-video")

    tracker = FailingTracker()
    config = make_config()

    with pytest.raises(RuntimeError, match="Tracker crashed mid-video"):
        VideoPipeline(config, tracker=tracker).run(source, output_dir)

    assert tracker.calls == 2
    assert tracker.raw_file_seen, "Test sırasında geçici video oluşturulmamış."
    assert not raw_path.exists(), "Hata sonrası geçici video temizlenmedi."
    assert list(output_dir.glob(".*_tracked_raw.mp4")) == []


def test_pipeline_includes_quality_sampling_report(tmp_path: Path) -> None:
    source = tmp_path / "quality_source.mp4"
    output_dir = tmp_path / "results"
    create_test_video(source, frame_count=5)

    config = make_config()
    config.output.save_video = False
    config.output.save_json = False
    config.output.save_mot_txt = False
    config.output.quality_every_n_frames = 2

    run = VideoPipeline(config, tracker=FakeTracker()).run(source, output_dir)

    quality = run.video_result.summary["quality"]
    assert quality["sampling"] == {
        "every_n_frames": 2,
        "frame_index_base": 0,
    }
    assert quality["sample_count"] == 3
    assert quality["sampled_frame_indices"] == [0, 2, 4]
    assert len(quality["samples"]) == 3
    assert set(quality["metrics"]) == {
        "brightness",
        "contrast",
        "sharpness",
        "noise_estimate",
    }
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("mode", ["none", "gamma"])
def test_pipeline_preprocesses_tracker_input_but_preserves_original_video_frames(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracklab.io.video import read_frames
    from tracklab.preprocessing.gamma import apply_gamma

    source = tmp_path / f"preprocessing_{mode}.mp4"
    output_dir = tmp_path / "results"
    create_test_video(source, frame_count=4)

    config = make_config()
    config.preprocessing.mode = mode
    config.preprocessing.gamma.value = 0.8
    config.output.save_json = False
    config.output.save_mot_txt = False

    tracked_frames: list[np.ndarray] = []
    drawn_frames: list[np.ndarray] = []

    class CapturingTracker:
        def track_frame(self, frame: np.ndarray) -> list[TrackedObject]:
            tracked_frames.append(frame.copy())
            return []

    def capture_draw(frame, objects, class_names=None):
        drawn_frames.append(frame.copy())
        return frame

    monkeypatch.setattr("tracklab.pipeline.draw_tracks", capture_draw)
    monkeypatch.setattr("tracklab.pipeline.transcode_video_h264", _fake_transcode)

    run = VideoPipeline(config, tracker=CapturingTracker()).run(source, output_dir)
    original_frames = [frame for _, frame in read_frames(source)]

    assert run.video_path is not None and run.video_path.is_file()
    assert len(tracked_frames) == len(original_frames) == 4
    assert len(drawn_frames) == len(original_frames) == 4

    for original, tracked, drawn in zip(
        original_frames, tracked_frames, drawn_frames, strict=True
    ):
        expected_tracking_input = (
            apply_gamma(original, gamma=config.preprocessing.gamma.value)
            if mode == "gamma"
            else original
        )
        np.testing.assert_array_equal(tracked, expected_tracking_input)
        # Visualization/encoding stays on the original frame, not the transformed one.
        np.testing.assert_array_equal(drawn, original)
        assert tracked.shape == original.shape
        assert tracked.dtype == original.dtype

    if mode == "gamma":
        assert any(
            not np.array_equal(original, tracked)
            for original, tracked in zip(original_frames, tracked_frames, strict=True)
        )
