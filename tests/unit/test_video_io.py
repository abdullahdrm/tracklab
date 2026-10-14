"""Unit tests for TrackLab video input and output."""

import numpy as np
import pytest

from tracklab.io.video import FrameWriter, probe_video, read_frames


def test_video_writer_reader_round_trip(tmp_path) -> None:
    path = tmp_path / "videos" / "sample.mp4"

    frames = []
    for index in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (20 * index, 40, 80)
        frames.append(frame)

    with FrameWriter(path, fps=10.0, size=(64, 48)) as writer:
        for frame in frames:
            writer.write(frame)

    assert path.is_file()
    assert path.stat().st_size > 0

    meta = probe_video(path)
    decoded = list(read_frames(path))

    assert meta.name == "sample.mp4"
    assert meta.width == 64
    assert meta.height == 48
    assert meta.fps == pytest.approx(10.0, abs=0.1)
    assert meta.frame_count == 3

    assert [index for index, _ in decoded] == [0, 1, 2]
    assert all(frame.shape == (48, 64, 3) for _, frame in decoded)


def test_probe_video_rejects_invalid_file(tmp_path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video")

    with pytest.raises(ValueError, match="Cannot open video"):
        probe_video(path)


def test_writer_rejects_wrong_frame_dimensions(tmp_path) -> None:
    path = tmp_path / "output.mp4"
    wrong_frame = np.zeros((24, 32, 3), dtype=np.uint8)

    with (
        pytest.raises(ValueError, match="Frame dimensions"),
        FrameWriter(path, fps=10.0, size=(64, 48)) as writer,
    ):
        writer.write(wrong_frame)
