"""Video input and output helpers for TrackLab."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import cv2
import numpy as np

from tracklab.schemas import VideoMeta


def _open_capture(path: str | Path) -> tuple[Path, cv2.VideoCapture]:
    """Open a video file or raise a useful error."""

    source = Path(path).expanduser()

    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {source}")

    return source, capture


def probe_video(path: str | Path) -> VideoMeta:
    """Read the basic metadata of a video file."""

    source, capture = _open_capture(path)

    try:
        width_value = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_value = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps_value = capture.get(cv2.CAP_PROP_FPS)
        count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if not all(
            math.isfinite(value) for value in (width_value, height_value, fps_value)
        ):
            raise ValueError(f"Invalid video metadata: {source}")

        width = round(width_value)
        height = round(height_value)
        fps = float(fps_value)

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {source}")

        if fps <= 0:
            raise ValueError(f"Invalid video FPS: {source}")

        frame_count = max(0, round(count_value)) if math.isfinite(count_value) else 0

        return VideoMeta(
            name=source.name,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
    finally:
        capture.release()


def read_frames(
    path: str | Path,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (zero-based frame index, BGR frame) pairs."""

    _, capture = _open_capture(path)

    try:
        frame_index = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame is None:
                raise ValueError(
                    f"Video decoder returned an empty frame at index {frame_index}"
                )

            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


class FrameWriter:
    """Write uint8 BGR frames to an MP4 video using the mp4v codec.

    Args:
        path: Destination video path.
        fps: Frames per second of the output video.
        size: Output dimensions as (width, height).
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be a finite positive number.")

        if len(size) != 2:
            raise ValueError("Size must be (width, height).")

        self.width, self.height = size

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Video dimensions must be positive.")

        self._closed = False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            float(fps),
            (self.width, self.height),
        )

        if not self._writer.isOpened():
            self._writer.release()
            self._closed = True
            raise RuntimeError(f"Cannot create output video: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        """Write one frame after validating its format and dimensions."""

        if self._closed:
            raise RuntimeError("Cannot write to a closed video writer.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array.")

        if frame.dtype != np.uint8:
            raise TypeError("Frame dtype must be uint8.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be a three-channel BGR image.")

        expected_shape = (self.height, self.width)
        if frame.shape[:2] != expected_shape:
            raise ValueError(
                f"Frame dimensions must be {expected_shape} "
                f"(height, width); got {frame.shape[:2]}."
            )

        self._writer.write(frame)

    def close(self) -> None:
        """Release the video encoder. Safe to call more than once."""

        if not self._closed:
            self._writer.release()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def transcode_video_h264(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Encode a video as browser-friendly H.264 MP4 with faststart."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(destination).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(f"Video to transcode not found: {source_path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required for browser-compatible video output. "
            "Install it with: brew install ffmpeg"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        detail = (completed.stderr or "Unknown FFmpeg error")[-2000:]
        raise RuntimeError(f"H.264 video encoding failed: {detail}")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg did not produce a valid output video.")

    return output_path
