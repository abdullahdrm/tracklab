"""Serialization helpers for TrackLab results."""

from __future__ import annotations

import gzip as gzip_module
import json
import zlib
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tracklab.schemas import (
    FrameResult,
    TrackedObject,
    VideoMeta,
    VideoResult,
)


def _json_default(value: Any) -> Any:
    """Convert a few common non-JSON values into serializable values."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (set, frozenset)):
        return sorted(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(
    result: VideoResult,
    path: str | Path,
    include_frames: bool = True,
    gzip: bool = False,
) -> Path:
    """Save a VideoResult as JSON, optionally compressed with gzip.

    When include_frames is False, the frames field is omitted.
    Gzip compression is controlled by the argument, not the filename.
    """

    output_path = Path(path).expanduser()

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "video": asdict(result.meta),
        "summary": result.summary,
        "config": result.config,
    }

    if include_frames:
        payload["frames"] = [asdict(frame) for frame in result.frames]

    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if gzip:
        with gzip_module.open(output_path, "wb") as compressed_file:
            compressed_file.write(content.encode("utf-8"))
    else:
        output_path.write_text(content, encoding="utf-8")

    return output_path


def load_json(path: str | Path) -> VideoResult:
    """Load a TrackLab JSON result, detecting gzip by its file signature."""

    input_path = Path(path).expanduser()
    raw = input_path.read_bytes()

    try:
        if raw.startswith(b"\x1f\x8b"):
            content = gzip_module.decompress(raw)
        else:
            content = raw

        payload = json.loads(content.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
        EOFError,
        zlib.error,
    ) as exc:
        raise ValueError(
            f"Invalid or unreadable TrackLab JSON file: {input_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError("The result JSON root must be an object.")

    video_data = payload.get("video")
    frames_data = payload.get("frames", [])
    summary = payload.get("summary", {})
    config = payload.get("config", {})

    if not isinstance(video_data, dict):
        raise TypeError("The result JSON must contain a video object.")

    if not isinstance(frames_data, list):
        raise TypeError("The frames field must be a list.")

    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise TypeError("The summary and config fields must be objects.")

    try:
        meta = VideoMeta(**video_data)
        frames: list[FrameResult] = []

        for frame_data in frames_data:
            if not isinstance(frame_data, dict):
                raise TypeError("Each frame entry must be an object.")

            objects_data = frame_data.get("objects")
            if not isinstance(objects_data, list):
                raise TypeError("Each frame must contain an objects list.")

            objects = []
            for object_data in objects_data:
                if not isinstance(object_data, dict):
                    raise TypeError("Each tracked object must be an object.")
                objects.append(TrackedObject(**object_data))

            frames.append(
                FrameResult(
                    frame_index=frame_data["frame_index"],
                    timestamp_s=frame_data["timestamp_s"],
                    objects=objects,
                )
            )

    except (KeyError, TypeError) as exc:
        raise ValueError(
            "The result JSON does not match the TrackLab data schema."
        ) from exc

    return VideoResult(
        meta=meta,
        frames=frames,
        summary=summary,
        config=config,
    )


def save_mot_txt(
    frames: Iterable[FrameResult],
    path: str | Path,
) -> Path:
    """Write detections in MOTChallenge text format.

    Frame indices in the exported file are one-based. Bounding boxes
    are written as top-left x, top-left y, width, height.
    """

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for frame in frames:
        mot_frame_index = frame.frame_index + 1

        for obj in frame.objects:
            x, y, width, height = obj.xywh

            lines.append(
                f"{mot_frame_index},{obj.track_id},"
                f"{x:.2f},{y:.2f},{width:.2f},{height:.2f},"
                f"{obj.score:.6f},-1,-1,-1\n"
            )

    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
