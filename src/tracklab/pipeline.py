"""End-to-end video processing pipeline for TrackLab."""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from tracklab.config import PipelineConfig
from tracklab.io.results import save_json, save_mot_txt
from tracklab.io.video import (
    FrameWriter,
    probe_video,
    read_frames,
    transcode_video_h264,
)
from tracklab.schemas import FrameResult, TrackedObject, VideoResult
from tracklab.tracking.yolo_bytetrack import YoloByteTrack
from tracklab.viz.draw import ClassNames, draw_tracks

Frame = NDArray[np.uint8]


class FrameTracker(Protocol):
    """Interface required by the pipeline's tracking component."""

    def track_frame(self, frame: Frame) -> list[TrackedObject]:
        """Return tracked objects for one frame."""


@dataclass(slots=True)
class PipelineRunResult:
    """Processing results and paths to any generated artifacts."""

    video_result: VideoResult
    video_path: Path | None
    json_path: Path | None
    mot_path: Path | None


class VideoPipeline:
    """Process a video using the configured detector/tracker."""

    def __init__(
        self,
        config: PipelineConfig,
        tracker: FrameTracker | None = None,
    ) -> None:
        """Initialize the pipeline with validated config and optional tracker."""

        self.config = config
        self._injected_tracker = tracker
        self.tracker: FrameTracker | None = tracker

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path = "outputs",
    ) -> PipelineRunResult:
        """Process a source video and write the enabled output artifacts.

        Output files are named using the source filename:
        *_tracked.mp4, *_result.json, and *_mot.txt.

        The returned VideoResult always includes the processed frame results.
        The include_frames setting controls only whether those details are
        included in the JSON file.
        """

        tracker = self._injected_tracker
        if tracker is None:
            tracker = YoloByteTrack(self.config)
        self.tracker = tracker

        source = Path(input_path).expanduser().resolve()
        output_root = Path(output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        meta = probe_video(source)

        video_path = (
            output_root / f"{source.stem}_tracked.mp4"
            if self.config.output.save_video
            else None
        )
        json_path = (
            output_root / f"{source.stem}_result.json"
            if self.config.output.save_json
            else None
        )
        mot_path = (
            output_root / f"{source.stem}_mot.txt"
            if self.config.output.save_mot_txt
            else None
        )

        raw_video_path = (
            output_root / f".{source.stem}_tracked_raw.mp4"
            if video_path is not None
            else None
        )

        frames: list[FrameResult] = []
        unique_track_ids: set[int] = set()
        objects_by_class_id: dict[str, int] = {}
        total_detections = 0

        model = getattr(tracker, "model", None)
        class_names: ClassNames = getattr(model, "names", None)

        try:
            with ExitStack() as stack:
                writer: FrameWriter | None = None

                for frame_index, frame in read_frames(source):
                    objects = tracker.track_frame(frame)

                    if raw_video_path is not None and writer is None:
                        writer = stack.enter_context(
                            FrameWriter(
                                raw_video_path,
                                fps=meta.fps,
                                size=(meta.width, meta.height),
                            )
                        )

                    frame_result = FrameResult(
                        frame_index=frame_index,
                        timestamp_s=frame_index / meta.fps,
                        objects=objects,
                    )
                    frames.append(frame_result)

                    total_detections += len(objects)

                    for obj in objects:
                        unique_track_ids.add(obj.track_id)
                        class_key = str(obj.class_id)
                        objects_by_class_id[class_key] = (
                            objects_by_class_id.get(class_key, 0) + 1
                        )

                    if writer is not None:
                        annotated = draw_tracks(
                            frame,
                            objects,
                            class_names=class_names,
                        )
                        writer.write(annotated)

            if not frames:
                raise ValueError(f"No readable frames found in video: {source}")

            if raw_video_path is not None and video_path is not None:
                transcode_video_h264(raw_video_path, video_path)
        finally:
            if raw_video_path is not None:
                raw_video_path.unlink(missing_ok=True)

        summary = {
            "processed_frames": len(frames),
            "total_detections": total_detections,
            "unique_tracks": len(unique_track_ids),
            "track_ids": sorted(unique_track_ids),
            "objects_by_class_id": objects_by_class_id,
        }

        result = VideoResult(
            meta=meta,
            frames=frames,
            summary=summary,
            config=self.config.model_dump(
                mode="json",
                exclude={"tracker_yaml"},
            ),
        )

        if json_path is not None:
            save_json(
                result,
                json_path,
                include_frames=self.config.output.include_frames,
            )

        if mot_path is not None:
            save_mot_txt(frames, mot_path)

        return PipelineRunResult(
            video_result=result,
            video_path=video_path,
            json_path=json_path,
            mot_path=mot_path,
        )
