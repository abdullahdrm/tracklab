"""End-to-end video processing pipeline for TrackLab."""

import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from tracklab.analytics.geometry import resolve_lines
from tracklab.analytics.line_counter import LineCounter
from tracklab.analytics.trajectories import TrajectoryStore
from tracklab.config import PipelineConfig
from tracklab.io.results import save_json, save_mot_txt
from tracklab.io.video import (
    FrameWriter,
    probe_video,
    read_frames,
    transcode_video_h264,
)
from tracklab.preprocessing.factory import build_preprocessor
from tracklab.quality.metrics import compute_frame_quality
from tracklab.quality.report import QUALITY_METRICS, summarize_quality_samples
from tracklab.schemas import FrameResult, TrackedObject, VideoResult
from tracklab.tracking.yolo_bytetrack import YoloByteTrack
from tracklab.viz.draw import ClassNames, draw_analytics, draw_tracks

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
        preprocessor = build_preprocessor(self.config.preprocessing)

        source = Path(input_path).expanduser().resolve()
        output_root = Path(output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        meta = probe_video(source)
        resolved_lines = resolve_lines(
            self.config.analytics.lines,
            width=meta.width,
            height=meta.height,
        )

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
        line_counter = LineCounter(self.config.analytics, lines=resolved_lines)
        trajectory_store = TrajectoryStore(self.config.analytics.trajectories)
        unique_track_ids: set[int] = set()
        objects_by_class_id: dict[str, int] = {}
        total_detections = 0
        quality_samples: list[dict[str, float | int]] = []
        latency_seconds = {
            "decode": 0.0,
            "quality": 0.0,
            "preprocessing": 0.0,
            "detector_tracker": 0.0,
            "analytics": 0.0,
            "draw": 0.0,
            "output_write": 0.0,
            "encode_transcode": 0.0,
        }
        run_started = time.perf_counter()

        model = getattr(tracker, "model", None)
        class_names: ClassNames = getattr(model, "names", None)

        try:
            with ExitStack() as stack:
                writer: FrameWriter | None = None

                frame_reader = iter(read_frames(source))
                while True:
                    decode_started = time.perf_counter()
                    try:
                        frame_index, frame = next(frame_reader)
                    except StopIteration:
                        break
                    latency_seconds["decode"] += time.perf_counter() - decode_started

                    quality_started = time.perf_counter()
                    if frame_index % self.config.output.quality_every_n_frames == 0:
                        quality_samples.append(
                            {
                                "frame_index": frame_index,
                                **compute_frame_quality(frame),
                            }
                        )
                    latency_seconds["quality"] += time.perf_counter() - quality_started

                    # Quality measures describe the original input frame;
                    # only the detector/tracker receives the transformed frame.
                    preprocessing_started = time.perf_counter()
                    tracking_frame = preprocessor(frame)
                    latency_seconds["preprocessing"] += (
                        time.perf_counter() - preprocessing_started
                    )
                    tracker_started = time.perf_counter()
                    objects = tracker.track_frame(tracking_frame)
                    latency_seconds["detector_tracker"] += (
                        time.perf_counter() - tracker_started
                    )
                    analytics_started = time.perf_counter()
                    line_counter.update(frame_index, objects)
                    trajectory_store.update(objects)
                    latency_seconds["analytics"] += (
                        time.perf_counter() - analytics_started
                    )

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
                        draw_started = time.perf_counter()
                        annotated = draw_tracks(
                            frame,
                            objects,
                            class_names=class_names,
                        )
                        annotated = draw_analytics(
                            annotated,
                            lines=resolved_lines,
                            line_counts=line_counter.counts,
                            trajectories=trajectory_store.visible_trajectories(),
                        )
                        latency_seconds["draw"] += time.perf_counter() - draw_started
                        output_started = time.perf_counter()
                        writer.write(annotated)
                        latency_seconds["output_write"] += (
                            time.perf_counter() - output_started
                        )

            if not frames:
                raise ValueError(f"No readable frames found in video: {source}")

            if raw_video_path is not None and video_path is not None:
                encode_started = time.perf_counter()
                transcode_video_h264(raw_video_path, video_path)
                latency_seconds["encode_transcode"] += (
                    time.perf_counter() - encode_started
                )
        finally:
            if raw_video_path is not None:
                raw_video_path.unlink(missing_ok=True)

        summary = {
            "processed_frames": len(frames),
            "total_detections": total_detections,
            "unique_tracks": len(unique_track_ids),
            "track_ids": sorted(unique_track_ids),
            "objects_by_class_id": objects_by_class_id,
            "analytics": {
                "line_counts": line_counter.summary(),
                "trajectories": trajectory_store.summary(),
            },
        }

        quality_metric_samples = [
            {name: float(sample[name]) for name in QUALITY_METRICS}
            for sample in quality_samples
        ]
        quality_summary = summarize_quality_samples(quality_metric_samples)
        summary["quality"] = {
            "sampling": {
                "every_n_frames": self.config.output.quality_every_n_frames,
                "frame_index_base": 0,
            },
            "sample_count": len(quality_samples),
            "sampled_frame_indices": [
                int(sample["frame_index"]) for sample in quality_samples
            ],
            "samples": quality_samples,
            "metrics": quality_summary["metrics"],
        }

        total_latency = time.perf_counter() - run_started
        frame_count = len(frames)
        summary["latency"] = {
            "frame_count": frame_count,
            "total_seconds": round(total_latency, 6),
            "stage_seconds": {
                name: round(seconds, 6) for name, seconds in latency_seconds.items()
            },
            "stage_ms_per_frame": {
                name: round((seconds / frame_count) * 1000, 6) if frame_count else 0.0
                for name, seconds in latency_seconds.items()
            },
            "effective_fps": round(frame_count / total_latency, 6)
            if total_latency > 0
            else 0.0,
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
