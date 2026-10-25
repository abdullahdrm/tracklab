# Analytics resolution audit — 2026-10-25

## Root cause and implementation

The original default line used absolute points `[100, 540]` and `[1800, 540]`.
Those coordinates are inside a 1920×1080 reference frame but the actual
`sample2.mp4` frame is 768×432, so the line was outside the image at `y=540`.
That explains the zero count in the original run; an isolated line moved into
the sample frame produced 23 crossings.

Line configuration now has an explicit `coordinate_space`:

- `pixels` is the default and preserves legacy configurations exactly.
- `normalized` accepts finite coordinates in `[0, 1]`.

Normalized coordinates are resolved once per video using the source dimensions
from `probe_video`. The conversion uses the inclusive pixel range: `0.0` maps
to `0` and `1.0` maps to `width - 1` or `height - 1`. Resolution returns a
copy, so a normalized configuration is not mutated between videos. The same
resolved `LineConfig` tuple is passed to `LineCounter` and `draw_analytics`.
Detector and tracker coordinates are not transformed by this change.

The default line now expresses the original proportional placement as a
normalized horizontal line at 50% of the frame height.

## Fine-tuning and MOT17 dependency audit

No dependency was found between this analytics defect and the historical
fine-tuning regression:

1. `training/finetune.py` consumes `configs/finetune.yaml`, the BDD100K
   `dataset.yaml`, and the selected checkpoint. It does not load
   `configs/analytics.yaml` or `VideoPipeline`.
2. `evaluation/mot17_detector_comparison.py` reads frames directly, calls
   YOLO for detections, updates Ultralytics `BYTETracker`, and writes
   MOTChallenge rows. `evaluation/trackeval_bridge.py` evaluates those rows
   with TrackEval. The line counter is not on this path.
3. `scripts/evaluate_mot.py` loads the production tracker settings, but it
   creates a direct evaluation namespace containing detector and tracker
   settings; it does not run production video analytics.
4. The evaluation path preserves the source frame shape for inference
   transforms, passes source dimensions to `Boxes`, and maps tracker output
   directly to MOT coordinates. Ground-truth rows are frame-validated and the
   TrackEval bridge has a ground-truth self-evaluation regression test.
5. `configs/eval_e3_coco.yaml` uses `models/yolo11n.pt`; the fine-tuned
   comparison uses the explicit
   `outputs/training/bdd100k_yolo11n_six_class/weights/best.pt` checkpoint.

The lower raw-detector precision and recall were observed before ByteTrack,
while line counting runs after tracking in the production pipeline. The line
bug therefore cannot explain the lower HOTA, DetA, AssA, IDF1, MOTA, or higher
IDSW in the recorded E3 and hold-out comparisons. No training or evaluation
bug was discovered, and no retraining was performed.

## Docker verification

The existing isolated validation directory and database were preserved. A
fresh `docker compose ... build` was attempted after the source changes, but
the dependency-install layer stalled while downloading the 454 MB ARM64 Torch
wheel. The Dockerfile now uses a BuildKit pip cache mount to avoid repeating
that download on later builds; dependency versions and runtime dependencies
were not changed.

For decisive application verification, `tracklab:resolution-fix` was built
from the already-built dependency-identical `tracklab:0.1.1` image and the
current source package was reinstalled with `--no-deps`. The isolated Compose
override then passed this real API workflow:

- `/health` returned `{"status":"ok"}`.
- `POST /jobs` accepted `data/sample2.mp4`.
- The job completed as `done` with 505 frames, 4,890 detections, and 145
  unique tracks.
- JSON schema `1.0` contained 505 frame records matching the processed count.
- The normalized line counted 12 positive-to-negative and 11
  negative-to-positive crossings, 23 total.
- The downloaded MP4 was non-empty H.264, 768×432, 25 FPS, 20.2 seconds and
  505 frames.
- Frame 250 was inspected visually; the overlay was inside the frame at the
  expected `y=216` position, from approximately `x=40` to `x=719`.

The project data, source data, original outputs, and original validation
artifacts were not deleted. The validation container remains available on
`127.0.0.1:18000`.

## Regression coverage

The new tests cover legacy pixels, normalized 1920×1080 and 768×432
resolution, inclusive boundaries, finite/range validation, directional
crossing, configuration reuse across resolutions, and the pipeline's shared
resolved geometry for counter and renderer. The full suite passed with 214
tests and the existing FastAPI/Starlette `httpx` deprecation warning.
