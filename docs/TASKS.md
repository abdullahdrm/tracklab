# TrackLab release checklist

This checklist records the maintained product and reproducibility surface. The
detailed MOT17 and fine-tuning measurements are maintained in `reports/`.

## Implemented

- [x] YOLO11n detection with ByteTrack identity tracking.
- [x] H.264 video, structured JSON and optional MOTChallenge output.
- [x] Directional line counting with pixel or normalized line coordinates.
- [x] Smoothed, time-bounded trajectory trails with complete JSON history.
- [x] FastAPI local demo, SQLite job state and Docker Compose deployment.
- [x] MOT17 tuning, tracker ablation, frozen hold-out and fine-tuning reports.
- [x] Pedestrian, urban-avenue, aerial-highway and rainy-road demo derivatives.

## Reproduction entry points

- `configs/pipeline.yaml` — default detector, tracker and analytics pipeline.
- `configs/pipeline_demo_urban_avenue.yaml` — urban traffic demo.
- `configs/pipeline_demo_aerial_highway.yaml` — aerial highway demo.
- `configs/pipeline_demo_rainy.yaml` — rainy-road demo.
- `scripts/evaluate_mot.py` and `scripts/run_final_holdout.py` — MOT17
  evaluation workflows described in the linked reports.

The default detector and tracker thresholds are unchanged by the published
demo-specific analytics configurations. Model weights and datasets remain
external dependencies and are intentionally excluded from Git.
