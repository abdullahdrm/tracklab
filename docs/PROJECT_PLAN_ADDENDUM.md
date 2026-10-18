# TrackLab Project Plan Addendum — As-built update (2026-10-17)

**Base document:** [`PROJECT_PLAN.pdf`](PROJECT_PLAN.pdf), v1.1 dated 12 October 2026. The PDF is kept unchanged as the original handover snapshot; this addendum records implementation changes made since then.

## Week 2 / Day 10: implemented scope

- E2 real-image and E2b synthetic-degradation ablations are complete and use only MOT17 tuning sequences 04/05/09/11. Hold-out sequences 02/10/13 were not evaluated.
- The reproducible result tables are in [`../reports/day10_ablation_results.md`](../reports/day10_ablation_results.md); raw JSON outputs remain under ignored `outputs/mot17/ablations/`.
- `evaluation/degradations.py`, `evaluation/ablation.py`, and `scripts/run_ablation.py` provide controlled synthetic degradations and separate real/synthetic stages.
- `evaluation/mot17_detector_comparison.py` accepts an optional frame transform for YOLO input. Precomputed MOT17 FRCNN detections are not transformed; they are already boxes generated upstream.
- The preprocessing implementation includes reusable gamma LUTs, reusable CLAHE instances, fixed chains, and frame shape/dtype validation.
- Additional exploratory Day 10 experiments compare processing order and LAB `L`, HSV `V`, and YCrCb `Y` channel variants, plus a focused real-time throughput benchmark. These are documented extensions of Day 10, not separate calendar days.

## Measured findings

- Baseline (no preprocessing): HOTA 0.4351, IDF1 0.5111.
- E2 CLAHE: HOTA 0.4512, IDF1 0.5373.
- E2 gamma → LAB CLAHE: HOTA 0.4542, IDF1 0.5354, but measured FPS is below 30 on some individual sequences.
- Color-space candidate YCrCb CLAHE → Gamma: HOTA 0.4518, IDF1 0.5395, measured 41.51 aggregate FPS in that run. This is a tuning candidate, not a final selected pipeline.
- Synthetic results depend on the degradation. CLAHE helps the low-light condition slightly but reduces HOTA under the tested low-contrast and Gaussian-noise conditions. E2b FPS includes time to synthesize the degradation and therefore is not production-only FPS.

## Boundaries and remaining work

- Current MOT17 evaluation scores pedestrian class 0 only. It does not establish vehicle/person classification performance.
- Synthetic low-light, low-contrast, Gaussian-noise and Gaussian-blur conditions do not prove performance on real fog or rain.
- Throughput measurements cover frame reading, preprocessing, YOLO, ByteTrack and prediction-file writing; full FastAPI, video encoding and end-to-end service latency still need profiling.
- Optuna search, YOLO dataset conversion, fine-tuning and final hold-out evaluation remain future tasks. Hold-out must remain untouched until Day 17.

## Source files added or extended

- Evaluation: `evaluation/degradations.py`, `evaluation/ablation.py`, `evaluation/mot17_detector_comparison.py`.
- Preprocessing: `src/tracklab/preprocessing/colorspace.py`, `clahe.py`, `gamma.py`, `factory.py`, and `src/tracklab/config.py`.
- Runners: `scripts/run_ablation.py`, `scripts/run_realtime_ablation.py`, `scripts/run_colorspace_ablation.py`.
- Tests: `tests/unit/test_ablation.py`, `test_degradations.py`, `test_colorspace.py`, and `test_preprocessing.py`.
- Result report: `reports/day10_ablation_results.md`.


## Week 3 / Day 11: YOLO dataset conversion

- Added `src/tracklab/datasets/to_yolo.py` and unit coverage in `tests/unit/test_to_yolo.py`.
- The generated dataset lives locally at `data/processed/MOT17_yolo_day11/`; images are symbolic links into the MOT17 source, so generated data is not committed.
- Sequence-level split: train uses MOT17-04/05/09-FRCNN (2,412 frames; 59,799 eligible pedestrian boxes), validation uses MOT17-11-FRCNN (900 frames; 9,436 eligible pedestrian boxes).
- The converter emits YOLO class `0` labels with normalized `xywh`; clipped boxes must have positive area and finite values in `[0, 1]`. Empty label files are retained for frames without an eligible pedestrian box.
- Dataset audit verified the YAML paths, frame/label parity, symbolic-link targets within the source MOT17 tree, and absence of hold-out sequences 02/10/13.
- Full local validation: 167 tests passed; Ruff lint and format passed. Fine-tuning has not started yet.
