# TrackLab project audit — 2026-10-25

This audit compares `docs/TASKS.md`, the current source tree, committed
experiment reports, available output artifacts, and the current test suite.
The status labels describe the evidence available in this checkout; a
historical result is not presented as a new rerun.

| Day | Scope | Status | Evidence and boundary |
|---:|---|---|---|
| 1 | Environment and initial YOLO/ByteTrack run | PASS | Repository structure, model assets, `.venv`, and setup history are present; the original smoke-run evidence is recorded in `docs/PROGRESS.md`. |
| 2 | Python video pipeline and JSON/MP4 output | PASS | `src/tracklab/pipeline.py`, I/O modules, CLI, and current pipeline tests are present. |
| 3 | FastAPI service and persistent jobs | PASS | API routes, JobManager, database models, and integration tests are present; the full suite passes. |
| 4 | Upload UI, H.264 output, validation limits | PASS | UI, FFmpeg path, validation logic, and integration coverage are present. |
| 5 | Docker, README, CI, v0.1.1 hardening | PASS (documented) | Docker files, CI workflow, release tag, and historical container verification are recorded. Docker was not rebuilt during this audit. |
| 6 | MOT17 reader and split manifest | PASS | Reader, split validation, and tuning/hold-out separation are present. Hold-out sequences remained untouched. |
| 7 | TrackEval bridge and E0 | PASS | Bridge, self-evaluation tests, and the documented perfect E0 result are present. |
| 8 | Baseline and `conf`/`imgsz` selection | PASS | Evaluation scripts, 12-run sweep report, and selected `0.10/960` configuration are present. |
| 9 | Quality metrics and preprocessing | PASS | Quality/preprocessing modules and tests are present; default remains `mode: none`. |
| 10 | Real/synthetic ablation and color-space experiments | PASS | `reports/day10_ablation_results.md`, ignored JSON outputs, tests, and progress records are present. |
| 11 | MOT17 YOLO conversion | PASS | Converter, split safeguards, tests, and the documented 3,312-frame dataset audit are present. |
| 12–13 | BDD100K six-class fine-tuning | PASS with caveat | Full 30-epoch artifacts exist and the checkpoint has the expected six class names. The recorded Ultralytics run used zero dataloader workers despite `workers: 4`; this affects throughput interpretation, not the checkpoint validity. |
| 14 | Fine-tuned vs COCO comparison | PASS | E3 JSON artifacts and report use the same declared inference/tracker settings on the four tuning sequences. Fine-tuning regressed the target metrics. |
| 15 | Line counting and trajectories | PASS | Analytics modules, pipeline integration, five analytics tests, 203-test full suite, and a 505-frame sample video demo are present. |

## Remaining project work

Days 16–20 remain open in `docs/TASKS.md`: tracker ablations, one-time
hold-out evaluation on Days 17, final error analysis, CI/documentation polish,
and the final demo/release decision. The hold-out sequences MOT17-02, -10, and
-13 were not used in this audit.

The fine-tuning decision and its uncertainty boundaries are documented in
[`fine_tuning_root_cause.md`](fine_tuning_root_cause.md). The default pipeline
continues to use COCO YOLO11n; the fine-tuned checkpoint is opt-in.
