# Fine-tuning root-cause investigation

## Decision

The original COCO-pretrained YOLO11n remains TrackLab's default detector. The
BDD100K fine-tuned checkpoint is retained as an explicit research comparison,
but it is not the production/default model for the current MOT17 objective.

This is a measured decision, not a claim that fine-tuning can never help on a
different target domain.

## Evidence reviewed

| Check | Result | Interpretation |
|---|---|---|
| BDD100K source-to-YOLO audit | 79,863/79,863 image-label pairs matched; 0 mismatches | No systematic pairing defect was found. |
| Fine-tuning artifact | 30/30 epochs; `best.pt`, `last.pt`, `results.csv` present | The intended run completed and produced traceable artifacts. |
| Class mapping | `args.yaml` records `cls_remap: true`; checkpoint names are `person, bicycle, car, motorcycle, bus, truck` | The COCO-to-six-class name mapping is not supported as the regression cause. |
| Runtime load-path recheck | A fresh target model built through the installed `DetectionTrainer` printed `Remapped 6/6 cls head rows` and `Transferred 499/499 items` | The actual Day 13 model-construction path applies the intended name-based transfer. |
| Comparable E3 evaluation | Same four tuning sequences, `imgsz=960`, `conf=0.1`, `iou=0.7`, and ByteTrack settings | The comparison is controlled at the declared configuration level. |
| Day 14 rerun | MOT17-05 rerun produced COCO HOTA `0.403436` and fine-tuned HOTA `0.234718`, matching the saved E3 artifacts | The reported regression is reproducible with the current evaluation code. |
| Tracking result | HOTA `0.4351 -> 0.3316`, DetA `0.4353 -> 0.3233`, IDF1 `0.5111 -> 0.3790`, IDSW `371 -> 476` | The fine-tuned model is worse across detection and association metrics. |
| Raw detector diagnostic | Fine-tuned precision/recall/F1 on MOT17-05 were below COCO (`0.3503/0.4334/0.3874` vs `0.5327/0.7009/0.6053`) | The regression starts before ByteTrack. |
| Size-stratified diagnostic | Fine-tuned recall collapsed for boxes >=150 px (`0.3320` vs `0.7793`) while small/medium bins were closer | The failure is not adequately described as only a small-object problem. |

The E3 source reports are `reports/e3_finetuned_vs_coco.md`,
`outputs/mot17/e3_coco/tuning_summary.json`, and
`outputs/mot17/e3_finetuned/tuning_summary.json`. Hold-out sequences were not
used. The raw-detector and size-stratified rows above are evidence recorded in
the audit notes supplied with this repository; their generator is not yet a
committed evaluation script, so they are treated as diagnostic evidence rather
than official TrackEval metrics.

## Root-cause classification

- **Confirmed dataset pairing bug:** No. The full pairing audit found no
  mismatch, and the visual spot check found no obvious image-label error.
- **Confirmed class-remapping bug:** No. The installed Ultralytics runtime
  remapped the six overlapping class rows by name, and the saved checkpoint
  exposes the expected target names.
- **Confirmed evaluation mismatch:** No evidence. The E3 JSON artifacts record
  the same declared inference and tracker settings for both models.
- **Confirmed software issue:** A reproducibility defect was found and fixed:
  the training wrapper previously relied on the installed Ultralytics default
  for `cls_remap` instead of passing the setting from project configuration.
  The historical run still recorded `cls_remap: true`, so this omission does
  not explain its measured regression. The project now pins the tested
  Ultralytics version (`8.4.175`) so the remapping implementation cannot change
  silently on a clean install.
- **Supported explanation:** The fine-tuned detector itself generalizes worse
  to MOT17, and the broad detector-level degradation is directly supported by
  raw-detection diagnostics.
- **Likely but not proven mechanism:** BDD100K driving-camera imagery and
  MOT17 surveillance imagery differ in viewpoint, scale, occlusion, and
  appearance. Class imbalance and fine-tuning-induced loss of useful COCO
  features are plausible contributors, but the current artifacts do not
  isolate their individual causal effects.

## Configuration policy

`configs/pipeline.yaml` and `configs/detector.yaml` continue to point to the
COCO checkpoint. The fine-tuned path is available only through the explicit
`configs/pipeline_finetuned.yaml` and `configs/detector_finetuned.yaml`
configuration pair. No new training run is justified by the current evidence.
