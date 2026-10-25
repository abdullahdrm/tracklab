# TrackLab model card

## Summary

TrackLab uses two detector checkpoints through the same YOLO + ByteTrack
pipeline:

1. **Default:** official COCO-pretrained Ultralytics YOLO11n, referenced as
   `models/yolo11n.pt`.
2. **Comparison:** YOLO11n fine-tuned for six BDD100K road-object classes,
   referenced as `outputs/training/bdd100k_yolo11n_six_class/weights/best.pt`.

The fine-tuned checkpoint is not the default because it reduced the MOT17
pedestrian tracking objective in both tuning and frozen hold-out evaluations.
Neither weight file is committed to Git.

## Intended use

- Local experimentation and reproducible pedestrian MOT evaluation.
- Video demonstrations with visible track IDs, line counts and trajectories.
- Research into detector, preprocessing and tracker interactions.

This is not a safety-certified surveillance, access-control, law-enforcement,
autonomous-driving or identity-recognition model. Track IDs are temporary
within a video and are not human identities.

## Training and data

The fine-tuned model was trained with Ultralytics YOLO11n, `imgsz=640`, batch 4,
30 epochs, MPS, and the BDD100K labelled train/validation split. The six output
classes are `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.
`pedestrian` and `rider` were mapped to `person` by the repository converter.

The MOT17 YOLO conversion used only tuning sequences for development and kept
MOT17-02/10/13 out of training and tuning. The final hold-out protocol was
committed before its results were inspected.

See [`DATA_LICENSES.md`](DATA_LICENSES.md) for dataset terms and attribution.

## Evaluation

All TrackEval results use pedestrian MOTChallenge evaluation with YOLO11n,
`imgsz=960`, `conf=0.10`, `IoU=0.70`, and the frozen ByteTrack baseline.

| Checkpoint / split | HOTA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| COCO / tuning 04,05,09,11 | 0.4351 | 0.4725 | 0.5111 | 371 |
| Fine-tuned / tuning 04,05,09,11 | 0.3316 | — | 0.3790 | 476 |
| COCO / hold-out 02,10,13 | 0.3237 | 0.3334 | 0.3931 | 354 |
| Fine-tuned / hold-out 02,10,13 | 0.3011 | 0.3389 | 0.3728 | 667 |

The fine-tuned BDD100K validation metrics were Precision=0.636, Recall=0.445,
mAP50=0.496 and mAP50-95=0.305. Those are BDD100K detection metrics and do
not predict MOT17 tracking performance.

## Limitations and failure modes

- BDD100K and MOT17 differ in camera viewpoint, scene distribution, density,
  annotation style and domain. Fine-tuning produced lower MOT17 performance.
- The fine-tuned model has high ID switch counts on hold-out, indicating that
  detector calibration and temporal box continuity matter to ByteTrack.
- MOT17 measures pedestrians only; vehicle class performance is not validated
  by the reported MOT metrics.
- Small, distant, occluded or low-contrast people can be missed. Track IDs can
  change after long occlusion or missed detections.
- Results are hardware, preprocessing, threshold and video-quality dependent.
- The E5 manual count is a bounded one-sequence audit, not a population-level
  human-count accuracy estimate.

## Bias, privacy and safety

The source datasets contain road scenes and people. Do not use TrackLab to make
decisions about people, infer identity, or retain uploaded footage without a
lawful purpose and appropriate consent/notice. The project does not provide
face recognition or identity persistence across videos. Review dataset terms,
privacy obligations and applicable law before deployment.

## Reproducibility

```bash
python scripts/run_tracker_ablation.py \
  --output-root outputs/mot17/tracker_ablation_day16
python scripts/run_final_holdout.py \
  --protocol configs/day17_final_protocol.yaml
```

The detailed investigation is in
[`reports/fine_tuning_root_cause.md`](reports/fine_tuning_root_cause.md), and
the final comparison is in
[`reports/day17_final_evaluation.md`](reports/day17_final_evaluation.md).

## Licensing

TrackLab source is AGPL-3.0-only. The detector code/model, BDD100K, MOT17,
TrackEval and local sample videos have separate terms; see
[`DATA_LICENSES.md`](DATA_LICENSES.md). This card does not grant rights to
redistribute weights or dataset files.
