# TrackLab current verification

TrackLab is presented as a configurable YOLO11n + ByteTrack video analytics
pipeline. The public README links the measured MOT17 results, fine-tuning
regression analysis, model card, license register and reproducible commands.

## Published demo verification

Each published preview is an annotated derivative with detector boxes, track
IDs, a yellow counting line, directional counters and smoothed trajectory
trails. The rendered video and pipeline JSON were checked as a pair.

| Demo | Source frames | Detections | Unique tracks | Line crossings |
|---|---:|---:|---:|---:|
| Urban avenue | 3,029 | 20,572 | 216 | 26 |
| Aerial highway | 911 | 19,752 | 287 | 44 |

The urban source is Pexels video 25865043 and the aerial source is Pexels video
3078508. Their source metadata, download routes, terms and published derivative
paths are recorded in `DATA_LICENSES.md`. The rainy-road and pedestrian source
records remain in the same register.

## Technical checks

- Default COCO YOLO11n and ByteTrack settings were used for both traffic runs;
  only analytics line geometry and the documented three-second trail settings
  were overridden.
- Output JSON frame counts match the corresponding H.264 frame counts.
- Final traffic MP4s are H.264 1280×720 encodings below 20 MB; GIF previews
  are below 8 MB and were visually inspected.
- Frame-level instrumentation confirmed changing counters and a maximum visible
  trail length of the configured three seconds at each source frame rate.
- MOT17 results, the fine-tuning regression and known service limitations remain
  documented in the linked experiment reports rather than repeated here.
