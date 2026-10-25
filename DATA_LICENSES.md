# Dataset licenses and attribution

## MOT17

- Source: https://motchallenge.net/data/MOT17/
- Dataset: MOT17 training sequences and annotations.
- License stated by MOTChallenge: Creative Commons Attribution-NonCommercial-
  ShareAlike 3.0 (CC BY-NC-SA 3.0).
- Required attribution: MOTChallenge / the dataset authors; retain the source
  URL and license information when describing results based on this dataset.
- Non-commercial restriction: do not assume that using MOT17-derived material
  is permitted in a commercial product or service.
- Share-alike: the source license specifies conditions for distributing
  altered, transformed, or derivative dataset material.
- Third-party warning: MOTChallenge notes that additional original-source
  conditions may apply to some sequences.

TrackLab code has its own project license, declared in `pyproject.toml`.
That project license does not replace or override the MOT17 dataset license.
The downloaded dataset, annotations, and dataset-derived artifacts are kept
outside Git under ignored data/output directories.

Before redistributing dataset files, annotations, derived data, or using
dataset-derived artifacts commercially, review the source license and any
applicable third-party terms.

## BDD100K

- Source and download instructions: https://bdd-data.berkeley.edu/ and
  https://github.com/bdd100k/bdd100k.
- This project used the BDD100K labelled train/validation images and detection
  annotations to train a six-class comparison checkpoint. Raw BDD100K files
  are not included in Git.
- The BDD100K toolkit repository is BSD-3-Clause, but the dataset has its own
  data license and download agreement. Read the official BDD100K license before
  copying, redistributing or using the data commercially:
  https://doc.bdd100k.com/license.html
- Attribution: Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen,
  Fangchen Liu, Vashisht Madhavan and Trevor Darrell, “BDD100K: A Diverse
  Driving Dataset for Heterogeneous Multitask Learning,” CVPR 2020.

## Ultralytics YOLO and model weights

- Runtime: https://github.com/ultralytics/ultralytics
- The pinned package is `ultralytics==8.4.175`; the official repository is
  AGPL-3.0. Ultralytics also offers an Enterprise license for uses that cannot
  satisfy AGPL obligations. Review the current terms at
  https://www.ultralytics.com/license.
- `models/yolo11n.pt` and the fine-tuned `best.pt` are excluded from Git. Do
  not redistribute them without checking the applicable model and dataset
  terms.

## TrackEval

- Source: https://github.com/JonathonLuiten/TrackEval
- TrackEval is MIT-licensed. The vendored/submodule evaluator is used for HOTA,
  CLEAR/MOTA and Identity/IDF1 calculations; its license remains separate from
  TrackLab's AGPL-3.0-only license.

## Published demo media

The repository distributes edited, annotated derivatives only. The original
stock downloads are not included.

### Pedestrian tracking

- Source page: [Pexels video 10531835](https://www.pexels.com/video/10531835/).
- Terms: [Pexels license](https://www.pexels.com/legal-pages/license/).
- The license permits free use and modification, including commercial use, and
  does not require attribution. It prohibits selling unaltered copies or
  redistributing the unmodified asset on another stock platform.
- Published derivatives: `docs/assets/demos/pedestrian-tracking.gif` and
  `docs/assets/demos/pedestrian-tracking.mp4`.

### Ordinary road traffic — urban avenue

- Source page: [Traffic on Avenida Afonso Schmidt, Santos — Pexels video
  25865043](https://www.pexels.com/video/transito-de-veiculos-na-avenida-afonso-schmidt-em-santos-25865043/).
- Download route used: [Pexels supported download endpoint](https://www.pexels.com/download/video/25865043/).
- Uploader shown on the source page: FÁBIO REIS DE ABREU (PORTUGA).
- Terms: [Pexels license](https://www.pexels.com/legal-pages/license/).
- Page metadata: 1920×1080, 59.94 FPS, 0:50. Measured downloaded stream:
  H.264, 1920×1080, 59.94 FPS, 50.53 seconds, 3,029 frames.
- Suitability check: busy side-on urban traffic with clearly observable vehicle
  motion. The default detector and tracker produced visible IDs, 216 unique
  tracks and 26 observed line crossings; the line placement is analytics-only.
- Published derivatives: `docs/assets/demos/traffic-urban-avenue.gif` and
  `docs/assets/demos/traffic-urban-avenue.mp4`.
- Pexels does not require attribution; the source page and uploader are listed
  here for provenance.

### Ordinary road traffic — aerial highway

- Source page: [Aerial View of Flow of Traffic in the Highway — Pexels video
  3078508](https://www.pexels.com/video/aerial-view-of-flow-of-traffic-in-the-highway-3078508/).
- Download route used: [Pexels supported download endpoint](https://www.pexels.com/download/video/3078508/).
- Uploader shown on the source page: Mostafa Meraji.
- Terms: [Pexels license](https://www.pexels.com/legal-pages/license/).
- Page metadata: 1920×1080, 29.97 FPS, 0:30. Measured downloaded stream:
  H.264, 1920×1080, 29.97 FPS, 30.40 seconds, 911 frames.
- Suitability check: elevated highway traffic with clear vehicle motion across
  multiple lanes. The default detector and tracker produced visible IDs, 287
  unique tracks and 44 observed line crossings; the line placement is
  analytics-only.
- Published derivatives: `docs/assets/demos/traffic-aerial-highway.gif` and
  `docs/assets/demos/traffic-aerial-highway.mp4`.
- Pexels does not require attribution; the source page and uploader are listed
  here for provenance.

### Rainy-road traffic

- Source page: [Night Highway Traffic on Rainy Road — Pexels video
  40346582](https://www.pexels.com/video/night-highway-traffic-on-rainy-road-40346582/).
- Download route used: [Pexels supported download endpoint](https://www.pexels.com/download/video/40346582/).
- Uploader shown on the source page: Harshal .
- Terms: [Pexels license](https://www.pexels.com/legal-pages/license/).
- Page metadata and measured downloaded stream: H.264, 1920×1080, 25 FPS,
  13 seconds, 325 frames.
- Suitability check: vehicles are visibly moving on a wet nighttime highway;
  the rainy-road analytics line produced one observed truck crossing.
- Published derivatives: `docs/assets/demos/traffic-rainy-road.gif` and
  `docs/assets/demos/traffic-rainy-road.mp4`.
- Pexels does not require attribution; the source page and uploader are listed
  here for provenance.

## Project source

TrackLab source code and repository-authored documentation are released under
[AGPL-3.0-only](LICENSE). This project license does not override dataset,
model, dependency or video-source terms listed above.
