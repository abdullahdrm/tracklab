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

## Local sample videos

- `data/sample.mp4` is a pedestrian street clip. macOS download metadata records
  the Pexels CDN URL for video `10531835`; the corresponding provider page is
  [Pexels video 10531835](https://www.pexels.com/video/10531835/). The
  [Pexels license](https://www.pexels.com/license/) permits free use and
  modification, including commercial use, without required attribution. It
  does prohibit redistributing the unmodified asset on another stock platform
  and does not grant rights to depicted third-party marks or people. The
  annotated, edited preview in `docs/assets/demos/` is therefore published as
  a project demonstration; no Pexels attribution is required. The exact
  uploader name was not recoverable from the local metadata.
- `data/sample2.mp4` is not a normal-weather traffic clip. Its download
  metadata identifies iStock/Getty item `1209071519`, [Motorway traffic
  driving in hazardous sleet](https://www.istockphoto.com/video/motorway-traffic-driving-in-hazardous-sleet-gm1209071519-349602656).
  The local file is a watermarked preview. The [iStock Content License
  Agreement](https://www.istockphoto.com/legal/license-agreement) permits use
  only under an applicable purchased license and expressly limits watermarked
  content to test/sample-layout use; it cannot be used in publicly available
  final materials. No purchase or redistribution permission is documented, so
  neither the source nor an annotated derivative is committed.
- The provided `sample3.mp4` is also identified from download metadata as an
  iStock/Getty item, `1251770963`, [Trucks and cars on a
  highway](https://www.istockphoto.com/tr/video/kamyonlar-ve-arabalar-y%C3%BCksek-yolda-gm1251770963-365061926).
  Visual inspection shows a sunny highway, not rainy-road traffic, and the
  file carries an iStock watermark. The same iStock terms apply; no purchase,
  attribution instruction, or public redistribution permission is documented.
  It remains local-only and is not committed.
- All three files are local sample inputs and remain ignored by Git. The three
  clips were processed for validation on 2026-10-25, but only the Pexels-based
  pedestrian derivative was eligible for publication on the evidence above.

## Project source

TrackLab source code and repository-authored documentation are released under
[AGPL-3.0-only](LICENSE). This project license does not override dataset,
model, dependency or video-source terms listed above.
