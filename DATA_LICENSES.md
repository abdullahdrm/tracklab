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

- `data/sample.mp4`, `data/sample1.mp4` and `data/sample2.mp4` are local sample
  inputs used for pipeline demonstrations and are ignored by Git.
- Their original source URLs and redistribution terms are not recorded in the
  repository. Treat them as local-only inputs until provenance and licensing
  are verified; the generated demo video is therefore not committed.

## Project source

TrackLab source code and repository-authored documentation are released under
[AGPL-3.0-only](LICENSE). This project license does not override dataset,
model, dependency or video-source terms listed above.
