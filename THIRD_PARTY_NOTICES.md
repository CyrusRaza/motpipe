# Third-party notices

This repository contains no third-party source code. Everything below is installed separately
from PyPI (or downloaded at first use) and keeps its own licence.

## Algorithms re-implemented from papers

- **SORT**: Bewley et al., "Simple Online and Realtime Tracking", 2016 (`trackers/sort.py`).

## Runtime dependencies

| Package | Tested version | Licence | Used for |
| --- | --- | --- | --- |
| [trackers](https://github.com/roboflow/trackers) (Roboflow) | 2.6.1 | Apache-2.0 | `bytetrack` tracker |
| [supervision](https://github.com/roboflow/supervision) | 0.31.2 | MIT | detection containers for `trackers` |
| [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) | 1.3.2 | MIT | `deepsort` tracker |
| [ultralytics](https://github.com/ultralytics/ultralytics) | 8.3.137 | AGPL-3.0 | `yolov8` detector |
| [ultralytics/yolov5](https://github.com/ultralytics/yolov5) | fetched via `torch.hub` | AGPL-3.0 | `yolov5` detector |
| [torch / torchvision](https://pytorch.org) | 2.7.0 / 0.22.0 | BSD-3-Clause | `fasterrcnn`, `ssd` detectors |
| OpenCV, NumPy, SciPy | – | Apache-2.0 / BSD | core |

ByteTrack algorithm: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022
(https://github.com/ifzhang/ByteTrack).

Ultralytics software is AGPL-3.0. It is only loaded when you select a YOLO detector; check its licence
terms before distributing an application that bundles it.

## Model weights

Weights are downloaded by the respective libraries on first use and are not stored in this repository.
They carry the licences of their publishers (Ultralytics for YOLO, PyTorch for the torchvision COCO models).
