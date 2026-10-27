# motpipe: swappable multi-object detection and tracking

[![CI](https://github.com/CyrusRaza/motpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/CyrusRaza/motpipe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Pick any detector and any tracker at runtime and run them on a video, webcam, or image sequence. The design follows the **strategy pattern**: detectors and trackers each sit behind a small interface, so every combination works without extra glue code.

| Detectors (`--detector`) | Trackers (`--tracker`) |
| --- | --- |
| `yolov8` (Ultralytics) | `sort` |
| `yolov5` (Ultralytics, via `torch.hub`) | `deepsort` (appearance embeddings) |
| `fasterrcnn` (torchvision, ResNet-50 FPN) | `bytetrack` |
| `ssd` (torchvision, SSD300 VGG16) | |

Tested with Python 3.10, `ultralytics` 8.3.137, `trackers` 2.6.1 (ByteTrack, from [Roboflow](https://github.com/roboflow/trackers), Apache-2.0), `deep-sort-realtime` 1.3.2 and PyTorch 2.7. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for all versions and licences.

## Install

```bash
git clone https://github.com/CyrusRaza/motpipe.git && cd motpipe
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# Install PyTorch for your platform first: https://pytorch.org/get-started/locally/
pip install -e ".[all]"          # everything, or pick extras: yolov8, yolov5, torchvision, deepsort, bytetrack
```

The base install is light (`numpy`, `opencv-python`, `scipy`). Each strategy imports its heavy dependencies only when selected.

Model weights are fetched automatically on first use (Ultralytics and torchvision caches). To use local weights, put them in `./models/` or set `MOTPIPE_WEIGHTS_DIR`.

## Usage

```bash
motpipe --video traffic.mp4 --detector yolov8 --tracker bytetrack --output out.mp4
motpipe --image-dir frames/ --detector fasterrcnn --tracker sort --output out.mp4 --fps 25
motpipe --video 0 --detector yolov8 --tracker deepsort --show          # webcam, press q to quit
motpipe --list                                                          # show available strategies
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--video` / `--image-dir` | - | Video file, webcam index, or folder of frames (one is required) |
| `--detector` | `yolov8` | Detection strategy |
| `--tracker` | `bytetrack` | Tracking strategy |
| `--conf` | `0.3` | Detection confidence threshold |
| `--device` | `auto` | `auto` picks CUDA if available, otherwise CPU |
| `--output` | - | Annotated output video |
| `--show` | off | Live preview window |

### Bring your own data

No sample data is shipped (the repo only contains code). Point the CLI at any video, webcam, or folder of frames (`.png`/`.jpg`/`.bmp`, played in filename order). By convention, put local files under `data/` and `models/`; both are git-ignored. Public datasets that work well for testing: [MOT17/MOT20](https://motchallenge.net/) and [UA-DETRAC](https://detrac-db.rit.albany.edu/) (check each dataset's licence before redistributing).

### As a library

```python
import cv2
from motpipe import create_pipeline

pipeline = create_pipeline("yolov8", "bytetrack", conf=0.3)
for result in pipeline.run(frames):          # any iterable of BGR frames
    print(result.index, result.tracks)       # rows of [x1, y1, x2, y2, track_id]

# Strategies can be swapped mid-stream:
from motpipe.trackers import TRACKERS
pipeline.tracker = TRACKERS.create("sort")
```

## Architecture

```
frames ──► Pipeline ──► BaseDetector.detect(frame)  ──► boxes, scores, class_ids
              │                                              │
              └────────► BaseTracker.update(boxes, scores, class_ids, frame) ──► [x1, y1, x2, y2, id]
```

- `motpipe/detectors/base.py`: `BaseDetector` interface. Implementations return `xyxy` boxes, scores and class ids for a BGR frame.
- `motpipe/trackers/base.py`: `BaseTracker` interface. `update()` is called once per frame and returns `[x1, y1, x2, y2, track_id]` rows.
- `motpipe/registry.py`: name → class lookup with lazy imports. This is what lets the user pick strategies by string at runtime.
- `motpipe/pipeline.py`: the context class. It depends only on the two interfaces.

### Adding your own detector or tracker

```python
import numpy as np
from motpipe.detectors import DETECTORS, BaseDetector

@DETECTORS.register("mydetector")
class MyDetector(BaseDetector):
    def detect(self, frame):
        ...  # return boxes [N,4] xyxy, scores [N], class_ids [N]
        return self.empty()
```

Once registered, `--detector mydetector` works. The same applies to trackers via `TRACKERS`. Note that the CLI offers only the built-in strategies in its `--detector`/`--tracker` choices, so register custom ones in your own entry script that calls the library API.

## Notes

- Class ids come straight from each model: Ultralytics models use 0-indexed COCO (80 classes), torchvision models use 1-indexed COCO (91 ids).
- The displayed FPS is per-frame processing throughput (detection + tracking), not the source video's frame rate.
- ByteTrack comes from the Roboflow [`trackers`](https://github.com/roboflow/trackers) package (pinned to `>=2.6.1,<3`); SORT is implemented in this repo from the paper.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Licence and credits

MIT, see [LICENSE](LICENSE). Third-party libraries and their versions are credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
