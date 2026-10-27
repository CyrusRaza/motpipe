from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detections, resolve_device, weights_dir


class YOLOv5Detector(BaseDetector):
    """Original Ultralytics YOLOv5, loaded through ``torch.hub``.

    ``repo`` is a GitHub ``owner/name`` (default ``ultralytics/yolov5``) or, with
    ``source="local"``, a path to a local clone of the repository.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.3,
        device: str = "auto",
        repo: str = "ultralytics/yolov5",
        source: str = "github",
    ) -> None:
        import torch

        super().__init__(conf_threshold, resolve_device(device))
        self.model = torch.hub.load(
            repo,
            "custom",
            path=model_path or str(weights_dir() / "yolov5s.pt"),
            source=source,
            verbose=False,
        )
        self.model.conf = conf_threshold
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> Detections:
        # YOLOv5's AutoShape wrapper expects RGB numpy input.
        results = self.model(frame[..., ::-1])
        det = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls
        if len(det) == 0:
            return self.empty()
        return det[:, :4].astype(float), det[:, 4].astype(float), det[:, 5].astype(int)
