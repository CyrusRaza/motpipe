from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detections, resolve_device, weights_dir


class YOLOv8Detector(BaseDetector):
    """Ultralytics YOLOv8. Weights are downloaded automatically if missing."""

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.3,
        device: str = "auto",
    ) -> None:
        from ultralytics import YOLO

        super().__init__(conf_threshold, resolve_device(device))
        self.model = YOLO(model_path or str(weights_dir() / "yolov8n.pt"))
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> Detections:
        result = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        boxes = result.boxes
        return (
            boxes.xyxy.cpu().numpy().astype(float),
            boxes.conf.cpu().numpy().astype(float),
            boxes.cls.cpu().numpy().astype(int),
        )
