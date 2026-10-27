from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, Detections, resolve_device, weights_dir


class _TorchvisionDetector(BaseDetector):
    """Shared logic for torchvision detection models (COCO, 91-index labels)."""

    default_weights_file: str

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.3,
        device: str = "auto",
    ) -> None:
        import torch

        super().__init__(conf_threshold, resolve_device(device))
        local = Path(model_path) if model_path else weights_dir() / self.default_weights_file
        if local.is_file():
            self.model = self._build(pretrained=False)
            self.model.load_state_dict(torch.load(local, map_location=self.device))
        else:  # fall back to torchvision's own download/cache
            self.model = self._build(pretrained=True)
        self.model.eval().to(self.device)
        self._torch = torch

    def _build(self, pretrained: bool):
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> Detections:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with self._torch.no_grad():
            out = self.model([tensor])[0]

        scores = out["scores"].cpu().numpy().astype(float)
        keep = scores >= self.conf_threshold
        return (
            out["boxes"].cpu().numpy().astype(float)[keep],
            scores[keep],
            out["labels"].cpu().numpy().astype(int)[keep],
        )


class FasterRCNNDetector(_TorchvisionDetector):
    default_weights_file = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

    def _build(self, pretrained: bool):
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        return fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)


class SSDDetector(_TorchvisionDetector):
    default_weights_file = "ssd300_vgg16_coco-b556d3b4.pth"

    def _build(self, pretrained: bool):
        from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16

        weights = SSD300_VGG16_Weights.COCO_V1 if pretrained else None
        return ssd300_vgg16(weights=weights, weights_backbone=None)
