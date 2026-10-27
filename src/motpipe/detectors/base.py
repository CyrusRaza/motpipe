from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

Detections = tuple[np.ndarray, np.ndarray, np.ndarray]


def resolve_device(device: str = "auto") -> str:
    """Turn ``"auto"`` into ``"cuda"`` when available, otherwise ``"cpu"``."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def weights_dir() -> Path:
    """Directory searched for model weights (``MOTPIPE_WEIGHTS_DIR`` or ``./models``)."""
    return Path(os.environ.get("MOTPIPE_WEIGHTS_DIR", "models"))


class BaseDetector(ABC):
    """Strategy interface for object detectors."""

    def __init__(self, conf_threshold: float = 0.3, device: str = "auto") -> None:
        self.conf_threshold = conf_threshold
        self.device = device

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Detections:
        """Detect objects in a BGR ``uint8`` frame (OpenCV convention).

        Returns:
            boxes: ``float [N, 4]`` in ``x1, y1, x2, y2`` pixel coordinates.
            scores: ``float [N]`` confidences, all >= ``conf_threshold``.
            class_ids: ``int [N]`` class indices of the underlying model.
        """

    @staticmethod
    def empty() -> Detections:
        return np.empty((0, 4), dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=int)
