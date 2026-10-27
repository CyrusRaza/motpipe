from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTracker(ABC):
    """Strategy interface for multi-object trackers."""

    def __init__(self, frame_rate: float = 30.0) -> None:
        self.frame_rate = frame_rate

    @abstractmethod
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Consume one frame of detections and return the active tracks.

        Must be called once per frame, including frames without detections.

        Returns:
            ``float [M, 5]`` rows of ``x1, y1, x2, y2, track_id``.
        """

    @staticmethod
    def empty() -> np.ndarray:
        return np.empty((0, 5), dtype=float)
