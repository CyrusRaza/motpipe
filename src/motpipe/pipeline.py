from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .detectors import BaseDetector
from .trackers import BaseTracker


@dataclass
class FrameResult:
    index: int
    frame: np.ndarray
    tracks: np.ndarray  # [M, 5]: x1, y1, x2, y2, track_id
    fps: float  # processing throughput for this frame (detect + track)


class Pipeline:
    """Context of the strategy pattern: runs any detector with any tracker.

    The pipeline only knows the ``BaseDetector`` / ``BaseTracker`` interfaces, so
    either strategy can be swapped without touching this class.
    """

    def __init__(self, detector: BaseDetector, tracker: BaseTracker) -> None:
        self.detector = detector
        self.tracker = tracker

    def process(self, frame: np.ndarray) -> np.ndarray:
        boxes, scores, class_ids = self.detector.detect(frame)
        return self.tracker.update(boxes, scores, class_ids, frame)

    def run(self, frames: Iterable[np.ndarray]) -> Iterator[FrameResult]:
        for index, frame in enumerate(frames):
            start = time.perf_counter()
            tracks = self.process(frame)
            elapsed = time.perf_counter() - start
            yield FrameResult(index, frame, tracks, 1.0 / elapsed if elapsed > 0 else 0.0)


def create_pipeline(
    detector: str,
    tracker: str,
    conf: float = 0.3,
    device: str = "auto",
    frame_rate: float = 30.0,
    detector_kwargs: dict | None = None,
    tracker_kwargs: dict | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Pipeline:
    """Build a pipeline from strategy names (see ``DETECTORS.names()`` / ``TRACKERS.names()``)."""
    from .detectors import DETECTORS
    from .trackers import TRACKERS

    if on_status:
        on_status(f"Loading detector '{detector}' and tracker '{tracker}'")
    det = DETECTORS.create(detector, conf_threshold=conf, device=device, **(detector_kwargs or {}))
    trk = TRACKERS.create(tracker, frame_rate=frame_rate, **(tracker_kwargs or {}))
    return Pipeline(det, trk)
