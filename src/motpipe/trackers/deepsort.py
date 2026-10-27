from __future__ import annotations

import numpy as np

from .base import BaseTracker


class DeepSORTTracker(BaseTracker):
    """DeepSORT via the ``deep-sort-realtime`` package (appearance embeddings + Kalman)."""

    def __init__(
        self, max_age: int = 30, n_init: int = 3, frame_rate: float = 30.0, **kwargs
    ) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        super().__init__(frame_rate)
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, **kwargs)

    def update(self, boxes, scores, class_ids, frame) -> np.ndarray:
        detections = [
            ([x1, y1, x2 - x1, y2 - y1], float(s), int(c))  # DeepSORT wants ltwh
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, class_ids, strict=True)
        ]
        tracks = self.tracker.update_tracks(detections, frame=frame)
        rows = [
            [*t.to_ltrb(), int(t.track_id)]
            for t in tracks
            if t.is_confirmed() and t.time_since_update == 0
        ]
        return np.array(rows, dtype=float).reshape(-1, 5)
