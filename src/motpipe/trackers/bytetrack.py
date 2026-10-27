from __future__ import annotations

import numpy as np

from .base import BaseTracker


class ByteTrackTracker(BaseTracker):
    """ByteTrack (Zhang et al., 2022) via the Roboflow ``trackers`` package.

    Uses low-confidence detections in a second association pass to recover tracks.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.5,
        lost_track_buffer: int = 30,
        minimum_consecutive_frames: int = 2,
        frame_rate: float = 30.0,
    ) -> None:
        from trackers import ByteTrackTracker as _ByteTrack

        super().__init__(frame_rate)
        self.tracker = _ByteTrack(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            high_conf_det_threshold=track_activation_threshold,
        )

    def update(self, boxes, scores, class_ids=None, frame=None) -> np.ndarray:
        import supervision as sv

        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        detections = sv.Detections(
            xyxy=boxes,
            confidence=np.asarray(scores, dtype=float).reshape(-1),
            class_id=None if class_ids is None else np.asarray(class_ids, dtype=int).reshape(-1),
        )
        tracked = self.tracker.update(detections)
        keep = tracked.tracker_id >= 0  # -1 marks tracks not yet confirmed
        return np.column_stack([tracked.xyxy[keep], tracked.tracker_id[keep]]).reshape(-1, 5)
