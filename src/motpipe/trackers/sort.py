"""SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Independent implementation of the algorithm described in the paper
(https://arxiv.org/abs/1602.00763): a constant-velocity Kalman filter per track
plus IoU-based Hungarian assignment.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import BaseTracker


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``[N, 4]`` and ``[M, 4]`` xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """xyxy -> measurement ``[cx, cy, area, aspect_ratio]``."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-9)])


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(x[2] * x[3], 0.0))
    h = x[2] / w if w > 0 else 0.0
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class _KalmanBoxTrack:
    """Constant-velocity Kalman filter over ``[cx, cy, s, r, vx, vy, vs]``."""

    F: np.ndarray  # transition, process-noise and measurement-noise matrices
    Q: np.ndarray  # are assigned below the class
    R: np.ndarray
    H = np.eye(4, 7)

    def __init__(self, box: np.ndarray, track_id: int) -> None:
        self.id = track_id
        self.x = np.zeros(7)
        self.x[:4] = _xyxy_to_z(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:  # keep the area non-negative
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _x_to_xyxy(self.x)

    def update(self, box: np.ndarray) -> None:
        y = _xyxy_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    @property
    def box(self) -> np.ndarray:
        return _x_to_xyxy(self.x)


_F = np.eye(7)
_F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0
_Q = np.eye(7)
_Q[-1, -1] *= 0.01
_Q[4:, 4:] *= 0.01
_R = np.eye(4)
_R[2:, 2:] *= 10.0
_KalmanBoxTrack.F, _KalmanBoxTrack.Q, _KalmanBoxTrack.R = _F, _Q, _R


class SORTTracker(BaseTracker):
    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        frame_rate: float = 30.0,
    ) -> None:
        super().__init__(frame_rate)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_KalmanBoxTrack] = []
        self._next_id = 1
        self._frame_count = 0

    def update(self, boxes, scores, class_ids, frame=None) -> np.ndarray:
        self._frame_count += 1
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)

        predicted = np.array([t.predict() for t in self._tracks]).reshape(-1, 4)
        valid = ~np.isnan(predicted).any(axis=1)
        self._tracks = [t for t, ok in zip(self._tracks, valid, strict=True) if ok]
        predicted = predicted[valid]

        matched_dets: set[int] = set()
        if len(predicted) and len(boxes):
            ious = iou_matrix(boxes, predicted)
            rows, cols = linear_sum_assignment(-ious)
            for d, t in zip(rows, cols, strict=True):
                if ious[d, t] >= self.iou_threshold:
                    self._tracks[t].update(boxes[d])
                    matched_dets.add(d)

        for d in range(len(boxes)):
            if d not in matched_dets:
                self._tracks.append(_KalmanBoxTrack(boxes[d], self._next_id))
                self._next_id += 1

        out = [
            [*t.box, t.id]
            for t in self._tracks
            if t.time_since_update < 1
            and (t.hit_streak >= self.min_hits or self._frame_count <= self.min_hits)
        ]
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        return np.array(out, dtype=float).reshape(-1, 5)
