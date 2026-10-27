from __future__ import annotations

import cv2
import numpy as np


def _color_for(track_id: int) -> tuple[int, int, int]:
    """Stable, visually distinct BGR color per track id."""
    hue = (track_id * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Draw ``[x1, y1, x2, y2, id]`` rows onto ``frame`` in place."""
    for x1, y1, x2, y2, track_id in tracks:
        color = _color_for(int(track_id))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, f"ID {int(track_id)}", (p1[0], max(p1[1] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return frame
