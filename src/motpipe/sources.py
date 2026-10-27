"""Frame sources (video file, webcam, image folder) and the output video writer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class FrameSource(Protocol):
    fps: float
    size: tuple[int, int]  # (width, height)

    def __iter__(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class VideoSource:
    """Video file path, or a webcam when given a digit string like ``"0"``."""

    def __init__(self, source: str) -> None:
        self._cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self._cap.isOpened():
            raise OSError(f"Unable to open video source: {source}")
        self.size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        self._cap.release()


class ImageFolderSource:
    """Image sequence, played in sorted filename order."""

    def __init__(self, folder: str, fps: float = 30.0) -> None:
        self._paths = sorted(
            p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._paths:
            raise OSError(f"No images found in: {folder}")
        height, width = cv2.imread(str(self._paths[0])).shape[:2]
        self.size = (width, height)
        self.fps = fps

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self._paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield frame

    def close(self) -> None:
        pass


def open_source(video: str | None = None, image_dir: str | None = None, fps: float = 30.0):
    if (video is None) == (image_dir is None):
        raise ValueError("Provide exactly one of video or image_dir")
    return VideoSource(video) if video is not None else ImageFolderSource(image_dir, fps)


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            raise OSError(f"Unable to open output video for writing: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
