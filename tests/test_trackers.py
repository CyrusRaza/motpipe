import numpy as np
import pytest

from motpipe.trackers import TRACKERS


def synthetic_scene(n_frames=40):
    """Two boxes moving in opposite directions, one detection frame at a time."""
    for t in range(n_frames):
        boxes = np.array(
            [
                [10 + 4 * t, 50, 60 + 4 * t, 120],
                [400 - 4 * t, 200, 450 - 4 * t, 280],
            ],
            dtype=float,
        )
        yield boxes, np.array([0.9, 0.85]), np.array([0, 2])


@pytest.mark.parametrize("name", ["sort", "bytetrack"])
def test_tracker_keeps_ids_stable(name):
    tracker = TRACKERS.create(name)
    frame = np.zeros((480, 640, 3), np.uint8)
    seen = set()
    for boxes, scores, classes in synthetic_scene():
        out = tracker.update(boxes, scores, classes, frame)
        assert out.ndim == 2 and out.shape[1] == 5
        seen.update(out[:, 4].astype(int))
    assert len(seen) == 2


@pytest.mark.parametrize("name", ["sort", "bytetrack"])
def test_tracker_handles_empty_frames(name):
    tracker = TRACKERS.create(name)
    frame = np.zeros((480, 640, 3), np.uint8)
    empty = (np.empty((0, 4)), np.empty(0), np.empty(0, int))
    assert tracker.update(*empty, frame).shape == (0, 5)
    for boxes, scores, classes in synthetic_scene(5):
        tracker.update(boxes, scores, classes, frame)
    assert tracker.update(*empty, frame).shape[1] == 5


@pytest.mark.parametrize("name", ["sort", "bytetrack"])
def test_tracker_recovers_identity_after_short_gap(name):
    tracker = TRACKERS.create(name)
    frame = np.zeros((480, 640, 3), np.uint8)
    ids = []
    box = np.array([[100.0, 100, 160, 200]])
    for t in range(30):
        if t in (15, 16):  # two missed detections
            out = tracker.update(np.empty((0, 4)), np.empty(0), np.empty(0, int), frame)
        else:
            out = tracker.update(box + [2 * t, 0, 2 * t, 0], np.array([0.9]), np.array([0]), frame)
        ids.extend(out[:, 4].astype(int))
    if name == "bytetrack":  # SORT's default max_age=1 deliberately forgets after 2 misses
        assert len(set(ids)) == 1


def test_tracker_ids_are_per_instance():
    boxes = np.array([[10.0, 10, 50, 90]])
    first = None
    for _ in range(2):
        tracker = TRACKERS.create("sort")
        for _ in range(4):
            out = tracker.update(boxes, np.array([0.9]), np.array([0]), None)
        first = first if first is not None else out[0, 4]
        assert out[0, 4] == first
