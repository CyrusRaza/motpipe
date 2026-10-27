import numpy as np
import pytest

from motpipe import BaseDetector, BaseTracker, Pipeline
from motpipe.detectors import DETECTORS
from motpipe.registry import Registry
from motpipe.trackers import TRACKERS


class FixedDetector(BaseDetector):
    def detect(self, frame):
        return np.array([[1.0, 2, 30, 40]]), np.array([0.9]), np.array([1])


class CountingTracker(BaseTracker):
    def __init__(self, frame_rate=30.0):
        super().__init__(frame_rate)
        self.calls = 0

    def update(self, boxes, scores, class_ids, frame):
        self.calls += 1
        return np.column_stack([boxes, np.full(len(boxes), self.calls)])


def test_pipeline_accepts_any_strategy_pair():
    pipe = Pipeline(FixedDetector(), CountingTracker())
    frames = [np.zeros((50, 50, 3), np.uint8)] * 3
    results = list(pipe.run(frames))
    assert [r.index for r in results] == [0, 1, 2]
    assert results[-1].tracks[0, 4] == 3
    assert all(r.fps > 0 for r in results)


def test_strategies_can_be_swapped_at_runtime():
    pipe = Pipeline(FixedDetector(), CountingTracker())
    frame = np.zeros((50, 50, 3), np.uint8)
    pipe.process(frame)
    pipe.tracker = TRACKERS.create("sort")
    assert pipe.process(frame).shape[1] == 5


def test_registry_lists_builtin_strategies():
    assert {"yolov8", "yolov5", "fasterrcnn", "ssd"} <= set(DETECTORS.names())
    assert {"sort", "bytetrack", "deepsort"} <= set(TRACKERS.names())


def test_registry_unknown_name_lists_options():
    with pytest.raises(ValueError, match="Available: .*sort"):
        TRACKERS.get("nope")


def test_registry_register_decorator_and_duplicates():
    reg = Registry("thing")

    @reg.register("a")
    class A:
        pass

    assert reg.create("A").__class__ is A
    with pytest.raises(ValueError):
        reg.register("a", A)
