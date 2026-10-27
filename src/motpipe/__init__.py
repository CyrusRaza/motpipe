"""Strategy-pattern multi-object detection and tracking."""

from .detectors import DETECTORS, BaseDetector
from .pipeline import FrameResult, Pipeline, create_pipeline
from .trackers import TRACKERS, BaseTracker

__version__ = "0.1.0"

__all__ = [
    "DETECTORS",
    "TRACKERS",
    "BaseDetector",
    "BaseTracker",
    "FrameResult",
    "Pipeline",
    "create_pipeline",
    "__version__",
]
