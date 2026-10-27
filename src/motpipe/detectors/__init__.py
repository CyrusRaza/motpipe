from ..registry import Registry
from .base import BaseDetector, Detections

DETECTORS: Registry[BaseDetector] = Registry("detector")

DETECTORS.register("yolov8", "motpipe.detectors.yolov8:YOLOv8Detector")
DETECTORS.register("yolov5", "motpipe.detectors.yolov5:YOLOv5Detector")
DETECTORS.register("fasterrcnn", "motpipe.detectors.torchvision_detectors:FasterRCNNDetector")
DETECTORS.register("ssd", "motpipe.detectors.torchvision_detectors:SSDDetector")

__all__ = ["BaseDetector", "Detections", "DETECTORS"]
