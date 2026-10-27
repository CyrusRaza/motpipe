from ..registry import Registry
from .base import BaseTracker

TRACKERS: Registry[BaseTracker] = Registry("tracker")

TRACKERS.register("sort", "motpipe.trackers.sort:SORTTracker")
TRACKERS.register("bytetrack", "motpipe.trackers.bytetrack:ByteTrackTracker")
TRACKERS.register("deepsort", "motpipe.trackers.deepsort:DeepSORTTracker")

__all__ = ["BaseTracker", "TRACKERS"]
