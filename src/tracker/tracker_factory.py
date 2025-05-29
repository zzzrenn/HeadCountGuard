from abc import ABC, abstractmethod
from typing import Dict, Type

from tracker.byte_tracker import ByteTrack


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections, img_info, img_size):
        pass


class TrackerFactory:
    _trackers: Dict[str, Type[BaseTracker]] = {"bytetrack": ByteTrack}

    @classmethod
    def create_tracker(cls, tracker_type: str, **kwargs) -> BaseTracker:
        """
        Create a tracker instance based on the specified type.

        Args:
            tracker_type: Type of tracker to create ('byte' for ByteTracker)
            **kwargs: Additional arguments to pass to the tracker constructor

        Returns:
            An instance of the requested tracker

        Raises:
            ValueError: If the tracker type is not supported
        """
        if tracker_type not in cls._trackers:
            raise ValueError(
                f"Unsupported tracker type: {tracker_type}. "
                f"Supported types are: {list(cls._trackers.keys())}"
            )

        return cls._trackers[tracker_type](**kwargs)
