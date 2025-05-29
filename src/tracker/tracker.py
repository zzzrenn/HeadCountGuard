from typing import Dict, List, Tuple

from loguru import logger

from tracker.tracker_factory import TrackerFactory


class PersonTracker:
    def __init__(self, tracker_type: str = "bytetrack", **kwargs):
        """
        Initialize the person tracker.

        Args:
            tracker_type: Type of tracker to use ('bytetrack' for ByteTracker)
            track_thresh: Tracking threshold
            track_buffer: Tracking buffer size
            match_thresh: Matching threshold
            frame_rate: Frame rate of the video
        """
        self.tracker = TrackerFactory.create_tracker(tracker_type, **kwargs)
        logger.info(f"Tracker initialized: {tracker_type}")

    def update(
        self, detections: List[Dict], img_info: List[int], img_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detection dictionaries
            img_info: List containing [height, width] of the original image
            img_size: Tuple of (target_height, target_width) for model input

        Returns:
            List of tracked objects with their IDs and states
        """
        return self.tracker.update(detections, img_info, img_size)
