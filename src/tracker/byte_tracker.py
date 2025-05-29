from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from tracker.bytetrack.byte_tracker import BYTETracker


class ByteTrack:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        """
        Initialize the ByteTracker.

        Args:
            track_thresh: Tracking threshold
            track_buffer: Tracking buffer size
            frame_rate: Frame rate of the video
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.match_thresh = match_thresh
        self.tracker = BYTETracker(
            args=SimpleNamespace(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                mot20=False,
            ),
            frame_rate=frame_rate,
        )
        self.tracked_objects = {}
        self.frame_id = 0

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
        if not detections:
            return []

        # Convert detections to ByteTrack format
        dets = []
        for det in detections:
            bbox = det["bbox"]
            dets.append([bbox[0], bbox[1], bbox[2], bbox[3], det["confidence"]])

        # Convert to numpy array
        dets = np.array(dets, dtype=np.float32)

        # Update tracks
        online_targets = self.tracker.update(dets, img_info, img_size)

        # Convert tracks to our format
        tracked_objects = []
        for t in online_targets:
            tracked_objects.append(
                {
                    "track_id": t.track_id,
                    "bbox": t.tlbr,  # top-left, bottom-right format
                    "tlwh": t.tlwh,  # top-left, width, height format
                    "class_id": t.class_id if hasattr(t, "class_id") else 0,
                    "score": t.score,
                }
            )

        return tracked_objects
