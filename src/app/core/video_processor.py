from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from counter import LineCrossingCounter
from detectors import PersonDetector
from tracker import PersonTracker


class VideoProcessor:
    """Handles video processing, detection, tracking, and counting."""

    def __init__(
        self,
        detector_config: Dict[str, Any],
        tracking_config: Dict[str, Any],
        line_crossing_config: Dict[str, Any],
    ):
        """Initialize video processor with configurations."""
        # Initialize detector
        logger.info("Initializing detector...")
        self.detector = PersonDetector(**detector_config)

        # Initialize tracker
        logger.info("Initializing tracker...")
        self.tracker = PersonTracker(**tracking_config)

        # Store configurations
        self.line_crossing_config = line_crossing_config

        # Video processing variables
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width = 0
        self.frame_height = 0
        self.video_fps = 30

        # Counter
        self.counter: Optional[LineCrossingCounter] = None

        # Processing state
        self.is_running = False
        self.current_frame_number = 0
        self.count = 0
        self.count_history: List[int] = []
        self.frame_history: List[int] = []

        # Callbacks
        self.frame_callback: Optional[Callable] = None
        self.count_callback: Optional[Callable] = None
        self.complete_callback: Optional[Callable] = None

        # ROI mask for bbox filtering (separate from counter's ROI)
        self.roi_mask_for_display: Optional[np.ndarray] = None
        self.bbox_filter_enabled = False

    def load_video(self, video_path: str) -> bool:
        """Load a video file and extract properties."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logger.error("Could not open video file")
            return False

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Ensure FPS is valid
        if self.video_fps <= 0:
            self.video_fps = 30

        # Reset processing state
        self.reset_processing_state()
        return True

    def reset_processing_state(self):
        """Reset processing state for new video or restart."""
        self.current_frame_number = 0
        self.count = 0
        self.count_history = []
        self.frame_history = []
        self.is_running = False

    def get_first_frame(self) -> Optional[np.ndarray]:
        """Get the first frame of the video for preview."""
        if not self.cap or not self.cap.isOpened():
            return None

        # Save current position
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Go to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()

        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def setup_counter(self, line_points: tuple, in_side: str):
        """Setup the line crossing counter."""
        self.counter = LineCrossingCounter(
            line_points=line_points,
            in_side=in_side,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            **self.line_crossing_config,
        )

    def setup_counter_with_roi(
        self, line_points: tuple, in_side: str, roi_mask: np.ndarray
    ):
        """Setup the line crossing counter with ROI mask."""
        # Ensure ROI mask has the correct dimensions
        if roi_mask.shape != (self.frame_height, self.frame_width):
            # Resize ROI mask to match frame dimensions
            roi_mask = cv2.resize(
                roi_mask,
                (self.frame_width, self.frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
            # Convert to binary mask
            roi_mask = (roi_mask > 0).astype(np.uint8) * 255

        self.counter = LineCrossingCounter(
            line_points=line_points,
            in_side=in_side,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            roi_mask=roi_mask,
            **self.line_crossing_config,
        )

    def set_callbacks(
        self,
        frame_callback: Callable = None,
        count_callback: Callable = None,
        complete_callback: Callable = None,
    ):
        """Set callback functions for frame updates, count updates, and completion."""
        self.frame_callback = frame_callback
        self.count_callback = count_callback
        self.complete_callback = complete_callback

    def start_processing(self):
        """Start video processing."""
        if not self.cap or not self.cap.isOpened():
            return False

        self.is_running = True
        return True

    def stop_processing(self):
        """Stop video processing."""
        self.is_running = False

    def process_next_frame(self) -> Optional[np.ndarray]:
        """Process the next frame and return the processed frame."""
        if not (self.cap and self.cap.isOpened() and self.is_running):
            return None

        ret, frame = self.cap.read()
        if not ret:
            # Video ended
            self.is_running = False
            if self.complete_callback:
                self.complete_callback(self.count_history, self.frame_history)
            return None

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect and track people
        detections = self.detector.detect(frame_rgb)
        tracked_objects = self.tracker.update(
            detections,
            [self.frame_height, self.frame_width],
            (self.frame_height, self.frame_width),
        )

        # Update counter
        if self.counter is not None:
            self.count = self.counter.update(tracked_objects)

            # Track count history
            self.count_history.append(self.count)
            self.frame_history.append(self.current_frame_number)

            if self.count_callback:
                self.count_callback(self.count, self.current_frame_number)

        # Draw bounding boxes and IDs
        for obj in tracked_objects:
            bbox = obj["bbox"]

            # Apply ROI filtering for display if enabled
            if self._bbox_intersects_roi(bbox):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_rgb,
                    f"ID: {obj['track_id']}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        self.current_frame_number += 1

        if self.frame_callback:
            self.frame_callback(frame_rgb)

        return frame_rgb

    def draw_counting_line(self, frame: np.ndarray, line_points: tuple) -> np.ndarray:
        """Draw the counting line on the frame."""
        if line_points:
            start, end = line_points
            cv2.line(frame, start, end, (255, 0, 0), 2)
        return frame

    def get_video_properties(self) -> Dict[str, Any]:
        """Get video properties."""
        return {
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.video_fps,
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.cap
            else 0,
        }

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_roi_for_display(self, roi_mask: np.ndarray):
        """Set ROI mask specifically for bounding box display filtering."""
        self.roi_mask_for_display = roi_mask

    def set_bbox_filter_enabled(self, enabled: bool):
        """Enable or disable bounding box filtering based on ROI intersection."""
        self.bbox_filter_enabled = enabled

    def _bbox_intersects_roi(self, bbox: List[float]) -> bool:
        """Check if a bounding box intersects with the ROI mask."""
        if self.roi_mask_for_display is None or not self.bbox_filter_enabled:
            return True

        x1, y1, x2, y2 = map(int, bbox)

        # Clamp coordinates to valid range
        x1 = max(0, min(x1, self.frame_width - 1))
        y1 = max(0, min(y1, self.frame_height - 1))
        x2 = max(0, min(x2, self.frame_width - 1))
        y2 = max(0, min(y2, self.frame_height - 1))

        # Extract the bounding box area from the ROI mask
        roi_region = self.roi_mask_for_display[y1 : y2 + 1, x1 : x2 + 1]

        # Check if any pixel in the bounding box area is part of the ROI
        return np.any(roi_region > 0) if roi_region.size > 0 else False
