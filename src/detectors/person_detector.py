from typing import Dict, List

import numpy as np
from loguru import logger

from detectors.detector_factory import DetectorFactory


class PersonDetector:
    def __init__(self, detector_type: str = "yolo", **kwargs):
        """
        Initialize the person detector.

        Args:
            detector_type: Type of detector to use ('yolo' for YOLO-based detector)
            conf_thresh: Detection confidence threshold
            **kwargs: Additional arguments for detector initialization
        """
        self.detector = DetectorFactory.create_detector(detector_type, **kwargs)
        logger.info(f"Detector initialized: {detector_type}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in the given frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of dictionaries containing person detection information:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'keypoints': List[Tuple[float, float]] (optional),
                'pose': str (optional)
            }
        """
        return self.detector.detect(frame)
