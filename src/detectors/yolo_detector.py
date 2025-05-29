from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from detectors.base_detector import BaseDetector


class YoloDetector(BaseDetector):
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        class_id: int = 0,
        conf_threshold: float = 0.5,
        input_height: int = 640,
        input_width: int = 640,
    ):
        """
        Initialize the person detector with YOLOv8.

        Args:
            model_path: Path to the ultralytics model weights
            conf_threshold: Confidence threshold for detections
            input_height: Height of the input frame
            input_width: Width of the input frame
        """
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to("cuda")
            logger.info("Using GPU for detection")
        else:
            logger.info("Using CPU for detection")
        self.model.eval()
        self.conf_threshold = conf_threshold
        self.person_class_id = class_id
        self.input_height = input_height
        self.input_width = input_width

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in the given frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of dictionaries containing detection information:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int
            }
        """
        results = self.model(
            frame, verbose=False, imgsz=[self.input_width, self.input_height]
        )[0]
        detections = []

        for box in results.boxes:
            if box.cls == self.person_class_id and box.conf > self.conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(box.conf),
                        "class_id": int(box.cls),
                    }
                )

        return detections
