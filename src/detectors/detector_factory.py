from detectors.base_detector import BaseDetector
from detectors.yolo_detector import YoloDetector


class DetectorFactory:
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> BaseDetector:
        """
        Create a detector instance based on the specified type.

        Args:
            detector_type: Type of detector ('yolo' or 'deepface')
            **kwargs: Additional arguments for detector initialization
                For YOLO:
                    - model_path: Path to the YOLO model
                    - conf_threshold: Confidence threshold

        Returns:
            An instance of the specified detector
        """
        if detector_type.lower() == "yolo":
            return YoloDetector(**kwargs)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
