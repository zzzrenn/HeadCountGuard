import numpy as np
import pytest

from detectors import PersonDetector
from detectors.yolo_detector import YoloDetector


@pytest.fixture
def sample_frame():
    # Create a sample frame (black image)
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def yolo_detector():
    return YoloDetector(
        model_path="models/yolov8s.pt",
        class_id=0,
        conf_threshold=0.25,
        input_height=640,
        input_width=640,
    )


class TestYoloDetector:
    def test_initialization(self, yolo_detector):
        assert yolo_detector.conf_threshold == 0.25
        assert yolo_detector.model is not None
        assert yolo_detector.person_class_id == 0
        assert yolo_detector.input_height == 640
        assert yolo_detector.input_width == 640

    def test_detect_empty_frame(self, yolo_detector, sample_frame):
        detections = yolo_detector.detect(sample_frame)
        assert isinstance(detections, list)
        # Empty frame should have no detections
        assert len(detections) == 0

    def test_detect_with_different_input_size(self):
        # Test with different input sizes
        detector = YoloDetector(
            model_path="models/yolov8n.pt",
            conf_threshold=0.5,
            input_height=320,
            input_width=320,
        )
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)


class TestPersonDetector:
    def test_factory_creation(self):
        detector = PersonDetector(
            detector_type="yolo",
            model_path="models/yolov8n.pt",
            conf_threshold=0.5,
            input_height=640,
            input_width=640,
            class_id=0,
        )
        assert isinstance(detector.detector, YoloDetector)

    def test_invalid_detector_type(self):
        with pytest.raises(ValueError):
            PersonDetector(detector_type="invalid_type")
