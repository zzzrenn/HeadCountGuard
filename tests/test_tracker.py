import pytest

from tracker import PersonTracker


@pytest.fixture
def tracker():
    return PersonTracker(
        tracker_type="bytetrack",
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
    )


@pytest.fixture
def sample_detections():
    return [
        {"bbox": [100, 100, 200, 200], "confidence": 0.9},  # Detection 1
        {"bbox": [300, 300, 400, 400], "confidence": 0.8},  # Detection 2
    ]


@pytest.fixture
def moving_detections():
    # Simulate a person moving across the frame
    return [
        {"bbox": [100, 100, 200, 200], "confidence": 0.9},  # Frame 1
        {"bbox": [150, 100, 250, 200], "confidence": 0.9},  # Frame 2
        {"bbox": [200, 100, 300, 200], "confidence": 0.9},  # Frame 3
    ]


@pytest.fixture
def low_confidence_detections():
    return [
        {"bbox": [100, 100, 200, 200], "confidence": 0.3},  # Low confidence
        {"bbox": [300, 300, 400, 400], "confidence": 0.2},  # Low confidence
    ]


@pytest.fixture
def occluded_detections():
    # Simulate a person being occluded
    return [
        {"bbox": [100, 100, 200, 200], "confidence": 0.9},  # Frame 1: visible
        {
            "bbox": [150, 100, 250, 200],
            "confidence": 0.4,
        },  # Frame 2: partially occluded
        {"bbox": [200, 100, 300, 200], "confidence": 0.9},  # Frame 3: visible again
    ]


class TestPersonTracker:
    def test_initialization(self, tracker):
        assert tracker.tracker.track_thresh == 0.5
        assert tracker.tracker.track_buffer == 30
        assert tracker.tracker.frame_rate == 30
        assert tracker.tracker.match_thresh == 0.8

    def test_update_with_empty_detections(self, tracker):
        tracked_objects = tracker.update([], [480, 640], (480, 640))
        assert isinstance(tracked_objects, list)
        assert len(tracked_objects) == 0

    def test_update_with_detections(self, tracker, sample_detections):
        tracked_objects = tracker.update(sample_detections, [480, 640], (480, 640))
        assert isinstance(tracked_objects, list)
        assert len(tracked_objects) > 0

        # Check if tracked objects have required attributes
        for obj in tracked_objects:
            assert "track_id" in obj
            assert "bbox" in obj
            assert "score" in obj
            assert len(obj["bbox"]) == 4

    def test_track_consistency(self, tracker, sample_detections):
        # First update
        tracked_objects1 = tracker.update(sample_detections, [480, 640], (480, 640))

        # Second update with same detections
        tracked_objects2 = tracker.update(sample_detections, [480, 640], (480, 640))

        # Track IDs should remain consistent
        track_ids1 = {obj["track_id"] for obj in tracked_objects1}
        track_ids2 = {obj["track_id"] for obj in tracked_objects2}
        assert track_ids1 == track_ids2

    def test_track_moving_object(self, tracker, moving_detections):
        # Track a moving object across multiple frames
        tracked_objects = []
        for det in moving_detections:
            result = tracker.update([det], [480, 640], (480, 640))
            tracked_objects.append(result)

        # Check that we have tracking results for each frame
        assert len(tracked_objects) == len(moving_detections)

        # Check that track IDs remain consistent
        track_ids = {obj[0]["track_id"] for obj in tracked_objects if obj}
        assert len(track_ids) == 1  # Should be tracking the same object

    def test_track_multiple_objects(self, tracker):
        # Test tracking multiple objects with different confidences
        detections1 = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"bbox": [300, 300, 400, 400], "confidence": 0.8},
        ]
        detections2 = [
            {"bbox": [110, 110, 210, 210], "confidence": 0.9},  # Moved slightly
            {"bbox": [300, 300, 400, 400], "confidence": 0.8},  # Same position
        ]

        # First frame
        tracked1 = tracker.update(detections1, [480, 640], (480, 640))
        assert len(tracked1) == 2

        # Second frame
        tracked2 = tracker.update(detections2, [480, 640], (480, 640))
        assert len(tracked2) == 2

        # Check track ID consistency
        track_ids1 = {obj["track_id"] for obj in tracked1}
        track_ids2 = {obj["track_id"] for obj in tracked2}
        assert track_ids1 == track_ids2
