import pytest

from counter import LineCrossingCounter


@pytest.fixture
def counter():
    # Create a counter with line at 50% of frame width
    return LineCrossingCounter(line_position=0.5)


@pytest.fixture
def frame_width():
    return 1000  # Example frame width


def test_right_crossing(counter, frame_width):
    # Simulate an object moving from left to right
    tracked_objects = [
        {"track_id": 1, "bbox": [200, 0, 300, 100]},  # Left of line (center at 0.25)
        {"track_id": 1, "bbox": [600, 0, 700, 100]},  # Right of line (center at 0.65)
    ]

    # First update should not count
    count = counter.update(tracked_objects[:1], frame_width)
    assert count == 0

    # Second update should count +1 for right crossing
    count = counter.update(tracked_objects[1:], frame_width)
    assert count == 1


def test_left_crossing(counter, frame_width):
    # Simulate an object moving from right to left
    tracked_objects = [
        {"track_id": 1, "bbox": [700, 0, 800, 100]},  # Right of line (center at 0.75)
        {"track_id": 1, "bbox": [300, 0, 400, 100]},  # Left of line (center at 0.35)
    ]

    # First update should not count
    count = counter.update(tracked_objects[:1], frame_width)
    assert count == 0

    # Second update should count -1 for left crossing
    count = counter.update(tracked_objects[1:], frame_width)
    assert count == -1


def test_multiple_objects(counter, frame_width):
    # Test multiple objects crossing in different directions
    tracked_objects = [
        # First object moving right
        {"track_id": 1, "bbox": [200, 0, 300, 100]},  # Left of line (center at 0.25)
        {"track_id": 1, "bbox": [600, 0, 700, 100]},  # Right of line (center at 0.65)
        # Second object moving left
        {"track_id": 2, "bbox": [700, 0, 800, 100]},  # Right of line (center at 0.75)
        {"track_id": 2, "bbox": [300, 0, 400, 100]},  # Left of line (center at 0.35)
    ]

    # First update - no crossings
    count = counter.update(tracked_objects[:1], frame_width)
    assert count == 0

    # Second update - right crossing (+1)
    count = counter.update(tracked_objects[1:2], frame_width)
    assert count == 1

    # Third update - no crossing
    count = counter.update(tracked_objects[2:3], frame_width)
    assert count == 1

    # Fourth update - left crossing (-1)
    count = counter.update(tracked_objects[3:], frame_width)
    assert count == 0
