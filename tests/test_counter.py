import pytest

from counter import LineCrossingCounter


@pytest.fixture
def frame_width():
    return 1000  # Example frame width


@pytest.fixture
def frame_height():
    return 1000  # Example frame height


@pytest.fixture
def counter_middle_vertical_right(frame_width, frame_height):
    # Create a counter with line at 50% of frame width
    return LineCrossingCounter(
        line_points=((0.5, 0), (0.5, 1)),
        in_side="right",
        frame_width=frame_width,
        frame_height=frame_height,
    )


@pytest.fixture
def counter_middle_horizontal_right(frame_width, frame_height):
    return LineCrossingCounter(
        line_points=((0, 0.5), (1, 0.5)),
        in_side="right",
        frame_width=frame_width,
        frame_height=frame_height,
    )


def test_vertical_right_crossing(counter_middle_vertical_right):
    # Simulate an object moving from left to right
    tracked_objects = [
        {"track_id": 1, "bbox": [200, 0, 300, 100]},  # Left of line (center at 0.25)
        {"track_id": 1, "bbox": [600, 0, 700, 100]},  # Right of line (center at 0.65)
    ]

    # First update should not count
    count = counter_middle_vertical_right.update(tracked_objects[:1])
    assert count == 0

    # Second update should count +1 for right crossing
    count = counter_middle_vertical_right.update(tracked_objects[1:])
    assert count == 1


def test_vertical_left_crossing(counter_middle_vertical_right):
    # Simulate an object moving from right to left
    tracked_objects = [
        {"track_id": 1, "bbox": [700, 0, 800, 100]},  # Right of line (center at 0.75)
        {"track_id": 1, "bbox": [300, 0, 400, 100]},  # Left of line (center at 0.35)
    ]

    # First update should not count
    count = counter_middle_vertical_right.update(tracked_objects[:1])
    assert count == 0

    # Second update should count -1 for left crossing
    count = counter_middle_vertical_right.update(tracked_objects[1:])
    assert count == -1


def test_vertical_crossing(counter_middle_vertical_right):
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
    count = counter_middle_vertical_right.update(tracked_objects[:1])
    assert count == 0

    # Second update - right crossing (+1)
    count = counter_middle_vertical_right.update(tracked_objects[1:2])
    assert count == 1

    # Third update - no crossing
    count = counter_middle_vertical_right.update(tracked_objects[2:3])
    assert count == 1

    # Fourth update - left crossing (-1)
    count = counter_middle_vertical_right.update(tracked_objects[3:])
    assert count == 0


def test_horizontal_top_crossing(counter_middle_horizontal_right):
    # Simulate an object moving from bottom to top
    tracked_objects = [
        {"track_id": 1, "bbox": [0, 700, 100, 800]},  # Below line (center at 0.75)
        {"track_id": 1, "bbox": [0, 300, 100, 400]},  # Above line (center at 0.35)
    ]

    # First update should not count
    count = counter_middle_horizontal_right.update(tracked_objects[:1])
    assert count == 0

    # Second update should count +1 for right crossing (bottom to top)
    count = counter_middle_horizontal_right.update(tracked_objects[1:])
    assert count == 1


def test_horizontal_bottom_crossing(counter_middle_horizontal_right):
    # Simulate an object moving from top to bottom
    tracked_objects = [
        {"track_id": 1, "bbox": [0, 300, 100, 400]},  # Above line (center at 0.35)
        {"track_id": 1, "bbox": [0, 700, 100, 800]},  # Below line (center at 0.75)
    ]

    # First update should not count because it's on the same side
    count = counter_middle_horizontal_right.update(tracked_objects[:1])
    assert count == 0

    # Second update should count -1 for left crossing (top to bottom)
    count = counter_middle_horizontal_right.update(tracked_objects[1:])
    assert count == -1


def test_horizontal_crossing(counter_middle_horizontal_right):
    # Test multiple objects crossing in different directions
    tracked_objects = [
        # First object moving up
        {"track_id": 1, "bbox": [0, 700, 100, 800]},  # Below line (center at 0.75)
        {"track_id": 1, "bbox": [0, 300, 100, 400]},  # Above line (center at 0.35)
        # Second object moving down
        {"track_id": 2, "bbox": [0, 300, 100, 400]},  # Above line (center at 0.35)
        {"track_id": 2, "bbox": [0, 700, 100, 800]},  # Below line (center at 0.75)
    ]

    # First update - no crossings
    count = counter_middle_horizontal_right.update(tracked_objects[:1])
    assert count == 0

    # Second update - up crossing (+1)
    count = counter_middle_horizontal_right.update(tracked_objects[1:2])
    assert count == 1

    # Third update - no crossing
    count = counter_middle_horizontal_right.update(tracked_objects[2:3])
    assert count == 1

    # Fourth update - down crossing (-1)
    count = counter_middle_horizontal_right.update(tracked_objects[3:])
    assert count == 0
