import pytest

from counter import CrossingCriteria, LineCrossingCounter


# Common fixtures
@pytest.fixture
def frame_dimensions():
    """Standard frame dimensions for testing."""
    return {"width": 1000, "height": 1000}


class TestSideCrossing:
    """Tests for side crossing detection with different center criterium."""

    @pytest.fixture
    def diagonal_line_params(self, frame_dimensions):
        """Common parameters for diagonal line counters."""
        return {
            "line_points": ((0.2, 0.2), (0.8, 0.8)),
            "frame_width": frame_dimensions["width"],
            "frame_height": frame_dimensions["height"],
        }

    @pytest.fixture
    def horizontal_movement_objects(self):
        """Shared horizontal movement path for diagonal line crossing tests.

        Sequence: IN -> IN/OUT -> IN/OUT -> OUT -> IN/OUT -> IN/OUT -> OUT
        Horizontal movement at y=0.5, crossing diagonal line at x=0.5
        Bbox: [x, y, x+w, y+h] -> center at [x+w/2, y+h/2]
        """
        return [
            # IN: bbox center at (0.35, 0.5) - clearly on left side
            {"track_id": 1, "bbox": [300, 450, 400, 550]},
            # IN/OUT: bbox center at (0.475, 0.5) - center slightly left, but bbox straddles line
            {"track_id": 1, "bbox": [425, 450, 525, 550]},
            # IN/OUT: bbox center at (0.505, 0.5) - center slightly right, but bbox straddles line
            {"track_id": 1, "bbox": [455, 450, 555, 550]},
            # OUT: bbox center at (0.75, 0.5) - clearly on right side
            {"track_id": 1, "bbox": [700, 450, 800, 550]},
            # IN/OUT: bbox center at (0.52, 0.5) - center right, bbox straddles line
            {"track_id": 1, "bbox": [470, 450, 570, 550]},
            # IN/OUT: bbox center at (0.48, 0.5) - center left, bbox straddles line
            {"track_id": 1, "bbox": [430, 450, 530, 550]},
            # IN: bbox center at (0.35, 0.5) - clearly on left side
            {"track_id": 1, "bbox": [300, 450, 400, 550]},
        ]

    def test_left_in_center(self, diagonal_line_params, horizontal_movement_objects):
        """Test left in center crossing."""
        counts = []
        counter = LineCrossingCounter(
            **diagonal_line_params,
            in_side="left",
            crossing_criteria=CrossingCriteria.CENTER,
        )
        for obj in horizontal_movement_objects:
            count = counter.update([obj])
            counts.append(count)

        # Verify crossing pattern: should cross from left to right once, then back
        expected_counts = [0, 0, -1, -1, -1, 0, 0]
        assert counts == expected_counts

    def test_right_in_center(self, diagonal_line_params, horizontal_movement_objects):
        """Test right in center crossing."""
        counts = []
        counter = LineCrossingCounter(
            **diagonal_line_params,
            in_side="right",
            crossing_criteria=CrossingCriteria.CENTER,
        )
        for obj in horizontal_movement_objects:
            count = counter.update([obj])
            counts.append(count)

        # Verify crossing pattern: should cross from left to right once, then back
        expected_counts = [0, 0, 1, 1, 1, 0, 0]
        assert counts == expected_counts


class TestDiagonalLineCrossing:
    """Tests for diagonal line crossing detection with different crossing criteria."""

    @pytest.fixture
    def diagonal_line_params(self, frame_dimensions):
        """Common parameters for diagonal line counters."""
        return {
            "line_points": ((0.2, 0.2), (0.8, 0.8)),
            "in_side": "left",
            "frame_width": frame_dimensions["width"],
            "frame_height": frame_dimensions["height"],
        }

    @pytest.fixture
    def horizontal_movement_objects(self):
        """Shared horizontal movement path for diagonal line crossing tests.

        Sequence: IN -> IN/OUT -> IN/OUT -> OUT -> IN/OUT -> IN/OUT -> OUT
        Horizontal movement at y=0.5, crossing diagonal line at x=0.5
        Bbox: [x, y, x+w, y+h] -> center at [x+w/2, y+h/2]
        """
        return [
            # IN: bbox center at (0.35, 0.5) - clearly on left side
            {"track_id": 1, "bbox": [300, 450, 400, 550]},
            # IN/OUT: bbox center at (0.475, 0.5) - center slightly left, but bbox straddles line
            {"track_id": 1, "bbox": [425, 450, 525, 550]},
            # IN/OUT: bbox center at (0.505, 0.5) - center slightly right, but bbox straddles line
            {"track_id": 1, "bbox": [455, 450, 555, 550]},
            # OUT: bbox center at (0.75, 0.5) - clearly on right side
            {"track_id": 1, "bbox": [700, 450, 800, 550]},
            # IN/OUT: bbox center at (0.52, 0.5) - center right, bbox straddles line
            {"track_id": 1, "bbox": [470, 450, 570, 550]},
            # IN/OUT: bbox center at (0.48, 0.5) - center left, bbox straddles line
            {"track_id": 1, "bbox": [430, 450, 530, 550]},
            # IN: bbox center at (0.35, 0.5) - clearly on left side
            {"track_id": 1, "bbox": [300, 450, 400, 550]},
        ]

    @pytest.fixture
    def counter_diagonal_center(self, diagonal_line_params):
        """Diagonal line counter using center point criteria."""
        return LineCrossingCounter(
            **diagonal_line_params,
            crossing_criteria=CrossingCriteria.CENTER,
        )

    @pytest.fixture
    def counter_diagonal_top(self, diagonal_line_params):
        """Diagonal line counter using top edge criteria."""
        return LineCrossingCounter(
            **diagonal_line_params,
            crossing_criteria=CrossingCriteria.TOP,
        )

    @pytest.fixture
    def counter_diagonal_bottom(self, diagonal_line_params):
        """Diagonal line counter using bottom edge criteria."""
        return LineCrossingCounter(
            **diagonal_line_params,
            crossing_criteria=CrossingCriteria.BOTTOM,
        )

    @pytest.fixture
    def counter_diagonal_left(self, diagonal_line_params):
        """Diagonal line counter using left edge criteria."""
        return LineCrossingCounter(
            **diagonal_line_params,
            crossing_criteria=CrossingCriteria.LEFT,
        )

    @pytest.fixture
    def counter_diagonal_right(self, diagonal_line_params):
        """Diagonal line counter using right edge criteria."""
        return LineCrossingCounter(
            **diagonal_line_params,
            crossing_criteria=CrossingCriteria.RIGHT,
        )

    def test_center_point_crossing(
        self, counter_diagonal_center, horizontal_movement_objects
    ):
        """Test center point crossing diagonal line."""
        counts = []
        for obj in horizontal_movement_objects:
            count = counter_diagonal_center.update([obj])
            counts.append(count)

        # Verify crossing pattern: should cross from left to right once, then back
        expected_counts = [0, 0, -1, -1, -1, 0, 0]
        assert counts == expected_counts

    def test_top_edge_crossing(self, counter_diagonal_top, horizontal_movement_objects):
        """Test top edge crossing diagonal line."""
        counts = []
        for obj in horizontal_movement_objects:
            count = counter_diagonal_top.update([obj])
            counts.append(count)

        # Verify crossing pattern based on top edge
        expected_counts = [0, -1, -1, -1, -1, -1, 0]
        assert counts == expected_counts

    def test_bottom_edge_crossing(
        self, counter_diagonal_bottom, horizontal_movement_objects
    ):
        """Test bottom edge crossing diagonal line."""
        counts = []
        for obj in horizontal_movement_objects:
            count = counter_diagonal_bottom.update([obj])
            counts.append(count)

        # Verify crossing pattern based on bottom edge
        expected_counts = [0, 0, 0, -1, 0, 0, 0]
        assert counts == expected_counts

    def test_left_edge_crossing(
        self, counter_diagonal_left, horizontal_movement_objects
    ):
        """Test left edge crossing diagonal line."""
        counts = []
        for obj in horizontal_movement_objects:
            count = counter_diagonal_left.update([obj])
            counts.append(count)

        # Verify crossing pattern based on left edge
        expected_counts = [0, 0, 0, -1, 0, 0, 0]
        assert counts == expected_counts

    def test_right_edge_crossing(
        self, counter_diagonal_right, horizontal_movement_objects
    ):
        """Test right edge crossing diagonal line."""
        counts = []
        for obj in horizontal_movement_objects:
            count = counter_diagonal_right.update([obj])
            counts.append(count)

        # Verify crossing pattern based on right edge
        expected_counts = [0, -1, -1, -1, -1, -1, 0]
        assert counts == expected_counts
