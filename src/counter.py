from typing import Dict, List, Tuple


class LineCrossingCounter:
    def __init__(
        self, line_points: Tuple[Tuple[float, float], Tuple[float, float]], in_side: str
    ):
        """
        Initialize the line crossing counter.

        Args:
            line_points: Tuple of two points defining the line ((x1, y1), (x2, y2))
                        Coordinates are normalized between 0 and 1
            in_side: Which side is considered "in": "left" or "right"
                    For horizontal lines, "left" means bottom
        """
        # Sort points by y-coordinate to ensure consistent behavior
        p1, p2 = line_points
        self.line_points = (p1, p2) if p1[1] <= p2[1] else (p2, p1)

        if in_side not in ["left", "right"]:
            raise ValueError("in_side must be either 'left' or 'right'")
        self.in_side = in_side
        self.count = 0
        self.tracked_positions = {}  # track_id -> previous position

    def _get_center(
        self, bbox: List[float], frame_width: int, frame_height: int
    ) -> Tuple[float, float]:
        """Calculate the normalized center coordinates of a bounding box."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return center_x / frame_width, center_y / frame_height

    def _is_point_on_in_side(self, point: Tuple[float, float]) -> bool:
        """Check if a point is on the 'in' side of the line."""
        x, y = point
        (x1, y1), (x2, y2) = self.line_points

        # Calculate line vector
        dx = x2 - x1
        dy = y2 - y1

        # Calculate cross product to determine which side of the line the point is on
        cross_product = dx * (y - y1) - dy * (x - x1)

        # For non-horizontal lines
        return (cross_product > 0) == (self.in_side == "left")

    def update(
        self, tracked_objects: List[Dict], frame_width: int, frame_height: int
    ) -> int:
        """
        Update the counter based on tracked objects crossing the line.

        Args:
            tracked_objects: List of tracked objects with their positions
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels

        Returns:
            Updated count
        """
        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_center = self._get_center(obj["bbox"], frame_width, frame_height)

            if track_id in self.tracked_positions:
                prev_center = self.tracked_positions[track_id]
                prev_on_in_side = self._is_point_on_in_side(prev_center)
                current_on_in_side = self._is_point_on_in_side(current_center)

                # Check for line crossing
                if prev_on_in_side != current_on_in_side:
                    # Moving from in to out decreases count, from out to in increases count
                    self.count += 1 if current_on_in_side else -1

            # Update previous position
            self.tracked_positions[track_id] = current_center

        return self.count
