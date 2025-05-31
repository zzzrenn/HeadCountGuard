from enum import Enum
from typing import Dict, List, Tuple, Union


class CrossingCriteria(Enum):
    """Enum defining different criteria for line crossing detection."""

    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class LineCrossingCounter:
    def __init__(
        self,
        line_points: Tuple[Tuple[float, float], Tuple[float, float]],
        in_side: str,
        frame_width: int,
        frame_height: int,
        crossing_criteria: Union[CrossingCriteria, str] = CrossingCriteria.CENTER,
    ):
        """
        Initialize the line crossing counter.

        Args:
            line_points: Tuple of two points defining the line ((x1, y1), (x2, y2))
                        Coordinates are normalized between 0 and 1
            in_side: Which side is considered "in": "left" or "right"
                    For horizontal lines, "left" means bottom
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
            crossing_criteria: What part of the bbox to use for crossing detection:
                             - CENTER: center point of bbox (default)
                             - TOP: top edge of bbox
                             - BOTTOM: bottom edge of bbox
                             - LEFT: left edge of bbox
                             - RIGHT: right edge of bbox
        """
        # Sort points by y-coordinate to ensure consistent behavior
        p1, p2 = line_points
        self.line_points = (p1, p2) if p1[1] <= p2[1] else (p2, p1)

        if in_side not in ["left", "right"]:
            raise ValueError("in_side must be either 'left' or 'right'")
        self.in_side = in_side

        # Handle crossing criteria
        if isinstance(crossing_criteria, str):
            self.crossing_criteria = CrossingCriteria(crossing_criteria)
        else:
            self.crossing_criteria = crossing_criteria

        self.count = 0
        self.tracked_positions = {}  # track_id -> previous position
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate the normalized center coordinates of a bounding box."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return center_x / self.frame_width, center_y / self.frame_height

    def _get_edge_point(self, bbox: List[float], edge: str) -> Tuple[float, float]:
        """Get the normalized coordinates of a specific edge of the bbox."""
        if edge == "top":
            return (bbox[0] + bbox[2]) / 2 / self.frame_width, bbox[
                1
            ] / self.frame_height
        elif edge == "bottom":
            return (bbox[0] + bbox[2]) / 2 / self.frame_width, bbox[
                3
            ] / self.frame_height
        elif edge == "left":
            return bbox[0] / self.frame_width, (
                bbox[1] + bbox[3]
            ) / 2 / self.frame_height
        elif edge == "right":
            return bbox[2] / self.frame_width, (
                bbox[1] + bbox[3]
            ) / 2 / self.frame_height

    def _get_reference_point(self, bbox: List[float]) -> Tuple[float, float]:
        """Get the reference point based on the crossing criteria."""
        if self.crossing_criteria == CrossingCriteria.CENTER:
            return self._get_center(bbox)
        elif self.crossing_criteria == CrossingCriteria.TOP:
            return self._get_edge_point(bbox, "top")
        elif self.crossing_criteria == CrossingCriteria.BOTTOM:
            return self._get_edge_point(bbox, "bottom")
        elif self.crossing_criteria == CrossingCriteria.LEFT:
            return self._get_edge_point(bbox, "left")
        elif self.crossing_criteria == CrossingCriteria.RIGHT:
            return self._get_edge_point(bbox, "right")

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

    def update(self, tracked_objects: List[Dict]) -> int:
        """
        Update the counter based on tracked objects crossing the line.

        Args:
            tracked_objects: List of tracked objects with their positions

        Returns:
            Updated count
        """
        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_reference = self._get_reference_point(obj["bbox"])

            if track_id in self.tracked_positions:
                prev_reference = self.tracked_positions[track_id]

                # Check for line crossing
                prev_on_in_side = self._is_point_on_in_side(prev_reference)
                current_on_in_side = self._is_point_on_in_side(current_reference)

                if prev_on_in_side != current_on_in_side:
                    # Moving from in to out decreases count, from out to in increases count
                    self.count += 1 if current_on_in_side else -1

            # Update previous position
            self.tracked_positions[track_id] = current_reference

        return self.count
