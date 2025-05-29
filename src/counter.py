from typing import Dict, List


class LineCrossingCounter:
    def __init__(self, line_position: float):
        """
        Initialize the line crossing counter.

        Args:
            line_position: x-coordinate of the vertical line (normalized between 0 and 1)
        """
        self.line_position = line_position
        self.count = 0
        self.tracked_positions = {}  # track_id -> previous position

    def _get_center_x(self, bbox: List[float], frame_width: int) -> float:
        """Calculate the normalized center x-coordinate of a bounding box."""
        center_x = (bbox[0] + bbox[2]) / 2
        return center_x / frame_width

    def update(self, tracked_objects: List[Dict], frame_width: int) -> int:
        """
        Update the counter based on tracked objects crossing the line.

        Args:
            tracked_objects: List of tracked objects with their positions
            frame_width: Width of the frame in pixels

        Returns:
            Updated count
        """
        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_center_x = self._get_center_x(obj["bbox"], frame_width)

            if track_id in self.tracked_positions:
                prev_center_x = self.tracked_positions[track_id]

                # Check for line crossing
                if (
                    prev_center_x < self.line_position
                    and current_center_x >= self.line_position
                ):
                    self.count += 1  # Moving right
                elif (
                    prev_center_x > self.line_position
                    and current_center_x <= self.line_position
                ):
                    self.count -= 1  # Moving left

            # Update previous position
            self.tracked_positions[track_id] = current_center_x

        return self.count
