#!/usr/bin/env python3
"""
ROI Manager for People Counter Application

Manages Region of Interest (ROI) drawing operations for enclosed polygon regions.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.utils.coordinate_utils import CoordinateTransformer


class ROIManager:
    """Manages drawing and operations for Region of Interest (ROI) polygons."""

    def __init__(self, coordinate_transformer: CoordinateTransformer):
        self.coord_transformer = coordinate_transformer
        self.roi_points: List[Tuple[int, int]] = []  # Points in frame coordinates
        self.is_drawing = False
        self.is_complete = False
        self.cached_drawing_info: Optional[Dict[str, Any]] = None
        self.last_canvas_size = (0, 0)

    def start_drawing(self):
        """Start drawing a new ROI polygon."""
        self.is_drawing = True
        self.is_complete = False
        self.roi_points.clear()
        self.cached_drawing_info = None

    def add_point(
        self, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int
    ):
        """Add a point to the ROI polygon."""
        if not self.is_drawing:
            return False

        # Convert canvas coordinates to frame coordinates
        frame_x, frame_y = self.coord_transformer.canvas_to_frame(
            canvas_x, canvas_y, canvas_width, canvas_height
        )

        # Check if this point is close to the first point (to close the polygon)
        if len(self.roi_points) >= 3:
            first_point_canvas = self.coord_transformer.frame_to_canvas(
                self.roi_points[0][0],
                self.roi_points[0][1],
                canvas_width,
                canvas_height,
            )
            distance = (
                (canvas_x - first_point_canvas[0]) ** 2
                + (canvas_y - first_point_canvas[1]) ** 2
            ) ** 0.5

            # If close to first point (within 15 pixels), close the polygon
            if distance < 15:
                self.finish_drawing()
                return True

        self.roi_points.append((frame_x, frame_y))
        self.cached_drawing_info = None  # Invalidate cache
        return False

    def finish_drawing(self):
        """Finish drawing the ROI polygon."""
        if len(self.roi_points) >= 3:
            self.is_drawing = False
            self.is_complete = True
            self.cached_drawing_info = None  # Invalidate cache

    def cancel_drawing(self):
        """Cancel current ROI drawing."""
        self.is_drawing = False
        self.is_complete = False
        self.roi_points.clear()
        self.cached_drawing_info = None

    def remove_last_point(self):
        """Remove the last added point."""
        if self.roi_points and self.is_drawing:
            self.roi_points.pop()
            self.cached_drawing_info = None

    def has_roi(self) -> bool:
        """Check if a complete ROI has been drawn."""
        return len(self.roi_points) >= 3  # Any polygon with 3+ points is valid

    def get_normalized_roi(self) -> Optional[List[Tuple[float, float]]]:
        """Get normalized ROI coordinates (0-1 range)."""
        if not self.has_roi():
            return None

        normalized_points = []
        for point in self.roi_points:
            norm_point = self.coord_transformer.normalize_coordinates(
                point[0], point[1]
            )
            normalized_points.append(norm_point)

        return normalized_points

    def get_roi_points(self) -> List[Tuple[int, int]]:
        """Get ROI points in frame coordinates."""
        return self.roi_points.copy()

    def create_roi_mask(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Create a binary mask for the ROI region."""
        if not self.has_roi():
            return np.zeros((frame_height, frame_width), dtype=np.uint8)

        # Create mask
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # Convert points to numpy array
        points = np.array(self.roi_points, dtype=np.int32)

        # Fill the polygon
        cv2.fillPoly(mask, [points], 255)

        return mask

    def point_in_roi(self, x: int, y: int, frame_width: int, frame_height: int) -> bool:
        """Check if a point is inside the ROI."""
        if not self.has_roi():
            return True  # If no ROI, consider all points valid

        # Create a small mask for the point
        mask = self.create_roi_mask(frame_width, frame_height)

        # Check bounds
        if 0 <= y < frame_height and 0 <= x < frame_width:
            return mask[y, x] > 0

        return False

    def compute_drawing_info(
        self, canvas_width: int, canvas_height: int
    ) -> Optional[Dict[str, Any]]:
        """Compute and cache drawing information for the ROI."""
        if not self.roi_points:
            return {"has_roi": False}

        preview_width, preview_height, x_offset, y_offset = (
            self.coord_transformer.get_preview_dimensions(canvas_width, canvas_height)
        )

        # Transform ROI points to canvas coordinates
        canvas_points = []
        for frame_x, frame_y in self.roi_points:
            canvas_x, canvas_y = self.coord_transformer.frame_to_canvas(
                frame_x, frame_y, canvas_width, canvas_height
            )
            canvas_points.append((canvas_x, canvas_y))

        return {
            "preview_width": preview_width,
            "preview_height": preview_height,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "canvas_points": canvas_points,
            "is_drawing": self.is_drawing,
            "is_complete": self.is_complete,
            "has_roi": True,
        }

    def get_drawing_info(
        self, canvas_width: int, canvas_height: int
    ) -> Optional[Dict[str, Any]]:
        """Get cached drawing info or compute if necessary."""
        current_canvas_size = (canvas_width, canvas_height)

        # Recompute if canvas size changed, ROI changed, or no cache exists
        if (
            self.cached_drawing_info is None
            or current_canvas_size != self.last_canvas_size
        ):
            self.cached_drawing_info = self.compute_drawing_info(
                canvas_width, canvas_height
            )
            self.last_canvas_size = current_canvas_size

        return self.cached_drawing_info

    def reset(self):
        """Reset all ROI data."""
        self.roi_points.clear()
        self.is_drawing = False
        self.is_complete = False
        self.cached_drawing_info = None
        self.last_canvas_size = (0, 0)
