from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.utils.coordinate_utils import CoordinateTransformer


class LineManager:
    """Manages drawing line operations and side selection (separating IN and OUT)."""

    def __init__(self, coordinate_transformer: CoordinateTransformer):
        self.coord_transformer = coordinate_transformer
        self.line_start: Optional[Tuple[int, int]] = None
        self.line_end: Optional[Tuple[int, int]] = None
        self.selected_side: Optional[str] = None
        self.drawing_line = False
        self.cached_drawing_info: Optional[Dict[str, Any]] = None
        self.last_canvas_size = (0, 0)
        self._finalized_for_processing = False

    @property
    def is_drawing(self) -> bool:
        """Property to check if currently drawing."""
        return self.drawing_line

    def start_drawing(
        self,
        canvas_x: int = None,
        canvas_y: int = None,
        canvas_width: int = None,
        canvas_height: int = None,
    ):
        """Start drawing a new line with optional initial coordinates."""
        if (
            canvas_x is not None
            and canvas_y is not None
            and canvas_width is not None
            and canvas_height is not None
        ):
            # Start with specific coordinates
            self.drawing_line = True
            self.line_start = None
            self.line_end = None
            self.cached_drawing_info = None
            self.set_line_start(canvas_x, canvas_y, canvas_width, canvas_height)
        else:
            # Original start_drawing method
            self.drawing_line = True
            self.line_start = None
            self.line_end = None
            self.cached_drawing_info = None

    def start_line(
        self, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int
    ):
        """Start drawing a line."""
        self.set_line_start(canvas_x, canvas_y, canvas_width, canvas_height)

    def update_line(
        self, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int
    ):
        """Update the line end point."""
        self.update_line_end(canvas_x, canvas_y, canvas_width, canvas_height)

    def finish_line(
        self, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int
    ):
        """Finish drawing the line."""
        self.update_line_end(canvas_x, canvas_y, canvas_width, canvas_height)
        self.finish_drawing()

    def set_line_start(
        self, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int
    ):
        """Set the starting point of the line."""
        frame_x, frame_y = self.coord_transformer.canvas_to_frame(
            canvas_x, canvas_y, canvas_width, canvas_height
        )
        self.line_start = (frame_x, frame_y)
        self.line_end = (frame_x, frame_y)

    def update_line_end(
        self, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int
    ):
        """Update the end point of the line."""
        if self.line_start:
            frame_x, frame_y = self.coord_transformer.canvas_to_frame(
                canvas_x, canvas_y, canvas_width, canvas_height
            )
            self.line_end = (frame_x, frame_y)
            self.cached_drawing_info = None  # Invalidate cache

    def finish_drawing(self):
        """Finish drawing the line."""
        self.drawing_line = False
        self.cached_drawing_info = (
            None  # Invalidate cache to recalculate with label positions
        )

    def select_side(self, side: str):
        """Select which side is the IN side."""
        self.selected_side = side

    def set_side_selection(self, side: str):
        """Set which side is the IN side (alias for select_side)."""
        self.select_side(side)

    def clear_side_selection(self):
        """Clear the side selection."""
        self.selected_side = None

    def get_side_selection(self) -> Optional[str]:
        """Get the currently selected side."""
        return self.selected_side

    def has_line(self) -> bool:
        """Check if a line has been drawn."""
        return self.line_start is not None and self.line_end is not None

    def get_normalized_line(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get normalized line coordinates (0-1 range)."""
        if not self.has_line():
            return None

        start_norm = self.coord_transformer.normalize_coordinates(*self.line_start)
        end_norm = self.coord_transformer.normalize_coordinates(*self.line_end)
        return start_norm, end_norm

    def get_counting_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get line coordinates for visualization."""
        if not self.has_line():
            return None
        return (
            (int(self.line_start[0]), int(self.line_start[1])),
            (int(self.line_end[0]), int(self.line_end[1])),
        )

    def compute_drawing_info(
        self, canvas_width: int, canvas_height: int
    ) -> Optional[Dict[str, Any]]:
        """Compute and cache drawing information."""
        if not self.has_line():
            return None

        preview_width, preview_height, x_offset, y_offset = (
            self.coord_transformer.get_preview_dimensions(canvas_width, canvas_height)
        )

        # Transform line points to preview coordinates
        start_x, start_y = self.coord_transformer.frame_to_canvas(
            self.line_start[0], self.line_start[1], canvas_width, canvas_height
        )
        end_x, end_y = self.coord_transformer.frame_to_canvas(
            self.line_end[0], self.line_end[1], canvas_width, canvas_height
        )

        # Calculate direction for extension
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx**2 + dy**2) ** 0.5

        if length > 0:
            dx /= length
            dy /= length

            # Calculate extended line points
            extension_length = max(canvas_width, canvas_height)
            ext_start_x = int(start_x - dx * extension_length)
            ext_start_y = int(start_y - dy * extension_length)
            ext_end_x = int(end_x + dx * extension_length)
            ext_end_y = int(end_y + dy * extension_length)

            # Only calculate expensive label positions if line drawing is finished
            label_positions = None
            if not self.drawing_line:  # Only when not actively drawing
                label_positions = self._calculate_cached_label_positions(
                    canvas_width, canvas_height
                )

            return {
                "preview_width": preview_width,
                "preview_height": preview_height,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "line_coords": (ext_start_x, ext_start_y, ext_end_x, ext_end_y),
                "label_positions": label_positions,
                "has_line": True,
            }

        return {"has_line": False}

    def _calculate_cached_label_positions(
        self, canvas_width: int, canvas_height: int
    ) -> Dict[str, tuple]:
        """Calculate label positions based on mask centroids (cached)."""

        # Get frame dimensions from coordinate transformer
        frame_width = self.coord_transformer.frame_width
        frame_height = self.coord_transformer.frame_height

        if frame_width == 0 or frame_height == 0:
            return {"left_pos": (0, 0), "right_pos": (0, 0)}

        # Create masks for both sides
        left_mask = self.create_side_mask("left", frame_width, frame_height)
        right_mask = self.create_side_mask("right", frame_width, frame_height)

        # Calculate centroids
        left_centroid = self._calculate_mask_centroid(left_mask)
        right_centroid = self._calculate_mask_centroid(right_mask)

        # Convert to canvas coordinates
        if left_centroid and right_centroid:
            left_canvas_x, left_canvas_y = self.coord_transformer.frame_to_canvas(
                left_centroid[0], left_centroid[1], canvas_width, canvas_height
            )
            right_canvas_x, right_canvas_y = self.coord_transformer.frame_to_canvas(
                right_centroid[0], right_centroid[1], canvas_width, canvas_height
            )

            return {
                "left_pos": (left_canvas_x, left_canvas_y),
                "right_pos": (right_canvas_x, right_canvas_y),
            }

        return {"left_pos": (0, 0), "right_pos": (0, 0)}

    def _calculate_mask_centroid(self, mask):
        """Calculate the centroid (center of mass) of a mask."""
        import numpy as np

        # Find all non-zero pixels in the mask
        y_coords, x_coords = np.where(mask > 0)

        if len(x_coords) == 0 or len(y_coords) == 0:
            return None

        # Calculate centroid
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))

        return (centroid_x, centroid_y)

    def finalize_for_processing(self, canvas_width: int, canvas_height: int):
        """Finalize all calculations for video processing - call this once before starting playback."""
        if self.has_line() and self.selected_side:
            # Force calculation and permanent caching of all drawing info
            self.cached_drawing_info = self.compute_drawing_info(
                canvas_width, canvas_height
            )
            # Mark as finalized so get_drawing_info won't recalculate
            self._finalized_for_processing = True

    def get_drawing_info(
        self, canvas_width: int, canvas_height: int
    ) -> Optional[Dict[str, Any]]:
        """Get cached drawing info or compute if necessary."""
        # If finalized for processing, always return cached info without recalculation
        if (
            hasattr(self, "_finalized_for_processing")
            and self._finalized_for_processing
        ):
            return self.cached_drawing_info

        current_canvas_size = (canvas_width, canvas_height)

        # Recompute if canvas size changed, line changed, or no cache exists
        if (
            self.cached_drawing_info is None
            or current_canvas_size != self.last_canvas_size
        ):
            self.cached_drawing_info = self.compute_drawing_info(
                canvas_width, canvas_height
            )
            self.last_canvas_size = current_canvas_size

        return self.cached_drawing_info

    def create_side_mask(
        self, side: str, frame_width: int, frame_height: int
    ) -> np.ndarray:
        """Create a mask for side highlighting."""
        if not self.has_line():
            return np.zeros((frame_height, frame_width), dtype=np.uint8)

        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # Get line points
        if self.line_start[1] > self.line_end[1]:
            self.line_start, self.line_end = self.line_end, self.line_start
        p1x, p1y = self.line_start
        p2x, p2y = self.line_end

        # Create grid of coordinates
        y_coords, x_coords = np.ogrid[:frame_height, :frame_width]

        # Calculate side of points relative to line P1->P2
        val_matrix = (p2x - p1x) * (y_coords - p1y) - (p2y - p1y) * (x_coords - p1x)

        if side == "left":
            mask[val_matrix > 0] = 255  # Highlight left side
        elif side == "right":
            mask[val_matrix < 0] = 255  # Highlight right side

        return mask

    def reset(self):
        """Reset all line data."""
        self.line_start = None
        self.line_end = None
        self.selected_side = None
        self.drawing_line = False
        self.cached_drawing_info = None
        self.last_canvas_size = (0, 0)
        # Reset finalization flag
        self._finalized_for_processing = False
