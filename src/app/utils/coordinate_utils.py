from typing import Tuple


class CoordinateTransformer:
    """Handles coordinate transformations between canvas and frame coordinates."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def update_frame_size(self, frame_width: int, frame_height: int):
        """Update frame dimensions."""
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_preview_dimensions(
        self, canvas_width: int, canvas_height: int
    ) -> Tuple[int, int, int, int]:
        """Calculate preview dimensions and offsets."""
        if self.frame_width == 0 or self.frame_height == 0:
            return 0, 0, 0, 0

        aspect_ratio = self.frame_width / self.frame_height

        if canvas_width / canvas_height > aspect_ratio:
            preview_height = canvas_height
            preview_width = int(canvas_height * aspect_ratio)
        else:
            preview_width = canvas_width
            preview_height = int(canvas_width / aspect_ratio)

        x_offset = (canvas_width - preview_width) // 2
        y_offset = (canvas_height - preview_height) // 2

        return preview_width, preview_height, x_offset, y_offset

    def canvas_to_frame(
        self, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int
    ) -> Tuple[int, int]:
        """Convert canvas coordinates to frame coordinates."""
        preview_width, preview_height, x_offset, y_offset = self.get_preview_dimensions(
            canvas_width, canvas_height
        )

        if preview_width == 0 or preview_height == 0:
            return 0, 0

        # Convert to frame coordinates
        frame_x = int((canvas_x - x_offset) * self.frame_width / preview_width)
        frame_y = int((canvas_y - y_offset) * self.frame_height / preview_height)

        # Clamp to frame bounds
        frame_x = max(0, min(frame_x, self.frame_width - 1))
        frame_y = max(0, min(frame_y, self.frame_height - 1))

        return frame_x, frame_y

    def frame_to_canvas(
        self, frame_x: int, frame_y: int, canvas_width: int, canvas_height: int
    ) -> Tuple[int, int]:
        """Convert frame coordinates to canvas coordinates."""
        preview_width, preview_height, x_offset, y_offset = self.get_preview_dimensions(
            canvas_width, canvas_height
        )

        if preview_width == 0 or preview_height == 0:
            return 0, 0

        canvas_x = int((frame_x / self.frame_width) * preview_width + x_offset)
        canvas_y = int((frame_y / self.frame_height) * preview_height + y_offset)

        return canvas_x, canvas_y

    def normalize_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """Normalize coordinates to 0-1 range."""
        if self.frame_width == 0 or self.frame_height == 0:
            return 0.0, 0.0
        return x / self.frame_width, y / self.frame_height
