import os
import sys
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

# Add parent directories to path for imports
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from app.core.line_manager import LineManager


class VideoDisplayWidget(QWidget):
    """Widget for displaying video frames and handling line drawing."""

    # Define signals
    line_finished = pyqtSignal()

    def __init__(self, line_manager: LineManager):
        super().__init__()

        self.line_manager = line_manager

        # Display state
        self.current_frame: Optional[np.ndarray] = None
        self.original_frame: Optional[np.ndarray] = None

        # Cached preview dimensions for optimization
        self.cached_preview_dims: Optional[Dict[str, int]] = None
        self.last_widget_size = (0, 0)
        self.last_frame_size = (0, 0)

        # Pre-allocated objects for performance
        self.display_pixmap: Optional[QPixmap] = None
        self.base_qimage: Optional[QImage] = None

        # Drawing state
        self.is_drawing = False
        self.start_point = None

        # Callbacks
        self.line_start_callback: Optional[Callable] = None
        self.line_update_callback: Optional[Callable] = None
        self.line_finish_callback: Optional[Callable] = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the video display UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create label for video display
        self.video_label = QLabel()
        self.video_label.setStyleSheet(
            "QLabel { background-color: black; border: 1px solid gray; }"
        )
        self.video_label.setScaledContents(False)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No video loaded")
        self.video_label.setMinimumSize(640, 480)

        # Enable mouse tracking for drawing
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self._on_mouse_press
        self.video_label.mouseMoveEvent = self._on_mouse_move
        self.video_label.mouseReleaseEvent = self._on_mouse_release

        layout.addWidget(self.video_label)

    def set_callbacks(
        self,
        line_start_callback: Callable = None,
        line_update_callback: Callable = None,
        line_finish_callback: Callable = None,
    ):
        """Set callback functions for line drawing events."""
        self.line_start_callback = line_start_callback
        self.line_update_callback = line_update_callback
        self.line_finish_callback = line_finish_callback

    def update_frame(self, frame: np.ndarray, is_original: bool = False):
        """Update the displayed frame."""
        self.current_frame = frame
        if is_original:
            self.original_frame = frame.copy()
        self.update_preview()

    def update_preview(self):
        """Highly optimized preview update with minimal object creation."""
        if self.current_frame is None:
            return

        # Get widget dimensions
        widget_width = self.video_label.width()
        widget_height = self.video_label.height()

        if widget_width <= 1 or widget_height <= 1:
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        current_frame_size = (frame_width, frame_height)
        current_widget_size = (widget_width, widget_height)

        # Check if we need to recalculate dimensions or reallocate pixmaps
        size_changed = (
            current_widget_size != self.last_widget_size
            or current_frame_size != self.last_frame_size
        )

        if self.cached_preview_dims is None or size_changed:
            # Update coordinate transformer with current frame size
            self.line_manager.coord_transformer.update_frame_size(
                frame_width, frame_height
            )

            # Calculate aspect ratio and new dimensions
            aspect_ratio = frame_width / frame_height

            if widget_width / widget_height > aspect_ratio:
                new_height = widget_height
                new_width = int(widget_height * aspect_ratio)
            else:
                new_width = widget_width
                new_height = int(widget_width / aspect_ratio)

            # Cache the calculated dimensions
            self.cached_preview_dims = {
                "new_width": new_width,
                "new_height": new_height,
                "offset_x": (widget_width - new_width) // 2,
                "offset_y": (widget_height - new_height) // 2,
            }

            # Pre-allocate display pixmap only when size changes
            self.display_pixmap = QPixmap(widget_width, widget_height)

            self.last_widget_size = current_widget_size
            self.last_frame_size = current_frame_size

        dims = self.cached_preview_dims

        # Resize frame using cached dimensions
        resized_frame = cv2.resize(
            self.current_frame, (dims["new_width"], dims["new_height"])
        )

        # Fast numpy to QImage conversion without extra copies
        height, width, channel = resized_frame.shape
        bytes_per_line = 3 * width

        # Create QImage directly from numpy data
        # Frames from video processor are already in RGB format, no conversion needed
        if resized_frame.flags["C_CONTIGUOUS"]:
            q_image = QImage(
                resized_frame.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
        else:
            # Fall back to previous method if not contiguous
            q_image = QImage(
                resized_frame.data, width, height, bytes_per_line, QImage.Format_RGB888
            )

        # Check if we need line overlay
        drawing_info = self.line_manager.get_drawing_info(widget_width, widget_height)
        has_line_overlay = drawing_info and drawing_info.get("has_line", False)

        if has_line_overlay:
            # Only use QPainter when we need to draw overlays
            self.display_pixmap.fill(Qt.black)
            painter = QPainter(self.display_pixmap)

            # Draw the video frame
            painter.drawImage(dims["offset_x"], dims["offset_y"], q_image)

            # Draw line overlay
            self._draw_line_overlay(painter, drawing_info)
            painter.end()

            self.video_label.setPixmap(self.display_pixmap)
        else:
            # Direct pixmap creation without QPainter for better performance
            frame_pixmap = QPixmap.fromImage(q_image)

            if dims["offset_x"] == 0 and dims["offset_y"] == 0:
                # No centering needed, use frame directly
                self.video_label.setPixmap(frame_pixmap)
            else:
                # Need centering
                self.display_pixmap.fill(Qt.black)
                painter = QPainter(self.display_pixmap)
                painter.drawPixmap(dims["offset_x"], dims["offset_y"], frame_pixmap)
                painter.end()
                self.video_label.setPixmap(self.display_pixmap)

    def _draw_line_overlay(self, painter: QPainter, drawing_info: Dict[str, Any]):
        """Draw line overlay using cached info."""
        if not drawing_info.get("has_line", False):
            return

        # Set up painter for line drawing
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)

        # Draw extended line using cached coordinates
        ext_start_x, ext_start_y, ext_end_x, ext_end_y = drawing_info["line_coords"]
        painter.drawLine(
            int(ext_start_x), int(ext_start_y), int(ext_end_x), int(ext_end_y)
        )

        # Draw labels if side is selected
        if self.line_manager.selected_side is not None:
            self._draw_labels_cached(painter, drawing_info)

    def _draw_labels_cached(self, painter: QPainter, drawing_info: Dict[str, Any]):
        """Draw IN/OUT labels using cached information."""
        # Get the pre-calculated label positions from drawing info
        label_positions = drawing_info.get("label_positions")
        if not label_positions:
            # No label positions available (likely still drawing the line)
            return

        left_pos = label_positions["left_pos"]
        right_pos = label_positions["right_pos"]

        # Set up font
        font = QFont("Arial", 16, QFont.Bold)
        painter.setFont(font)

        # Determine which side should be IN based on user selection
        if self.line_manager.selected_side == "left":
            in_pos = left_pos
            out_pos = right_pos
        else:  # right side
            in_pos = right_pos
            out_pos = left_pos

        # Draw IN label with background for better visibility
        self._draw_text_with_outline(
            painter, int(in_pos[0]), int(in_pos[1]), "IN", QColor(0, 255, 0)
        )

        # Draw OUT label with background for better visibility
        self._draw_text_with_outline(
            painter, int(out_pos[0]), int(out_pos[1]), "OUT", QColor(255, 0, 0)
        )

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

    def _draw_text_with_outline(self, painter, x, y, text, color):
        """Draw text with a white outline for better visibility."""
        # Get font metrics for proper text centering
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.width(text)
        text_height = font_metrics.height()

        # Center the text at the given position
        centered_x = x - text_width // 2
        centered_y = y + text_height // 4  # Slight adjustment for visual centering

        # Draw white outline by drawing text multiple times with offsets
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:  # Don't draw at center position yet
                    painter.drawText(centered_x + dx, centered_y + dy, text)

        # Draw the colored text on top
        painter.setPen(QPen(color))
        painter.drawText(centered_x, centered_y, text)

    def preview_side(self, side: str, frame_width: int, frame_height: int):
        """Show mask highlight for side selection."""
        if self.original_frame is not None and self.line_manager.has_line():
            # Create a copy of the original frame for preview
            preview_img = self.original_frame.copy()

            # Create mask for side highlighting
            mask = self.line_manager.create_side_mask(side, frame_width, frame_height)

            # Apply green overlay to the selected side
            overlay_color = np.array([0, 255, 0], dtype=preview_img.dtype)
            overlay_img = np.zeros_like(preview_img)
            overlay_img[mask > 0] = overlay_color

            # Blend the original frame with the green overlay
            highlighted_frame = cv2.addWeighted(overlay_img, 0.3, preview_img, 0.7, 0)

            # Temporarily update the display
            self.current_frame = highlighted_frame
            self.update_preview()

    def clear_preview(self):
        """Clear side preview and restore original frame."""
        if self.original_frame is not None:
            self.current_frame = self.original_frame.copy()
            self.update_preview()

    def reset_display(self):
        """Reset the display to initial state."""
        self.current_frame = None
        self.original_frame = None
        self.display_pixmap = None
        self.base_qimage = None
        self.cached_preview_dims = None
        self.video_label.clear()
        self.video_label.setText("No video loaded")

    def _on_mouse_press(self, event):
        """Handle mouse press events for line drawing."""
        if event.button() == Qt.LeftButton and self.line_manager.is_drawing:
            self.is_drawing = True
            self.start_point = (event.x(), event.y())

            # Convert widget coordinates to canvas coordinates for line manager
            widget_width = self.video_label.width()
            widget_height = self.video_label.height()
            self.line_manager.start_line(
                event.x(), event.y(), widget_width, widget_height
            )

            if self.line_start_callback:
                self.line_start_callback()

    def _on_mouse_move(self, event):
        """Handle mouse move events for line drawing."""
        if self.is_drawing and self.line_manager.is_drawing:
            widget_width = self.video_label.width()
            widget_height = self.video_label.height()
            self.line_manager.update_line(
                event.x(), event.y(), widget_width, widget_height
            )
            self.update_preview()

            if self.line_update_callback:
                self.line_update_callback()

    def _on_mouse_release(self, event):
        """Handle mouse release events for line drawing."""
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False

            widget_width = self.video_label.width()
            widget_height = self.video_label.height()
            self.line_manager.finish_line(
                event.x(), event.y(), widget_width, widget_height
            )
            self.update_preview()

            if self.line_finish_callback:
                self.line_finish_callback()

            self.line_finished.emit()

    def resizeEvent(self, event):
        """Handle resize events to update preview."""
        super().resizeEvent(event)
        if self.current_frame is not None:
            # Clear cached dimensions on resize
            self.cached_preview_dims = None
            self.update_preview()
