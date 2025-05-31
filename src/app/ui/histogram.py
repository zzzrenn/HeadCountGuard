from typing import List

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QWidget


class HistogramWidget(QWidget):
    """Widget for displaying count history as a histogram/line plot."""

    def __init__(self):
        super().__init__()

        # Performance optimization variables
        self.line_artist = None
        self.points_artist = None
        self.last_data_size = 0
        self.background = None

        # Data caching for incremental updates
        self.cached_count_history = []
        self.cached_frame_history = []

        self.setup_ui()

    def setup_ui(self):
        """Setup the histogram UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create histogram group
        histogram_group = QGroupBox("Count History")
        histogram_layout = QVBoxLayout(histogram_group)

        # Create matplotlib figure with Qt backend
        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Frame Number")
        self.ax.set_ylabel("Count")
        self.ax.set_title("People Count Over Time")
        self.ax.grid(True, alpha=0.3)

        # Create canvas for matplotlib
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(250)
        histogram_layout.addWidget(self.canvas)

        layout.addWidget(histogram_group)

        # Set tight layout to optimize space usage
        self.fig.tight_layout()

        # Initialize empty plot elements for performance
        (self.line_artist,) = self.ax.plot([], [], "b-", linewidth=2, animated=True)
        (self.points_artist,) = self.ax.plot([], [], "bo", markersize=3, animated=True)

        # Initial draw to create background
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def update_plot(self, count_history: List[int], frame_history: List[int]):
        """Update the histogram plot with new data using optimized drawing."""
        if len(count_history) == 0:
            return

        # Check if data has changed significantly
        data_changed = len(count_history) != len(self.cached_count_history) or (
            len(count_history) > 0
            and len(self.cached_count_history) > 0
            and count_history[-1] != self.cached_count_history[-1]
        )

        if not data_changed:
            return  # No need to redraw

        # Cache the current data
        self.cached_count_history = count_history.copy()
        self.cached_frame_history = frame_history.copy()

        # Determine if we need to update axis limits
        current_count_range = (min(count_history), max(count_history))
        current_frame_range = (
            (min(frame_history), max(frame_history)) if frame_history else (0, 1)
        )

        # Get current axis limits
        current_ylim = self.ax.get_ylim()
        current_xlim = self.ax.get_xlim()

        # Check if we need to update axis limits (with some padding)
        y_margin = max(1, abs(current_count_range[1] - current_count_range[0]) * 0.1)
        new_ylim = (
            current_count_range[0] - y_margin,
            current_count_range[1] + y_margin,
        )
        new_xlim = (0, current_frame_range[1] + 50)

        need_axis_update = (
            new_ylim[0] < current_ylim[0]
            or new_ylim[1] > current_ylim[1]
            or new_xlim[1] > current_xlim[1]
            or abs(new_xlim[1] - current_xlim[1]) > 100
        )  # Significant frame change

        if need_axis_update or self.background is None:
            # Full redraw needed for axis changes
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)

            # Redraw static elements
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # Use blitting for fast animated updates
        self.canvas.restore_region(self.background)

        # Update line data
        self.line_artist.set_data(frame_history, count_history)
        self.points_artist.set_data(frame_history, count_history)

        # Draw only the animated artists
        self.ax.draw_artist(self.line_artist)
        self.ax.draw_artist(self.points_artist)

        # Blit the changes
        self.canvas.blit(self.ax.bbox)

    def update_plot_incremental(self, new_count: int, new_frame: int):
        """Optimized incremental update for single new data point."""
        # Add new data point
        self.cached_count_history.append(new_count)
        self.cached_frame_history.append(new_frame)

        # Limit history to prevent memory issues (keep last 5000 points)
        if len(self.cached_count_history) > 5000:
            self.cached_count_history = self.cached_count_history[-5000:]
            self.cached_frame_history = self.cached_frame_history[-5000:]

        # Check if axis limits need updating
        current_ylim = self.ax.get_ylim()
        current_xlim = self.ax.get_xlim()

        y_margin = 1
        need_y_update = new_count < current_ylim[0] or new_count > current_ylim[1]
        need_x_update = new_frame > current_xlim[1] - 50

        if need_y_update or need_x_update:
            # Update axis limits
            if need_y_update:
                all_counts = self.cached_count_history
                y_range = max(all_counts) - min(all_counts)
                y_margin = max(1, y_range * 0.1)
                new_ylim = (min(all_counts) - y_margin, max(all_counts) + y_margin)
                self.ax.set_ylim(new_ylim)

            if need_x_update:
                self.ax.set_xlim(0, new_frame + 100)

            # Redraw background
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # Fast blit update
        if self.background is not None:
            self.canvas.restore_region(self.background)

            # Update line data
            self.line_artist.set_data(
                self.cached_frame_history, self.cached_count_history
            )
            self.points_artist.set_data(
                self.cached_frame_history, self.cached_count_history
            )

            # Draw animated artists
            self.ax.draw_artist(self.line_artist)
            self.ax.draw_artist(self.points_artist)

            # Blit the changes
            self.canvas.blit(self.ax.bbox)

    def clear_plot(self):
        """Clear the histogram plot."""
        self.cached_count_history.clear()
        self.cached_frame_history.clear()

        # Clear the plot
        self.ax.clear()
        self.ax.set_xlabel("Frame Number")
        self.ax.set_ylabel("Count")
        self.ax.set_title("People Count Over Time")
        self.ax.grid(True, alpha=0.3)

        # Reinitialize plot elements
        (self.line_artist,) = self.ax.plot([], [], "b-", linewidth=2, animated=True)
        (self.points_artist,) = self.ax.plot([], [], "bo", markersize=3, animated=True)

        self.fig.tight_layout()
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def resizeEvent(self, event):
        """Handle resize events to update background."""
        super().resizeEvent(event)
        if hasattr(self, "canvas") and self.canvas is not None:
            # Invalidate background on resize
            self.background = None
            self.canvas.draw()
            if hasattr(self, "ax"):
                self.background = self.canvas.copy_from_bbox(self.ax.bbox)
