import os
import sys
import time
from collections import deque

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Add parent directories to path for imports
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from app.core.config_manager import ConfigManager
from app.core.line_manager import LineManager
from app.core.video_processor import VideoProcessor
from app.ui.controls import ControlsWidget
from app.ui.histogram import HistogramWidget
from app.ui.video_display import VideoDisplayWidget
from app.utils.coordinate_utils import CoordinateTransformer


class PeopleCounterApp(QMainWindow):
    """Main application window that coordinates all components."""

    def __init__(self):
        super().__init__()

        # Initialize core components
        self.config_manager = ConfigManager()
        self.video_processor = VideoProcessor(
            self.config_manager.get_detector_config(),
            self.config_manager.get_tracking_config(),
            self.config_manager.get_line_crossing_config(),
        )
        self.coord_transformer = CoordinateTransformer(0, 0)
        self.line_manager = LineManager(self.coord_transformer)

        # UI state
        self.is_processing = False

        # FPS tracking
        self.frame_times = deque(maxlen=30)  # Keep last 30 frame times for averaging
        self.last_frame_time = None

        # Timer for video processing
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_video_frame)

        self.setup_ui()
        self.setup_callbacks()

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("People Counter - PyQt")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Create video display widget
        self.video_display = VideoDisplayWidget(self.line_manager)
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.StyledPanel)
        video_layout = QVBoxLayout(video_frame)
        video_layout.addWidget(self.video_display)

        # Create right panel for controls and histogram
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Create controls widget
        self.controls = ControlsWidget()
        right_layout.addWidget(self.controls)

        # Create histogram widget
        self.histogram = HistogramWidget()
        right_layout.addWidget(self.histogram)

        # Add to splitter
        splitter.addWidget(video_frame)
        splitter.addWidget(right_panel)

        # Set splitter proportions (video takes more space)
        splitter.setSizes([1000, 400])

        # Set minimum sizes
        video_frame.setMinimumWidth(600)
        right_panel.setMinimumWidth(350)

    def setup_callbacks(self):
        """Setup callbacks between components."""
        # Video display callbacks
        self.video_display.set_callbacks(
            line_finish_callback=self.on_line_drawing_finished
        )

        # Controls callbacks
        self.controls.set_callbacks(
            video_select_callback=self.on_video_selected,
            draw_line_callback=self.on_draw_line_clicked,
            side_select_callback=self.on_side_selected,
            side_preview_callback=self.on_side_preview,
            side_clear_callback=self.on_side_clear,
        )

        # Video processor callbacks
        self.video_processor.set_callbacks(
            frame_callback=self.on_frame_processed,
            count_callback=self.on_count_updated,
            complete_callback=self.on_video_completed,
        )

    def on_video_selected(self, video_path: str):
        """Handle video selection."""
        success = self.video_processor.load_video(video_path)
        if not success:
            return

        # Update coordinate transformer with video dimensions
        props = self.video_processor.get_video_properties()
        self.coord_transformer.update_frame_size(props["width"], props["height"])

        # Reset line manager for new video
        self.line_manager.reset()

        # Reset video display
        self.video_display.reset_display()

        # Clear histogram
        self.histogram.clear_plot()

        # Reset FPS tracking
        self.frame_times.clear()
        self.last_frame_time = None
        self.controls.update_fps_display(0.0)

        # Reset count display
        self.controls.update_count_display(0)

        # Get and display first frame
        first_frame = self.video_processor.get_first_frame()
        if first_frame is not None:
            self.video_display.update_frame(first_frame, is_original=True)
            self.controls.enable_line_drawing()

    def on_draw_line_clicked(self):
        """Handle draw line button click."""
        self.line_manager.start_drawing()

    def on_line_drawing_finished(self):
        """Handle completion of line drawing."""
        if self.line_manager.has_line():
            self.controls.enable_side_selection()

    def on_side_selected(self, side: str):
        """Handle side selection."""
        self.line_manager.select_side(side)

        # Setup counter with normalized line coordinates
        normalized_line = self.line_manager.get_normalized_line()
        if normalized_line:
            self.video_processor.setup_counter(normalized_line, side)

        # Start video processing
        self.start_video_processing()

    def on_side_preview(self, side: str):
        """Handle side preview on hover."""
        if not self.is_processing:  # Only show preview if not currently processing
            props = self.video_processor.get_video_properties()
            self.video_display.preview_side(side, props["width"], props["height"])

    def on_side_clear(self):
        """Handle clearing side preview."""
        if not self.is_processing:  # Only clear preview if not currently processing
            self.video_display.clear_preview()

    def start_video_processing(self):
        """Start the video processing loop."""
        if self.video_processor.start_processing():
            self.is_processing = True

            # Finalize all line/mask calculations for video processing
            # This ensures no expensive calculations happen during playback
            widget_width = self.video_display.video_label.width()
            widget_height = self.video_display.video_label.height()
            self.line_manager.finalize_for_processing(widget_width, widget_height)

            # Reset FPS tracking for processing
            self.frame_times.clear()
            self.last_frame_time = None
            # Calculate frame rate for timer
            props = self.video_processor.get_video_properties()
            frame_interval = int(1000 / props["fps"])  # Convert to milliseconds
            self.processing_timer.start(frame_interval)

    def process_video_frame(self):
        """Process a single video frame."""
        if not self.is_processing:
            self.processing_timer.stop()
            return

        frame = self.video_processor.process_next_frame()
        if frame is not None:
            # Draw counting line on frame
            counting_line = self.line_manager.get_counting_line()
            if counting_line:
                frame = self.video_processor.draw_counting_line(frame, counting_line)
        else:
            # Video processing completed
            self.is_processing = False
            self.processing_timer.stop()

    def on_frame_processed(self, frame: np.ndarray):
        """Handle processed frame."""
        # Calculate FPS
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_interval = current_time - self.last_frame_time
            self.frame_times.append(frame_interval)

            # Calculate average FPS from recent frame times
            if len(self.frame_times) > 0:
                avg_interval = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
                self.controls.update_fps_display(fps)

        self.last_frame_time = current_time

        self.video_display.update_frame(frame)

    def on_count_updated(self, count: int, frame_number: int):
        """Handle count update."""
        self.controls.update_count_display(count)

        # Use incremental update for real-time performance
        self.histogram.update_plot_incremental(count, frame_number)

        # Only do full batch updates occasionally for data consistency
        if frame_number % 300 == 0:  # Every 300 frames instead of 30
            self.histogram.update_plot(
                self.video_processor.count_history, self.video_processor.frame_history
            )

    def on_video_completed(self, count_history: list, frame_history: list):
        """Handle video processing completion."""
        self.is_processing = False
        self.processing_timer.stop()
        # Final histogram update
        self.histogram.update_plot(count_history, frame_history)
        print("Video playback completed")

    def closeEvent(self, event):
        """Handle window close event."""
        self.cleanup()
        event.accept()

    def cleanup(self):
        """Cleanup resources."""
        if self.processing_timer.isActive():
            self.processing_timer.stop()
        self.video_processor.release()
