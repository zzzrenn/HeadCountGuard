import time
from collections import deque

import numpy as np
from loguru import logger
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.core.config_manager import ConfigManager
from app.core.line_manager import LineManager
from app.core.roi_manager import ROIManager
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
        self.roi_manager = ROIManager(self.coord_transformer)

        # UI state
        self.is_processing = False
        self.line_drawn = False
        self.side_selected = False
        self.selected_side = None
        self.bbox_filter_enabled = True  # Enable by default

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
        self.showFullScreen()
        self.setWindowState(Qt.WindowFullScreen)

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

        # Create video display widget with both managers
        self.video_display = VideoDisplayWidget(self.line_manager, self.roi_manager)
        # Always start in line mode
        self.video_display.set_drawing_mode("line")
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

        # Setup keyboard shortcuts
        self.setup_shortcuts()

    def setup_callbacks(self):
        """Setup callbacks between components."""
        # Video display callbacks
        self.video_display.set_callbacks(
            line_finish_callback=self.on_line_drawing_finished,
            roi_point_added_callback=self.on_roi_point_added,
            roi_finish_callback=self.on_roi_drawing_finished,
        )

        # Controls callbacks
        self.controls.set_callbacks(
            video_select_callback=self.on_video_selected,
            draw_line_callback=self.on_draw_line_clicked,
            draw_roi_callback=self.on_draw_roi_clicked,
            cancel_roi_callback=self.on_cancel_roi_clicked,
            skip_roi_callback=self.on_skip_roi_clicked,
            confirm_callback=self.on_confirm_clicked,
            side_select_callback=self.on_side_selected,
            side_preview_callback=self.on_side_preview,
            side_clear_callback=self.on_side_clear,
            exit_callback=self.on_exit_clicked,
            bbox_filter_callback=self.on_bbox_filter_toggled,
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

        # Reset processing state
        self.is_processing = False
        if self.processing_timer.isActive():
            self.processing_timer.stop()

        # Update coordinate transformer with video dimensions
        props = self.video_processor.get_video_properties()
        self.coord_transformer.update_frame_size(props["width"], props["height"])

        # Reset managers for new video
        self.line_manager.reset()
        self.roi_manager.reset()
        self.line_drawn = False
        self.side_selected = False
        self.selected_side = None

        # Reset video display to line mode
        self.video_display.set_drawing_mode("line")
        self.video_display.set_processing_active(False)  # Allow canvas interactions
        self.video_display.reset_display()

        # Reset all UI controls to initial state
        self.controls.draw_line_btn.setEnabled(False)  # Will be enabled below
        self.controls.side_group.setEnabled(False)
        self.controls.roi_group.setEnabled(False)

        # Enable bbox filter checkbox and set it checked by default
        self.controls.enable_bbox_filter(True)
        self.controls.bbox_filter_checkbox.setChecked(True)
        self.bbox_filter_enabled = True

        # Reset ROI button states specifically
        self.controls.draw_roi_btn.setEnabled(
            True
        )  # Default enabled state within group
        self.controls.skip_roi_btn.setEnabled(
            True
        )  # Default enabled state within group
        self.controls.cancel_roi_btn.setVisible(False)
        self.controls.confirm_btn.setEnabled(False)

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
            # Enable line drawing (step 2)
            self.controls.enable_line_drawing()
            logger.info("New video loaded, line drawing enabled")

    def on_draw_line_clicked(self):
        """Handle draw line button click."""
        logger.info("Draw line clicked")
        # Reset states when drawing a new line
        self.line_manager.reset()
        self.roi_manager.reset()
        self.line_drawn = False
        self.side_selected = False
        self.selected_side = None

        # Reset UI states for fresh start
        self.controls.side_group.setEnabled(False)
        self.controls.roi_group.setEnabled(False)
        self.controls.confirm_btn.setEnabled(False)

        # Keep bbox filter enabled and checked
        self.controls.enable_bbox_filter(True)
        self.controls._roi_available = False

        # Start line drawing
        self.line_manager.start_drawing()

    def on_draw_roi_clicked(self):
        """Handle draw ROI button click."""
        self.video_display.set_drawing_mode("roi")
        self.roi_manager.start_drawing()

    def on_cancel_roi_clicked(self):
        """Handle cancel ROI button click."""
        logger.info("Cancel ROI clicked")
        self.roi_manager.cancel_drawing()
        self.video_display.set_drawing_mode("line")
        self.video_display.update_preview()

        # Reset confirm button state - only enable if we had skipped ROI before
        # For now, disable it and let user choose skip or draw ROI again
        self.controls.confirm_btn.setEnabled(False)

        # Update ROI availability but keep bbox filter enabled
        self.controls._roi_available = False
        self.controls.enable_bbox_filter(True)

    def on_skip_roi_clicked(self):
        """Handle skip ROI button click - enable confirm button for processing without ROI."""
        logger.info("Skip ROI clicked - enabling confirm button")
        # Just enable the confirm button, don't start processing automatically
        self.controls.confirm_btn.setEnabled(True)
        # Keep ROI drawing options enabled in case user changes their mind
        # Don't disable the draw_roi_btn and skip_roi_btn

        # Update ROI availability but keep bbox filter enabled
        self.controls._roi_available = False
        self.controls.enable_bbox_filter(True)

    def on_confirm_clicked(self):
        """Handle confirm button click - start processing with ROI."""
        logger.info("Confirm clicked - starting processing")
        self.start_processing()

    def on_line_drawing_finished(self):
        """Handle completion of line drawing."""
        if self.line_manager.has_line():
            self.line_drawn = True
            # Enable side selection (step 3) but keep line drawing enabled for re-drawing
            self.controls.enable_side_selection()
            self.controls.enable_line_drawing()  # Allow re-drawing line

    def on_roi_drawing_finished(self):
        """Handle completion of ROI drawing."""
        logger.info("Main window - ROI drawing finished!")
        if self.roi_manager.has_roi():
            logger.info(
                "Main window - ROI manager has ROI, calling controls.on_roi_finished()"
            )
            self.controls.on_roi_finished()
            # Update ROI availability and keep bbox filter enabled
            self.controls._roi_available = True
            self.controls.enable_bbox_filter(True)
            # The filter should already be checked by default
            if not self.bbox_filter_enabled:
                self.controls.bbox_filter_checkbox.setChecked(True)
                self.bbox_filter_enabled = True
        else:
            logger.info("Main window - ROI manager does NOT have ROI")
            self.controls._roi_available = False
            self.controls.enable_bbox_filter(True)

    def on_side_selected(self, side: str):
        """Handle side selection."""
        logger.info(f"Side selected: {side}")
        self.line_manager.select_side(side)
        self.side_selected = True
        self.selected_side = side

        # Enable ROI options (step 4) but keep previous steps enabled
        logger.info("Calling enable_roi_options")
        self.controls.enable_roi_options()
        # Keep line drawing and side selection enabled for modifications
        self.controls.enable_line_drawing()
        self.controls.enable_side_selection()

    def start_processing(self):
        """Start video processing with current line and optional ROI."""
        if not self.line_drawn or not self.side_selected:
            return

        # Disable canvas interactions during processing
        self.video_display.set_processing_active(True)

        # Setup counter based on whether ROI is drawn
        normalized_line = self.line_manager.get_normalized_line()
        if not normalized_line:
            return

        # Get video properties for ROI mask creation
        props = self.video_processor.get_video_properties()

        if self.roi_manager.has_roi():
            # Use line + ROI
            roi_mask = self.roi_manager.create_roi_mask(props["width"], props["height"])
            self.video_processor.setup_counter_with_roi(
                normalized_line, self.selected_side, roi_mask
            )

            # Set ROI mask for display filtering
            self.video_processor.set_roi_for_display(roi_mask)
            self.video_processor.set_bbox_filter_enabled(self.bbox_filter_enabled)

            # Keep bbox filter available during processing
            self.controls.enable_bbox_filter(True)

            logger.info("Starting processing with line and ROI filter")
        else:
            # Use line only
            self.video_processor.setup_counter(normalized_line, self.selected_side)

            # No ROI available, but keep bbox filter enabled (it will show all boxes)
            self.video_processor.set_roi_for_display(None)
            self.video_processor.set_bbox_filter_enabled(self.bbox_filter_enabled)
            self.controls.enable_bbox_filter(True)

            logger.info("Starting processing with line only")

        # Disable all drawing controls during processing but keep bbox filter if available
        self.controls.disable_all_drawing()

        # Start video processing
        self.start_video_processing()

    def on_side_preview(self, side: str):
        """Handle side preview on hover."""
        if not self.is_processing and self.line_drawn:
            props = self.video_processor.get_video_properties()
            self.video_display.preview_side(side, props["width"], props["height"])

    def on_side_clear(self):
        """Handle clearing side preview."""
        if not self.is_processing:
            self.video_display.clear_preview()

    def start_video_processing(self):
        """Start the video processing timer."""
        if not self.video_processor.start_processing():
            logger.error("Failed to start video processing")
            return

        # Start processing timer
        # Use a shorter interval for better responsiveness
        self.processing_timer.start(16)  # ~60 FPS

        # Update UI state
        self.is_processing = True

        # Show ROI overlay if available
        if self.roi_manager.has_roi():
            props = self.video_processor.get_video_properties()
            self.video_display.show_roi_overlay(props["width"], props["height"])

        logger.info("Video processing started")

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

        # Re-enable canvas interactions
        self.video_display.set_processing_active(False)

        # Clear ROI overlay
        self.video_display.clear_roi_overlay()

        # Re-enable controls based on current state
        if self.line_drawn:
            self.controls.enable_line_drawing()
        if self.side_selected:
            self.controls.enable_side_selection()
        if self.line_drawn and self.side_selected:
            self.controls.enable_roi_options()

        # Update bbox filter availability based on ROI state
        self._update_bbox_filter_availability()

        # Final histogram update
        self.histogram.update_plot(count_history, frame_history)
        logger.info("Video playback completed - ready for new video")

    def on_roi_point_added(self):
        """Handle when a ROI point is added."""
        # Check if we have enough points for a valid ROI
        if self.roi_manager.has_roi():
            logger.info(
                f"ROI now has {len(self.roi_manager.roi_points)} points - enabling confirm button"
            )
            self.controls.confirm_btn.setEnabled(True)
        else:
            logger.info(
                f"ROI has {len(self.roi_manager.roi_points)} points - need at least 3"
            )

    def closeEvent(self, event):
        """Handle window close event."""
        self.cleanup()
        event.accept()

    def cleanup(self):
        """Cleanup resources."""
        if self.processing_timer.isActive():
            self.processing_timer.stop()
        self.video_processor.release()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Escape key to exit full screen (go to windowed mode)
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.escape_shortcut.activated.connect(self.exit_full_screen)

        # F11 to toggle full screen
        self.f11_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        self.f11_shortcut.activated.connect(self.toggle_full_screen)

        # Ctrl+Q to quit application
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(self.close)

    def on_exit_clicked(self):
        """Handle exit button click."""
        self.close()

    def exit_full_screen(self):
        """Exit full screen mode and go to windowed mode."""
        if self.isFullScreen():
            self.showNormal()
            self.setGeometry(100, 100, 1400, 900)  # Restore original size

    def toggle_full_screen(self):
        """Toggle between full screen and windowed mode."""
        if self.isFullScreen():
            self.exit_full_screen()
        else:
            self.showFullScreen()

    def on_bbox_filter_toggled(self, enabled: bool):
        """Handle bbox filter toggled."""
        self.bbox_filter_enabled = enabled
        logger.info(f"Bbox filter enabled: {self.bbox_filter_enabled}")

        # If processing is active, immediately apply the filter change
        if self.is_processing:
            self.video_processor.set_bbox_filter_enabled(self.bbox_filter_enabled)

    def _update_bbox_filter_availability(self):
        """Update bbox filter availability based on current ROI state."""
        # Always keep the bbox filter available and checked by default
        self.controls.enable_bbox_filter(True)
        if not self.bbox_filter_enabled:
            self.controls.bbox_filter_checkbox.setChecked(True)
            self.bbox_filter_enabled = True
