from typing import Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ControlsWidget(QWidget):
    """Widget containing control buttons and count display."""

    # Define signals
    video_selected = pyqtSignal(str)
    draw_line_clicked = pyqtSignal()
    draw_roi_clicked = pyqtSignal()
    cancel_roi_clicked = pyqtSignal()
    skip_roi_clicked = pyqtSignal()
    confirm_clicked = pyqtSignal()
    side_selected = pyqtSignal(str)
    side_preview = pyqtSignal(str)
    side_clear = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Callbacks
        self.video_select_callback: Optional[Callable] = None
        self.draw_line_callback: Optional[Callable] = None
        self.draw_roi_callback: Optional[Callable] = None
        self.cancel_roi_callback: Optional[Callable] = None
        self.skip_roi_callback: Optional[Callable] = None
        self.confirm_callback: Optional[Callable] = None
        self.side_select_callback: Optional[Callable] = None
        self.side_preview_callback: Optional[Callable] = None
        self.side_clear_callback: Optional[Callable] = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the controls UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Create control buttons section
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout(button_group)

        # Step 1: Video selection
        self.select_video_btn = QPushButton("1. Select Video")
        self.select_video_btn.setMinimumHeight(40)
        self.select_video_btn.clicked.connect(self._on_select_video)
        button_layout.addWidget(self.select_video_btn)

        # Step 2: Line drawing
        self.draw_line_btn = QPushButton("2. Draw Counting Line")
        self.draw_line_btn.setMinimumHeight(40)
        self.draw_line_btn.clicked.connect(self._on_draw_line)
        self.draw_line_btn.setEnabled(False)
        button_layout.addWidget(self.draw_line_btn)

        layout.addWidget(button_group)

        # Step 3: Side selection section
        self.side_group = QGroupBox("3. Select IN Side (which side people enter from)")
        side_layout = QHBoxLayout(self.side_group)

        self.left_btn = QPushButton("Left Side")
        self.left_btn.setMinimumHeight(40)
        self.left_btn.clicked.connect(lambda: self._on_side_select("left"))
        self.left_btn.enterEvent = lambda event: self._on_side_hover("left")
        self.left_btn.leaveEvent = lambda event: self._on_side_leave()
        side_layout.addWidget(self.left_btn)

        self.right_btn = QPushButton("Right Side")
        self.right_btn.setMinimumHeight(40)
        self.right_btn.clicked.connect(lambda: self._on_side_select("right"))
        self.right_btn.enterEvent = lambda event: self._on_side_hover("right")
        self.right_btn.leaveEvent = lambda event: self._on_side_leave()
        side_layout.addWidget(self.right_btn)

        self.side_group.setEnabled(False)
        layout.addWidget(self.side_group)

        # Step 4: Optional ROI section
        self.roi_group = QGroupBox("4. Optional: Draw Region of Interest (ROI)")
        roi_layout = QVBoxLayout(self.roi_group)

        # ROI explanation
        roi_explanation = QLabel(
            "Draw an ROI to limit counting to a specific area.\nOnly people inside this region will be counted."
        )
        roi_explanation.setWordWrap(True)
        roi_explanation.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        roi_layout.addWidget(roi_explanation)

        # ROI buttons
        roi_button_layout = QHBoxLayout()

        self.draw_roi_btn = QPushButton("Draw ROI")
        self.draw_roi_btn.setMinimumHeight(35)
        self.draw_roi_btn.clicked.connect(self._on_draw_roi)
        roi_button_layout.addWidget(self.draw_roi_btn)

        self.skip_roi_btn = QPushButton("Skip ROI")
        self.skip_roi_btn.setMinimumHeight(35)
        self.skip_roi_btn.clicked.connect(self._on_skip_roi)
        roi_button_layout.addWidget(self.skip_roi_btn)

        roi_layout.addLayout(roi_button_layout)

        # Cancel and Confirm buttons (side by side)
        cancel_confirm_layout = QHBoxLayout()

        self.cancel_roi_btn = QPushButton("Cancel ROI")
        self.cancel_roi_btn.setMinimumHeight(30)
        self.cancel_roi_btn.clicked.connect(self._on_cancel_roi)
        self.cancel_roi_btn.setVisible(False)
        self.cancel_roi_btn.setStyleSheet(
            "QPushButton { background-color: #ff6b6b; color: white; }"
        )
        cancel_confirm_layout.addWidget(self.cancel_roi_btn)

        # Confirm button (visible but disabled initially, enabled after ROI is drawn)
        self.confirm_btn = QPushButton("✓ Confirm & Start Processing")
        self.confirm_btn.setMinimumHeight(30)
        self.confirm_btn.clicked.connect(self._on_confirm)
        self.confirm_btn.setEnabled(False)  # Disabled initially
        self.confirm_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        cancel_confirm_layout.addWidget(self.confirm_btn)

        roi_layout.addLayout(cancel_confirm_layout)

        self.roi_group.setEnabled(False)
        layout.addWidget(self.roi_group)

        # Counter display section
        counter_group = QGroupBox("Statistics")
        counter_layout = QVBoxLayout(counter_group)

        self.counter_label = QLabel("Count: 0")
        counter_font = QFont("Arial", 20, QFont.Bold)
        self.counter_label.setFont(counter_font)
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        counter_layout.addWidget(self.counter_label)

        # Add FPS display
        self.fps_label = QLabel("FPS: 0.0")
        fps_font = QFont("Arial", 12)
        self.fps_label.setFont(fps_font)
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet("""
            QLabel {
                background-color: #e8e8e8;
                border: 1px solid #bbb;
                border-radius: 3px;
                padding: 5px;
                margin-top: 5px;
            }
        """)
        counter_layout.addWidget(self.fps_label)

        layout.addWidget(counter_group)

        # Add stretch to push everything up
        layout.addStretch()

    def set_callbacks(
        self,
        video_select_callback: Callable = None,
        draw_line_callback: Callable = None,
        draw_roi_callback: Callable = None,
        cancel_roi_callback: Callable = None,
        skip_roi_callback: Callable = None,
        confirm_callback: Callable = None,
        side_select_callback: Callable = None,
        side_preview_callback: Callable = None,
        side_clear_callback: Callable = None,
    ):
        """Set callback functions for control events."""
        self.video_select_callback = video_select_callback
        self.draw_line_callback = draw_line_callback
        self.draw_roi_callback = draw_roi_callback
        self.cancel_roi_callback = cancel_roi_callback
        self.skip_roi_callback = skip_roi_callback
        self.confirm_callback = confirm_callback
        self.side_select_callback = side_select_callback
        self.side_preview_callback = side_preview_callback
        self.side_clear_callback = side_clear_callback

    def update_count_display(self, count: int):
        """Update the count display."""
        self.counter_label.setText(f"Count: {count}")

    def update_fps_display(self, fps: float):
        """Update the FPS display."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_select_video(self):
        """Handle video selection button click."""
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All files (*.*)",
        )
        if video_path and self.video_select_callback:
            self.video_select_callback(video_path)

    def _on_draw_line(self):
        """Handle draw line button click."""
        if self.draw_line_callback:
            self.draw_line_callback()

    def _on_draw_roi(self):
        """Handle draw ROI button click."""
        if self.draw_roi_callback:
            self.draw_roi_callback()
        self.draw_roi_btn.setEnabled(False)
        self.skip_roi_btn.setEnabled(False)
        self.cancel_roi_btn.setVisible(True)
        self.confirm_btn.setEnabled(False)  # Disable until ROI is complete

    def _on_cancel_roi(self):
        """Handle cancel ROI button click."""
        if self.cancel_roi_callback:
            self.cancel_roi_callback()
        self.draw_roi_btn.setEnabled(True)
        self.skip_roi_btn.setEnabled(True)
        self.cancel_roi_btn.setVisible(False)
        self.confirm_btn.setEnabled(False)  # Disable when canceling

    def _on_skip_roi(self):
        """Handle skip ROI button click."""
        if self.skip_roi_callback:
            self.skip_roi_callback()

    def _on_side_select(self, side: str):
        """Handle side selection button click."""
        if self.side_select_callback:
            self.side_select_callback(side)

    def _on_side_hover(self, side: str):
        """Handle side button hover."""
        if self.side_preview_callback:
            self.side_preview_callback(side)

    def _on_side_leave(self):
        """Handle side button leave."""
        if self.side_clear_callback:
            self.side_clear_callback()

    def enable_line_drawing(self):
        """Enable the draw line button after video is loaded."""
        self.draw_line_btn.setEnabled(True)

    def enable_side_selection(self):
        """Enable the side selection buttons after line is drawn."""
        self.side_group.setEnabled(True)

    def enable_roi_options(self):
        """Enable ROI drawing options."""
        self.roi_group.setEnabled(True)
        self.draw_roi_btn.setEnabled(True)
        self.cancel_roi_btn.setEnabled(True)
        self.skip_roi_btn.setEnabled(True)

    def reset_roi_buttons(self):
        """Reset ROI buttons to initial state."""
        self.roi_group.setEnabled(False)
        self.draw_roi_btn.setEnabled(False)
        self.cancel_roi_btn.setEnabled(False)
        self.skip_roi_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)

    def on_roi_finished(self):
        """Called when ROI drawing is finished."""
        self.draw_roi_btn.setEnabled(False)
        self.cancel_roi_btn.setEnabled(False)
        self.confirm_btn.setEnabled(True)

    def disable_all_drawing(self):
        """Disable all drawing controls when processing starts."""
        self.draw_line_btn.setEnabled(False)
        self.side_group.setEnabled(False)
        self.roi_group.setEnabled(False)

    def _on_confirm(self):
        """Handle confirm button click."""
        if self.confirm_callback:
            self.confirm_callback()
