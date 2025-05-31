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
    side_selected = pyqtSignal(str)
    side_preview = pyqtSignal(str)
    side_clear = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Callbacks
        self.video_select_callback: Optional[Callable] = None
        self.draw_line_callback: Optional[Callable] = None
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

        self.select_video_btn = QPushButton("Select Video")
        self.select_video_btn.setMinimumHeight(40)
        self.select_video_btn.clicked.connect(self._on_select_video)
        button_layout.addWidget(self.select_video_btn)

        self.draw_line_btn = QPushButton("Draw Line")
        self.draw_line_btn.setMinimumHeight(40)
        self.draw_line_btn.clicked.connect(self._on_draw_line)
        button_layout.addWidget(self.draw_line_btn)

        layout.addWidget(button_group)

        # Create side selection section
        self.side_group = QGroupBox("Select IN Side")
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

        layout.addWidget(self.side_group)

        # Create counter display section
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

        # Initially disable buttons
        self.set_button_states(draw_line_enabled=False, side_buttons_enabled=False)

    def set_callbacks(
        self,
        video_select_callback: Callable = None,
        draw_line_callback: Callable = None,
        side_select_callback: Callable = None,
        side_preview_callback: Callable = None,
        side_clear_callback: Callable = None,
    ):
        """Set callback functions for control events."""
        self.video_select_callback = video_select_callback
        self.draw_line_callback = draw_line_callback
        self.side_select_callback = side_select_callback
        self.side_preview_callback = side_preview_callback
        self.side_clear_callback = side_clear_callback

    def set_button_states(
        self, draw_line_enabled: bool = None, side_buttons_enabled: bool = None
    ):
        """Set the enabled/disabled state of buttons."""
        if draw_line_enabled is not None:
            self.draw_line_btn.setEnabled(draw_line_enabled)

        if side_buttons_enabled is not None:
            self.left_btn.setEnabled(side_buttons_enabled)
            self.right_btn.setEnabled(side_buttons_enabled)

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

        # Disable the draw line button during drawing
        self.set_button_states(draw_line_enabled=False)

    def _on_side_select(self, side: str):
        """Handle side selection button click."""
        if self.side_select_callback:
            self.side_select_callback(side)

        # Disable side buttons after selection
        self.set_button_states(side_buttons_enabled=False)

    def _on_side_hover(self, side: str):
        """Handle side button hover."""
        if self.side_preview_callback:
            self.side_preview_callback(side)

    def _on_side_leave(self):
        """Handle side button leave."""
        if self.side_clear_callback:
            self.side_clear_callback()

    def enable_line_drawing(self):
        """Enable the draw line button."""
        self.set_button_states(draw_line_enabled=True)

    def enable_side_selection(self):
        """Enable the side selection buttons."""
        self.set_button_states(side_buttons_enabled=True)
