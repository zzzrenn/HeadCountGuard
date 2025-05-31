#!/usr/bin/env python3
"""
People Counter Application - PyQt Version Entry Point

A high-performance people counting application with video processing,
line drawing interface, and real-time histogram visualization using PyQt.
"""

import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui.main_window import PeopleCounterApp


def main():
    """Main application entry point."""
    try:
        # Create PyQt application
        app = QApplication(sys.argv)
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        # Create the main window
        main_window = PeopleCounterApp()
        main_window.show()

        # Start the application event loop
        sys.exit(app.exec_())

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
