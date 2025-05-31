#!/usr/bin/env python3
"""
People Counter Application - PyQt Version Entry Point

A high-performance people counting application with video processing,
line drawing interface, and real-time histogram visualization using PyQt.
"""

import sys

from loguru import logger
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

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

        logger.info("People Counter application started successfully")

        # Start the application event loop
        sys.exit(app.exec_())

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
