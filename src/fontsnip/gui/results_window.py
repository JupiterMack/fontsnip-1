# -*- coding: utf-8 -*-
"""
src/fontsnip/gui/results_window.py

Defines the ResultsWindow widget for displaying font matching results.

This module provides a small, temporary PyQt6 QWidget that appears near the
snipped area. It shows the top font matches and is designed to be
non-intrusive, closing automatically or on user interaction.
"""

import sys
from typing import List

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QFrame
from PyQt6.QtCore import Qt, QRect, QTimer
from PyQt6.QtGui import QFont, QMouseEvent

# Configuration for the results window
WINDOW_TIMEOUT_MS = 5000  # 5 seconds
WINDOW_OFFSET_Y = 10      # Pixels below the snip to show the window


class ResultsWindow(QWidget):
    """
    A temporary, frameless window to display the top font matches.
    """

    def __init__(self, matches: List[str], snip_rect: QRect, parent: QWidget = None):
        """
        Initializes the results window.

        Args:
            matches (List[str]): A list of font name strings, with the best
                                 match first.
            snip_rect (QRect): The geometry of the screen capture, used for
                               positioning the window.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        if not matches:
            # Nothing to show, so don't bother creating the window.
            # This can be handled by the caller, but it's a good safeguard.
            return

        self.matches = matches
        self.snip_rect = snip_rect

        self._setup_window_properties()
        self._setup_ui()
        self._position_window()

        # Set up a timer to automatically close the window
        self.close_timer = QTimer(self)
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.close)
        self.close_timer.start(WINDOW_TIMEOUT_MS)

    def _setup_window_properties(self):
        """Sets the window flags and styling."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.ToolTip  # ToolTip hint helps it behave like a temporary popup
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #E0E0E0;
                border: 1px solid #555555;
                border-radius: 5px;
                font-family: sans-serif;
            }
        """)

    def _setup_ui(self):
        """Creates and arranges the widgets within the window."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(5)

        # Top Match Label (most prominent)
        top_match_label = QLabel(self.matches[0])
        top_match_font = QFont()
        top_match_font.setBold(True)
        top_match_font.setPointSize(11)
        top_match_label.setFont(top_match_font)
        top_match_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(top_match_label)

        # Clipboard Info Label
        clipboard_label = QLabel("(Copied to clipboard)")
        clipboard_font = QFont()
        clipboard_font.setPointSize(8)
        clipboard_font.setItalic(True)
        clipboard_label.setFont(clipboard_font)
        clipboard_label.setStyleSheet("color: #AAAAAA; padding-bottom: 4px;")
        layout.addWidget(clipboard_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #555;")
        layout.addWidget(separator)

        # Other Matches
        for i, match in enumerate(self.matches[1:3], start=2):
            match_label = QLabel(f"{i}. {match}")
            match_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(match_label)

        self.setLayout(layout)

    def _position_window(self):
        """
        Positions the window just below the original snip rectangle,
        ensuring it stays within the screen boundaries.
        """
        screen_geometry = QApplication.primaryScreen().availableGeometry()

        # Adjust size to content
        self.adjustSize()
        window_size = self.size()

        # Initial position: below the snip
        pos_x = self.snip_rect.x()
        pos_y = self.snip_rect.bottom() + WINDOW_OFFSET_Y

        # Adjust X position to prevent going off-screen right
        if pos_x + window_size.width() > screen_geometry.right():
            pos_x = screen_geometry.right() - window_size.width()

        # Adjust X position to prevent going off-screen left
        if pos_x < screen_geometry.left():
            pos_x = screen_geometry.left()

        # Adjust Y position to prevent going off-screen bottom
        # If it goes off the bottom, try placing it above the snip instead
        if pos_y + window_size.height() > screen_geometry.bottom():
            pos_y = self.snip_rect.top() - window_size.height() - WINDOW_OFFSET_Y

        # Final check for top of screen
        if pos_y < screen_geometry.top():
            pos_y = screen_geometry.top()

        self.move(pos_x, pos_y)

    def mousePressEvent(self, event: QMouseEvent):
        """Closes the window when the user clicks anywhere on it."""
        self.close()
        event.accept()


if __name__ == '__main__':
    # This block allows for standalone testing of the ResultsWindow widget.
    app = QApplication(sys.argv)

    # --- Test Case 1: Snip in the middle of the screen ---
    dummy_matches_1 = ["Arial", "Helvetica Neue", "Roboto"]
    screen = QApplication.primaryScreen().geometry()
    dummy_snip_rect_1 = QRect(
        screen.width() // 2 - 150,
        screen.height() // 2 - 50,
        300,
        100
    )
    results_win_1 = ResultsWindow(matches=dummy_matches_1, snip_rect=dummy_snip_rect_1)
    results_win_1.show()

    # --- Test Case 2: Snip near the bottom-right corner to test repositioning ---
    dummy_matches_2 = ["Times New Roman", "Georgia", "Garamond"]
    dummy_snip_rect_2 = QRect(
        screen.width() - 350,
        screen.height() - 150,
        300,
        100
    )
    results_win_2 = ResultsWindow(matches=dummy_matches_2, snip_rect=dummy_snip_rect_2)
    results_win_2.show()

    # --- Test Case 3: Snip near the top-left corner ---
    dummy_matches_3 = ["Courier New", "Consolas", "Lucida Console"]
    dummy_snip_rect_3 = QRect(20, 20, 300, 100)
    results_win_3 = ResultsWindow(matches=dummy_matches_3, snip_rect=dummy_snip_rect_3)
    results_win_3.show()

    sys.exit(app.exec())
```