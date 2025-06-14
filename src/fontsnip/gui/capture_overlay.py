# -*- coding: utf-8 -*-
"""
src/fontsnip/gui/capture_overlay.py

Defines the CaptureOverlay widget for the screen capture process.

This module provides a PyQt6 QWidget that creates a borderless, full-screen,
semi-transparent window. It handles mouse events to allow the user to select
a rectangular region of the screen, providing live visual feedback. Upon
selection, it uses the 'mss' library to capture the screen area and emits a
signal with the resulting image data as a NumPy array.
"""

import logging
import mss
import numpy as np
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QCursor


class CaptureOverlay(QWidget):
    """
    A full-screen, semi-transparent overlay for selecting a screen region.

    This widget covers the entire screen, allowing the user to draw a
    rectangle to select an area. Once the selection is made (on mouse
    release), it captures the selected area using mss and emits a signal
    containing the image data as a NumPy array.
    """
    # Signal emitted when a screen region is successfully captured.
    # The payload is the captured image as a NumPy array (in BGRA format from mss).
    image_captured = pyqtSignal(np.ndarray)

    def __init__(self):
        """Initializes the capture overlay widget."""
        super().__init__()
        logging.info("Initializing CaptureOverlay.")

        # Get screen geometry to make the widget full-screen
        screen = QApplication.primaryScreen()
        if not screen:
            logging.error("No primary screen found. Cannot initialize CaptureOverlay.")
            # Fallback to a default size if no screen is available
            self.setGeometry(0, 0, 800, 600)
        else:
            self.setGeometry(screen.geometry())

        # Set window flags for a borderless, stay-on-top overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Prevents it from appearing in the taskbar
        )

        # Set attributes for transparency and to ensure it's deleted on close
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Set the cursor to a crosshair
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        # Variables to store the selection rectangle coordinates
        self.begin = QPoint()
        self.end = QPoint()
        self.is_selecting = False

        # mss instance for screen capturing
        self.sct = mss.mss()

    def showEvent(self, event):
        """Ensure the widget is active and ready when shown."""
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        logging.debug("CaptureOverlay shown and activated.")

    def paintEvent(self, event):
        """Draws the semi-transparent overlay and the selection rectangle."""
        painter = QPainter(self)

        # Draw the semi-transparent black overlay
        overlay_color = QColor(0, 0, 0, 120)  # Black with ~50% opacity
        painter.fillRect(self.rect(), QBrush(overlay_color))

        # If selecting, draw the selection rectangle
        if self.is_selecting:
            selection_rect = QRect(self.begin, self.end).normalized()

            # Clear the area inside the selection rectangle to make it fully visible
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(selection_rect, Qt.BrushStyle.SolidPattern)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Draw a border around the selection rectangle for better visibility
            pen = QPen(QColor(50, 150, 255, 220), 1, Qt.PenStyle.SolidLine)  # A light blue border
            painter.setPen(pen)
            painter.drawRect(selection_rect)

    def mousePressEvent(self, event):
        """Starts the selection process when the left mouse button is pressed."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin = event.position().toPoint()
            self.end = self.begin
            self.is_selecting = True
            self.update()  # Trigger a repaint
            logging.debug(f"Selection started at: {self.begin.x()},{self.begin.y()}")

    def mouseMoveEvent(self, event):
        """Updates the selection rectangle as the mouse moves."""
        if self.is_selecting:
            self.end = event.position().toPoint()
            self.update()  # Trigger a repaint to show live feedback

    def mouseReleaseEvent(self, event):
        """Finalizes the selection, captures the screen, and closes the overlay."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            logging.debug(f"Selection ended at: {self.end.x()},{self.end.y()}")

            # Immediately hide the overlay to return control to the user
            self.hide()

            # Create the final rectangle and ensure it has a valid size
            selection_rect = QRect(self.begin, self.end).normalized()
            if selection_rect.width() > 1 and selection_rect.height() > 1:
                self.capture_screen(selection_rect)
            else:
                logging.warning("Selection was too small, capture aborted.")

            # Ensure the widget is closed and resources are released
            self.close()

    def keyPressEvent(self, event):
        """Allows the user to cancel the capture with the Escape key."""
        if event.key() == Qt.Key.Key_Escape:
            logging.info("Capture cancelled by user (Escape key).")
            self.is_selecting = False
            self.close()

    def capture_screen(self, rect: QRect):
        """
        Captures the specified rectangular area of the screen using mss.

        Args:
            rect (QRect): The rectangle defining the area to capture.
        """
        try:
            # Define the monitor and coordinates for mss
            # rect.left() etc. are in logical coordinates, which for the primary
            # screen should map directly to physical pixels.
            monitor = {
                "top": rect.top(),
                "left": rect.left(),
                "width": rect.width(),
                "height": rect.height(),
            }
            logging.info(f"Capturing screen area: {monitor}")

            # Grab the data
            sct_img = self.sct.grab(monitor)

            # Convert to a NumPy array
            # mss returns a BGRA image, which is what OpenCV expects (BGR)
            # if we ignore the alpha channel.
            img_array = np.array(sct_img)

            logging.info(f"Image captured with shape: {img_array.shape}")
            self.image_captured.emit(img_array)

        except mss.exception.ScreenShotError as e:
            logging.error(f"Failed to capture screen: {e}")
        finally:
            # The sct instance is managed by the class instance.
            # No need to close it here if we plan to reuse it.
            pass


if __name__ == '__main__':
    # A simple test case to run the overlay independently.
    import sys

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    app = QApplication(sys.argv)
    overlay = CaptureOverlay()

    def on_capture(image_data: np.ndarray):
        """A slot to handle the captured image for testing."""
        print(f"Signal received! Image data shape: {image_data.shape}, dtype: {image_data.dtype}")
        # For testing, let's try to save the image.
        try:
            from PIL import Image
            # mss gives BGRA, Pillow needs RGBA for saving with transparency.
            # The channel order is already correct for this conversion.
            img = Image.fromarray(image_data, 'RGBA')
            save_path = "capture_test.png"
            img.save(save_path)
            print(f"Captured image saved as '{save_path}'")
        except ImportError:
            print("Pillow is not installed. Cannot save image for verification.")
        except Exception as e:
            print(f"Error saving image: {e}")
        finally:
            app.quit()

    overlay.image_captured.connect(on_capture)
    # If the capture is cancelled (e.g., Esc key), the app should also quit.
    overlay.destroyed.connect(app.quit)
    overlay.show()
    sys.exit(app.exec())
```