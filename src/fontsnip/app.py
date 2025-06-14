# -*- coding: utf-8 -*-
"""
src/fontsnip/app.py

Core application controller for FontSnip.

This module contains the main application class, `FontSnipApp`, which manages
the application's state machine, system tray icon, global hotkey listener,
and orchestrates the entire capture-process-match-display workflow.

For modularity, helper classes for capture, processing, and matching are
defined here but are intended to be moved to their own respective files.
"""

import sys
import os
import pickle
import threading
from pathlib import Path

# --- Core Dependencies ---
import numpy as np
import cv2
import mss
import easyocr
import pyperclip
from pynput import keyboard

# --- PyQt6 Dependencies ---
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QWidget
from PyQt6.QtGui import QIcon, QAction, QPainter, QPen, QColor, QCursor, QGuiApplication
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint

# --- Constants ---
APP_NAME = "FontSnip"
CURRENT_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_PATH.parent.parent
ASSETS_PATH = PROJECT_ROOT / "assets"
DATA_PATH = PROJECT_ROOT / "data"
ICON_FILE = ASSETS_PATH / "icon.png"
FONT_DB_FILE = DATA_PATH / "font_features.pkl"

# Hotkey configuration (Ctrl+Alt+S)
# Using a set for tracking pressed keys
HOTKEY_COMBINATION = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode.from_char('s')}
# A set to keep track of currently pressed keys
current_keys = set()


# =============================================================================
# Helper Class: Capture Overlay (Intended for src/fontsnip/capture_overlay.py)
# =============================================================================
class CaptureOverlay(QWidget):
    """
    A full-screen, semi-transparent overlay for selecting a screen region.
    """
    # Signal emitted when a region is successfully captured
    # Emits the captured image as a numpy array
    capture_finished = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_drawing = False

        # Get primary screen geometry
        primary_screen = QGuiApplication.primaryScreen()
        screen_geometry = primary_screen.geometry()
        self.setGeometry(screen_geometry)

        # Configure window properties for an overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Prevents it from appearing in the taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.is_drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.hide()  # Hide immediately for a responsive feel
            self.capture_screen()
            self.close() # Ensure the widget is destroyed

    def paintEvent(self, event):
        painter = QPainter(self)
        # Semi-transparent black overlay
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self.is_drawing:
            selection_rect = QRect(self.start_point, self.end_point).normalized()
            # Clear the selected area (make it fully transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(selection_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            # Draw a white border around the selection
            painter.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.SolidLine))
            painter.drawRect(selection_rect)

    def capture_screen(self):
        """Captures the selected region using mss."""
        selection_rect = QRect(self.start_point, self.end_point).normalized()

        # Adjust for screen geometry if multiple monitors are used
        screen_pos = self.mapToGlobal(QPoint(0,0))
        capture_x = screen_pos.x() + selection_rect.x()
        capture_y = screen_pos.y() + selection_rect.y()

        monitor = {
            "top": capture_y,
            "left": capture_x,
            "width": selection_rect.width(),
            "height": selection_rect.height(),
        }

        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            # Convert to NumPy array (BGRA to BGR)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.capture_finished.emit(img)


# =============================================================================
# Helper Class: Image Processor (Intended for src/fontsnip/image_processor.py)
# =============================================================================
class ImageProcessor:
    """
    Handles the image processing pipeline from raw capture to OCR data.
    """
    def __init__(self):
        # Initialize EasyOCR reader once to avoid model reloading
        # Using only English for now.
        print("Initializing EasyOCR... (This may take a moment)")
        self.reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if CUDA is available
        print("EasyOCR initialized.")

    def process_image(self, image_np: np.ndarray):
        """
        Applies preprocessing and runs OCR on the input image.
        Returns a list of tuples: (character_image, character_text, confidence)
        """
        if image_np.size == 0 or image_np.shape[0] < 10 or image_np.shape[1] < 10:
            print("Warning: Captured image is too small to process.")
            return []

        # 1. Upscaling
        scale_factor = 3
        width = int(image_np.shape[1] * scale_factor)
        height = int(image_np.shape[0] * scale_factor)
        upscaled = cv2.resize(image_np, (width, height), interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale Conversion
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # 3. Binarization (Adaptive Thresholding)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert image: OCR works best with black text on white background
        binary = cv2.bitwise_not(binary)

        # 4. Optical Character Recognition (OCR)
        # detail=1 provides character-level bounding boxes
        ocr_results = self.reader.readtext(binary, detail=1, paragraph=False)

        # 5. Filter and Isolate Characters
        char_data = []
        for (bbox, text, conf) in ocr_results:
            if conf > 0.6 and len(text) == 1 and text.isalnum(): # Filter for high-confidence single alphanumeric chars
                # Extract each character based on its bounding box
                (tl, tr, br, bl) = bbox
                x_min, y_min = int(tl[0]), int(tl[1])
                x_max, y_max = int(br[0]), int(br[1])
                
                # Crop the character from the binarized image
                char_image = binary[y_min:y_max, x_min:x_max]
                if char_image.size > 0:
                    char_data.append((char_image, text, conf))
        
        return char_data


# =============================================================================
# Helper Class: Font Matcher (Intended for src/fontsnip/font_matcher.py)
# =============================================================================
class FontMatcher:
    """
    Extracts features from characters and matches them against a font database.
    """
    def __init__(self, font_database: dict):
        self.font_database = font_database

    def _extract_features(self, char_image: np.ndarray) -> np.ndarray:
        """
        Computes a feature vector for a single character image.
        This logic MUST match the one in `scripts/build_font_database.py`.
        """
        h, w = char_image.shape
        if h == 0 or w == 0:
            return np.zeros(7) # Return a zero vector for empty images

        # 1. Aspect Ratio
        aspect_ratio = w / h

        # 2. Pixel Density
        pixel_density = np.sum(char_image == 0) / (w * h) # Black pixels (text)

        # 3. Centroid
        moments = cv2.moments(255 - char_image) # Moments work on white objects
        if moments["m00"] != 0:
            centroid_x = moments["m10"] / moments["m00"] / w
            centroid_y = moments["m01"] / moments["m00"] / h
        else:
            centroid_x, centroid_y = 0.5, 0.5 # Default to center

        # 4. Contour Analysis
        contours, _ = cv2.findContours(char_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_holes = 0
        total_perimeter = 0
        total_area = 0
        if contours:
            # The first contour is the external one
            c0 = contours[0]
            total_perimeter = cv2.arcLength(c0, True)
            total_area = cv2.contourArea(c0)
            # Holes are contours with parents (the external contour)
            # This is a simplified hole count
            num_holes = len(contours) - 1

        # Normalize perimeter and area
        norm_perimeter = total_perimeter / (2 * (w + h)) if (w + h) > 0 else 0
        norm_area = total_area / (w * h) if (w * h) > 0 else 0
        
        return np.array([
            aspect_ratio, pixel_density, centroid_x, centroid_y,
            num_holes, norm_perimeter, norm_area
        ])

    def find_best_matches(self, char_data: list) -> list:
        """
        Finds the best font matches for the given set of characters.
        """
        if not char_data:
            return []

        # Calculate the average feature vector for the snipped characters
        feature_vectors = [self._extract_features(img) for img, _, _ in char_data]
        target_vector = np.mean(feature_vectors, axis=0)

        # Compare with the database using Euclidean distance
        distances = []
        for font_name, db_vector in self.font_database.items():
            dist = np.linalg.norm(target_vector - db_vector)
            distances.append((font_name, dist))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])

        return distances[:5] # Return top 5 matches


# =============================================================================
# Main Application Class
# =============================================================================
class FontSnipApp:
    """
    The main application controller. Manages state, UI, and workflow.
    """
    # Signal to trigger capture from the hotkey thread
    trigger_capture = pyqtSignal()

    def __init__(self, app: QApplication):
        self.app = app
        self.is_capturing = False
        self.capture_overlay = None
        self.hotkey_listener = None

        # Load font database
        self.font_database = self.load_font_database()
        if not self.font_database:
            # No need for a message box, as the tray icon can show the error
            print("Error: Font database not found or failed to load.")
            # Allow app to run so tray icon can show error and quit option
        
        # Initialize core components
        self.image_processor = ImageProcessor()
        self.font_matcher = FontMatcher(self.font_database)

        # Setup UI and listeners
        self.setup_tray_icon()
        self.setup_hotkey_listener()
        
        # Connect the thread-safe signal to the slot
        self.trigger_capture.connect(self.start_capture_mode)

    def setup_tray_icon(self):
        """Creates and configures the system tray icon and its menu."""
        self.tray_icon = QSystemTrayIcon()

        if ICON_FILE.exists():
            self.tray_icon.setIcon(QIcon(str(ICON_FILE)))
        else:
            print(f"Warning: Icon file not found at {ICON_FILE}")
            # Use a default system icon if available
            self.tray_icon.setIcon(QIcon.fromTheme("applications-graphics"))

        self.tray_icon.setToolTip(f"{APP_NAME} - Press Ctrl+Alt+S to capture")
        
        menu = QMenu()
        
        # Action to trigger capture
        capture_action = QAction("Snip Font (Ctrl+Alt+S)", self.app)
        capture_action.triggered.connect(self.start_capture_mode)
        menu.addAction(capture_action)
        
        menu.addSeparator()
        
        # Quit action
        quit_action = QAction("Quit", self.app)
        quit_action.triggered.connect(self.quit_app)
        menu.addAction(quit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()

        if not self.font_database:
            self.tray_icon.showMessage(
                f"{APP_NAME} Error",
                f"Could not load font database from {FONT_DB_FILE}. Please run the build script.",
                QSystemTrayIcon.MessageIcon.Critical
            )

    def setup_hotkey_listener(self):
        """Initializes and starts the global hotkey listener."""
        self.hotkey_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.hotkey_listener.start()
        print(f"Hotkey listener started. Press {'+'.join(str(k).replace('Key.', '') for k in HOTKEY_COMBINATION)} to snip.")

    def _on_press(self, key):
        """Callback for key press events."""
        if key in HOTKEY_COMBINATION:
            current_keys.add(key)
            if all(k in current_keys for k in HOTKEY_COMBINATION):
                # Emit signal to trigger capture on the main GUI thread
                self.trigger_capture.emit()

    def _on_release(self, key):
        """Callback for key release events."""
        try:
            current_keys.remove(key)
        except KeyError:
            pass

    def start_capture_mode(self):
        """Creates and shows the capture overlay."""
        if self.is_capturing or not self.font_database:
            if not self.font_database:
                print("Cannot capture: Font database is not loaded.")
            return
            
        self.is_capturing = True
        self.capture_overlay = CaptureOverlay()
        self.capture_overlay.capture_finished.connect(self.on_capture_finished)
        self.capture_overlay.show()

    def on_capture_finished(self, image_data: np.ndarray):
        """Handles the captured image and starts the processing thread."""
        print("Capture finished. Processing image...")
        self.is_capturing = False # Reset state
        self.capture_overlay = None # Allow garbage collection
        
        # Run processing in a separate thread to keep the GUI responsive
        # For this simple case, direct call is okay, but threading is better practice
        # for long operations.
        self.process_and_match(image_data)

    def process_and_match(self, image_data: np.ndarray):
        """Orchestrates the image processing and font matching workflow."""
        char_data = self.image_processor.process_image(image_data)
        
        if not char_data:
            print("No valid characters found in the snip.")
            self.tray_icon.showMessage(
                "No Text Found",
                "Could not detect any valid characters in the selected area.",
                QSystemTrayIcon.MessageIcon.Warning
            )
            return

        print(f"Found {len(char_data)} characters. Matching against database...")
        matches = self.font_matcher.find_best_matches(char_data)
        
        if matches:
            self.show_results(matches)
        else:
            print("Could not find any matching fonts.")
            self.tray_icon.showMessage(
                "No Match Found",
                "Could not find a matching font in the database.",
                QSystemTrayIcon.MessageIcon.Information
            )

    def show_results(self, matches: list):
        """Displays the top font matches in a system notification."""
        top_match_name = Path(matches[0][0]).stem
        
        # Copy top match to clipboard
        try:
            pyperclip.copy(top_match_name)
            clipboard_msg = f"\n'{top_match_name}' copied to clipboard."
        except pyperclip.PyperclipException:
            clipboard_msg = "\nCould not copy to clipboard."
            print("Warning: pyperclip could not access the clipboard.")

        # Format results for display
        title = f"Font Found: {top_match_name}"
        message = "Top Matches:\n"
        for i, (font_path, dist) in enumerate(matches):
            font_name = Path(font_path).stem
            message += f"{i+1}. {font_name}\n"
        
        message += clipboard_msg
        
        print(f"--- Match Results ---\n{message}")

        self.tray_icon.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 4000)

    def load_font_database(self):
        """Loads the pre-computed font features from a pickle file."""
        if not FONT_DB_FILE.exists():
            print(f"Error: Font database file not found at {FONT_DB_FILE}")
            return None
        try:
            with open(FONT_DB_FILE, 'rb') as f:
                database = pickle.load(f)
            print(f"Font database loaded successfully with {len(database)} fonts.")
            return database
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            print(f"Error loading font database: {e}")
            return None

    def quit_app(self):
        """Stops all background threads and quits the application."""
        print("Quitting FontSnip...")
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        self.tray_icon.hide()
        self.app.quit()

```