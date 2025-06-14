# -*- coding: utf-8 -*-
"""
src/fontsnip/core/image_processor.py

Implements the image processing pipeline as described in State 3 of the
project architecture. This module is responsible for taking a raw image capture,
preprocessing it for clarity, and running it through an OCR engine to
extract character data.
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import easyocr
import numpy as np

# --- Module-level Configuration ---
# These parameters can be fine-tuned for performance and accuracy.
# They could be moved to a central config file in a future iteration.

# Factor by which to upscale the image before OCR. Helps with small text.
UPSCALE_FACTOR = 2

# Confidence threshold for OCR results. Characters below this are discarded.
OCR_CONFIDENCE_THRESHOLD = 0.60

# Parameters for cv2.adaptiveThreshold.
# Block size must be an odd number. It's the size of the pixel neighborhood
# used to calculate a threshold value.
ADAPTIVE_THRESH_BLOCK_SIZE = 15
# A constant subtracted from the mean or weighted mean.
ADAPTIVE_THRESH_C = 7


class ImageProcessor:
    """
    Encapsulates the entire image processing pipeline from a raw screen
    capture to recognized character data.

    This class holds the EasyOCR reader instance to avoid reloading the model
    on every call, which is a significant performance optimization.
    """

    def __init__(self, languages: List[str] = None):
        """
        Initializes the ImageProcessor, which includes loading the EasyOCR model.
        This can be time-consuming, so the instance should be created once and
        reused throughout the application's lifecycle.

        Args:
            languages (List[str]): A list of language codes for EasyOCR to use.
                                   Defaults to ['en'].
        """
        if languages is None:
            languages = ['en']

        logging.info(f"Initializing EasyOCR Reader for languages: {languages}...")
        # gpu=False is a safe default for cross-platform desktop apps
        # where CUDA might not be available or configured.
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
            logging.info("EasyOCR Reader initialized successfully.")
        except Exception as e:
            logging.critical(f"Failed to initialize EasyOCR Reader: {e}")
            logging.critical("Please ensure you have the necessary model files and dependencies.")
            # In a real app, this might trigger a user-facing error message.
            self.reader = None

    def process_image(self, raw_screenshot_np: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executes the full image processing pipeline on a raw screenshot.

        Args:
            raw_screenshot_np (np.ndarray): The raw image data from mss,
                                            expected in BGRA format.

        Returns:
            A tuple containing:
            - processed_image (np.ndarray): The final black and white (binarized)
                                            image used for feature extraction.
            - filtered_results (List[Dict[str, Any]]): A list of dictionaries,
                                                      where each dictionary contains
                                                      the bounding box, recognized
                                                      text, and confidence for a
                                                      character.
        """
        if self.reader is None:
            logging.error("ImageProcessor is not initialized. Cannot process image.")
            return np.array([]), []

        if raw_screenshot_np is None or raw_screenshot_np.size == 0:
            logging.warning("process_image called with an empty image.")
            return np.array([]), []

        # 1. Convert BGRA from mss to BGR for OpenCV
        # mss provides a 4-channel image (BGRA), but most OpenCV functions
        # work with 3-channel (BGR) or 1-channel (grayscale). We drop the alpha channel.
        bgr_image = cv2.cvtColor(raw_screenshot_np, cv2.COLOR_BGRA2BGR)

        # --- Preprocessing for OCR ---

        # 2. Upscaling
        # Improves OCR accuracy, especially on small text. INTER_CUBIC is a
        # high-quality interpolation method that preserves edges better than linear.
        h, w, _ = bgr_image.shape
        upscaled_image = cv2.resize(
            bgr_image,
            (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR),
            interpolation=cv2.INTER_CUBIC
        )

        # 3. Grayscale Conversion
        gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

        # (Optional) Denoising - can be useful for noisy sources
        # gray_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)

        # 4. Binarization
        # Adaptive thresholding is crucial for handling non-uniform backgrounds.
        # It calculates a threshold for small regions of the image, making it
        # robust to lighting changes. We invert the threshold (THRESH_BINARY_INV)
        # so that the text becomes white (255) and the background black (0),
        # which is a common convention for contour and feature analysis.
        binarized_image = cv2.adaptiveThreshold(
            gray_image,
            255,  # Max value to assign
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Method to calculate threshold
            cv2.THRESH_BINARY_INV,  # Invert the threshold (text becomes white)
            ADAPTIVE_THRESH_BLOCK_SIZE,
            ADAPTIVE_THRESH_C
        )

        # --- Optical Character Recognition (OCR) ---

        # 5. Perform OCR
        # We use `readtext` which returns bounding boxes, text, and confidence.
        # `paragraph=False` treats the snip as a single line or disparate
        # characters, which is more common for this use case.
        try:
            # The `detail=1` parameter ensures we get the full output with coordinates.
            ocr_results = self.reader.readtext(
                binarized_image,
                detail=1,
                paragraph=False,
            )
        except Exception as e:
            logging.error(f"An error occurred during OCR processing: {e}")
            return binarized_image, []

        # 6. Filter results based on confidence
        filtered_results = []
        for (bbox, text, conf) in ocr_results:
            if conf >= OCR_CONFIDENCE_THRESHOLD:
                # The bbox from easyocr is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                result_dict = {
                    'bbox': bbox,
                    'text': text,
                    'confidence': conf
                }
                filtered_results.append(result_dict)
                logging.debug(f"Accepted char: '{text}' with confidence {conf:.2f}")
            else:
                logging.debug(f"Rejected char: '{text}' with confidence {conf:.2f}")

        if not filtered_results:
            logging.warning("OCR did not find any characters with sufficient confidence.")

        # The feature extractor will need the binarized image to crop characters from.
        return binarized_image, filtered_results


if __name__ == '__main__':
    # This block is for demonstration and testing purposes.
    # It will not run when imported as a module. To run, execute:
    # python -m src.fontsnip.core.image_processor
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a dummy image for testing (white text on a gray background)
    width, height = 400, 100
    dummy_image_bgr = np.full((height, width, 3), 128, dtype=np.uint8)  # Gray background
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dummy_image_bgr, 'Test Font 123', (20, 60), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert to BGRA to simulate mss input
    dummy_image_bgra = cv2.cvtColor(dummy_image_bgr, cv2.COLOR_BGR2BGRA)

    print("Initializing ImageProcessor for testing...")
    processor = ImageProcessor(languages=['en'])

    if processor.reader:
        print("\nProcessing dummy image...")
        processed_img, results = processor.process_image(dummy_image_bgra)

        print(f"\nFound {len(results)} characters with confidence >= {OCR_CONFIDENCE_THRESHOLD}")
        for res in results:
            print(f"  - Text: '{res['text']}', Confidence: {res['confidence']:.4f}")

        # Display the processed image for visual verification
        if processed_img.size > 0:
            cv2.imshow('Original Dummy Image (BGR)', dummy_image_bgr)
            cv2.imshow('Binarized and Upscaled Image', processed_img)
            print("\nDisplaying processed image. Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("\nCould not run test because ImageProcessor failed to initialize.")