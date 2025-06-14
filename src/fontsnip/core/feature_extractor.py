# -*- coding: utf-8 -*-
"""
src/fontsnip/core/feature_extractor.py

Defines functions for extracting a feature vector from an individual
character glyph image. This logic is a critical component used by both the
font matcher (for user-snipped characters) and the database builder (for
pre-rendered font characters).

The goal is to create a "fingerprint" of the glyph's shape that is
invariant to scale.
"""

import numpy as np
import cv2


def extract_features(glyph_image: np.ndarray) -> np.ndarray:
    """
    Computes a feature vector for a single character glyph image.

    The input image is expected to be a binarized, single-channel NumPy array
    where the glyph is white (255) and the background is black (0).

    The feature vector is designed to be scale-invariant and includes:
    1.  Aspect Ratio: width / height of the bounding box.
    2.  Pixel Density: Ratio of white pixels to total pixels.
    3.  Normalized Centroid (X): Center of mass x-coordinate, normalized by width.
    4.  Normalized Centroid (Y): Center of mass y-coordinate, normalized by height.
    5.  Number of Holes: Count of internal contours (e.g., in 'o', 'B').
    6.  Normalized Contour Area: Area of the largest contour, normalized by image area.
    7.  Normalized Contour Perimeter: Perimeter of the largest contour, normalized.

    Args:
        glyph_image (np.ndarray): A 2D NumPy array representing the binarized
                                  character glyph.

    Returns:
        np.ndarray: A 1D NumPy array containing the computed features. Returns
                    a zero vector if the image is invalid or empty.
    """
    # Ensure the image is a 2D array of type uint8, as required by OpenCV
    if glyph_image.ndim != 2 or glyph_image.dtype != np.uint8:
        try:
            if glyph_image.max() <= 1.0 and glyph_image.min() >= 0.0:  # If it's a boolean or 0-1 float array
                glyph_image = (glyph_image * 255).astype(np.uint8)
            else:
                glyph_image = glyph_image.astype(np.uint8)
        except (ValueError, TypeError):
            # Return a zero vector of the expected feature length
            return np.zeros(7, dtype=np.float32)

    h, w = glyph_image.shape

    # --- Basic Sanity Checks ---
    if h == 0 or w == 0 or np.sum(glyph_image) == 0:
        # Return a zero vector if the image is empty
        return np.zeros(7, dtype=np.float32)

    # --- Feature 1: Aspect Ratio ---
    aspect_ratio = w / h

    # --- Feature 2: Pixel Density ---
    # Total number of white pixels (the character) / total pixels in the image
    pixel_density = np.count_nonzero(glyph_image) / (w * h)

    # --- Features 3 & 4: Normalized Centroid ---
    moments = cv2.moments(glyph_image)
    if moments["m00"] != 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        norm_center_x = center_x / w
        norm_center_y = center_y / h
    else:
        # Should not happen if we passed the sum check, but for safety
        norm_center_x = 0.5
        norm_center_y = 0.5

    # --- Contour-based Features ---
    # Use RETR_TREE to get the full hierarchy, which is needed to count holes.
    contours, hierarchy = cv2.findContours(
        glyph_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # If no contours are found, return the features calculated so far
        # and zeros for the contour-based ones.
        return np.array([
            aspect_ratio,
            pixel_density,
            norm_center_x,
            norm_center_y,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

    # --- Feature 5: Number of Holes ---
    # A hole is an inner contour. The hierarchy is [Next, Previous, First_Child, Parent].
    # We count contours that have a parent (i.e., hierarchy[0][i][3] is not -1).
    num_holes = 0
    if hierarchy is not None and len(hierarchy) > 0:
        # The first dimension of hierarchy corresponds to the contours
        for i in range(len(hierarchy[0])):
            # If the parent index is valid, it's an inner contour (a hole).
            if hierarchy[0][i][3] != -1:
                num_holes += 1

    # --- Features 6 & 7: Normalized Area and Perimeter of the largest contour ---
    # Assume the largest contour is the main character outline.
    largest_contour = max(contours, key=cv2.contourArea)

    # Feature 6: Normalized Contour Area
    contour_area = cv2.contourArea(largest_contour)
    norm_contour_area = contour_area / (w * h)

    # Feature 7: Normalized Contour Perimeter
    contour_perimeter = cv2.arcLength(largest_contour, True)
    # Normalize by a factor that is scale-invariant, like the sum of dimensions
    norm_contour_perimeter = contour_perimeter / (2 * (w + h))

    # --- Assemble the final feature vector ---
    feature_vector = np.array([
        aspect_ratio,
        pixel_density,
        norm_center_x,
        norm_center_y,
        float(num_holes),
        norm_contour_area,
        norm_contour_perimeter
    ], dtype=np.float32)

    return feature_vector


if __name__ == '__main__':
    # This block is for testing the feature extractor directly.
    # We can create a few simple synthetic glyphs to test the logic.
    import os
    import platform
    from PIL import Image, ImageDraw, ImageFont

    print("--- Running Feature Extractor Test ---")

    def create_test_glyph(char, size=64):
        """Creates a binarized numpy array of a character."""
        image = Image.new("L", (size, size), "black")
        draw = ImageDraw.Draw(image)

        # Try to find a common font, fallback to default
        font_path = None
        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/arial.ttf"
        elif platform.system() == "Darwin":  # macOS
            font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        else:  # Linux
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        try:
            if not os.path.exists(font_path):
                raise IOError
            font = ImageFont.truetype(font_path, size - 10)
        except IOError:
            print(f"Warning: Test font not found at {font_path}. Using default.")
            font = ImageFont.load_default()

        # Get text bounding box to center the character
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        position = ((size - text_width) / 2 - bbox[0], (size - text_height) / 2 - bbox[1])
        draw.text(position, char, font=font, fill="white")

        # Convert to numpy and binarize
        np_image = np.array(image)
        _, binarized_image = cv2.threshold(np_image, 1, 255, cv2.THRESH_BINARY)

        # Crop to the bounding box of the character for a tight fit
        contours, _ = cv2.findContours(binarized_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Combine all contours to get the overall bounding box
            all_points = np.concatenate([c for c in contours])
            x, y, w, h = cv2.boundingRect(all_points)
            return binarized_image[y:y + h, x:x + w]
        return None

    # Test with a few characters that have different topological features
    test_chars = ['T', 'O', 'B', 'i', 'g']
    for char in test_chars:
        print(f"\n--- Character: '{char}' ---")
        glyph = create_test_glyph(char)
        if glyph is not None and glyph.size > 0:
            features = extract_features(glyph)
            print(f"Glyph Shape: {glyph.shape}")
            print(f"Feature Vector (len={len(features)}):")
            print(f"  1. Aspect Ratio:         {features[0]:.4f}")
            print(f"  2. Pixel Density:        {features[1]:.4f}")
            print(f"  3. Norm Centroid (X):    {features[2]:.4f}")
            print(f"  4. Norm Centroid (Y):    {features[3]:.4f}")
            print(f"  5. Num Holes:            {features[4]:.0f}")
            print(f"  6. Norm Contour Area:    {features[5]:.4f}")
            print(f"  7. Norm Contour Perim:   {features[6]:.4f}")

            # To visually inspect the glyphs during testing, uncomment below:
            # resized_glyph = cv2.resize(glyph, (128, 128), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow(f"Glyph '{char}'", resized_glyph)
            # cv2.waitKey(0)
        else:
            print("Could not generate glyph.")

    # cv2.destroyAllWindows()
    print("\n--- Test Complete ---")
```