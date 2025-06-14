import os
import sys
import platform
import string
import pickle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- Path Setup ---
# This script is intended to be run from the project root directory.
# We add the project root to the Python path to allow importing from the 'src' package.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except NameError:
    # Fallback for environments where __file__ is not defined
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))


# --- Project-specific Import ---
# The core feature extraction logic is defined in a separate module to be shared
# with the main application. This ensures that the features generated for the
# database are calculated in the exact same way as the features from a user's snip.
try:
    from src.fontsnip.core.feature_extractor import extract_features_from_char_image
except ImportError:
    # If the script is run standalone or the module doesn't exist yet, this block
    # will be executed. It provides a dummy function to illustrate the expected
    # interface and allows the script to run for structural validation.
    # In the final, integrated project, this 'except' block should not be hit.
    print("WARNING: Could not import 'extract_features_from_char_image' from 'src.fontsnip.core'.")
    print("Using a dummy placeholder function. The generated database will NOT be functional.")
    print("Please ensure 'src/fontsnip/core/feature_extractor.py' exists and is correct.")

    def extract_features_from_char_image(image: np.ndarray) -> np.ndarray | None:
        """
        Dummy feature extractor. The real implementation is in src/fontsnip/core/
        This is a placeholder to make the script runnable for inspection.
        It extracts a minimal feature set: aspect ratio and pixel density.
        """
        if image is None or image.size == 0 or len(image.shape) != 2:
            return None
        h, w = image.shape
        if h == 0 or w == 0:
            return None

        # Feature 1: Aspect Ratio
        aspect_ratio = w / h

        # Feature 2: Pixel Density (ratio of black pixels to total pixels)
        # The image is binarized with black text (0) on a white background (255).
        pixel_density = np.sum(image == 0) / image.size

        return np.array([aspect_ratio, pixel_density])

# --- Constants ---
# The set of characters to render for each font to create its "fingerprint".
# Using a standard set ensures comparability across all fonts.
CHAR_SET = string.ascii_letters + string.digits

# The size (in points) to render the font. This should be a reasonably high
# resolution to capture detail, similar to what the OCR upscaling achieves.
FONT_SIZE = 48

# The directory to save the generated database file.
DATA_DIR = PROJECT_ROOT / "data"
# The final output file.
OUTPUT_DB_PATH = DATA_DIR / "font_features.pkl"


def get_system_font_paths() -> list[str]:
    """
    Scans system-specific directories to find all .ttf and .otf font files.

    Returns:
        A sorted list of absolute paths to found font files.
    """
    font_paths = set()
    system = platform.system()
    font_dirs = []

    if system == "Windows":
        # The 'windir' environment variable is the most reliable way to find C:\Windows
        win_dir = os.environ.get("windir", "C:/Windows")
        font_dirs.append(Path(win_dir) / "Fonts")
    elif system == "Darwin":  # macOS
        font_dirs.extend([
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library/Fonts",
        ])
    else:  # Linux and other Unix-like systems
        font_dirs.extend([
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".local/share/fonts",
            Path.home() / ".fonts",
        ])

    print(f"Scanning for fonts in the following directories for {system}:")
    for d in font_dirs:
        if d.exists():
            print(f"  - {d}")

    for font_dir in font_dirs:
        if font_dir.is_dir():
            # Recursively search for .ttf and .otf files
            for ext in ["*.ttf", "*.otf", "*.TTF", "*.OTF"]:
                for font_file in font_dir.rglob(ext):
                    font_paths.add(str(font_file))

    print(f"\nFound {len(font_paths)} unique font files.")
    return sorted(list(font_paths))


def render_char_image(font_path: str, char: str, font_size: int) -> np.ndarray | None:
    """
    Renders a single character using a given font file and returns it as a
    binarized, tightly-cropped NumPy array.

    Args:
        font_path: The absolute path to the .ttf or .otf font file.
        char: The character to render (e.g., 'A').
        font_size: The font size in points.

    Returns:
        A 2D NumPy array (uint8) representing the binarized image of the
        character (black text on white background), or None if the character
        cannot be rendered or has no visible glyph.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Pillow cannot open or read the font file.
        return None

    # Get the bounding box of the character to determine its exact size.
    # The box is (left, top, right, bottom).
    try:
        bbox = font.getbbox(char)
    except (TypeError, ValueError):
        # Some fonts might not have a glyph for a specific character.
        return None

    if bbox is None:
        return None

    left, top, right, bottom = bbox
    char_width = right - left
    char_height = bottom - top

    # If the character has no size (e.g., a space), it has no visual features.
    if char_width == 0 or char_height == 0:
        return None

    # Create an image perfectly sized for the character glyph.
    image = Image.new("L", (char_width, char_height), color=255)  # 'L' mode is 8-bit grayscale.
    draw = ImageDraw.Draw(image)

    # Draw the character. The position (-left, -top) aligns the character's
    # top-left corner of its bounding box to the (0, 0) of the new image.
    draw.text((-left, -top), char, font=font, fill=0)  # Black text (0) on white background (255)

    # Convert the Pillow image to a NumPy array. The image is already binarized.
    return np.array(image)


def main():
    """
    Main script execution function.
    1. Finds all system fonts.
    2. For each font, renders a standard character set.
    3. Extracts features for each rendered character image.
    4. Averages the features to create a single "fingerprint" vector for the font.
    5. Saves the resulting database (a dictionary mapping font paths to vectors) to a pickle file.
    """
    print("--- Starting Font Database Build Process ---")

    # Ensure the output directory exists
    DATA_DIR.mkdir(exist_ok=True)

    font_paths = get_system_font_paths()
    if not font_paths:
        print("\nERROR: No font files found. Cannot build the database.")
        sys.exit(1)

    font_database = {}
    total_fonts = len(font_paths)

    for i, font_path in enumerate(font_paths):
        font_name = Path(font_path).name
        print(f"\n[{i + 1}/{total_fonts}] Processing: {font_name}", flush=True)

        char_feature_vectors = []
        try:
            # A quick check to see if the font is loadable at all.
            ImageFont.truetype(font_path, 10)
        except Exception as e:
            print(f"  -> SKIPPING: Font cannot be loaded by Pillow. Reason: {e}")
            continue

        for char in CHAR_SET:
            # 1. Render character to a binarized, cropped image (NumPy array)
            char_image = render_char_image(font_path, char, FONT_SIZE)

            if char_image is None:
                continue  # Skip characters that can't be rendered

            # 2. Extract features from the rendered image using the shared function
            feature_vector = extract_features_from_char_image(char_image)

            if feature_vector is not None and feature_vector.size > 0:
                char_feature_vectors.append(feature_vector)

        # 3. If we have successfully processed characters, average their feature vectors
        if not char_feature_vectors:
            print(f"  -> SKIPPING: No valid characters could be rendered or processed for this font.")
            continue

        try:
            # Use np.vstack to create a 2D array, then calculate the mean along axis 0
            avg_feature_vector = np.mean(np.vstack(char_feature_vectors), axis=0)
        except Exception as e:
            print(f"  -> SKIPPING: Could not compute average vector. Reason: {e}")
            continue

        # 4. Store the result in our database dictionary
        # We use the full, absolute path as the key to guarantee uniqueness.
        font_database[font_path] = avg_feature_vector
        print(f"  -> Success. Processed {len(char_feature_vectors)} characters. Vector shape: {avg_feature_vector.shape}")

    if not font_database:
        print("\nERROR: Database is empty. No fonts were processed successfully.")
        sys.exit(1)

    # 5. Save the completed database to the specified pickle file
    print(f"\nSaving database with {len(font_database)} font entries to: {OUTPUT_DB_PATH}")
    try:
        with open(OUTPUT_DB_PATH, "wb") as f:
            pickle.dump(font_database, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("--- Font Database Build Complete! ---")
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to save database file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```