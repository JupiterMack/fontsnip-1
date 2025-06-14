# -*- coding: utf-8 -*-
"""
The Core Processing Package for FontSnip.

This package encapsulates the primary logic of the FontSnip application,
handling the pipeline from screen capture to font matching. It is designed
to be a modular and self-contained unit that can be orchestrated by the
main application controller (`app.py`).

Modules within this package will include:
- `capture`: Manages the screen capture overlay and grabs pixel data.
- `image_processing`: Contains the OCR and image preprocessing pipeline.
- `feature_extractor`: Defines the logic for extracting feature vectors from glyphs.
- `font_matcher`: Handles loading the font database and finding the best font matches.
"""

# By importing key functions or classes here, we create a simplified public API
# for the 'core' package. This allows other parts of the application, like
# the main app controller, to import them directly from `fontsnip.core`
# instead of needing to know the internal module structure.

# The following imports are placeholders to illustrate the intended architecture.
# They will be implemented as the corresponding modules are developed.

# from .capture import ScreenCapturer
# from .image_processing import ImageProcessor
# from .feature_extractor import FeatureExtractor
# from .font_matcher import FontMatcher

# __all__ = [
#     "ScreenCapturer",
#     "ImageProcessor",
#     "FeatureExtractor",
#     "FontMatcher",
# ]