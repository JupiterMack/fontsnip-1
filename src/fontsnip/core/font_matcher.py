# -*- coding: utf-8 -*-
"""
src/fontsnip/core/font_matcher.py

This module defines the FontMatcher class, responsible for comparing a captured
text's features against a pre-computed font database to find the best matches.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# Assuming feature_extractor is in the same 'core' package
from .feature_extractor import extract_features

# Configure logging
logger = logging.getLogger(__name__)


class FontMatcher:
    """
    Handles loading a font feature database and matching new font snippets
    against it.
    """

    def __init__(self, db_path: Path):
        """
        Initializes the FontMatcher by loading the pre-computed font database.

        Args:
            db_path (Path): The path to the font features database file
                            (e.g., 'font_features.pkl').
        """
        self.db_path = db_path
        self.font_database: Optional[Dict[str, np.ndarray]] = self._load_database()

    def _load_database(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Loads the font feature database from the specified pickle file.

        Returns:
            A dictionary mapping font names to their feature vectors, or None
            if the database file cannot be found or loaded.
        """
        if not self.db_path.exists():
            logger.error(f"Font database not found at '{self.db_path}'.")
            logger.error("Please run the 'scripts/build_font_database.py' script first.")
            return None
        try:
            with open(self.db_path, "rb") as f:
                database = pickle.load(f)
                logger.info(f"Successfully loaded font database with {len(database)} fonts.")
                return database
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            logger.error(f"Error loading or parsing font database '{self.db_path}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading font database: {e}")
            return None

    @staticmethod
    def _calculate_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates the cosine distance between two vectors.
        Distance = 1 - Cosine Similarity.
        A smaller distance indicates higher similarity.

        Args:
            vec_a (np.ndarray): The first vector.
            vec_b (np.ndarray): The second vector.

        Returns:
            The cosine distance, a float between 0.0 and 2.0.
            Returns 1.0 if either vector has a zero norm to avoid division by zero.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            # If one vector is a zero vector, they are dissimilar.
            # Cosine similarity is undefined, but distance can be treated as maximal (or neutral).
            # Returning 1.0 is a safe neutral value.
            return 1.0

        similarity = dot_product / (norm_a * norm_b)

        # Clamp similarity to [-1, 1] to handle potential floating point inaccuracies
        similarity = np.clip(similarity, -1.0, 1.0)

        # Distance is 1 - similarity. Ranges from 0 (identical) to 2 (opposite).
        return 1.0 - similarity

    def find_best_matches(
        self,
        character_glyphs: List[np.ndarray],
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Finds the top N best font matches for a list of character glyphs.

        Args:
            character_glyphs (List[np.ndarray]): A list of preprocessed, binarized
                                                 images of individual characters.
            top_n (int): The number of top matches to return.

        Returns:
            A list of tuples, where each tuple contains a font name and its
            distance score (lower is better). The list is sorted by score.
            Returns an empty list if no matches can be found or if the
            database is not loaded.
        """
        if not self.font_database:
            logger.warning("Font database is not loaded. Cannot perform matching.")
            return []

        # 1. Extract features for each glyph from the user's snip
        snip_features = [extract_features(glyph) for glyph in character_glyphs]
        valid_features = [f for f in snip_features if f is not None]

        if not valid_features:
            logger.warning("Could not extract any valid features from the provided glyphs.")
            return []

        # 2. Compute the average feature vector for the snip
        target_vector = np.mean(valid_features, axis=0)
        logger.debug(f"Computed target feature vector: {target_vector}")

        # 3. Calculate the distance to every font in the database
        distances = []
        for font_name, db_vector in self.font_database.items():
            # Ensure vectors have the same dimension
            if target_vector.shape != db_vector.shape:
                logger.warning(f"Shape mismatch between target vector and DB vector for font '{font_name}'. Skipping.")
                continue

            distance = self._calculate_cosine_distance(target_vector, db_vector)
            distances.append((font_name, distance))

        # 4. Sort by distance (ascending) and return the top N
        distances.sort(key=lambda item: item[1])

        return distances[:top_n]
```