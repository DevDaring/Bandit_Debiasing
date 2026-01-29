"""
Language detection for multilingual inputs.
Returns language code and confidence score.
Supports: English (en), Hindi (hi), Bengali (bn)
"""

import logging
from typing import Dict, List
import langdetect
from langdetect import detect_langs

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detect language of input text.

    Uses langdetect library for detection.
    Returns one-hot encoding for en/hi/bn/other.
    """

    def __init__(self):
        """Initialize language detector."""
        self.target_languages = ["en", "hi", "bn"]
        langdetect.DetectorFactory.seed = 0  # For reproducibility

    def detect(self, text: str) -> Dict:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Dict with 'language_code', 'confidence', 'one_hot_vector'
        """
        if not text or len(text.strip()) < 3:
            return {"language_code": "other", "confidence": 0.0, "one_hot_vector": [0, 0, 0, 1]}

        try:
            # Detect language with confidence
            lang_probs = detect_langs(text)

            # Get most probable language
            top_lang = lang_probs[0]
            language_code = top_lang.lang
            confidence = top_lang.prob

            # Map to one-hot vector
            one_hot = self._to_one_hot(language_code)

            return {"language_code": language_code, "confidence": confidence, "one_hot_vector": one_hot}

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {"language_code": "other", "confidence": 0.0, "one_hot_vector": [0, 0, 0, 1]}

    def _to_one_hot(self, language_code: str) -> List[float]:
        """
        Convert language code to one-hot vector.

        Order: [en, hi, bn, other]
        """
        if language_code == "en":
            return [1.0, 0.0, 0.0, 0.0]
        elif language_code == "hi":
            return [0.0, 1.0, 0.0, 0.0]
        elif language_code == "bn":
            return [0.0, 0.0, 1.0, 0.0]
        else:
            return [0.0, 0.0, 0.0, 1.0]

    def get_feature_vector(self, text: str) -> List[float]:
        """
        Get feature vector for bandit context.

        Returns:
            4-dim vector [en, hi, bn, other]
        """
        result = self.detect(text)
        return result["one_hot_vector"]
