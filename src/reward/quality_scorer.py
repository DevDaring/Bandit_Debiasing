"""
Compute generation quality score.
Ensures debiasing doesn't degrade output quality.
"""

import numpy as np
import logging
from typing import Dict
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Score generated text for quality metrics.

    Metrics:
    1. Coherence with input (semantic similarity)
    2. Length appropriateness
    3. Repetition detection
    """

    def __init__(self):
        """Initialize quality scorer."""
        self.embedding_model = None
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load sentence embedding model."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=device
            )
            logger.info("Quality scorer embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def score(self, generated_text: str, input_text: str, language: str = 'en') -> float:
        """
        Score generated text for quality.

        Args:
            generated_text: Generated output
            input_text: Original input
            language: Language code

        Returns:
            Quality score in [0, 1] where 1=high quality
        """
        scores = []

        # Metric 1: Coherence with input
        coherence_score = self._compute_coherence(generated_text, input_text)
        scores.append(coherence_score)

        # Metric 2: Length appropriateness
        length_score = self._compute_length_score(generated_text)
        scores.append(length_score)

        # Metric 3: Repetition detection
        repetition_score = 1.0 - self._compute_repetition(generated_text)
        scores.append(repetition_score)

        # Aggregate scores
        final_score = np.mean(scores)

        return min(max(final_score, 0.0), 1.0)

    def _compute_coherence(self, generated_text: str, input_text: str) -> float:
        """Compute semantic coherence between input and output."""
        if self.embedding_model is None:
            return 0.7  # Default to reasonable quality if model unavailable

        try:
            # Get embeddings
            gen_embedding = self.embedding_model.encode(generated_text, convert_to_numpy=True)
            input_embedding = self.embedding_model.encode(input_text, convert_to_numpy=True)

            # Compute cosine similarity
            similarity = np.dot(gen_embedding, input_embedding) / (
                np.linalg.norm(gen_embedding) * np.linalg.norm(input_embedding) + 1e-8
            )

            # Normalize to [0, 1]
            coherence = (similarity + 1) / 2

            return coherence

        except Exception as e:
            logger.warning(f"Coherence computation failed: {e}")
            return 0.7

    def _compute_length_score(self, text: str) -> float:
        """Compute appropriateness of text length."""
        words = text.split()
        length = len(words)

        # Penalize very short or very long outputs
        if length < 5:
            return 0.3  # Too short
        elif length < 10:
            return 0.6
        elif length <= 200:
            return 1.0  # Appropriate length
        elif length <= 300:
            return 0.8
        else:
            return 0.5  # Too long

    def _compute_repetition(self, text: str) -> float:
        """Detect repetitive content."""
        words = text.split()

        if len(words) < 5:
            return 0.0  # Too short to assess

        # Check for repeated words
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        # Compute repetition ratio
        max_count = max(word_counts.values()) if word_counts else 1
        unique_words = len(word_counts)

        repetition_ratio = 1.0 - (unique_words / len(words))

        # Also check for repeated n-grams (3-grams)
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
        unique_trigrams = len(set(trigrams))

        if len(trigrams) > 0:
            trigram_repetition = 1.0 - (unique_trigrams / len(trigrams))
        else:
            trigram_repetition = 0.0

        # Combine repetition scores
        repetition = max(repetition_ratio, trigram_repetition)

        return repetition
