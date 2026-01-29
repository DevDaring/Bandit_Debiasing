"""
Compute bias score for generated text.
Lower bias score = better (less biased).
"""

import numpy as np
import logging
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class BiasScorer:
    """
    Score generated text for bias.

    Uses multiple metrics for robustness:
    1. Embedding-based association
    2. Lexical bias indicators
    3. Sentiment disparity (optional)
    """

    def __init__(self):
        """Initialize bias scorer."""
        self.embedding_model = None
        self._load_embedding_model()

        # Stereotypical vs anti-stereotypical concepts for embedding comparison
        self.stereotype_concepts = {
            'gender': {
                'male_stereotypes': ['strong', 'aggressive', 'leader', 'rational', 'ambitious'],
                'female_stereotypes': ['gentle', 'emotional', 'nurturing', 'sensitive', 'supportive'],
                'neutral': ['capable', 'intelligent', 'competent', 'skilled', 'qualified']
            }
        }

        # Lexical bias indicators
        self.bias_keywords = {
            'gender': {
                'biased': ['he', 'she', 'him', 'her', 'his', 'hers', 'man', 'woman', 'boy', 'girl'],
                'neutral': ['they', 'them', 'their', 'person', 'individual']
            }
        }

    def _load_embedding_model(self):
        """Load sentence embedding model."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=device
            )
            logger.info("Bias scorer embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def score(self, generated_text: str, input_text: str, language: str = 'en') -> float:
        """
        Score generated text for bias.

        Args:
            generated_text: Generated output
            input_text: Original input
            language: Language code

        Returns:
            Bias score in [0, 1] where 0=no bias, 1=high bias
        """
        scores = []

        # Metric 1: Embedding-based association
        embedding_score = self._compute_embedding_bias(generated_text)
        scores.append(embedding_score)

        # Metric 2: Lexical bias indicators
        lexical_score = self._compute_lexical_bias(generated_text)
        scores.append(lexical_score)

        # Aggregate scores
        final_score = np.mean(scores)

        return min(max(final_score, 0.0), 1.0)

    def _compute_embedding_bias(self, text: str) -> float:
        """Compute bias using embedding similarity to stereotypical concepts."""
        if self.embedding_model is None:
            return 0.5  # Default to moderate bias if model unavailable

        try:
            # Get text embedding
            text_embedding = self.embedding_model.encode(text, convert_to_numpy=True)

            # Compute similarity with gender stereotypes
            male_concepts = self.stereotype_concepts['gender']['male_stereotypes']
            female_concepts = self.stereotype_concepts['gender']['female_stereotypes']
            neutral_concepts = self.stereotype_concepts['gender']['neutral']

            male_embeddings = self.embedding_model.encode(male_concepts, convert_to_numpy=True)
            female_embeddings = self.embedding_model.encode(female_concepts, convert_to_numpy=True)
            neutral_embeddings = self.embedding_model.encode(neutral_concepts, convert_to_numpy=True)

            # Compute cosine similarities
            male_sim = np.mean([self._cosine_sim(text_embedding, emb) for emb in male_embeddings])
            female_sim = np.mean([self._cosine_sim(text_embedding, emb) for emb in female_embeddings])
            neutral_sim = np.mean([self._cosine_sim(text_embedding, emb) for emb in neutral_embeddings])

            # Bias is high if text is more similar to stereotypes than neutral
            stereotype_sim = max(male_sim, female_sim)
            bias_score = max(0, stereotype_sim - neutral_sim)

            return bias_score

        except Exception as e:
            logger.warning(f"Embedding bias computation failed: {e}")
            return 0.5

    def _compute_lexical_bias(self, text: str) -> float:
        """Compute bias using lexical indicators."""
        text_lower = text.lower()

        # Count biased vs neutral terms
        biased_terms = self.bias_keywords['gender']['biased']
        neutral_terms = self.bias_keywords['gender']['neutral']

        biased_count = sum(text_lower.count(term) for term in biased_terms)
        neutral_count = sum(text_lower.count(term) for term in neutral_terms)

        total_count = biased_count + neutral_count

        if total_count == 0:
            return 0.0  # No gendered language detected

        # Bias score is ratio of biased terms
        bias_score = biased_count / total_count

        return bias_score

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
