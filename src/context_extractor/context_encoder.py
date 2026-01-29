"""
Encode all extracted features into fixed-dimension context vector.
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class ContextEncoder:
    """Combine all context features into bandit input vector."""

    def __init__(self, output_dim: int = 128):
        """
        Initialize context encoder.

        Args:
            output_dim: Output dimension (default 128)
        """
        self.output_dim = output_dim
        self.embedding_model = None
        self._load_embedding_model()

        # Simple random projection matrix
        self.projection_matrix = None

    def _load_embedding_model(self):
        """Load sentence embedding model."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=device
            )
            logger.info("Sentence transformer loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.embedding_model = None

    def encode(
        self,
        language_features: List[float],
        demographic_features: List[float],
        topic_features: List[float],
        bias_risk: float,
        text: str,
        additional_features: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Encode all features into context vector.

        Args:
            language_features: 4-dim language one-hot
            demographic_features: 12-dim demographic features
            topic_features: 10-dim topic probabilities
            bias_risk: 1-dim bias risk score
            text: Original text for embedding
            additional_features: Optional additional features

        Returns:
            Context vector of shape (128,)
        """
        # Get text embedding (384-dim)
        if self.embedding_model is not None:
            text_emb = self.embedding_model.encode(text, convert_to_numpy=True)
            # Compress to 16 dims
            text_emb_compressed = self._compress_embedding(text_emb, target_dim=16)
        else:
            text_emb_compressed = np.zeros(16)

        # Additional features
        if additional_features is None:
            additional_features = self._compute_additional_features(text)

        # Concatenate all features
        all_features = np.concatenate([
            language_features,      # 4
            demographic_features,   # 12
            topic_features,         # 10
            [bias_risk],           # 1
            text_emb_compressed,   # 16
            additional_features    # 5
        ])  # Total: 48 dims

        # Project to output dimension
        context_vector = self._project_to_output_dim(all_features)

        # L2 normalize
        context_vector = self._normalize(context_vector)

        return context_vector

    def encode_text(self, text: str) -> np.ndarray:
        """
        Simplified encoding from text only.

        Args:
            text: Input text

        Returns:
            Context vector of shape (128,)
        """
        from .language_detector import LanguageDetector
        from .demographic_detector import DemographicDetector
        from .topic_classifier import TopicClassifier
        from .bias_risk_scorer import BiasRiskScorer

        # Initialize detectors
        lang_detector = LanguageDetector()
        demo_detector = DemographicDetector()
        topic_classifier = TopicClassifier()
        bias_scorer = BiasRiskScorer()

        # Extract features
        lang_result = lang_detector.detect(text)
        language = lang_result['language_code']

        demo_result = demo_detector.detect(text, language)
        topic_result = topic_classifier.classify(text)

        bias_risk = bias_scorer.compute_risk(
            demo_result,
            topic_result,
            text,
            language
        )

        # Get feature vectors
        lang_features = lang_result['one_hot_vector']
        demo_features = demo_detector.get_feature_vector(text, language)
        topic_features = topic_classifier.get_feature_vector(text)

        return self.encode(
            lang_features,
            demo_features,
            topic_features,
            bias_risk,
            text
        )

    def _compress_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Compress embedding to target dimension using average pooling."""
        chunk_size = len(embedding) // target_dim
        compressed = []
        for i in range(target_dim):
            start = i * chunk_size
            end = start + chunk_size if i < target_dim - 1 else len(embedding)
            compressed.append(np.mean(embedding[start:end]))
        return np.array(compressed)

    def _compute_additional_features(self, text: str) -> List[float]:
        """Compute additional features from text."""
        words = text.split()

        # Text length (normalized)
        length_norm = min(len(words) / 100.0, 1.0)

        # Is question
        is_question = 1.0 if '?' in text else 0.0

        # Has exclamation
        has_exclamation = 1.0 if '!' in text else 0.0

        # Capitalization ratio
        cap_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Punctuation density
        punct_density = sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)

        return [length_norm, is_question, has_exclamation, cap_ratio, punct_density]

    def _project_to_output_dim(self, features: np.ndarray) -> np.ndarray:
        """Project features to output dimension."""
        input_dim = len(features)

        # Initialize projection matrix if needed
        if self.projection_matrix is None:
            np.random.seed(42)
            self.projection_matrix = np.random.randn(self.output_dim, input_dim) * 0.01

        # Linear projection
        return self.projection_matrix @ features

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize vector."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
