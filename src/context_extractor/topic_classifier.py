"""
Classify input text into bias-sensitive topic categories using zero-shot classification.
"""

import logging
from typing import Dict, List
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

TOPIC_CATEGORIES = [
    'employment_career',
    'crime_justice',
    'relationships_family',
    'education',
    'healthcare_medical',
    'politics_government',
    'sports_entertainment',
    'technology_science',
    'finance_business',
    'general_other'
]

TOPIC_BIAS_RISK = {
    'employment_career': 0.9,
    'crime_justice': 0.95,
    'relationships_family': 0.7,
    'education': 0.5,
    'healthcare_medical': 0.6,
    'politics_government': 0.85,
    'sports_entertainment': 0.3,
    'technology_science': 0.2,
    'finance_business': 0.5,
    'general_other': 0.2
}


class TopicClassifier:
    """Classify text into bias-sensitive topics using zero-shot classification."""

    def __init__(self):
        """Initialize zero-shot classifier."""
        self.classifier = None
        self.categories = TOPIC_CATEGORIES
        self.bias_risk_map = TOPIC_BIAS_RISK

    def _load_classifier(self):
        """Lazily load classifier."""
        if self.classifier is None:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Topic classifier loaded")
            except Exception as e:
                logger.error(f"Failed to load topic classifier: {e}")
                self.classifier = None

    def classify(self, text: str) -> Dict:
        """
        Classify text into topic categories.

        Args:
            text: Input text

        Returns:
            Dict with topic probabilities and bias risk
        """
        if len(text.strip()) < 10:
            return self._default_classification()

        self._load_classifier()

        if self.classifier is None:
            return self._default_classification()

        try:
            result = self.classifier(
                text,
                candidate_labels=self.categories,
                multi_label=True
            )

            # Convert to probability dict
            probs = {label: score for label, score in zip(result['labels'], result['scores'])}

            # Compute weighted bias risk
            bias_risk = sum(probs.get(topic, 0) * risk
                          for topic, risk in self.bias_risk_map.items())

            return {
                'topic_probabilities': probs,
                'top_topic': result['labels'][0],
                'bias_risk': bias_risk
            }

        except Exception as e:
            logger.warning(f"Topic classification failed: {e}")
            return self._default_classification()

    def _default_classification(self) -> Dict:
        """Return default classification."""
        probs = {topic: 1.0/len(self.categories) for topic in self.categories}
        return {
            'topic_probabilities': probs,
            'top_topic': 'general_other',
            'bias_risk': 0.5
        }

    def get_feature_vector(self, text: str) -> List[float]:
        """
        Get topic probability vector for bandit context.

        Returns:
            10-dim probability vector (one per topic)
        """
        result = self.classify(text)
        return [result['topic_probabilities'].get(topic, 0.0)
                for topic in self.categories]

    def get_bias_risk(self, text: str) -> float:
        """Get bias risk score based on topic."""
        result = self.classify(text)
        return result['bias_risk']

    def unload(self):
        """Unload classifier to free memory."""
        if self.classifier is not None:
            del self.classifier
            self.classifier = None
            torch.cuda.empty_cache()
            logger.info("Topic classifier unloaded")
