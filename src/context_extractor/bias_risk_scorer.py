"""
Compute overall bias risk score for input text.
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class BiasRiskScorer:
    """Compute aggregate bias risk score from multiple signals."""

    def __init__(self):
        """Initialize with default weights."""
        self.weights = {
            'demographic_density': 0.25,
            'topic_bias_risk': 0.35,
            'stereotype_patterns': 0.25,
            'sensitive_context': 0.15
        }

        # Stereotype patterns to detect
        self.stereotype_patterns = [
            r'\b(all|every|most|typical)\s+\w+\s+(are|is|should|must|always|never)',
            r'\b\w+\s+(are always|are never|should always|should never)',
            r'\bstereotype\b',
            r'\bgeneralization\b'
        ]

    def compute_risk(
        self,
        demographic_features: Dict,
        topic_features: Dict,
        text: str,
        language: str = 'en'
    ) -> float:
        """
        Compute overall bias risk score.

        Args:
            demographic_features: From DemographicDetector
            topic_features: From TopicClassifier
            text: Input text
            language: Language code

        Returns:
            Risk score in [0, 1]
        """
        # Component 1: Demographic density
        demo_risk = demographic_features.get('demographic_density', 0.0)

        # Component 2: Topic bias risk
        topic_risk = topic_features.get('bias_risk', 0.5)

        # Component 3: Stereotype patterns
        pattern_risk = self._detect_stereotype_patterns(text)

        # Component 4: Sensitive context
        sensitive_risk = self._detect_sensitive_context(text)

        # Weighted combination
        total_risk = (
            self.weights['demographic_density'] * demo_risk +
            self.weights['topic_bias_risk'] * topic_risk +
            self.weights['stereotype_patterns'] * pattern_risk +
            self.weights['sensitive_context'] * sensitive_risk
        )

        return min(max(total_risk, 0.0), 1.0)

    def _detect_stereotype_patterns(self, text: str) -> float:
        """Detect stereotypical language patterns."""
        text_lower = text.lower()

        matches = 0
        for pattern in self.stereotype_patterns:
            if re.search(pattern, text_lower):
                matches += 1

        return min(matches / len(self.stereotype_patterns), 1.0)

    def _detect_sensitive_context(self, text: str) -> float:
        """Detect sensitive contexts (questions, comparisons)."""
        text_lower = text.lower()

        score = 0.0

        # Questions about groups
        if re.search(r'\b(why do|why are|how do|how are)\s+\w+', text_lower):
            score += 0.5

        # Comparisons
        if re.search(r'\b(better than|worse than|compared to|versus|vs)', text_lower):
            score += 0.3

        # Attributions
        if re.search(r'\b(because|due to|caused by)\s+\w+', text_lower):
            score += 0.2

        return min(score, 1.0)
