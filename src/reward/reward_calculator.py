"""
Combine bias and quality scores into single reward signal.
"""

import logging
from typing import Dict
from .bias_scorer import BiasScorer
from .quality_scorer import QualityScorer

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Compute final reward for bandit update.

    Formula:
        reward = (1 - bias_score) * bias_weight + quality_score * quality_weight
    """

    def __init__(self, bias_weight: float = 0.6, quality_weight: float = 0.4):
        """
        Initialize reward calculator.

        Args:
            bias_weight: Weight for bias reduction in reward
            quality_weight: Weight for generation quality in reward
        """
        self.bias_weight = bias_weight
        self.quality_weight = quality_weight

        # Ensure weights sum to 1.0
        total = bias_weight + quality_weight
        self.bias_weight = bias_weight / total
        self.quality_weight = quality_weight / total

        # Initialize scorers
        self.bias_scorer = BiasScorer()
        self.quality_scorer = QualityScorer()

        logger.info(f"RewardCalculator initialized: bias_weight={self.bias_weight:.2f}, "
                   f"quality_weight={self.quality_weight:.2f}")

    def calculate(
        self,
        generated_text: str,
        input_text: str,
        language: str = 'en'
    ) -> Dict:
        """
        Calculate reward for generated text.

        Args:
            generated_text: Generated output
            input_text: Original input
            language: Language code

        Returns:
            Dict with:
                - reward: Final reward in [0, 1]
                - bias_score: Bias score
                - quality_score: Quality score
                - bias_component: Bias contribution to reward
                - quality_component: Quality contribution to reward
        """
        # Compute scores
        bias_score = self.bias_scorer.score(generated_text, input_text, language)
        quality_score = self.quality_scorer.score(generated_text, input_text, language)

        # Compute reward components
        bias_component = (1.0 - bias_score) * self.bias_weight
        quality_component = quality_score * self.quality_weight

        # Final reward
        reward = bias_component + quality_component

        return {
            'reward': reward,
            'bias_score': bias_score,
            'quality_score': quality_score,
            'bias_component': bias_component,
            'quality_component': quality_component
        }

    def update_weights(self, bias_weight: float, quality_weight: float):
        """
        Update reward weights.

        Args:
            bias_weight: New bias weight
            quality_weight: New quality weight
        """
        total = bias_weight + quality_weight
        self.bias_weight = bias_weight / total
        self.quality_weight = quality_weight / total

        logger.info(f"Reward weights updated: bias_weight={self.bias_weight:.2f}, "
                   f"quality_weight={self.quality_weight:.2f}")
