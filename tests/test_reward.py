"""
Tests for reward calculation.
"""

import pytest
import numpy as np

from src.reward.bias_scorer import BiasScorer
from src.reward.quality_scorer import QualityScorer
from src.reward.reward_calculator import RewardCalculator


@pytest.mark.slow
class TestBiasScorer:
    """Test bias scoring."""

    def test_initialization(self):
        """Test bias scorer initialization."""
        scorer = BiasScorer()
        assert scorer is not None

    def test_score_returns_float(self):
        """Test that scoring returns float in [0, 1]."""
        scorer = BiasScorer()

        text = "The engineer completed the project."
        input_text = "Tell me about an engineer."

        score = scorer.score(text, input_text, language='en')

        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_gendered_text_higher_bias(self):
        """Test that gendered text gets higher bias score."""
        scorer = BiasScorer()

        neutral_text = "The person completed the work successfully."
        gendered_text = "He completed the work successfully."

        input_text = "Tell me about a worker."

        neutral_score = scorer.score(neutral_text, input_text, language='en')
        gendered_score = scorer.score(gendered_text, input_text, language='en')

        # Gendered text should have higher bias score
        assert gendered_score >= neutral_score


@pytest.mark.slow
class TestQualityScorer:
    """Test quality scoring."""

    def test_initialization(self):
        """Test quality scorer initialization."""
        scorer = QualityScorer()
        assert scorer is not None

    def test_score_returns_float(self):
        """Test that scoring returns float in [0, 1]."""
        scorer = QualityScorer()

        text = "The engineer completed the project with great attention to detail."
        input_text = "Tell me about an engineer."

        score = scorer.score(text, input_text, language='en')

        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_quality_penalizes_short_text(self):
        """Test that very short text gets lower quality score."""
        scorer = QualityScorer()

        input_text = "Tell me a story."

        short_text = "Yes."
        normal_text = "Once upon a time, there was a brave knight who went on many adventures."

        short_score = scorer.score(short_text, input_text, language='en')
        normal_score = scorer.score(normal_text, input_text, language='en')

        # Normal text should have higher quality
        assert normal_score >= short_score

    def test_quality_penalizes_repetition(self):
        """Test that repetitive text gets lower quality score."""
        scorer = QualityScorer()

        input_text = "Tell me about something."

        normal_text = "The cat sat on the mat and looked around curiously."
        repetitive_text = "The cat cat cat sat sat sat on on on the the the mat mat mat."

        normal_score = scorer.score(normal_text, input_text, language='en')
        repetitive_score = scorer.score(repetitive_text, input_text, language='en')

        # Normal text should have higher quality
        assert normal_score > repetitive_score


class TestRewardCalculator:
    """Test reward calculator."""

    def test_initialization(self):
        """Test reward calculator initialization."""
        calculator = RewardCalculator(bias_weight=0.6, quality_weight=0.4)

        assert calculator.bias_weight > 0
        assert calculator.quality_weight > 0
        # Weights should sum to 1
        assert abs(calculator.bias_weight + calculator.quality_weight - 1.0) < 0.01

    def test_weights_normalization(self):
        """Test that weights are normalized to sum to 1."""
        calculator = RewardCalculator(bias_weight=3.0, quality_weight=2.0)

        # Should normalize to 0.6 and 0.4
        assert abs(calculator.bias_weight - 0.6) < 0.01
        assert abs(calculator.quality_weight - 0.4) < 0.01

    @pytest.mark.slow
    def test_calculate_returns_dict(self):
        """Test that calculate returns proper dictionary."""
        calculator = RewardCalculator()

        generated_text = "The person completed the work successfully."
        input_text = "Tell me about a worker."

        result = calculator.calculate(generated_text, input_text, language='en')

        assert isinstance(result, dict)
        assert 'reward' in result
        assert 'bias_score' in result
        assert 'quality_score' in result
        assert 'bias_component' in result
        assert 'quality_component' in result

    @pytest.mark.slow
    def test_reward_in_valid_range(self):
        """Test that reward is in [0, 1]."""
        calculator = RewardCalculator()

        generated_text = "The engineer completed the project."
        input_text = "Tell me about an engineer."

        result = calculator.calculate(generated_text, input_text, language='en')

        assert 0.0 <= result['reward'] <= 1.0
        assert 0.0 <= result['bias_score'] <= 1.0
        assert 0.0 <= result['quality_score'] <= 1.0

    @pytest.mark.slow
    def test_reward_components_sum_correctly(self):
        """Test that reward components sum to total reward."""
        calculator = RewardCalculator()

        generated_text = "The person worked on the task."
        input_text = "Tell me about work."

        result = calculator.calculate(generated_text, input_text, language='en')

        # Components should sum to total reward (within floating point error)
        component_sum = result['bias_component'] + result['quality_component']
        assert abs(component_sum - result['reward']) < 0.01

    @pytest.mark.slow
    def test_better_text_higher_reward(self):
        """Test that better text gets higher reward."""
        calculator = RewardCalculator()

        input_text = "Tell me about a worker."

        # Good text: neutral and high quality
        good_text = "The person completed the work successfully with great attention to detail."

        # Bad text: biased and lower quality
        bad_text = "He did it."

        good_result = calculator.calculate(good_text, input_text, language='en')
        bad_result = calculator.calculate(bad_text, input_text, language='en')

        # Good text should have higher reward
        assert good_result['reward'] >= bad_result['reward']

    def test_update_weights(self):
        """Test updating reward weights."""
        calculator = RewardCalculator(bias_weight=0.6, quality_weight=0.4)

        calculator.update_weights(bias_weight=0.8, quality_weight=0.2)

        assert abs(calculator.bias_weight - 0.8) < 0.01
        assert abs(calculator.quality_weight - 0.2) < 0.01
        assert abs(calculator.bias_weight + calculator.quality_weight - 1.0) < 0.01
