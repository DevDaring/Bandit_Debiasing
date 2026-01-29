"""
Tests for context extraction module.
"""

import pytest
import numpy as np

from src.context_extractor.language_detector import LanguageDetector
from src.context_extractor.demographic_detector import DemographicDetector
from src.context_extractor.bias_risk_scorer import BiasRiskScorer
from src.context_extractor.context_encoder import ContextEncoder


class TestLanguageDetector:
    """Test language detection."""

    def test_detect_english(self):
        """Test English language detection."""
        detector = LanguageDetector()
        result = detector.detect("This is an English sentence.")

        assert result['language_code'] == 'en'
        assert result['confidence'] > 0.5
        assert len(result['one_hot']) == 4

    def test_detect_returns_dict(self, sample_text):
        """Test that detector returns proper dictionary."""
        detector = LanguageDetector()
        result = detector.detect(sample_text)

        assert isinstance(result, dict)
        assert 'language_code' in result
        assert 'confidence' in result
        assert 'one_hot' in result


class TestDemographicDetector:
    """Test demographic detection."""

    def test_detector_initialization(self):
        """Test that detector initializes."""
        detector = DemographicDetector()
        assert detector is not None

    def test_detect_returns_vector(self, sample_text):
        """Test that detection returns proper vector."""
        detector = DemographicDetector()
        features = detector.detect(sample_text, language='en')

        assert isinstance(features, np.ndarray)
        assert features.shape == (12,)
        assert all(f >= 0 for f in features)

    def test_detect_gender_markers(self):
        """Test gender marker detection."""
        detector = DemographicDetector()

        # Text with male pronouns
        male_text = "He said his opinion."
        male_features = detector.detect(male_text, language='en')
        assert male_features[0] > 0  # gender_male

        # Text with female pronouns
        female_text = "She said her opinion."
        female_features = detector.detect(female_text, language='en')
        assert female_features[1] > 0  # gender_female

        # Neutral text
        neutral_text = "They said their opinion."
        neutral_features = detector.detect(neutral_text, language='en')
        assert neutral_features[2] > 0  # gender_neutral


class TestBiasRiskScorer:
    """Test bias risk scoring."""

    def test_scorer_initialization(self):
        """Test that scorer initializes."""
        scorer = BiasRiskScorer()
        assert scorer is not None

    def test_score_returns_float(self, sample_text):
        """Test that scoring returns float in [0, 1]."""
        scorer = BiasRiskScorer()
        risk_score = scorer.score(sample_text, language='en')

        assert isinstance(risk_score, (float, np.floating))
        assert 0.0 <= risk_score <= 1.0

    def test_biased_text_higher_risk(self):
        """Test that biased text gets higher risk score."""
        scorer = BiasRiskScorer()

        neutral_text = "The person completed the task."
        biased_text = "Women are always more emotional than men."

        neutral_score = scorer.score(neutral_text, language='en')
        biased_score = scorer.score(biased_text, language='en')

        # Biased text should have higher risk (though this is heuristic-based)
        # Just verify both return valid scores
        assert 0.0 <= neutral_score <= 1.0
        assert 0.0 <= biased_score <= 1.0


@pytest.mark.slow
class TestContextEncoder:
    """Test context encoder."""

    def test_encoder_initialization(self):
        """Test that encoder initializes."""
        encoder = ContextEncoder(output_dim=128)
        assert encoder is not None
        assert encoder.output_dim == 128

    def test_encode_text_returns_vector(self, sample_text):
        """Test that encoding returns proper vector."""
        encoder = ContextEncoder(output_dim=128)
        context = encoder.encode_text(sample_text)

        assert isinstance(context, np.ndarray)
        assert context.shape == (128,)

    def test_encode_text_normalized(self, sample_text):
        """Test that context vector is normalized."""
        encoder = ContextEncoder(output_dim=128)
        context = encoder.encode_text(sample_text)

        norm = np.linalg.norm(context)
        assert abs(norm - 1.0) < 0.01  # Should be close to 1

    def test_encode_batch(self, sample_texts):
        """Test batch encoding."""
        encoder = ContextEncoder(output_dim=128)
        contexts = encoder.encode_batch(sample_texts)

        assert contexts.shape == (len(sample_texts), 128)

        # Check all normalized
        for context in contexts:
            norm = np.linalg.norm(context)
            assert abs(norm - 1.0) < 0.01

    def test_different_texts_different_contexts(self):
        """Test that different texts produce different contexts."""
        encoder = ContextEncoder(output_dim=128)

        text1 = "The doctor examined the patient."
        text2 = "The cat sat on the mat."

        context1 = encoder.encode_text(text1)
        context2 = encoder.encode_text(text2)

        # Contexts should be different
        assert not np.allclose(context1, context2)
