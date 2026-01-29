"""
Tests for debiasing arms.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.debiasing_arms.base_arm import BaseArm
from src.debiasing_arms.no_intervention import NoInterventionArm
from src.debiasing_arms.prompt_prefix_arm import PromptPrefixArm
from src.debiasing_arms.output_adjustment_arm import OutputAdjustmentArm


class TestNoInterventionArm:
    """Test no intervention arm."""

    def test_initialization(self):
        """Test arm initialization."""
        arm = NoInterventionArm()

        assert arm.name == "NoIntervention"
        assert arm.requires_model_access is True

    def test_apply_returns_unchanged(self):
        """Test that apply returns unchanged text."""
        arm = NoInterventionArm()

        input_text = "This is a test."
        modified_text = arm.apply(input_text, language='en')

        assert modified_text == input_text


class TestPromptPrefixArm:
    """Test prompt prefix arm."""

    def test_initialization(self):
        """Test arm initialization."""
        arm = PromptPrefixArm()

        assert arm.name == "PromptPrefix"
        assert arm.requires_model_access is True

    def test_apply_adds_prefix_english(self):
        """Test that apply adds prefix for English."""
        arm = PromptPrefixArm()

        input_text = "Tell me about doctors."
        modified_text = arm.apply(input_text, language='en')

        assert len(modified_text) > len(input_text)
        assert input_text in modified_text

    def test_apply_adds_prefix_hindi(self):
        """Test that apply adds prefix for Hindi."""
        arm = PromptPrefixArm()

        input_text = "डॉक्टरों के बारे में बताओ।"
        modified_text = arm.apply(input_text, language='hi')

        assert len(modified_text) > len(input_text)
        assert input_text in modified_text

    def test_apply_adds_prefix_bengali(self):
        """Test that apply adds prefix for Bengali."""
        arm = PromptPrefixArm()

        input_text = "ডাক্তারদের সম্পর্কে বলুন।"
        modified_text = arm.apply(input_text, language='bn')

        assert len(modified_text) > len(input_text)
        assert input_text in modified_text

    def test_apply_unknown_language_defaults_to_english(self):
        """Test that unknown language defaults to English."""
        arm = PromptPrefixArm()

        input_text = "Test text."
        modified_text = arm.apply(input_text, language='unknown')

        assert len(modified_text) > len(input_text)


class TestOutputAdjustmentArm:
    """Test output adjustment arm."""

    def test_initialization(self):
        """Test arm initialization."""
        arm = OutputAdjustmentArm()

        assert arm.name == "OutputAdjustment"
        assert arm.requires_model_access is True

    def test_logits_processor_creation(self):
        """Test that logits processor is created."""
        arm = OutputAdjustmentArm()

        # This would require a real tokenizer to test properly
        # For now, just verify the arm initializes
        assert arm is not None


class TestArmInterface:
    """Test that all arms implement the required interface."""

    def test_all_arms_have_name(self):
        """Test that all arms have a name."""
        arms = [
            NoInterventionArm(),
            PromptPrefixArm(),
            OutputAdjustmentArm()
        ]

        for arm in arms:
            assert hasattr(arm, 'name')
            assert isinstance(arm.name, str)
            assert len(arm.name) > 0

    def test_all_arms_have_requires_model_access(self):
        """Test that all arms have requires_model_access."""
        arms = [
            NoInterventionArm(),
            PromptPrefixArm(),
            OutputAdjustmentArm()
        ]

        for arm in arms:
            assert hasattr(arm, 'requires_model_access')
            assert isinstance(arm.requires_model_access, bool)

    def test_all_arms_have_apply(self):
        """Test that all arms have apply method."""
        arms = [
            NoInterventionArm(),
            PromptPrefixArm(),
            OutputAdjustmentArm()
        ]

        input_text = "Test input."

        for arm in arms:
            assert hasattr(arm, 'apply')
            # Verify apply can be called
            result = arm.apply(input_text, language='en')
            assert isinstance(result, str)

    def test_all_arms_serializable(self):
        """Test that all arms can be represented as strings."""
        arms = [
            NoInterventionArm(),
            PromptPrefixArm(),
            OutputAdjustmentArm()
        ]

        for arm in arms:
            # Should be able to convert to string
            arm_str = str(arm)
            assert isinstance(arm_str, str)
            assert len(arm_str) > 0


@pytest.mark.slow
class TestSteeringVectorArm:
    """Test steering vector arm (requires actual vectors)."""

    def test_initialization_without_vector(self):
        """Test that arm can be initialized (even without real vector)."""
        # This test is limited without actual steering vectors
        # Just verify the import works
        from src.debiasing_arms.steering_vector_arm import GenderSteeringArm

        # Would need real vector file to fully initialize
        assert GenderSteeringArm is not None

    def test_steering_arm_subclasses_exist(self):
        """Test that all steering arm subclasses exist."""
        from src.debiasing_arms.steering_vector_arm import (
            GenderSteeringArm,
            RaceSteeringArm,
            ReligionSteeringArm
        )

        assert GenderSteeringArm is not None
        assert RaceSteeringArm is not None
        assert ReligionSteeringArm is not None
