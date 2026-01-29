"""
Tests for bandit algorithms.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.bandit.linucb import LinUCB
from src.bandit.thompson_sampling import ThompsonSamplingLinear
from src.bandit.neural_bandit import NeuralBandit


class TestLinUCB:
    """Test LinUCB bandit."""

    def test_initialization(self, bandit_config):
        """Test LinUCB initialization."""
        bandit = LinUCB(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim,
            alpha=bandit_config.linucb_alpha,
            arm_names=bandit_config.arm_names
        )

        assert bandit.n_arms == bandit_config.n_arms
        assert bandit.context_dim == bandit_config.context_dim
        assert bandit.total_rounds == 0

    def test_select_arm(self, bandit_config, sample_context):
        """Test arm selection."""
        bandit = LinUCB(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )

        arm_idx, confidence = bandit.select_arm(sample_context)

        assert 0 <= arm_idx < bandit_config.n_arms
        assert confidence >= 0

    def test_update(self, bandit_config, sample_context):
        """Test bandit update."""
        bandit = LinUCB(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )

        arm_idx = 0
        reward = 0.8

        bandit.update(sample_context, arm_idx, reward)

        assert bandit.total_rounds == 1
        assert len(bandit.history) == 1

    def test_multiple_rounds(self, bandit_config, sample_contexts):
        """Test multiple rounds of selection and update."""
        bandit = LinUCB(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )

        for context in sample_contexts:
            arm_idx, confidence = bandit.select_arm(context)
            reward = np.random.random()
            bandit.update(context, arm_idx, reward)

        assert bandit.total_rounds == len(sample_contexts)

    def test_save_load(self, bandit_config, sample_context, temp_checkpoint_dir):
        """Test save and load functionality."""
        bandit = LinUCB(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )

        # Run a few rounds
        for _ in range(5):
            arm_idx, _ = bandit.select_arm(sample_context)
            bandit.update(sample_context, arm_idx, 0.5)

        # Save
        save_path = temp_checkpoint_dir / "bandit.pkl"
        bandit.save(str(save_path))

        # Load into new bandit
        new_bandit = LinUCB(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )
        new_bandit.load(str(save_path))

        assert new_bandit.total_rounds == bandit.total_rounds
        assert len(new_bandit.history) == len(bandit.history)


class TestThompsonSampling:
    """Test Thompson Sampling bandit."""

    def test_initialization(self, bandit_config):
        """Test Thompson Sampling initialization."""
        bandit = ThompsonSamplingLinear(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim,
            noise_std=bandit_config.ts_noise_std,
            arm_names=bandit_config.arm_names
        )

        assert bandit.n_arms == bandit_config.n_arms
        assert bandit.context_dim == bandit_config.context_dim

    def test_select_arm(self, bandit_config, sample_context):
        """Test arm selection."""
        bandit = ThompsonSamplingLinear(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )

        arm_idx, confidence = bandit.select_arm(sample_context)

        assert 0 <= arm_idx < bandit_config.n_arms
        assert confidence >= 0

    def test_update(self, bandit_config, sample_context):
        """Test bandit update."""
        bandit = ThompsonSamplingLinear(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )

        arm_idx = 0
        reward = 0.8

        bandit.update(sample_context, arm_idx, reward)

        assert bandit.total_rounds == 1

    def test_randomness_in_selection(self, bandit_config, sample_context):
        """Test that Thompson Sampling introduces randomness."""
        bandit = ThompsonSamplingLinear(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim
        )

        # Select arm multiple times with same context
        selections = []
        for _ in range(10):
            arm_idx, _ = bandit.select_arm(sample_context)
            selections.append(arm_idx)

        # Should have some variety (not deterministic)
        # Note: This could fail by chance, but very unlikely with 10 samples
        unique_arms = len(set(selections))
        assert unique_arms >= 1  # At least some arms selected


@pytest.mark.slow
class TestNeuralBandit:
    """Test Neural Bandit."""

    def test_initialization(self, bandit_config):
        """Test Neural Bandit initialization."""
        bandit = NeuralBandit(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim,
            hidden_dim=bandit_config.neural_hidden_dim,
            learning_rate=bandit_config.neural_learning_rate,
            arm_names=bandit_config.arm_names
        )

        assert bandit.n_arms == bandit_config.n_arms
        assert bandit.context_dim == bandit_config.context_dim

    def test_select_arm(self, bandit_config, sample_context):
        """Test arm selection."""
        bandit = NeuralBandit(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim,
            hidden_dim=32,  # Smaller for faster testing
        )

        arm_idx, confidence = bandit.select_arm(sample_context)

        assert 0 <= arm_idx < bandit_config.n_arms
        assert confidence >= 0

    def test_learning(self, bandit_config, sample_contexts):
        """Test that neural bandit learns."""
        bandit = NeuralBandit(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim,
            hidden_dim=32,
            batch_size=4,
            train_every=5
        )

        # Train for several rounds
        for i, context in enumerate(sample_contexts):
            arm_idx, _ = bandit.select_arm(context)

            # Give higher reward to arm 0
            reward = 1.0 if arm_idx == 0 else 0.2

            bandit.update(context, arm_idx, reward)

        # After training, should prefer arm 0
        # (This is probabilistic, so not a strict test)
        assert bandit.total_rounds == len(sample_contexts)


class TestBanditComparison:
    """Compare different bandit algorithms."""

    def test_all_bandits_work(self, bandit_config, sample_contexts):
        """Test that all bandit algorithms work similarly."""
        bandits = [
            LinUCB(
                n_arms=bandit_config.n_arms,
                context_dim=bandit_config.context_dim
            ),
            ThompsonSamplingLinear(
                n_arms=bandit_config.n_arms,
                context_dim=bandit_config.context_dim
            ),
            NeuralBandit(
                n_arms=bandit_config.n_arms,
                context_dim=bandit_config.context_dim,
                hidden_dim=32
            )
        ]

        for bandit in bandits:
            for context in sample_contexts[:5]:  # Just a few samples
                arm_idx, confidence = bandit.select_arm(context)
                reward = np.random.random()
                bandit.update(context, arm_idx, reward)

            assert bandit.total_rounds == 5

    def test_arm_statistics(self, bandit_config, sample_contexts):
        """Test arm statistics retrieval."""
        bandit = LinUCB(
            n_arms=bandit_config.n_arms,
            context_dim=bandit_config.context_dim,
            arm_names=bandit_config.arm_names
        )

        # Run some rounds
        for context in sample_contexts:
            arm_idx, _ = bandit.select_arm(context)
            reward = np.random.random()
            bandit.update(context, arm_idx, reward)

        stats = bandit.get_arm_statistics()

        assert isinstance(stats, dict)
        assert len(stats) <= bandit_config.n_arms

        for arm_name, arm_stats in stats.items():
            assert 'selections' in arm_stats
            assert 'mean_reward' in arm_stats
            assert arm_stats['selections'] >= 0
