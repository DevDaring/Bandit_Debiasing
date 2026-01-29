"""
Pytest configuration and fixtures.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from config.model_config import ModelConfig
from config.bandit_config import BanditConfig


@pytest.fixture
def model_config():
    """Provide model configuration."""
    return ModelConfig()


@pytest.fixture
def bandit_config():
    """Provide bandit configuration."""
    return BanditConfig()


@pytest.fixture
def sample_context():
    """Provide sample context vector."""
    context = torch.randn(128)
    context = context / context.norm()  # Normalize
    return context.numpy()


@pytest.fixture
def sample_contexts():
    """Provide batch of sample context vectors."""
    contexts = torch.randn(10, 128)
    contexts = contexts / contexts.norm(dim=1, keepdim=True)  # Normalize
    return contexts.numpy()


@pytest.fixture
def sample_text():
    """Provide sample input text."""
    return "The doctor said she would help the patient."


@pytest.fixture
def sample_texts():
    """Provide batch of sample texts."""
    return [
        "The doctor said she would help the patient.",
        "The engineer fixed his code.",
        "The nurse checked on her patients.",
        "The teacher graded the assignments.",
        "The scientist conducted an experiment."
    ]


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Provide temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_results_dir(tmp_path):
    """Provide temporary results directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def mock_steering_vector():
    """Provide mock steering vector."""
    n_layers = 32
    hidden_dim = 4096
    return torch.randn(n_layers, hidden_dim) * 0.01


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


# Skip markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
