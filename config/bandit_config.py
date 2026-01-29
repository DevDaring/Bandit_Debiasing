"""
Configuration for Multi-Armed Bandit algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BanditConfig:
    """Hyperparameters for contextual bandit algorithms."""

    # Number of arms (debiasing strategies)
    n_arms: int = 6

    # Arm definitions
    arm_names: List[str] = field(
        default_factory=lambda: [
            "no_intervention",  # Arm 0: Baseline - no debiasing
            "gender_steering",  # Arm 1: Apply gender debiasing steering vector
            "race_steering",  # Arm 2: Apply race/ethnicity debiasing steering vector
            "religion_steering",  # Arm 3: Apply religion debiasing steering vector
            "prompt_prefix",  # Arm 4: Add debiasing instruction prefix to prompt
            "output_adjustment",  # Arm 5: Post-hoc probability adjustment on output
        ]
    )

    # Context feature dimension
    context_dim: int = 128

    # LinUCB specific
    linucb_alpha: float = 0.5  # Exploration parameter (higher = more exploration)

    # Thompson Sampling specific
    ts_prior_mean: float = 0.0
    ts_prior_variance: float = 1.0
    ts_noise_std: float = 0.5

    # Neural Bandit specific
    neural_hidden_dim: int = 64
    neural_learning_rate: float = 0.001
    neural_dropout_rate: float = 0.1
    neural_n_mc_samples: int = 10
    neural_buffer_size: int = 10000
    neural_batch_size: int = 32
    neural_update_frequency: int = 10

    # Training settings
    warmup_rounds: int = 100  # Random exploration before using learned policy
    update_frequency: int = 1  # Update bandit every N samples

    # Reward weights
    bias_weight: float = 0.6  # Weight for bias reduction in reward
    quality_weight: float = 0.4  # Weight for generation quality in reward


@dataclass
class SteeringVectorConfig:
    """Configuration for steering vector arms."""

    # Layer range to apply steering (typically middle-to-late layers work best)
    # For 7B model with 32 layers, apply to layers 12-24
    start_layer: int = 12
    end_layer: int = 24

    # Steering strength multipliers (tunable per bias type)
    gender_strength: float = 1.5
    race_strength: float = 1.2
    religion_strength: float = 1.0

    # Direction of steering (positive = towards fair, negative = towards biased)
    steering_direction: int = -1  # Subtract bias direction to debias


@dataclass
class WandbConfig:
    """Weights & Biases configuration for experiment tracking."""

    project_name: str = "mab-debiasing"
    entity: Optional[str] = None  # Your wandb username/team
    log_frequency: int = 1  # Log every N steps
    log_model: bool = True  # Save model checkpoints to wandb
    enabled: bool = True  # Set to False to disable wandb logging


from typing import Optional
