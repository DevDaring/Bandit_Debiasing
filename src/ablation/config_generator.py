"""
Ablation configuration generator.

Generates systematic configurations for ablation studies:
1. Component-level ablations
2. Hyperparameter sensitivity
3. Bandit algorithm comparisons
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from itertools import product
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str

    # Component flags
    use_context_extractor: bool = True
    use_steering_vectors: bool = True
    use_prompt_prefix: bool = True
    use_output_adjustment: bool = True

    # Bandit configuration
    bandit_algorithm: str = 'linucb'  # linucb, thompson, neural, random, static
    static_arm: Optional[int] = None  # For static baseline

    # Hyperparameters
    alpha: float = 1.0  # LinUCB exploration parameter
    lambda_fairness: float = 0.5  # Fairness weight
    bias_threshold: float = 0.3  # Violation threshold
    context_dim: int = 128

    # Training
    n_warmup_steps: int = 50
    learning_rate: float = 0.01

    # Arms to enable (subset of 0-5)
    enabled_arms: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'use_context_extractor': self.use_context_extractor,
            'use_steering_vectors': self.use_steering_vectors,
            'use_prompt_prefix': self.use_prompt_prefix,
            'use_output_adjustment': self.use_output_adjustment,
            'bandit_algorithm': self.bandit_algorithm,
            'static_arm': self.static_arm,
            'alpha': self.alpha,
            'lambda_fairness': self.lambda_fairness,
            'bias_threshold': self.bias_threshold,
            'context_dim': self.context_dim,
            'n_warmup_steps': self.n_warmup_steps,
            'learning_rate': self.learning_rate,
            'enabled_arms': self.enabled_arms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AblationConfig':
        """Create from dictionary."""
        return cls(**data)


# Standard ablation configurations as specified in the prompts
STANDARD_ABLATION_CONFIGS = {
    # Full system
    'full': AblationConfig(
        name='full',
        description='Full Fair-CB system with all components enabled'
    ),

    # Baseline comparisons
    'random': AblationConfig(
        name='random',
        description='Random arm selection (no learning)',
        bandit_algorithm='random'
    ),

    'static_baseline': AblationConfig(
        name='static_baseline',
        description='Static baseline (always arm 0 - no intervention)',
        bandit_algorithm='static',
        static_arm=0
    ),

    'static_gender': AblationConfig(
        name='static_gender',
        description='Static gender steering (always arm 1)',
        bandit_algorithm='static',
        static_arm=1
    ),

    'static_prompt': AblationConfig(
        name='static_prompt',
        description='Static prompt prefix (always arm 4)',
        bandit_algorithm='static',
        static_arm=4
    ),

    # Component ablations
    'no_context': AblationConfig(
        name='no_context',
        description='Without context extraction (random features)',
        use_context_extractor=False
    ),

    'no_steering': AblationConfig(
        name='no_steering',
        description='Without steering vectors (arms 1,2,3 disabled)',
        use_steering_vectors=False,
        enabled_arms=[0, 4, 5]
    ),

    'no_prompt': AblationConfig(
        name='no_prompt',
        description='Without prompt prefix (arm 4 disabled)',
        use_prompt_prefix=False,
        enabled_arms=[0, 1, 2, 3, 5]
    ),

    'no_output_adjust': AblationConfig(
        name='no_output_adjust',
        description='Without output adjustment (arm 5 disabled)',
        use_output_adjustment=False,
        enabled_arms=[0, 1, 2, 3, 4]
    ),

    # Bandit algorithm comparisons
    'linucb': AblationConfig(
        name='linucb',
        description='LinUCB algorithm',
        bandit_algorithm='linucb'
    ),

    'thompson': AblationConfig(
        name='thompson',
        description='Thompson Sampling algorithm',
        bandit_algorithm='thompson'
    ),

    'neural': AblationConfig(
        name='neural',
        description='Neural contextual bandit',
        bandit_algorithm='neural'
    ),

    # Hyperparameter sensitivity
    'alpha_0.5': AblationConfig(
        name='alpha_0.5',
        description='LinUCB with alpha=0.5 (less exploration)',
        alpha=0.5
    ),

    'alpha_2.0': AblationConfig(
        name='alpha_2.0',
        description='LinUCB with alpha=2.0 (more exploration)',
        alpha=2.0
    ),

    'lambda_0.0': AblationConfig(
        name='lambda_0.0',
        description='No fairness weight (pure reward optimization)',
        lambda_fairness=0.0
    ),

    'lambda_1.0': AblationConfig(
        name='lambda_1.0',
        description='High fairness weight',
        lambda_fairness=1.0
    ),
}


class AblationConfigGenerator:
    """
    Generate ablation configurations systematically.

    Provides:
    1. Standard ablation configs
    2. Hyperparameter grid search
    3. Leave-one-out component ablations
    """

    def __init__(self, base_config: Optional[AblationConfig] = None):
        """
        Initialize config generator.

        Args:
            base_config: Base configuration to modify for ablations
        """
        self.base_config = base_config or STANDARD_ABLATION_CONFIGS['full']

    def get_standard_configs(self) -> List[AblationConfig]:
        """Get all standard ablation configurations."""
        return list(STANDARD_ABLATION_CONFIGS.values())

    def generate_component_ablations(self) -> List[AblationConfig]:
        """
        Generate leave-one-out component ablations.

        Returns:
            List of configs, each with one component disabled
        """
        configs = [self.base_config]  # Full system

        # Component ablations
        components = [
            ('use_context_extractor', 'no_context'),
            ('use_steering_vectors', 'no_steering'),
            ('use_prompt_prefix', 'no_prompt'),
            ('use_output_adjustment', 'no_output_adjust'),
        ]

        for attr, name in components:
            config_dict = self.base_config.to_dict()
            config_dict['name'] = name
            config_dict['description'] = f'Ablation: {attr}=False'
            config_dict[attr] = False

            # Update enabled arms for steering ablation
            if attr == 'use_steering_vectors':
                config_dict['enabled_arms'] = [0, 4, 5]
            elif attr == 'use_prompt_prefix':
                config_dict['enabled_arms'] = [0, 1, 2, 3, 5]
            elif attr == 'use_output_adjustment':
                config_dict['enabled_arms'] = [0, 1, 2, 3, 4]

            configs.append(AblationConfig.from_dict(config_dict))

        return configs

    def generate_bandit_comparison(self) -> List[AblationConfig]:
        """
        Generate bandit algorithm comparison configs.

        Returns:
            Configs for each bandit algorithm
        """
        algorithms = ['linucb', 'thompson', 'neural', 'random']
        configs = []

        for algo in algorithms:
            config_dict = self.base_config.to_dict()
            config_dict['name'] = f'bandit_{algo}'
            config_dict['description'] = f'Bandit algorithm: {algo}'
            config_dict['bandit_algorithm'] = algo
            configs.append(AblationConfig.from_dict(config_dict))

        return configs

    def generate_hyperparameter_grid(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[AblationConfig]:
        """
        Generate configs for hyperparameter grid search.

        Args:
            param_grid: Dict mapping parameter name to list of values
                Example: {'alpha': [0.5, 1.0, 2.0], 'lambda_fairness': [0.0, 0.5, 1.0]}

        Returns:
            List of configs for all parameter combinations
        """
        if not param_grid:
            return [self.base_config]

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        configs = []
        for combo in combinations:
            config_dict = self.base_config.to_dict()

            # Create descriptive name
            name_parts = []
            for name, value in zip(param_names, combo):
                config_dict[name] = value
                name_parts.append(f"{name}={value}")

            config_dict['name'] = '_'.join(name_parts)
            config_dict['description'] = f"Grid search: {', '.join(name_parts)}"

            configs.append(AblationConfig.from_dict(config_dict))

        return configs

    def generate_static_arm_baselines(self) -> List[AblationConfig]:
        """
        Generate static arm baselines (non-adaptive).

        Returns:
            Configs for each static arm
        """
        arm_names = [
            'No Intervention',
            'Gender Steering',
            'Race Steering',
            'Religion Steering',
            'Prompt Prefix',
            'Output Adjustment'
        ]

        configs = []
        for arm_id, arm_name in enumerate(arm_names):
            config_dict = self.base_config.to_dict()
            config_dict['name'] = f'static_arm_{arm_id}'
            config_dict['description'] = f'Static: {arm_name}'
            config_dict['bandit_algorithm'] = 'static'
            config_dict['static_arm'] = arm_id
            configs.append(AblationConfig.from_dict(config_dict))

        return configs

    def generate_all(self) -> List[AblationConfig]:
        """
        Generate all ablation configurations.

        Returns:
            Complete list of unique ablation configs
        """
        all_configs = []

        # Standard configs
        all_configs.extend(self.get_standard_configs())

        # Component ablations
        all_configs.extend(self.generate_component_ablations())

        # Bandit comparisons
        all_configs.extend(self.generate_bandit_comparison())

        # Static baselines
        all_configs.extend(self.generate_static_arm_baselines())

        # Remove duplicates by name
        seen_names = set()
        unique_configs = []
        for config in all_configs:
            if config.name not in seen_names:
                seen_names.add(config.name)
                unique_configs.append(config)

        return unique_configs

    def save_configs(self, filepath: str, configs: List[AblationConfig] = None):
        """Save configurations to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if configs is None:
            configs = self.generate_all()

        data = [c.to_dict() for c in configs]

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(configs)} ablation configs to {path}")

    @classmethod
    def load_configs(cls, filepath: str) -> List[AblationConfig]:
        """Load configurations from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return [AblationConfig.from_dict(d) for d in data]

    def __repr__(self) -> str:
        return f"AblationConfigGenerator(base={self.base_config.name})"
