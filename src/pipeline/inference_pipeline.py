"""
Complete inference pipeline integrating all components.
"""

import logging
import json
from typing import Optional, Dict, List, Union
import numpy as np

from config.model_config import ModelConfig, clear_gpu_memory
from config.bandit_config import BanditConfig
from src.llm.model_loader import ModelLoader
from src.llm.generator import Generator
from src.context_extractor.context_encoder import ContextEncoder
from src.reward.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)


class MABDebiasInferencePipeline:
    """
    Main inference pipeline for MAB debiasing system.

    Pipeline flow:
        1. Extract context features
        2. Query bandit for arm selection
        3. Apply selected intervention
        4. Generate response
        5. Compute reward
        6. Update bandit (if learning enabled)
        7. Return response
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        bandit_type: str = 'linucb',
        bandit_config: Optional[BanditConfig] = None,
        enable_learning: bool = True
    ):
        """
        Initialize inference pipeline.

        Args:
            model_name: Model identifier
            bandit_type: 'linucb', 'thompson', or 'neural'
            bandit_config: Bandit configuration
            enable_learning: Whether to update bandit
        """
        self.model_config = ModelConfig()
        self.model_name = model_name or self.model_config.model_name

        self.bandit_config = bandit_config or BanditConfig()
        self.bandit_type = bandit_type
        self.enable_learning = enable_learning

        # Components (loaded later)
        self.model_loader = None
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.context_encoder = None
        self.bandit = None
        self.arms = {}
        self.reward_calculator = None

        logger.info(f"MABDebiasInferencePipeline initialized: model={self.model_name}, "
                   f"bandit={self.bandit_type}, learning={self.enable_learning}")

    def load_components(self):
        """Load all components sequentially with memory management."""
        logger.info("Loading pipeline components...")

        # 1. Load model
        self.model_loader = ModelLoader()
        self.model, self.tokenizer = self.model_loader.load(self.model_name)
        self.generator = Generator(self.model, self.tokenizer)
        logger.info("✓ Model loaded")

        # 2. Load context encoder
        self.context_encoder = ContextEncoder(output_dim=self.bandit_config.context_dim)
        logger.info("✓ Context encoder loaded")

        # 3. Load bandit
        self.bandit = self._create_bandit()
        logger.info(f"✓ Bandit loaded: {self.bandit_type}")

        # 4. Load arms
        self.arms = self._create_arms()
        logger.info(f"✓ Arms loaded: {list(self.arms.keys())}")

        # 5. Load reward calculator
        self.reward_calculator = RewardCalculator(
            bias_weight=self.bandit_config.bias_weight,
            quality_weight=self.bandit_config.quality_weight
        )
        logger.info("✓ Reward calculator loaded")

        logger.info("All components loaded successfully!")

    def _create_bandit(self):
        """Create bandit algorithm instance."""
        from src.bandit.linucb import LinUCB
        from src.bandit.thompson_sampling import ThompsonSamplingLinear
        from src.bandit.neural_bandit import NeuralBandit

        if self.bandit_type == 'linucb':
            return LinUCB(
                n_arms=self.bandit_config.n_arms,
                context_dim=self.bandit_config.context_dim,
                alpha=self.bandit_config.linucb_alpha,
                arm_names=self.bandit_config.arm_names
            )
        elif self.bandit_type == 'thompson':
            return ThompsonSamplingLinear(
                n_arms=self.bandit_config.n_arms,
                context_dim=self.bandit_config.context_dim,
                noise_std=self.bandit_config.ts_noise_std,
                arm_names=self.bandit_config.arm_names
            )
        elif self.bandit_type == 'neural':
            return NeuralBandit(
                n_arms=self.bandit_config.n_arms,
                context_dim=self.bandit_config.context_dim,
                hidden_dim=self.bandit_config.neural_hidden_dim,
                learning_rate=self.bandit_config.neural_learning_rate,
                arm_names=self.bandit_config.arm_names
            )
        else:
            raise ValueError(f"Unknown bandit type: {self.bandit_type}")

    def _create_arms(self) -> Dict:
        """Create all debiasing arms."""
        from src.debiasing_arms.no_intervention import NoInterventionArm
        from src.debiasing_arms.steering_vector_arm import (
            GenderSteeringArm, RaceSteeringArm, ReligionSteeringArm
        )
        from src.debiasing_arms.prompt_prefix_arm import PromptPrefixArm
        from src.debiasing_arms.output_adjustment_arm import OutputAdjustmentArm

        arms = {
            0: NoInterventionArm(),
            1: GenderSteeringArm(),
            2: RaceSteeringArm(),
            3: ReligionSteeringArm(),
            4: PromptPrefixArm(),
            5: OutputAdjustmentArm()
        }

        return arms

    def process(self, input_text: str, return_details: bool = False) -> Union[str, Dict]:
        """
        Process input and return response.

        Args:
            input_text: Input text
            return_details: Whether to return detailed information

        Returns:
            Generated text or dict with details
        """
        # 1. Extract context features
        context = self.context_encoder.encode_text(input_text)
        logger.debug(f"Context extracted: shape={context.shape}")

        # 2. Select arm via bandit
        arm_idx, confidence = self.bandit.select_arm(context)
        arm = self.arms[arm_idx]
        logger.info(f"Selected arm: {arm.name} (confidence={confidence:.3f})")

        # 3. Generate response with intervention
        result = self.generator.generate(
            input_text=input_text,
            intervention=arm,
            generation_config=None,
            language='en'  # TODO: Detect from context
        )

        generated_text = result['text']

        # 4. Compute reward
        reward_data = self.reward_calculator.calculate(generated_text, input_text)
        logger.info(f"Reward: {reward_data['reward']:.3f} "
                   f"(bias={reward_data['bias_score']:.3f}, "
                   f"quality={reward_data['quality_score']:.3f})")

        # 5. Update bandit (if learning enabled)
        if self.enable_learning:
            self.bandit.update(context, arm_idx, reward_data['reward'])
            logger.debug(f"Bandit updated")

        # 6. Return response
        if return_details:
            return {
                'response': generated_text,
                'selected_arm': arm.name,
                'arm_index': arm_idx,
                'arm_confidence': confidence,
                'reward': reward_data['reward'],
                'bias_score': reward_data['bias_score'],
                'quality_score': reward_data['quality_score'],
                'generation_time': result['time_seconds']
            }
        else:
            return generated_text

    def process_batch(self, inputs: List[str], return_details: bool = False) -> List:
        """
        Process batch of inputs sequentially.

        Args:
            inputs: List of input texts
            return_details: Whether to return details

        Returns:
            List of responses
        """
        results = []
        for input_text in inputs:
            result = self.process(input_text, return_details)
            results.append(result)
        return results

    def save_state(self, path: str):
        """Save bandit state."""
        self.bandit.save(path)
        logger.info(f"Pipeline state saved to {path}")

    def load_state(self, path: str):
        """Load bandit state."""
        self.bandit.load(path)
        logger.info(f"Pipeline state loaded from {path}")

    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        arm_stats = self.bandit.get_arm_statistics()

        return {
            'total_rounds': self.bandit.total_rounds,
            'arm_statistics': arm_stats,
            'bandit_type': self.bandit_type,
            'learning_enabled': self.enable_learning
        }

    def unload(self):
        """Unload models to free memory."""
        if self.model_loader:
            self.model_loader.unload()
        clear_gpu_memory()
        logger.info("Pipeline unloaded")
