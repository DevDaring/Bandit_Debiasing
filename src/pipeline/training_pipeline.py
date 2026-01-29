"""
Training pipeline with warmup, W&B logging, and checkpointing.
"""

import logging
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import wandb

from .inference_pipeline import MABDebiasInferencePipeline

logger = logging.getLogger(__name__)


class MABDebiasTrainingPipeline:
    """
    Training pipeline for MAB debiasing system.

    Features:
    - Warmup phase with random exploration
    - W&B logging for remote monitoring
    - Periodic evaluation on held-out set
    - Checkpoint saving every N steps
    - Graceful shutdown handling
    """

    def __init__(
        self,
        inference_pipeline: MABDebiasInferencePipeline,
        warmup_samples: int = 100,
        eval_every: int = 100,
        save_every: int = 500,
        checkpoint_dir: str = './checkpoints',
        results_dir: str = './results',
        wandb_project: Optional[str] = 'mab-debiasing',
        wandb_run_name: Optional[str] = None,
        enable_wandb: bool = True
    ):
        """
        Initialize training pipeline.

        Args:
            inference_pipeline: Inference pipeline instance
            warmup_samples: Number of samples for random exploration
            eval_every: Evaluate every N samples
            save_every: Save checkpoint every N samples
            checkpoint_dir: Directory for checkpoints
            results_dir: Directory for results
            wandb_project: W&B project name
            wandb_run_name: W&B run name (optional)
            enable_wandb: Whether to use W&B logging
        """
        self.pipeline = inference_pipeline
        self.warmup_samples = warmup_samples
        self.eval_every = eval_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        self.enable_wandb = enable_wandb
        if self.enable_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'model_name': self.pipeline.model_name,
                    'bandit_type': self.pipeline.bandit_type,
                    'warmup_samples': warmup_samples,
                    'eval_every': eval_every,
                    'save_every': save_every,
                    'bandit_config': self.pipeline.bandit_config.__dict__
                }
            )
            logger.info(f"W&B initialized: project={wandb_project}, run={wandb_run_name}")

        # Training state
        self.total_samples = 0
        self.warmup_complete = False
        self.training_history = []

        logger.info(f"TrainingPipeline initialized: warmup={warmup_samples}, "
                   f"eval_every={eval_every}, save_every={save_every}")

    def warmup(self, train_data: List[str]):
        """
        Warmup phase with random arm exploration.

        Args:
            train_data: Training data (list of input texts)
        """
        logger.info(f"Starting warmup phase: {self.warmup_samples} samples")

        # Temporarily disable learning
        original_learning_state = self.pipeline.enable_learning
        self.pipeline.enable_learning = False

        # Random exploration
        warmup_count = min(self.warmup_samples, len(train_data))

        with tqdm(total=warmup_count, desc="Warmup") as pbar:
            for i in range(warmup_count):
                input_text = train_data[i]

                # Select random arm
                random_arm = np.random.randint(0, len(self.pipeline.arms))

                # Get context
                context = self.pipeline.context_encoder.encode_text(input_text)

                # Generate with random arm
                arm = self.pipeline.arms[random_arm]
                result = self.pipeline.generator.generate(
                    input_text=input_text,
                    intervention=arm,
                    generation_config=None,
                    language='en'  # TODO: Detect language
                )

                # Compute reward
                reward_data = self.pipeline.reward_calculator.calculate(
                    result['text'], input_text
                )

                # Update bandit
                self.pipeline.bandit.update(context, random_arm, reward_data['reward'])

                # Log
                self.total_samples += 1
                pbar.update(1)

                if self.enable_wandb:
                    wandb.log({
                        'warmup/arm': random_arm,
                        'warmup/reward': reward_data['reward'],
                        'warmup/bias_score': reward_data['bias_score'],
                        'warmup/quality_score': reward_data['quality_score'],
                        'step': self.total_samples
                    })

        # Restore learning state
        self.pipeline.enable_learning = original_learning_state
        self.warmup_complete = True

        logger.info(f"Warmup complete: {warmup_count} samples processed")

    def train(
        self,
        train_data: List[str],
        eval_data: Optional[List[str]] = None,
        n_epochs: int = 1
    ) -> Dict:
        """
        Train the MAB system.

        Args:
            train_data: Training data (list of input texts)
            eval_data: Evaluation data (optional)
            n_epochs: Number of training epochs

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Starting training: {len(train_data)} samples, {n_epochs} epochs")

        # Warmup if not done
        if not self.warmup_complete:
            self.warmup(train_data)

        # Enable learning
        self.pipeline.enable_learning = True

        # Training loop
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")

            # Shuffle data
            indices = np.random.permutation(len(train_data))

            with tqdm(total=len(train_data), desc=f"Epoch {epoch + 1}") as pbar:
                for idx in indices:
                    input_text = train_data[idx]

                    # Process sample
                    result = self.pipeline.process(input_text, return_details=True)

                    # Store history
                    self.training_history.append({
                        'step': self.total_samples,
                        'epoch': epoch,
                        'arm': result['selected_arm'],
                        'arm_index': result['arm_index'],
                        'confidence': result['arm_confidence'],
                        'reward': result['reward'],
                        'bias_score': result['bias_score'],
                        'quality_score': result['quality_score'],
                        'generation_time': result['generation_time']
                    })

                    # Log to W&B
                    if self.enable_wandb:
                        wandb.log({
                            'train/reward': result['reward'],
                            'train/bias_score': result['bias_score'],
                            'train/quality_score': result['quality_score'],
                            'train/arm': result['arm_index'],
                            'train/confidence': result['arm_confidence'],
                            'train/generation_time': result['generation_time'],
                            'epoch': epoch,
                            'step': self.total_samples
                        })

                    # Periodic evaluation
                    if eval_data and self.total_samples % self.eval_every == 0:
                        eval_metrics = self.evaluate(eval_data[:50])  # Eval on subset

                        if self.enable_wandb:
                            wandb.log({
                                'eval/reward': eval_metrics['mean_reward'],
                                'eval/bias_score': eval_metrics['mean_bias_score'],
                                'eval/quality_score': eval_metrics['mean_quality_score'],
                                'step': self.total_samples
                            })

                        logger.info(f"Eval @ step {self.total_samples}: "
                                   f"reward={eval_metrics['mean_reward']:.3f}, "
                                   f"bias={eval_metrics['mean_bias_score']:.3f}, "
                                   f"quality={eval_metrics['mean_quality_score']:.3f}")

                    # Periodic checkpoint
                    if self.total_samples % self.save_every == 0:
                        self._save_checkpoint(epoch, self.total_samples)

                    self.total_samples += 1
                    pbar.update(1)

            # End-of-epoch checkpoint
            self._save_checkpoint(epoch, self.total_samples, is_epoch_end=True)

        # Final checkpoint
        self._save_checkpoint(n_epochs - 1, self.total_samples, is_final=True)

        # Compute final metrics
        final_metrics = self._compute_training_metrics()

        # Save training history
        self._save_training_history()

        logger.info("Training complete!")

        return final_metrics

    def evaluate(self, eval_data: List[str]) -> Dict:
        """
        Evaluate on held-out data.

        Args:
            eval_data: Evaluation data

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on {len(eval_data)} samples")

        # Disable learning for evaluation
        original_learning_state = self.pipeline.enable_learning
        self.pipeline.enable_learning = False

        results = []

        for input_text in tqdm(eval_data, desc="Evaluation"):
            result = self.pipeline.process(input_text, return_details=True)
            results.append(result)

        # Restore learning state
        self.pipeline.enable_learning = original_learning_state

        # Compute metrics
        metrics = {
            'mean_reward': np.mean([r['reward'] for r in results]),
            'mean_bias_score': np.mean([r['bias_score'] for r in results]),
            'mean_quality_score': np.mean([r['quality_score'] for r in results]),
            'std_reward': np.std([r['reward'] for r in results]),
            'arm_distribution': self._compute_arm_distribution(results),
            'n_samples': len(results)
        }

        return metrics

    def _compute_arm_distribution(self, results: List[Dict]) -> Dict:
        """Compute arm selection distribution."""
        arm_counts = {}
        for r in results:
            arm = r['selected_arm']
            arm_counts[arm] = arm_counts.get(arm, 0) + 1

        # Normalize
        total = sum(arm_counts.values())
        arm_distribution = {arm: count / total for arm, count in arm_counts.items()}

        return arm_distribution

    def _save_checkpoint(self, epoch: int, step: int, is_epoch_end: bool = False, is_final: bool = False):
        """Save checkpoint."""
        if is_final:
            checkpoint_name = f"bandit_{self.pipeline.bandit_type}_final.pkl"
        elif is_epoch_end:
            checkpoint_name = f"bandit_{self.pipeline.bandit_type}_epoch_{epoch + 1}.pkl"
        else:
            checkpoint_name = f"bandit_{self.pipeline.bandit_type}_step_{step}.pkl"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save bandit state
        self.pipeline.save_state(str(checkpoint_path))

        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        metadata = {
            'epoch': epoch,
            'step': step,
            'total_samples': self.total_samples,
            'timestamp': time.time(),
            'bandit_type': self.pipeline.bandit_type,
            'model_name': self.pipeline.model_name
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _compute_training_metrics(self) -> Dict:
        """Compute overall training metrics."""
        if not self.training_history:
            return {}

        rewards = [h['reward'] for h in self.training_history]
        bias_scores = [h['bias_score'] for h in self.training_history]
        quality_scores = [h['quality_score'] for h in self.training_history]

        # Compute improvement over time
        window_size = 100
        if len(rewards) >= window_size:
            early_reward = np.mean(rewards[:window_size])
            late_reward = np.mean(rewards[-window_size:])
            improvement = late_reward - early_reward
        else:
            improvement = 0.0

        metrics = {
            'total_samples': self.total_samples,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_bias_score': np.mean(bias_scores),
            'mean_quality_score': np.mean(quality_scores),
            'reward_improvement': improvement,
            'final_arm_distribution': self._compute_arm_distribution(self.training_history[-100:]),
            'bandit_statistics': self.pipeline.get_statistics()
        }

        return metrics

    def _save_training_history(self):
        """Save training history to file."""
        history_path = self.results_dir / f'training_history_{self.pipeline.bandit_type}.json'

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"Training history saved: {history_path}")

    def finish(self):
        """Cleanup and finish training."""
        if self.enable_wandb:
            wandb.finish()

        logger.info("Training pipeline finished")
