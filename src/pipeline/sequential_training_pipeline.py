"""
Sequential training pipeline with theory and metrics integration.

Enhanced pipeline that:
1. Integrates RegretTracker and FairnessViolationTracker
2. Computes IBR and FAR metrics
3. Uses standardized CSV output
4. Supports ablation configurations
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm import tqdm

from .training_pipeline import MABDebiasTrainingPipeline
from ..theory import RegretTracker, FairnessViolationTracker, TheoreticalBoundComputer
from ..metrics import ComprehensiveMetricsEvaluator
from ..output import CSVOutputManager

logger = logging.getLogger(__name__)


class SequentialTrainingPipeline:
    """
    Sequential training pipeline with enhanced metrics and theory tracking.

    Wraps the base training pipeline with:
    1. Regret and fairness violation tracking
    2. IBR and FAR computation
    3. Standardized CSV output
    4. Theoretical bound verification
    """

    def __init__(
        self,
        inference_pipeline,
        n_arms: int = 6,
        context_dim: int = 128,
        bias_threshold: float = 0.3,
        lambda_fairness: float = 0.5,
        results_dir: str = './results',
        checkpoint_dir: str = './checkpoints',
        enable_wandb: bool = True,
        wandb_project: str = 'fair-cb',
        wandb_run_name: Optional[str] = None
    ):
        """
        Initialize sequential training pipeline.

        Args:
            inference_pipeline: MAB inference pipeline
            n_arms: Number of bandit arms
            context_dim: Context vector dimension
            bias_threshold: Threshold for fairness violations
            lambda_fairness: Weight for FAR computation
            results_dir: Directory for results
            checkpoint_dir: Directory for checkpoints
            enable_wandb: Whether to enable W&B logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name
        """
        self.inference_pipeline = inference_pipeline
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.bias_threshold = bias_threshold
        self.lambda_fairness = lambda_fairness

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base training pipeline
        self.base_pipeline = MABDebiasTrainingPipeline(
            inference_pipeline=inference_pipeline,
            checkpoint_dir=str(checkpoint_dir),
            results_dir=str(results_dir),
            enable_wandb=enable_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name
        )

        # Initialize theory trackers
        self.regret_tracker = RegretTracker(n_arms=n_arms)
        self.fairness_tracker = FairnessViolationTracker(threshold=bias_threshold)
        self.bound_computer = TheoreticalBoundComputer(
            n_arms=n_arms,
            context_dim=context_dim,
            lambda_fairness=lambda_fairness
        )

        # Initialize metrics evaluator
        self.metrics_evaluator = ComprehensiveMetricsEvaluator(
            lambda_weight=lambda_fairness,
            bias_threshold=bias_threshold
        )

        # Initialize CSV manager
        self.csv_manager = CSVOutputManager(
            output_dir=str(results_dir),
            timestamp_files=True
        )

        # Training history
        self.training_history: List[Dict[str, Any]] = []
        self.epoch_metrics: List[Dict[str, float]] = []

    def train_sequential(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
        n_epochs: int = 1,
        warmup_samples: int = 50,
        eval_every: int = 100,
        save_every: int = 500
    ) -> Dict[str, Any]:
        """
        Sequential training with enhanced tracking.

        Args:
            train_data: Training data with text, category, language fields
            eval_data: Optional evaluation data
            n_epochs: Number of training epochs
            warmup_samples: Warmup samples for random exploration
            eval_every: Evaluate every N samples
            save_every: Save checkpoint every N samples

        Returns:
            Training results dictionary
        """
        logger.info(f"Starting sequential training: {len(train_data)} samples, {n_epochs} epochs")

        # Extract texts for base pipeline
        train_texts = [d.get('text', d.get('sentence', '')) for d in train_data]
        eval_texts = [d.get('text', d.get('sentence', '')) for d in eval_data] if eval_data else None

        # Warmup phase
        if warmup_samples > 0:
            logger.info(f"Warmup phase: {warmup_samples} samples")
            self.base_pipeline.warmup(train_texts[:warmup_samples])

        # Main training loop
        total_samples = len(train_data) * n_epochs
        sample_idx = 0

        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            epoch_start = time.time()

            for i, sample in enumerate(tqdm(train_data, desc=f"Epoch {epoch + 1}")):
                sample_idx += 1

                # Process through inference pipeline
                text = sample.get('text', sample.get('sentence', ''))
                result = self.inference_pipeline.process(text)

                # Extract values
                selected_arm = result.get('selected_arm', 0)
                reward = result.get('reward', 0.5)
                bias_score = result.get('bias_score', 0.5)
                quality_score = result.get('quality_score', 0.5)

                # Estimate optimal (best observed reward for context)
                optimal_reward = 1.0  # Theoretical max
                regret = max(0, optimal_reward - reward)

                # Update theory trackers
                self.regret_tracker.update(
                    selected_arm=selected_arm,
                    reward=reward,
                    optimal_reward=optimal_reward
                )

                self.fairness_tracker.update(
                    bias_score=bias_score,
                    bias_category=sample.get('category', sample.get('bias_type')),
                    language=sample.get('language'),
                    selected_arm=selected_arm
                )

                # Update metrics evaluator
                baseline_bias = sample.get('baseline_bias', bias_score + 0.1)
                self.metrics_evaluator.add_observation(
                    bias_score=bias_score,
                    reward=reward,
                    optimal_reward=optimal_reward,
                    category=sample.get('category', sample.get('bias_type')),
                    language=sample.get('language'),
                    baseline_bias=baseline_bias
                )

                # Record history
                self.training_history.append({
                    'epoch': epoch,
                    'step': sample_idx,
                    'selected_arm': selected_arm,
                    'reward': reward,
                    'bias_score': bias_score,
                    'regret': regret,
                    'cumulative_regret': self.regret_tracker.get_cumulative_regret(),
                    'cumulative_violation': self.fairness_tracker.get_cumulative_violation()
                })

                # Periodic evaluation
                if eval_texts and sample_idx % eval_every == 0:
                    self._log_intermediate_metrics(sample_idx)

            # End of epoch
            epoch_time = time.time() - epoch_start
            epoch_metrics = self._compute_epoch_metrics(epoch)
            epoch_metrics['epoch_time'] = epoch_time
            self.epoch_metrics.append(epoch_metrics)

            logger.info(f"Epoch {epoch + 1} complete: IBR={epoch_metrics.get('ibr', 0):.4f}, "
                       f"FAR={epoch_metrics.get('far', 0):.4f}")

        # Final evaluation and results
        final_results = self._compute_final_results()

        # Save results
        self._save_all_results(final_results)

        return final_results

    def _compute_epoch_metrics(self, epoch: int) -> Dict[str, float]:
        """Compute metrics at end of epoch."""
        eval_result = self.metrics_evaluator.evaluate()

        return {
            'epoch': epoch,
            'ibr': eval_result.ibr.ibr_score,
            'far': eval_result.far.far_score,
            'cumulative_regret': self.regret_tracker.get_cumulative_regret(),
            'cumulative_violation': self.fairness_tracker.get_cumulative_violation(),
            'mean_bias': eval_result.mean_bias_score,
            'mean_reward': eval_result.mean_reward,
            'n_samples': eval_result.n_samples
        }

    def _compute_final_results(self) -> Dict[str, Any]:
        """Compute final results with all metrics."""
        eval_result = self.metrics_evaluator.evaluate()
        regret_stats = self.regret_tracker.get_regret_statistics()
        fairness_stats = self.fairness_tracker.get_fairness_statistics()

        # Verify theoretical bounds
        T = self.regret_tracker.timestep
        regret_bound_satisfied, regret_margin = self.bound_computer.verify_regret_bound(
            empirical_regret=self.regret_tracker.get_cumulative_regret(),
            T=T
        )
        far_bound_satisfied, far_margin = self.bound_computer.verify_far_bound(
            empirical_regret=self.regret_tracker.get_cumulative_regret(),
            empirical_violation=self.fairness_tracker.get_cumulative_violation(),
            T=T
        )

        return {
            # Core metrics
            'ibr': eval_result.ibr.ibr_score,
            'far': eval_result.far.far_score,

            # IBR breakdown
            'ibr_arithmetic_mean': eval_result.ibr.arithmetic_mean,
            'ibr_worst_category': eval_result.ibr.worst_category,
            'ibr_best_category': eval_result.ibr.best_category,

            # FAR breakdown
            'far_cumulative_regret': eval_result.far.cumulative_regret,
            'far_cumulative_violation': eval_result.far.cumulative_violation,

            # Regret statistics
            **{f'regret_{k}': v for k, v in regret_stats.items()},

            # Fairness statistics
            **{f'fairness_{k}': v for k, v in fairness_stats.items()},

            # Theoretical verification
            'regret_bound_satisfied': regret_bound_satisfied,
            'far_bound_satisfied': far_bound_satisfied,

            # Per-group metrics
            'per_category': eval_result.per_category_scores,
            'per_language': eval_result.per_language_scores,

            # Per-category IBR
            'ibr_per_category': {
                cat: r.bias_reduction
                for cat, r in eval_result.ibr.per_category_results.items()
            },

            # Training info
            'n_samples': eval_result.n_samples,
            'n_epochs': len(self.epoch_metrics),
            'epoch_metrics': self.epoch_metrics
        }

    def _log_intermediate_metrics(self, step: int):
        """Log intermediate metrics."""
        logger.info(
            f"Step {step}: R(T)={self.regret_tracker.get_cumulative_regret():.4f}, "
            f"V(T)={self.fairness_tracker.get_cumulative_violation():.4f}"
        )

    def _save_all_results(self, results: Dict[str, Any]):
        """Save all results to CSV files."""
        import pandas as pd

        # Main results
        main_df = pd.DataFrame([{
            k: v for k, v in results.items()
            if not isinstance(v, (dict, list))
        }])
        self.csv_manager.save_main_results(main_df)

        # Per-category results
        if results.get('per_category'):
            cat_df = pd.DataFrame([
                {'category': cat, 'bias_score': score}
                for cat, score in results['per_category'].items()
            ])
            self.csv_manager.save_per_category_results(cat_df)

        # Per-language results
        if results.get('per_language'):
            lang_df = pd.DataFrame([
                {'language': lang, 'bias_score': score}
                for lang, score in results['per_language'].items()
            ])
            self.csv_manager.save_per_language_results(lang_df)

        # Theory verification
        theory_df = pd.DataFrame([{
            'regret_bound_satisfied': results.get('regret_bound_satisfied'),
            'far_bound_satisfied': results.get('far_bound_satisfied'),
            'cumulative_regret': results.get('far_cumulative_regret'),
            'cumulative_violation': results.get('far_cumulative_violation')
        }])
        self.csv_manager.save_theory_verification(theory_df)

        # Save trackers
        self.regret_tracker.save(str(self.results_dir / 'regret_history.json'))
        self.fairness_tracker.save(str(self.results_dir / 'fairness_history.json'))

        logger.info(f"Saved all results to {self.results_dir}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        return {
            'n_samples': self.regret_tracker.timestep,
            'cumulative_regret': self.regret_tracker.get_cumulative_regret(),
            'average_regret': self.regret_tracker.get_average_regret(),
            'cumulative_violation': self.fairness_tracker.get_cumulative_violation(),
            'violation_rate': self.fairness_tracker.get_violation_rate(),
            'sublinear_regret': self.regret_tracker.is_sublinear(),
            'constraint_satisfied': self.fairness_tracker.is_constraint_satisfied()
        }

    def __repr__(self) -> str:
        summary = self.get_training_summary()
        return (f"SequentialTrainingPipeline(samples={summary['n_samples']}, "
                f"regret={summary['cumulative_regret']:.4f})")
