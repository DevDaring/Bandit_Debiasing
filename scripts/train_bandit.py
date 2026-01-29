"""
Main training script with command-line arguments.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List

from config.model_config import ModelConfig
from config.bandit_config import BanditConfig
from src.pipeline.inference_pipeline import MABDebiasInferencePipeline
from src.pipeline.training_pipeline import MABDebiasTrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(dataset_path: str, max_samples: int = None) -> List[str]:
    """Load training data from JSON file."""
    logger.info(f"Loading training data from {dataset_path}...")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract input texts
    if isinstance(data, list):
        texts = [item['input'] if isinstance(item, dict) else item for item in data]
    else:
        raise ValueError(f"Unexpected data format in {dataset_path}")

    if max_samples:
        texts = texts[:max_samples]

    logger.info(f"Loaded {len(texts)} training samples")
    return texts


def load_eval_data(dataset_path: str, max_samples: int = None) -> List[str]:
    """Load evaluation data from JSON file."""
    logger.info(f"Loading evaluation data from {dataset_path}...")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract input texts
    if isinstance(data, list):
        texts = [item['input'] if isinstance(item, dict) else item for item in data]
    else:
        raise ValueError(f"Unexpected data format in {dataset_path}")

    if max_samples:
        texts = texts[:max_samples]

    logger.info(f"Loaded {len(texts)} evaluation samples")
    return texts


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MAB Debiasing System')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')

    # Bandit arguments
    parser.add_argument('--bandit_type', type=str, default='linucb',
                        choices=['linucb', 'thompson', 'neural'],
                        help='Bandit algorithm type')
    parser.add_argument('--linucb_alpha', type=float, default=0.5,
                        help='LinUCB exploration parameter')
    parser.add_argument('--ts_noise_std', type=float, default=0.1,
                        help='Thompson Sampling noise std')
    parser.add_argument('--neural_hidden_dim', type=int, default=64,
                        help='Neural bandit hidden dimension')
    parser.add_argument('--neural_learning_rate', type=float, default=0.001,
                        help='Neural bandit learning rate')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data JSON file')
    parser.add_argument('--eval_data', type=str, default=None,
                        help='Path to evaluation data JSON file')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum training samples to use')
    parser.add_argument('--max_eval_samples', type=int, default=100,
                        help='Maximum evaluation samples to use')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--warmup_samples', type=int, default=100,
                        help='Number of warmup samples')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N samples')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N samples')

    # Reward arguments
    parser.add_argument('--bias_weight', type=float, default=0.6,
                        help='Weight for bias reduction in reward')
    parser.add_argument('--quality_weight', type=float, default=0.4,
                        help='Weight for quality in reward')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for results')

    # W&B arguments
    parser.add_argument('--wandb_project', type=str, default='mab-debiasing',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')

    args = parser.parse_args()

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("MAB DEBIASING TRAINING")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Bandit: {args.bandit_type}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Epochs: {args.n_epochs}")
    logger.info(f"Warmup samples: {args.warmup_samples}")
    logger.info("="*60)

    # Load data
    train_data = load_training_data(args.train_data, args.max_train_samples)

    eval_data = None
    if args.eval_data:
        eval_data = load_eval_data(args.eval_data, args.max_eval_samples)

    # Create bandit config
    bandit_config = BanditConfig()
    bandit_config.linucb_alpha = args.linucb_alpha
    bandit_config.ts_noise_std = args.ts_noise_std
    bandit_config.neural_hidden_dim = args.neural_hidden_dim
    bandit_config.neural_learning_rate = args.neural_learning_rate
    bandit_config.bias_weight = args.bias_weight
    bandit_config.quality_weight = args.quality_weight

    # Create inference pipeline
    logger.info("Initializing inference pipeline...")
    inference_pipeline = MABDebiasInferencePipeline(
        model_name=args.model_name,
        bandit_type=args.bandit_type,
        bandit_config=bandit_config,
        enable_learning=True
    )

    # Load components
    logger.info("Loading pipeline components...")
    inference_pipeline.load_components()
    logger.info("All components loaded successfully!")

    # Create training pipeline
    logger.info("Initializing training pipeline...")
    training_pipeline = MABDebiasTrainingPipeline(
        inference_pipeline=inference_pipeline,
        warmup_samples=args.warmup_samples,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        enable_wandb=not args.no_wandb
    )

    # Train
    logger.info("Starting training...")
    try:
        metrics = training_pipeline.train(
            train_data=train_data,
            eval_data=eval_data,
            n_epochs=args.n_epochs
        )

        # Save final metrics
        metrics_path = Path(args.results_dir) / f'final_metrics_{args.bandit_type}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Final metrics saved to {metrics_path}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total samples processed: {metrics['total_samples']}")
        logger.info(f"Mean reward: {metrics['mean_reward']:.4f}")
        logger.info(f"Mean bias score: {metrics['mean_bias_score']:.4f}")
        logger.info(f"Mean quality score: {metrics['mean_quality_score']:.4f}")
        logger.info(f"Reward improvement: {metrics['reward_improvement']:.4f}")
        logger.info("\nFinal arm distribution:")
        for arm, prob in metrics['final_arm_distribution'].items():
            logger.info(f"  {arm}: {prob:.3f}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        training_pipeline.finish()
        inference_pipeline.unload()

    logger.info("Training script finished!")


if __name__ == "__main__":
    main()
