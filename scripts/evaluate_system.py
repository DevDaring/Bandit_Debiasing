"""
Comprehensive evaluation script.
Compare trained bandit against baselines.
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.model_config import ModelConfig
from config.bandit_config import BanditConfig
from src.pipeline.inference_pipeline import MABDebiasInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(dataset_path: str, max_samples: int = None) -> List[str]:
    """Load test data from JSON file."""
    logger.info(f"Loading test data from {dataset_path}...")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract input texts
    if isinstance(data, list):
        texts = [item['input'] if isinstance(item, dict) else item for item in data]
    else:
        raise ValueError(f"Unexpected data format in {dataset_path}")

    if max_samples:
        texts = texts[:max_samples]

    logger.info(f"Loaded {len(texts)} test samples")
    return texts


def evaluate_pipeline(
    pipeline: MABDebiasInferencePipeline,
    test_data: List[str],
    mode: str = 'learned'
) -> Dict:
    """
    Evaluate pipeline on test data.

    Args:
        pipeline: Inference pipeline
        test_data: Test data
        mode: 'learned', 'random', 'no_intervention', or arm index

    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating in {mode} mode on {len(test_data)} samples...")

    results = []
    start_time = time.time()

    # Disable learning during evaluation
    pipeline.enable_learning = False

    for input_text in tqdm(test_data, desc=f"Eval ({mode})"):
        # Process based on mode
        if mode == 'learned':
            # Use trained bandit
            result = pipeline.process(input_text, return_details=True)

        elif mode == 'random':
            # Random arm selection
            context = pipeline.context_encoder.encode_text(input_text)
            random_arm_idx = np.random.randint(0, len(pipeline.arms))
            arm = pipeline.arms[random_arm_idx]

            gen_result = pipeline.generator.generate(
                input_text=input_text,
                intervention=arm,
                generation_config=None,
                language='en'
            )

            reward_data = pipeline.reward_calculator.calculate(gen_result['text'], input_text)

            result = {
                'response': gen_result['text'],
                'selected_arm': arm.name,
                'arm_index': random_arm_idx,
                'arm_confidence': 0.0,
                'reward': reward_data['reward'],
                'bias_score': reward_data['bias_score'],
                'quality_score': reward_data['quality_score'],
                'generation_time': gen_result['time_seconds']
            }

        elif mode == 'no_intervention':
            # Force no intervention arm (Arm 0)
            arm = pipeline.arms[0]
            gen_result = pipeline.generator.generate(
                input_text=input_text,
                intervention=arm,
                generation_config=None,
                language='en'
            )

            reward_data = pipeline.reward_calculator.calculate(gen_result['text'], input_text)

            result = {
                'response': gen_result['text'],
                'selected_arm': arm.name,
                'arm_index': 0,
                'arm_confidence': 1.0,
                'reward': reward_data['reward'],
                'bias_score': reward_data['bias_score'],
                'quality_score': reward_data['quality_score'],
                'generation_time': gen_result['time_seconds']
            }

        elif isinstance(mode, int):
            # Force specific arm
            arm = pipeline.arms[mode]
            gen_result = pipeline.generator.generate(
                input_text=input_text,
                intervention=arm,
                generation_config=None,
                language='en'
            )

            reward_data = pipeline.reward_calculator.calculate(gen_result['text'], input_text)

            result = {
                'response': gen_result['text'],
                'selected_arm': arm.name,
                'arm_index': mode,
                'arm_confidence': 1.0,
                'reward': reward_data['reward'],
                'bias_score': reward_data['bias_score'],
                'quality_score': reward_data['quality_score'],
                'generation_time': gen_result['time_seconds']
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

        results.append(result)

    total_time = time.time() - start_time

    # Compute metrics
    metrics = {
        'mode': mode,
        'n_samples': len(results),
        'mean_reward': np.mean([r['reward'] for r in results]),
        'std_reward': np.std([r['reward'] for r in results]),
        'mean_bias_score': np.mean([r['bias_score'] for r in results]),
        'std_bias_score': np.std([r['bias_score'] for r in results]),
        'mean_quality_score': np.mean([r['quality_score'] for r in results]),
        'std_quality_score': np.std([r['quality_score'] for r in results]),
        'mean_generation_time': np.mean([r['generation_time'] for r in results]),
        'total_time': total_time,
        'arm_distribution': compute_arm_distribution(results),
        'results': results
    }

    return metrics


def compute_arm_distribution(results: List[Dict]) -> Dict:
    """Compute arm selection distribution."""
    arm_counts = {}
    for r in results:
        arm = r['selected_arm']
        arm_counts[arm] = arm_counts.get(arm, 0) + 1

    total = sum(arm_counts.values())
    arm_distribution = {arm: count / total for arm, count in arm_counts.items()}

    return arm_distribution


def compare_methods(all_metrics: Dict[str, Dict]) -> Dict:
    """Compare different evaluation methods."""
    comparison = {
        'methods': list(all_metrics.keys()),
        'rewards': {method: metrics['mean_reward'] for method, metrics in all_metrics.items()},
        'bias_scores': {method: metrics['mean_bias_score'] for method, metrics in all_metrics.items()},
        'quality_scores': {method: metrics['mean_quality_score'] for method, metrics in all_metrics.items()}
    }

    # Compute improvements over baseline
    if 'no_intervention' in all_metrics:
        baseline_reward = all_metrics['no_intervention']['mean_reward']
        baseline_bias = all_metrics['no_intervention']['mean_bias_score']
        baseline_quality = all_metrics['no_intervention']['mean_quality_score']

        comparison['improvements'] = {}
        for method, metrics in all_metrics.items():
            if method != 'no_intervention':
                comparison['improvements'][method] = {
                    'reward_improvement': metrics['mean_reward'] - baseline_reward,
                    'bias_reduction': baseline_bias - metrics['mean_bias_score'],
                    'quality_preservation': metrics['mean_quality_score'] / baseline_quality
                }

    return comparison


def plot_comparison(all_metrics: Dict[str, Dict], output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(all_metrics.keys())
    rewards = [all_metrics[m]['mean_reward'] for m in methods]
    bias_scores = [all_metrics[m]['mean_bias_score'] for m in methods]
    quality_scores = [all_metrics[m]['mean_quality_score'] for m in methods]

    # Plot 1: Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(methods, rewards)
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Mean Reward by Method')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(methods, bias_scores)
    axes[1].set_ylabel('Bias Score')
    axes[1].set_title('Mean Bias Score by Method (lower is better)')
    axes[1].tick_params(axis='x', rotation=45)

    axes[2].bar(methods, quality_scores)
    axes[2].set_ylabel('Quality Score')
    axes[2].set_title('Mean Quality Score by Method')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comparison plot to {output_dir / 'method_comparison.png'}")

    # Plot 2: Scatter plot (bias vs quality)
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.scatter(
            all_metrics[method]['mean_bias_score'],
            all_metrics[method]['mean_quality_score'],
            s=100,
            label=method
        )

    plt.xlabel('Bias Score (lower is better)')
    plt.ylabel('Quality Score (higher is better)')
    plt.title('Bias vs Quality Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'bias_quality_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved scatter plot to {output_dir / 'bias_quality_tradeoff.png'}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate MAB Debiasing System')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained bandit checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data JSON file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum test samples to use')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--bandit_type', type=str, default='linucb',
                        choices=['linucb', 'thompson', 'neural'],
                        help='Bandit algorithm type')
    parser.add_argument('--results_dir', type=str, default='./results/evaluation',
                        help='Directory for results')
    parser.add_argument('--compare_baselines', action='store_true',
                        help='Compare against baseline methods')

    args = parser.parse_args()

    # Create output directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("MAB DEBIASING EVALUATION")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test data: {args.test_data}")
    logger.info("="*60)

    # Load test data
    test_data = load_test_data(args.test_data, args.max_samples)

    # Create pipeline
    logger.info("Initializing pipeline...")
    pipeline = MABDebiasInferencePipeline(
        model_name=args.model_name,
        bandit_type=args.bandit_type,
        enable_learning=False
    )

    # Load components
    logger.info("Loading components...")
    pipeline.load_components()

    # Load trained checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    pipeline.load_state(args.checkpoint)

    # Evaluate learned policy
    logger.info("\nEvaluating learned policy...")
    learned_metrics = evaluate_pipeline(pipeline, test_data, mode='learned')

    all_metrics = {'learned': learned_metrics}

    # Compare with baselines
    if args.compare_baselines:
        logger.info("\nEvaluating baseline: no intervention...")
        baseline_metrics = evaluate_pipeline(pipeline, test_data, mode='no_intervention')
        all_metrics['no_intervention'] = baseline_metrics

        logger.info("\nEvaluating baseline: random selection...")
        random_metrics = evaluate_pipeline(pipeline, test_data, mode='random')
        all_metrics['random'] = random_metrics

    # Compute comparison
    comparison = compare_methods(all_metrics)

    # Save results
    results_path = results_dir / f'evaluation_results_{args.bandit_type}.json'
    with open(results_path, 'w') as f:
        # Remove full results for cleaner JSON
        save_metrics = {}
        for method, metrics in all_metrics.items():
            save_metrics[method] = {k: v for k, v in metrics.items() if k != 'results'}

        output_data = {
            'metrics': save_metrics,
            'comparison': comparison
        }
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Create plots
    if args.compare_baselines:
        logger.info("\nGenerating comparison plots...")
        plot_comparison(all_metrics, results_dir / 'figures')

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    for method, metrics in all_metrics.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        logger.info(f"  Mean Bias Score: {metrics['mean_bias_score']:.4f} ± {metrics['std_bias_score']:.4f}")
        logger.info(f"  Mean Quality Score: {metrics['mean_quality_score']:.4f} ± {metrics['std_quality_score']:.4f}")
        logger.info(f"  Arm Distribution:")
        for arm, prob in metrics['arm_distribution'].items():
            logger.info(f"    {arm}: {prob:.3f}")

    if 'improvements' in comparison:
        logger.info("\nIMPROVEMENTS OVER BASELINE:")
        for method, improvements in comparison['improvements'].items():
            logger.info(f"\n{method.upper()}:")
            logger.info(f"  Reward improvement: {improvements['reward_improvement']:.4f}")
            logger.info(f"  Bias reduction: {improvements['bias_reduction']:.4f}")
            logger.info(f"  Quality preservation: {improvements['quality_preservation']:.2%}")

    logger.info("="*60)

    # Cleanup
    pipeline.unload()

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
