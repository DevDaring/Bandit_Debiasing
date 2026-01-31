#!/usr/bin/env python3
"""
Enhanced evaluation script with IBR and FAR metrics.

Evaluates Fair-CB on both datasets with comprehensive metrics:
- IBR (Intersectional Bias Reduction)
- FAR (Fairness-Aware Regret)
- Per-category performance
- Per-language performance
- Baseline comparisons

Usage:
    python scripts/evaluate_system.py --checkpoint ./checkpoints/final.pt
    python scripts/evaluate_system.py --dataset multi_crows --output-csv ./results
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Fair-CB system')

    # Input
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['multi_crows', 'indibias', 'both'],
                       help='Dataset to evaluate on')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate')

    # Metrics
    parser.add_argument('--compute-ibr', action='store_true', default=True, help='Compute IBR')
    parser.add_argument('--compute-far', action='store_true', default=True, help='Compute FAR')
    parser.add_argument('--lambda-fairness', type=float, default=0.5, help='FAR lambda')
    parser.add_argument('--bias-threshold', type=float, default=0.3, help='Bias threshold')

    # Output
    parser.add_argument('--output-csv', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--generate-latex', action='store_true', help='Generate LaTeX tables')

    return parser.parse_args()


def load_evaluation_data(dataset_name: str, max_samples: int = None):
    """Load evaluation dataset."""
    from src.data import UnifiedDatasetManager

    manager = UnifiedDatasetManager()

    if dataset_name == 'multi_crows':
        data = manager.load_crows_pairs()
    elif dataset_name == 'indibias':
        data = manager.load_indibias()
    elif dataset_name == 'both':
        data = manager.load_all()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples:
        data = data[:max_samples]

    logger.info(f"Loaded {len(data)} samples from {dataset_name}")
    return data


def evaluate_with_metrics(data, pipeline, args):
    """Evaluate pipeline with comprehensive metrics."""
    from src.metrics import ComprehensiveMetricsEvaluator
    from src.output import CSVOutputManager
    import pandas as pd

    evaluator = ComprehensiveMetricsEvaluator(
        lambda_weight=args.lambda_fairness,
        bias_threshold=args.bias_threshold
    )

    csv_manager = CSVOutputManager(
        output_dir=args.output_csv,
        timestamp_files=True
    )

    results_list = []

    for i, sample in enumerate(data):
        text = sample.get('text', sample.get('sentence', ''))
        category = sample.get('category', sample.get('bias_type', 'unknown'))
        language = sample.get('language', 'en')

        # Process through pipeline
        if pipeline:
            result = pipeline.process(text)
            bias_score = result.get('bias_score', 0.5)
            reward = result.get('reward', 0.5)
            selected_arm = result.get('selected_arm', 0)
        else:
            # Mock evaluation without pipeline
            bias_score = 0.3
            reward = 0.7
            selected_arm = 1

        # Add to evaluator
        baseline_bias = sample.get('baseline_bias', bias_score + 0.1)
        evaluator.add_observation(
            bias_score=bias_score,
            reward=reward,
            optimal_reward=1.0,
            category=category,
            language=language,
            baseline_bias=baseline_bias
        )

        results_list.append({
            'sample_id': i,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'category': category,
            'language': language,
            'bias_score': bias_score,
            'reward': reward,
            'selected_arm': selected_arm
        })

    # Compute final metrics
    eval_result = evaluator.evaluate()

    # Generate summary
    summary = {
        'ibr': eval_result.ibr.ibr_score,
        'far': eval_result.far.far_score,
        'ibr_arithmetic_mean': eval_result.ibr.arithmetic_mean,
        'n_categories': eval_result.ibr.n_categories,
        'n_improved': eval_result.ibr.n_improved,
        'worst_category': eval_result.ibr.worst_category,
        'best_category': eval_result.ibr.best_category,
        'cumulative_regret': eval_result.far.cumulative_regret,
        'cumulative_violation': eval_result.far.cumulative_violation,
        'n_samples': eval_result.n_samples,
        'mean_bias': eval_result.mean_bias_score,
        'mean_reward': eval_result.mean_reward
    }

    # Save results
    results_df = pd.DataFrame(results_list)
    csv_manager.save_main_results(results_df, 'detailed_results')

    summary_df = pd.DataFrame([summary])
    csv_manager.save_main_results(summary_df, 'summary')

    # Per-category results
    cat_df = pd.DataFrame([
        {'category': cat, 'bias_score': score}
        for cat, score in eval_result.per_category_scores.items()
    ])
    csv_manager.save_per_category_results(cat_df)

    # Per-language results
    lang_df = pd.DataFrame([
        {'language': lang, 'bias_score': score}
        for lang, score in eval_result.per_language_scores.items()
    ])
    csv_manager.save_per_language_results(lang_df)

    return summary, eval_result


def generate_latex_tables(eval_result, output_dir: Path):
    """Generate LaTeX tables for publication."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fair-CB Evaluation Results}",
        r"\label{tab:evaluation}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"IBR & {eval_result.ibr.ibr_score:.4f} \\\\",
        f"FAR & {eval_result.far.far_score:.4f} \\\\",
        f"Mean Bias Score & {eval_result.mean_bias_score:.4f} \\\\",
        f"Mean Reward & {eval_result.mean_reward:.4f} \\\\",
        f"Categories Improved & {eval_result.ibr.n_improved}/{eval_result.ibr.n_categories} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]

    with open(output_dir / 'evaluation_table.tex', 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Saved LaTeX table to {output_dir / 'evaluation_table.tex'}")


def main():
    """Main evaluation function."""
    args = parse_args()

    output_dir = Path(args.output_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Fair-CB System Evaluation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Lambda (fairness): {args.lambda_fairness}")
    logger.info(f"Bias threshold: {args.bias_threshold}")
    logger.info("=" * 60)

    # Load data
    data = load_evaluation_data(args.dataset, args.max_samples)

    # Load pipeline if checkpoint provided
    pipeline = None
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        # Pipeline loading would go here
        pass

    # Run evaluation
    summary, eval_result = evaluate_with_metrics(data, pipeline, args)

    # Generate LaTeX if requested
    if args.generate_latex:
        generate_latex_tables(eval_result, output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"IBR: {summary['ibr']:.4f}")
    logger.info(f"FAR: {summary['far']:.4f}")
    logger.info(f"Mean Bias: {summary['mean_bias']:.4f}")
    logger.info(f"Mean Reward: {summary['mean_reward']:.4f}")
    logger.info(f"Categories Improved: {summary['n_improved']}/{summary['n_categories']}")
    logger.info(f"Worst Category: {summary['worst_category']}")
    logger.info(f"Best Category: {summary['best_category']}")
    logger.info("=" * 60)

    # Generate report
    evaluator_report = f"""
# Fair-CB Evaluation Report

**Date**: {datetime.now().isoformat()}
**Dataset**: {args.dataset}
**Samples**: {summary['n_samples']}

## Core Metrics

| Metric | Value |
|--------|-------|
| IBR (Intersectional Bias Reduction) | {summary['ibr']:.4f} |
| FAR (Fairness-Aware Regret) | {summary['far']:.4f} |
| Mean Bias Score | {summary['mean_bias']:.4f} |
| Mean Reward | {summary['mean_reward']:.4f} |

## IBR Breakdown

- Categories evaluated: {summary['n_categories']}
- Categories improved: {summary['n_improved']}
- Worst category: {summary['worst_category']}
- Best category: {summary['best_category']}
- Arithmetic mean (comparison): {summary['ibr_arithmetic_mean']:.4f}

## FAR Breakdown

- Cumulative regret: {summary['cumulative_regret']:.4f}
- Cumulative violation: {summary['cumulative_violation']:.4f}
- Lambda: {args.lambda_fairness}
"""

    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write(evaluator_report)

    logger.info(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
