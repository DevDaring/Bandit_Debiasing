"""
Master orchestrator for running complete experiment on GCP.

Usage:
    python scripts/run_experiment.py --language en --n_epochs 3

This script:
1. Prepares datasets (if not exists)
2. Creates steering vectors (if not exists)
3. Trains all bandit algorithms
4. Evaluates each trained model
5. Generates comparison report
6. Saves all results
"""

import argparse
import logging
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run command and log output.

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)

        logger.info(f"✓ {description} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed!")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def check_file_exists(path: Path) -> bool:
    """Check if file exists."""
    exists = path.exists()
    if exists:
        logger.info(f"✓ Found: {path}")
    else:
        logger.info(f"✗ Not found: {path}")
    return exists


def prepare_datasets(force: bool = False) -> bool:
    """Prepare evaluation datasets."""
    data_dir = Path('./data/bias_evaluation_sets')

    if not force and data_dir.exists() and any(data_dir.iterdir()):
        logger.info("Datasets already exist, skipping preparation")
        return True

    logger.info("Preparing datasets...")
    return run_command(
        ['python', 'scripts/prepare_evaluation_data.py'],
        "Dataset preparation"
    )


def create_steering_vectors(force: bool = False) -> bool:
    """Create steering vectors."""
    vectors_dir = Path('./data/steering_vectors')

    if not force:
        # Check if all vectors exist
        required_vectors = ['gender_steering.pt', 'race_steering.pt', 'religion_steering.pt']
        all_exist = all((vectors_dir / v).exists() for v in required_vectors)

        if all_exist:
            logger.info("Steering vectors already exist, skipping creation")
            return True

    logger.info("Creating steering vectors...")
    return run_command(
        ['python', 'scripts/create_steering_vectors.py'],
        "Steering vector creation"
    )


def train_bandit(
    bandit_type: str,
    train_data: str,
    eval_data: str,
    n_epochs: int,
    warmup_samples: int,
    max_train_samples: int = None
) -> bool:
    """Train a single bandit algorithm."""
    cmd = [
        'python', 'scripts/train_bandit.py',
        '--bandit_type', bandit_type,
        '--train_data', train_data,
        '--eval_data', eval_data,
        '--n_epochs', str(n_epochs),
        '--warmup_samples', str(warmup_samples),
        '--wandb_run_name', f'mab_debiasing_{bandit_type}'
    ]

    if max_train_samples:
        cmd.extend(['--max_train_samples', str(max_train_samples)])

    return run_command(
        cmd,
        f"Training {bandit_type} bandit"
    )


def evaluate_bandit(bandit_type: str, test_data: str, max_samples: int = None) -> bool:
    """Evaluate a trained bandit."""
    checkpoint_path = f'./checkpoints/bandit_{bandit_type}_final.pkl'

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    cmd = [
        'python', 'scripts/evaluate_system.py',
        '--checkpoint', checkpoint_path,
        '--test_data', test_data,
        '--bandit_type', bandit_type,
        '--compare_baselines'
    ]

    if max_samples:
        cmd.extend(['--max_samples', str(max_samples)])

    return run_command(
        cmd,
        f"Evaluating {bandit_type} bandit"
    )


def generate_comparison_report(results_dir: Path) -> Dict:
    """Generate comparison report across all bandit algorithms."""
    logger.info("\nGenerating comparison report...")

    bandit_types = ['linucb', 'thompson', 'neural']
    comparison_data = {}

    for bandit_type in bandit_types:
        eval_file = results_dir / 'evaluation' / f'evaluation_results_{bandit_type}.json'

        if eval_file.exists():
            with open(eval_file, 'r') as f:
                data = json.load(f)
                comparison_data[bandit_type] = data
        else:
            logger.warning(f"Evaluation results not found for {bandit_type}")

    # Create comparison summary
    summary = {
        'timestamp': time.time(),
        'algorithms': list(comparison_data.keys()),
        'comparison': {}
    }

    if comparison_data:
        # Compare learned policies
        for bandit_type, data in comparison_data.items():
            if 'metrics' in data and 'learned' in data['metrics']:
                learned = data['metrics']['learned']
                summary['comparison'][bandit_type] = {
                    'mean_reward': learned['mean_reward'],
                    'mean_bias_score': learned['mean_bias_score'],
                    'mean_quality_score': learned['mean_quality_score'],
                    'arm_distribution': learned['arm_distribution']
                }

                # Add improvements if available
                if 'comparison' in data and 'improvements' in data['comparison']:
                    improvements = data['comparison']['improvements'].get('learned', {})
                    summary['comparison'][bandit_type]['improvements'] = improvements

        # Find best algorithm
        if summary['comparison']:
            best_algorithm = max(
                summary['comparison'].keys(),
                key=lambda k: summary['comparison'][k]['mean_reward']
            )
            summary['best_algorithm'] = best_algorithm
            summary['best_reward'] = summary['comparison'][best_algorithm]['mean_reward']

    # Save comparison report
    report_path = results_dir / 'comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Comparison report saved to {report_path}")

    return summary


def print_final_summary(summary: Dict):
    """Print final experiment summary."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    if 'comparison' in summary and summary['comparison']:
        print("\nAlgorithm Performance:")
        print("-"*60)

        for algo in summary['algorithms']:
            if algo in summary['comparison']:
                comp = summary['comparison'][algo]
                print(f"\n{algo.upper()}:")
                print(f"  Mean Reward: {comp['mean_reward']:.4f}")
                print(f"  Mean Bias Score: {comp['mean_bias_score']:.4f} (lower is better)")
                print(f"  Mean Quality Score: {comp['mean_quality_score']:.4f}")

                if 'improvements' in comp:
                    imp = comp['improvements']
                    print(f"  Improvements over baseline:")
                    print(f"    Reward: +{imp.get('reward_improvement', 0):.4f}")
                    print(f"    Bias reduction: {imp.get('bias_reduction', 0):.4f}")
                    print(f"    Quality preservation: {imp.get('quality_preservation', 1):.2%}")

        if 'best_algorithm' in summary:
            print("\n" + "-"*60)
            print(f"Best Algorithm: {summary['best_algorithm'].upper()}")
            print(f"Best Reward: {summary['best_reward']:.4f}")
            print("-"*60)

    print("\n" + "="*60)


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Run complete MAB debiasing experiment')

    parser.add_argument('--language', type=str, default='en',
                        choices=['en', 'hi', 'bn'],
                        help='Language to use for training/evaluation')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--warmup_samples', type=int, default=100,
                        help='Number of warmup samples')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum training samples (for quick testing)')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='Maximum evaluation samples')

    parser.add_argument('--skip_data_prep', action='store_true',
                        help='Skip dataset preparation')
    parser.add_argument('--skip_steering', action='store_true',
                        help='Skip steering vector creation')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training (only evaluate)')
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['linucb', 'thompson', 'neural'],
                        choices=['linucb', 'thompson', 'neural'],
                        help='Bandit algorithms to train/evaluate')

    parser.add_argument('--force_data_prep', action='store_true',
                        help='Force dataset preparation even if exists')
    parser.add_argument('--force_steering', action='store_true',
                        help='Force steering vector creation even if exists')

    args = parser.parse_args()

    # Create necessary directories
    Path('./logs').mkdir(exist_ok=True)
    Path('./results').mkdir(exist_ok=True)
    Path('./checkpoints').mkdir(exist_ok=True)

    logger.info("="*60)
    logger.info("MAB DEBIASING - COMPLETE EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Language: {args.language}")
    logger.info(f"Epochs: {args.n_epochs}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info("="*60)

    start_time = time.time()

    # Step 1: Prepare datasets
    if not args.skip_data_prep:
        success = prepare_datasets(force=args.force_data_prep)
        if not success:
            logger.error("Dataset preparation failed!")
            return 1
    else:
        logger.info("Skipping dataset preparation")

    # Step 2: Create steering vectors
    if not args.skip_steering:
        success = create_steering_vectors(force=args.force_steering)
        if not success:
            logger.error("Steering vector creation failed!")
            return 1
    else:
        logger.info("Skipping steering vector creation")

    # Define data paths
    data_dir = Path(f'./data/bias_evaluation_sets/{args.language}')
    train_data = str(data_dir / 'train.json')
    eval_data = str(data_dir / 'validation.json')
    test_data = str(data_dir / 'test.json')

    # Verify data files exist
    for path_str in [train_data, eval_data, test_data]:
        if not check_file_exists(Path(path_str)):
            logger.error(f"Required data file missing: {path_str}")
            return 1

    # Step 3: Train all bandit algorithms
    if not args.skip_training:
        for bandit_type in args.algorithms:
            success = train_bandit(
                bandit_type=bandit_type,
                train_data=train_data,
                eval_data=eval_data,
                n_epochs=args.n_epochs,
                warmup_samples=args.warmup_samples,
                max_train_samples=args.max_train_samples
            )

            if not success:
                logger.error(f"Training failed for {bandit_type}")
                # Continue with other algorithms
    else:
        logger.info("Skipping training")

    # Step 4: Evaluate all trained models
    for bandit_type in args.algorithms:
        success = evaluate_bandit(
            bandit_type=bandit_type,
            test_data=test_data,
            max_samples=args.max_eval_samples
        )

        if not success:
            logger.warning(f"Evaluation failed for {bandit_type}")
            # Continue with other algorithms

    # Step 5: Generate comparison report
    results_dir = Path('./results')
    summary = generate_comparison_report(results_dir)

    # Print final summary
    print_final_summary(summary)

    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info(f"\nTotal experiment time: {hours}h {minutes}m {seconds}s")
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("="*60)
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Logs saved to: ./logs/")
    logger.info(f"Checkpoints saved to: ./checkpoints/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
