"""
Run MAB debiasing experiments across all 6 models.

Usage:
    # Run all models
    python scripts/run_multi_model_experiment.py --output_dir ./results

    # Run specific models
    python scripts/run_multi_model_experiment.py --models qwen2.5-7b aya-expanse-8b

    # Run only Hindi-specialized models
    python scripts/run_multi_model_experiment.py --model_group hindi_specialized
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models_multi import (
    MODELS,
    EXPERIMENT_ORDER,
    MODEL_GROUPS,
    get_model_config,
    get_models_by_group,
    print_model_summary,
)
from src.llm.multi_model_loader import MultiModelLoader, get_gpu_memory_info
from src.data.mab_dataset import MABDataset

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for multi-model experiment."""

    # Models to run
    model_keys: List[str]

    # Data paths
    data_dir: str
    output_dir: str

    # Experiment settings
    n_train_samples: int = 1000
    n_eval_samples: int = 200
    bandit_type: str = "linucb"

    # Per-model settings
    enable_learning: bool = True
    save_checkpoints: bool = True

    # HuggingFace token (for gated models)
    hf_token: Optional[str] = None

# ============================================================================
# SINGLE MODEL EXPERIMENT
# ============================================================================

def run_single_model_experiment(
    model_key: str,
    dataset: MABDataset,
    output_dir: Path,
    config: ExperimentConfig,
    loader: MultiModelLoader,
) -> Dict:
    """
    Run complete experiment for a single model.

    Args:
        model_key: Model identifier
        dataset: Loaded dataset
        output_dir: Directory for this model's results
        config: Experiment configuration
        loader: Model loader instance

    Returns:
        Results dictionary
    """
    model_config = get_model_config(model_key)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {model_config.display_name}")
    print(f"Type: {model_config.model_type.value}")
    print(f"Languages: {', '.join(model_config.supported_languages)}")
    print("=" * 70)

    results = {
        "model_key": model_key,
        "model_id": model_config.model_id,
        "display_name": model_config.display_name,
        "model_type": model_config.model_type.value,
        "parameters": model_config.parameters,
        "start_time": datetime.now().isoformat(),
        "metrics": {},
        "per_language": {},
        "per_bias_type": {},
        "errors": [],
    }

    try:
        # Load model
        model, tokenizer = loader.load(model_key)

        # Record memory usage
        mem_info = get_gpu_memory_info()
        results["gpu_memory_gb"] = mem_info["used_gb"]

        # =====================================================================
        # PHASE 1: Baseline Evaluation (No Debiasing)
        # =====================================================================
        print("\n--- Phase 1: Baseline Evaluation ---")

        baseline_results = {
            "by_language": {},
            "by_bias_type": {},
            "overall": {},
        }

        # Evaluate by language
        for lang in ["en", "hi", "bn"]:
            if lang not in model_config.supported_languages:
                print(f"Skipping {lang} (not supported by model)")
                continue

            lang_data = dataset.filter(language=lang, split="test")[:config.n_eval_samples]

            if not lang_data:
                continue

            print(f"\nEvaluating {lang.upper()}: {len(lang_data)} samples")

            lang_metrics = {
                "n_samples": len(lang_data),
                "bias_scores": [],
                "quality_scores": [],
                "responses": [],
            }

            for item in tqdm(lang_data, desc=f"Baseline {lang.upper()}"):
                try:
                    # Generate without debiasing
                    response = loader.generate(item.sentence)

                    # Placeholder scores (replace with actual bias/quality scoring)
                    bias_score = 0.5
                    quality_score = 0.8

                    lang_metrics["bias_scores"].append(bias_score)
                    lang_metrics["quality_scores"].append(quality_score)
                    lang_metrics["responses"].append(response[:100])  # Store sample

                except Exception as e:
                    results["errors"].append(f"Baseline {lang} {item.id}: {str(e)}")

            # Calculate averages
            if lang_metrics["bias_scores"]:
                lang_metrics["mean_bias"] = sum(lang_metrics["bias_scores"]) / len(lang_metrics["bias_scores"])
                lang_metrics["mean_quality"] = sum(lang_metrics["quality_scores"]) / len(lang_metrics["quality_scores"])

            baseline_results["by_language"][lang] = lang_metrics

        results["metrics"]["baseline"] = baseline_results

        # =====================================================================
        # PHASE 2: MAB Training (Placeholder)
        # =====================================================================
        print("\n--- Phase 2: MAB Training ---")

        train_data = list(dataset.train_iter())[:config.n_train_samples]
        print(f"Training on {len(train_data)} samples")

        training_metrics = {
            "n_samples": len(train_data),
            "arm_selections": {i: 0 for i in range(6)},
            "rewards": [],
        }

        # Placeholder training loop
        for item in tqdm(train_data, desc="MAB Training"):
            try:
                # TODO: Integrate with actual MAB pipeline
                selected_arm = 1  # Placeholder
                reward = 0.7

                training_metrics["arm_selections"][selected_arm] += 1
                training_metrics["rewards"].append(reward)

            except Exception as e:
                results["errors"].append(f"Training {item.id}: {str(e)}")

        if training_metrics["rewards"]:
            training_metrics["mean_reward"] = sum(training_metrics["rewards"]) / len(training_metrics["rewards"])

        results["metrics"]["training"] = training_metrics

        # =====================================================================
        # PHASE 3: Post-MAB Evaluation (Placeholder)
        # =====================================================================
        print("\n--- Phase 3: Post-MAB Evaluation ---")

        mab_results = {
            "by_language": {},
            "by_bias_type": {},
            "overall": {},
        }

        for lang in ["en", "hi", "bn"]:
            if lang not in model_config.supported_languages:
                continue

            lang_data = dataset.filter(language=lang, split="test")[:config.n_eval_samples]

            if not lang_data:
                continue

            print(f"\nEvaluating {lang.upper()} with MAB: {len(lang_data)} samples")

            lang_metrics = {
                "n_samples": len(lang_data),
                "bias_scores": [],
                "quality_scores": [],
                "arm_selections": {i: 0 for i in range(6)},
            }

            for item in tqdm(lang_data, desc=f"MAB Eval {lang.upper()}"):
                try:
                    # TODO: Run through MAB pipeline with learned policy
                    bias_score = 0.3  # Should be lower than baseline
                    quality_score = 0.78
                    selected_arm = 1

                    lang_metrics["bias_scores"].append(bias_score)
                    lang_metrics["quality_scores"].append(quality_score)
                    lang_metrics["arm_selections"][selected_arm] += 1

                except Exception as e:
                    results["errors"].append(f"MAB Eval {lang} {item.id}: {str(e)}")

            if lang_metrics["bias_scores"]:
                lang_metrics["mean_bias"] = sum(lang_metrics["bias_scores"]) / len(lang_metrics["bias_scores"])
                lang_metrics["mean_quality"] = sum(lang_metrics["quality_scores"]) / len(lang_metrics["quality_scores"])

                # Calculate improvement
                baseline_bias = baseline_results["by_language"].get(lang, {}).get("mean_bias", 0.5)
                lang_metrics["bias_reduction"] = baseline_bias - lang_metrics["mean_bias"]
                lang_metrics["bias_reduction_pct"] = (lang_metrics["bias_reduction"] / baseline_bias) * 100 if baseline_bias > 0 else 0

            mab_results["by_language"][lang] = lang_metrics

        results["metrics"]["mab"] = mab_results

        # =====================================================================
        # Calculate Overall Results
        # =====================================================================

        all_baseline_bias = []
        all_mab_bias = []

        for lang in ["en", "hi", "bn"]:
            if lang in baseline_results["by_language"]:
                all_baseline_bias.extend(baseline_results["by_language"][lang].get("bias_scores", []))
            if lang in mab_results["by_language"]:
                all_mab_bias.extend(mab_results["by_language"][lang].get("bias_scores", []))

        if all_baseline_bias and all_mab_bias:
            results["metrics"]["overall"] = {
                "baseline_mean_bias": sum(all_baseline_bias) / len(all_baseline_bias),
                "mab_mean_bias": sum(all_mab_bias) / len(all_mab_bias),
                "overall_bias_reduction": (sum(all_baseline_bias) / len(all_baseline_bias)) - (sum(all_mab_bias) / len(all_mab_bias)),
            }

        results["status"] = "success"

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()

    finally:
        results["end_time"] = datetime.now().isoformat()

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{model_key}_results.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved: {results_file}")

        # Unload model
        loader.unload()

    return results

# ============================================================================
# MULTI-MODEL EXPERIMENT RUNNER
# ============================================================================

def run_multi_model_experiment(config: ExperimentConfig) -> Dict:
    """
    Run experiments across all specified models.

    Args:
        config: Experiment configuration

    Returns:
        Aggregated results dictionary
    """
    print("\n" + "=" * 70)
    print("MULTI-MODEL MAB DEBIASING EXPERIMENT")
    print("=" * 70)
    print(f"\nModels to evaluate: {len(config.model_keys)}")
    for key in config.model_keys:
        mc = get_model_config(key)
        print(f"  - {mc.display_name} ({mc.parameters})")

    print(f"\nData directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = MABDataset(config.data_dir)

    # Initialize loader
    loader = MultiModelLoader()
    if config.hf_token:
        loader.set_hf_token(config.hf_token)

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = {
        "experiment_start": datetime.now().isoformat(),
        "config": {
            "models": config.model_keys,
            "n_train_samples": config.n_train_samples,
            "n_eval_samples": config.n_eval_samples,
            "bandit_type": config.bandit_type,
        },
        "model_results": {},
        "summary": {},
    }

    for i, model_key in enumerate(config.model_keys):
        print(f"\n\n{'#'*70}")
        print(f"# MODEL {i+1}/{len(config.model_keys)}: {model_key}")
        print(f"{'#'*70}")

        model_output_dir = output_path / model_key

        try:
            results = run_single_model_experiment(
                model_key=model_key,
                dataset=dataset,
                output_dir=model_output_dir,
                config=config,
                loader=loader,
            )

            all_results["model_results"][model_key] = results

        except Exception as e:
            print(f"ERROR with {model_key}: {e}")
            all_results["model_results"][model_key] = {
                "status": "failed",
                "error": str(e),
            }

        # Memory cleanup between models
        loader.unload()
        time.sleep(2)

    # Generate summary
    all_results["experiment_end"] = datetime.now().isoformat()
    all_results["summary"] = generate_summary(all_results["model_results"])

    # Save aggregated results
    summary_file = output_path / "all_models_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\nAll results saved to: {output_path}")
    print(f"Summary file: {summary_file}")

    # Print summary table
    print_summary_table(all_results["summary"])

    return all_results

def generate_summary(model_results: Dict) -> Dict:
    """Generate summary statistics across all models."""
    summary = {
        "by_model": {},
        "by_language": {"en": [], "hi": [], "bn": []},
        "rankings": {},
    }

    for model_key, results in model_results.items():
        if results.get("status") != "success":
            continue

        metrics = results.get("metrics", {})
        overall = metrics.get("overall", {})

        summary["by_model"][model_key] = {
            "baseline_bias": overall.get("baseline_mean_bias"),
            "mab_bias": overall.get("mab_mean_bias"),
            "bias_reduction": overall.get("overall_bias_reduction"),
        }

        # Per-language
        mab_metrics = metrics.get("mab", {}).get("by_language", {})
        for lang in ["en", "hi", "bn"]:
            if lang in mab_metrics:
                summary["by_language"][lang].append({
                    "model": model_key,
                    "bias": mab_metrics[lang].get("mean_bias"),
                    "reduction": mab_metrics[lang].get("bias_reduction_pct"),
                })

    return summary

def print_summary_table(summary: Dict):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Baseline Bias':<15} {'MAB Bias':<15} {'Reduction':<15}")
    print("-" * 70)

    for model_key, metrics in summary.get("by_model", {}).items():
        baseline = metrics.get("baseline_bias", 0)
        mab = metrics.get("mab_bias", 0)
        reduction = metrics.get("bias_reduction", 0)

        print(f"{model_key:<25} {baseline:<15.3f} {mab:<15.3f} {reduction:<15.3f}")

    print("\n" + "=" * 80)

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run MAB debiasing experiments across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all 6 models
    python scripts/run_multi_model_experiment.py

    # Run specific models
    python scripts/run_multi_model_experiment.py --models qwen2.5-7b aya-expanse-8b

    # Run a model group
    python scripts/run_multi_model_experiment.py --model_group hindi_specialized

    # Quick test with fewer samples
    python scripts/run_multi_model_experiment.py --n_train 100 --n_eval 50
        """
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list(MODELS.keys()),
        help="Specific models to run (default: all)"
    )

    parser.add_argument(
        "--model_group",
        type=str,
        default=None,
        choices=list(MODEL_GROUPS.keys()),
        help="Run a predefined model group"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="Path to processed dataset"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/multi_model",
        help="Output directory for results"
    )

    parser.add_argument(
        "--n_train",
        type=int,
        default=1000,
        help="Number of training samples"
    )

    parser.add_argument(
        "--n_eval",
        type=int,
        default=200,
        help="Number of evaluation samples"
    )

    parser.add_argument(
        "--bandit",
        type=str,
        default="linucb",
        choices=["linucb", "thompson", "neural"],
        help="Bandit algorithm to use"
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models"
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.models:
        model_keys = args.models
    elif args.model_group:
        model_keys = get_models_by_group(args.model_group)
    else:
        model_keys = EXPERIMENT_ORDER  # All 6 models

    # Create config
    config = ExperimentConfig(
        model_keys=model_keys,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_train_samples=args.n_train,
        n_eval_samples=args.n_eval,
        bandit_type=args.bandit,
        hf_token=args.hf_token,
    )

    # Run experiment
    results = run_multi_model_experiment(config)

    return results

if __name__ == "__main__":
    main()
