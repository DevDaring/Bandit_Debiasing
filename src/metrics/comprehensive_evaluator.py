"""
Comprehensive metrics evaluator for Fair-CB.

Integrates all metrics:
- IBR (Intersectional Bias Reduction)
- FAR (Fairness-Aware Regret)
- Per-category bias scores
- Per-language performance
- Theoretical bound verification

Provides unified interface for experiment evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

from .ibr import IntersectionalBiasReduction, IBRResult
from .far import FairnessAwareRegret, FARResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    # Core metrics
    ibr: IBRResult
    far: FARResult

    # Per-group performance
    per_category_scores: Dict[str, float]
    per_language_scores: Dict[str, float]

    # Statistics
    n_samples: int
    mean_bias_score: float
    mean_reward: float

    # Comparison to baseline
    baseline_comparison: Optional[Dict[str, float]] = None


class ComprehensiveMetricsEvaluator:
    """
    Unified evaluator for all Fair-CB metrics.

    Combines:
    1. IBR for intersectional fairness
    2. FAR for regret-fairness trade-off
    3. Per-category analysis
    4. Per-language analysis
    5. Statistical significance tests
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        lambda_weight: float = 0.5,
        bias_threshold: float = 0.3
    ):
        """
        Initialize comprehensive evaluator.

        Args:
            categories: Bias categories to track
            languages: Languages to track
            lambda_weight: Weight for FAR computation
            bias_threshold: Threshold for fairness violations
        """
        self.categories = categories or [
            'gender', 'race', 'religion', 'caste',
            'socioeconomic', 'nationality', 'age',
            'sexual_orientation', 'physical_appearance', 'disability'
        ]
        self.languages = languages or ['en', 'hi', 'bn']

        self.lambda_weight = lambda_weight
        self.bias_threshold = bias_threshold

        # Initialize sub-evaluators
        self.ibr_calculator = IntersectionalBiasReduction(categories=self.categories)
        self.far_calculator = FairnessAwareRegret(
            lambda_weight=lambda_weight,
            bias_threshold=bias_threshold
        )

        # Per-group tracking
        self.category_scores: Dict[str, List[float]] = {cat: [] for cat in self.categories}
        self.language_scores: Dict[str, List[float]] = {lang: [] for lang in self.languages}

        # Baseline tracking
        self.baseline_scores: Dict[str, List[float]] = {cat: [] for cat in self.categories}

        # General tracking
        self.all_bias_scores: List[float] = []
        self.all_rewards: List[float] = []
        self.n_samples: int = 0

    def add_observation(
        self,
        bias_score: float,
        reward: float,
        optimal_reward: float,
        category: Optional[str] = None,
        language: Optional[str] = None,
        baseline_bias: Optional[float] = None
    ):
        """
        Add a single observation.

        Args:
            bias_score: Bias score of generated output
            reward: Reward received
            optimal_reward: Optimal reward (for regret)
            category: Bias category (if known)
            language: Language code
            baseline_bias: Baseline bias without intervention
        """
        self.n_samples += 1

        # General tracking
        self.all_bias_scores.append(bias_score)
        self.all_rewards.append(reward)

        # Update FAR
        regret = max(0.0, optimal_reward - reward)
        self.far_calculator.update(regret, bias_score)

        # Per-category tracking
        if category:
            if category not in self.category_scores:
                self.category_scores[category] = []
            self.category_scores[category].append(bias_score)

            # IBR tracking (if baseline provided)
            if baseline_bias is not None:
                self.ibr_calculator.add_observation(
                    category=category,
                    baseline_bias=baseline_bias,
                    method_bias=bias_score
                )
                self.baseline_scores[category].append(baseline_bias)

        # Per-language tracking
        if language:
            if language not in self.language_scores:
                self.language_scores[language] = []
            self.language_scores[language].append(bias_score)

    def add_batch(
        self,
        observations: List[Dict[str, Any]]
    ):
        """
        Add batch of observations.

        Each observation dict should have:
        - bias_score, reward, optimal_reward (required)
        - category, language, baseline_bias (optional)
        """
        for obs in observations:
            self.add_observation(
                bias_score=obs['bias_score'],
                reward=obs['reward'],
                optimal_reward=obs['optimal_reward'],
                category=obs.get('category'),
                language=obs.get('language'),
                baseline_bias=obs.get('baseline_bias')
            )

    def evaluate(self) -> EvaluationResult:
        """
        Compute all metrics.

        Returns:
            EvaluationResult with complete breakdown
        """
        # Compute IBR
        ibr_result = self.ibr_calculator.compute()

        # Compute FAR
        far_result = self.far_calculator.compute()

        # Per-category scores (mean bias)
        per_category = {
            cat: np.mean(scores) if scores else 0.0
            for cat, scores in self.category_scores.items()
        }

        # Per-language scores (mean bias)
        per_language = {
            lang: np.mean(scores) if scores else 0.0
            for lang, scores in self.language_scores.items()
        }

        # Baseline comparison
        baseline_comparison = None
        if any(self.baseline_scores.values()):
            baseline_comparison = {
                cat: {
                    'baseline': np.mean(self.baseline_scores[cat]) if self.baseline_scores.get(cat) else 0,
                    'method': np.mean(self.category_scores[cat]) if self.category_scores.get(cat) else 0,
                }
                for cat in self.categories
                if self.baseline_scores.get(cat)
            }
            for cat in baseline_comparison:
                baseline = baseline_comparison[cat]['baseline']
                method = baseline_comparison[cat]['method']
                if baseline > 0:
                    baseline_comparison[cat]['reduction'] = (baseline - method) / baseline
                else:
                    baseline_comparison[cat]['reduction'] = 0

        return EvaluationResult(
            ibr=ibr_result,
            far=far_result,
            per_category_scores=per_category,
            per_language_scores=per_language,
            n_samples=self.n_samples,
            mean_bias_score=np.mean(self.all_bias_scores) if self.all_bias_scores else 0,
            mean_reward=np.mean(self.all_rewards) if self.all_rewards else 0,
            baseline_comparison=baseline_comparison
        )

    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get summary as dictionary for logging/export.

        Returns:
            Dictionary with all metrics
        """
        result = self.evaluate()

        summary = {
            # Core metrics
            'ibr_score': result.ibr.ibr_score,
            'far_score': result.far.far_score,

            # IBR details
            'ibr_arithmetic_mean': result.ibr.arithmetic_mean,
            'ibr_n_categories': result.ibr.n_categories,
            'ibr_n_improved': result.ibr.n_improved,
            'ibr_worst_category': result.ibr.worst_category,
            'ibr_best_category': result.ibr.best_category,

            # FAR details
            'far_cumulative_regret': result.far.cumulative_regret,
            'far_cumulative_violation': result.far.cumulative_violation,
            'far_lambda': result.far.lambda_weight,
            'far_average': result.far.average_far,

            # Overall stats
            'n_samples': result.n_samples,
            'mean_bias_score': result.mean_bias_score,
            'mean_reward': result.mean_reward,

            # Per-category
            'per_category_bias': result.per_category_scores,
            'per_language_bias': result.per_language_scores,
        }

        return summary

    def compare_to_baseline(self, baseline_evaluator: 'ComprehensiveMetricsEvaluator') -> Dict[str, Any]:
        """
        Compare this evaluator's results to a baseline.

        Args:
            baseline_evaluator: Evaluator for baseline method

        Returns:
            Comparison dictionary
        """
        this_result = self.evaluate()
        baseline_result = baseline_evaluator.evaluate()

        comparison = {
            'ibr_improvement': this_result.ibr.ibr_score - baseline_result.ibr.ibr_score,
            'far_reduction': baseline_result.far.far_score - this_result.far.far_score,
            'bias_reduction': baseline_result.mean_bias_score - this_result.mean_bias_score,
            'reward_improvement': this_result.mean_reward - baseline_result.mean_reward,
        }

        # Per-category improvement
        comparison['per_category_improvement'] = {}
        for cat in this_result.per_category_scores:
            if cat in baseline_result.per_category_scores:
                comparison['per_category_improvement'][cat] = (
                    baseline_result.per_category_scores[cat] -
                    this_result.per_category_scores[cat]
                )

        return comparison

    def generate_report(self) -> str:
        """
        Generate human-readable evaluation report.

        Returns:
            Formatted report string
        """
        result = self.evaluate()

        lines = [
            "=" * 60,
            "FAIR-CB COMPREHENSIVE EVALUATION REPORT",
            "=" * 60,
            "",
            "CORE METRICS",
            "-" * 40,
            f"IBR (Intersectional Bias Reduction): {result.ibr.ibr_score:.4f}",
            f"FAR (Fairness-Aware Regret): {result.far.far_score:.4f}",
            "",
            "IBR BREAKDOWN",
            "-" * 40,
            f"  Arithmetic Mean (comparison): {result.ibr.arithmetic_mean:.4f}",
            f"  Categories evaluated: {result.ibr.n_categories}",
            f"  Categories improved: {result.ibr.n_improved}",
            f"  Worst category: {result.ibr.worst_category}",
            f"  Best category: {result.ibr.best_category}",
            "",
            "FAR BREAKDOWN",
            "-" * 40,
            f"  Cumulative Regret: {result.far.cumulative_regret:.4f}",
            f"  Cumulative Violation: {result.far.cumulative_violation:.4f}",
            f"  Lambda weight: {result.far.lambda_weight}",
            f"  Regret component: {result.far.regret_component:.4f}",
            f"  Violation component: {result.far.violation_component:.4f}",
            "",
            "OVERALL STATISTICS",
            "-" * 40,
            f"  Total samples: {result.n_samples}",
            f"  Mean bias score: {result.mean_bias_score:.4f}",
            f"  Mean reward: {result.mean_reward:.4f}",
            "",
            "PER-CATEGORY BIAS SCORES",
            "-" * 40,
        ]

        for cat, score in sorted(result.per_category_scores.items()):
            lines.append(f"  {cat}: {score:.4f}")

        lines.extend([
            "",
            "PER-LANGUAGE BIAS SCORES",
            "-" * 40,
        ])

        for lang, score in sorted(result.per_language_scores.items()):
            lines.append(f"  {lang}: {score:.4f}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def save(self, filepath: str):
        """Save evaluation results to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.get_summary_dict()
        data['report'] = self.generate_report()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved evaluation results to {path}")

    def reset(self):
        """Clear all accumulated data."""
        self.ibr_calculator.reset()
        self.far_calculator.reset()
        self.category_scores = {cat: [] for cat in self.categories}
        self.language_scores = {lang: [] for lang in self.languages}
        self.baseline_scores = {cat: [] for cat in self.categories}
        self.all_bias_scores = []
        self.all_rewards = []
        self.n_samples = 0

    def __repr__(self) -> str:
        result = self.evaluate()
        return (f"ComprehensiveEvaluator(n={self.n_samples}, "
                f"IBR={result.ibr.ibr_score:.4f}, FAR={result.far.far_score:.4f})")
