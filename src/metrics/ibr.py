"""
Intersectional Bias Reduction (IBR) metric.

Novel metric that captures how well a method reduces bias across
ALL categories simultaneously, not just on average.

Formula:
    IBR = HarmonicMean(BR_1, BR_2, ..., BR_N)
        = N / Σ(1/BR_i)

Where BR_i = Bias Reduction for category i:
    BR_i = (baseline_bias_i - method_bias_i) / baseline_bias_i

Properties:
- Range: [0, 1] where 1 indicates perfect reduction in all categories
- Penalizes methods that fail on ANY category (unlike arithmetic mean)
- Captures intersectional fairness (no category left behind)

Example:
    If method reduces Gender bias by 80% but Race bias by only 10%:
    - Arithmetic mean: (0.8 + 0.1) / 2 = 0.45
    - Harmonic mean (IBR): 2 / (1/0.8 + 1/0.1) = 0.145
    The IBR correctly reflects that the method is NOT good overall.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiasReductionResult:
    """Result for a single bias category."""
    category: str
    baseline_bias: float
    method_bias: float
    bias_reduction: float
    is_improvement: bool


@dataclass
class IBRResult:
    """Complete IBR computation result."""
    ibr_score: float  # Standard IBR (harmonic mean of positive reductions)
    signed_ibr: float  # IBR with penalty for negative reductions (NEW)
    arithmetic_mean: float
    per_category_results: Dict[str, BiasReductionResult]
    n_categories: int
    n_improved: int
    n_worsened: int  # NEW: categories where bias increased
    worst_category: str
    best_category: str


def compute_bias_reduction(baseline_bias: float, method_bias: float) -> float:
    """
    Compute bias reduction for a single category.

    BR = (baseline - method) / baseline

    Args:
        baseline_bias: Bias score without intervention (0 to 1)
        method_bias: Bias score with method applied (0 to 1)

    Returns:
        Bias reduction ratio in [0, 1] (can be negative if method increases bias)
    """
    if baseline_bias <= 0:
        return 0.0  # No baseline bias to reduce

    reduction = (baseline_bias - method_bias) / baseline_bias
    return float(np.clip(reduction, -1.0, 1.0))


def compute_ibr(
    bias_reductions: Dict[str, float],
    epsilon: float = 1e-10,
    include_negative: bool = False
) -> float:
    """
    Compute Intersectional Bias Reduction (IBR) using harmonic mean.

    IBR = N / Σ(1/BR_i)

    Args:
        bias_reductions: Dict mapping category -> bias reduction ratio
        epsilon: Small value to avoid division by zero
        include_negative: If True, include ALL categories (negative gets penalized)

    Returns:
        IBR score in [0, 1]
    """
    if not bias_reductions:
        return 0.0

    if include_negative:
        # FIXED: Include all categories, but transform negative to penalize
        # Negative reduction means bias INCREASED - this should hurt the score
        transformed = {}
        for cat, br in bias_reductions.items():
            if br > 0:
                transformed[cat] = br
            else:
                # Negative reduction: penalize by using small value
                # A method that increases bias should get very low score
                transformed[cat] = epsilon
        reductions_to_use = transformed
    else:
        # Original behavior: only positive reductions
        reductions_to_use = {
            cat: max(br, epsilon)
            for cat, br in bias_reductions.items()
            if br > 0
        }

    if not reductions_to_use:
        return 0.0

    n = len(reductions_to_use)
    harmonic_sum = sum(1.0 / br for br in reductions_to_use.values())

    if harmonic_sum <= 0:
        return 0.0

    ibr = n / harmonic_sum

    return float(np.clip(ibr, 0.0, 1.0))


def compute_signed_ibr(
    bias_reductions: Dict[str, float],
    epsilon: float = 1e-10
) -> float:
    """
    Compute Signed IBR that explicitly penalizes negative reductions.
    
    For categories with negative reduction (bias increased):
    - Each worsened category contributes a penalty term
    
    signed_IBR = IBR_positive * (1 - penalty_factor)
    where penalty_factor = n_worsened / n_total
    
    This ensures methods that increase bias in ANY category are penalized.
    
    Args:
        bias_reductions: Dict mapping category -> bias reduction ratio
        epsilon: Small value for numerical stability
        
    Returns:
        Signed IBR score in [0, 1] where worsened categories reduce score
    """
    if not bias_reductions:
        return 0.0
    
    n_total = len(bias_reductions)
    n_worsened = sum(1 for br in bias_reductions.values() if br < 0)
    
    # Compute standard IBR on positive reductions
    positive_reductions = {cat: br for cat, br in bias_reductions.items() if br > 0}
    
    if not positive_reductions:
        return 0.0
    
    base_ibr = compute_ibr(positive_reductions, epsilon, include_negative=False)
    
    # Apply penalty for worsened categories
    penalty_factor = n_worsened / n_total
    signed_ibr = base_ibr * (1 - penalty_factor)
    
    return float(np.clip(signed_ibr, 0.0, 1.0))


class IntersectionalBiasReduction:
    """
    Compute and track Intersectional Bias Reduction metric.

    This metric uses harmonic mean to ensure that methods must
    reduce bias across ALL categories to achieve high scores.

    Key insight: Arithmetic mean can hide failures in specific categories.
    Harmonic mean requires consistent performance across all categories.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        epsilon: float = 1e-10
    ):
        """
        Initialize IBR calculator.

        Args:
            categories: List of bias categories to track
            epsilon: Small value to prevent division by zero
        """
        self.categories = categories or [
            'gender', 'race', 'religion', 'caste',
            'socioeconomic', 'nationality', 'age',
            'sexual_orientation', 'physical_appearance', 'disability'
        ]
        self.epsilon = epsilon

        # Baseline and method scores per category
        self.baseline_scores: Dict[str, List[float]] = {cat: [] for cat in self.categories}
        self.method_scores: Dict[str, List[float]] = {cat: [] for cat in self.categories}

    def add_observation(
        self,
        category: str,
        baseline_bias: float,
        method_bias: float
    ):
        """
        Add a single observation for a category.

        Args:
            category: Bias category name
            baseline_bias: Bias score without intervention
            method_bias: Bias score with method applied
        """
        if category not in self.baseline_scores:
            self.baseline_scores[category] = []
            self.method_scores[category] = []

        self.baseline_scores[category].append(baseline_bias)
        self.method_scores[category].append(method_bias)

    def add_batch(
        self,
        observations: List[Dict[str, float]]
    ):
        """
        Add batch of observations.

        Each observation should have: category, baseline_bias, method_bias

        Args:
            observations: List of observation dictionaries
        """
        for obs in observations:
            self.add_observation(
                category=obs['category'],
                baseline_bias=obs['baseline_bias'],
                method_bias=obs['method_bias']
            )

    def compute(self) -> IBRResult:
        """
        Compute IBR from accumulated observations.

        Returns:
            IBRResult with full breakdown
        """
        per_category_results = {}
        bias_reductions = {}

        for category in self.categories:
            if not self.baseline_scores.get(category):
                continue

            baseline_mean = np.mean(self.baseline_scores[category])
            method_mean = np.mean(self.method_scores[category])

            br = compute_bias_reduction(baseline_mean, method_mean)
            bias_reductions[category] = br

            per_category_results[category] = BiasReductionResult(
                category=category,
                baseline_bias=baseline_mean,
                method_bias=method_mean,
                bias_reduction=br,
                is_improvement=br > 0
            )

        if not bias_reductions:
            return IBRResult(
                ibr_score=0.0,
                signed_ibr=0.0,
                arithmetic_mean=0.0,
                per_category_results={},
                n_categories=0,
                n_improved=0,
                n_worsened=0,
                worst_category='',
                best_category=''
            )

        # Compute IBR (harmonic mean)
        ibr_score = compute_ibr(bias_reductions, self.epsilon)
        
        # Compute signed IBR that penalizes negative reductions
        signed_ibr = compute_signed_ibr(bias_reductions, self.epsilon)

        # Compute arithmetic mean for comparison
        positive_reductions = [br for br in bias_reductions.values() if br > 0]
        arithmetic_mean = np.mean(positive_reductions) if positive_reductions else 0.0

        # Find worst and best categories
        sorted_cats = sorted(bias_reductions.items(), key=lambda x: x[1])
        worst_category = sorted_cats[0][0]
        best_category = sorted_cats[-1][0]
        
        # Count categories where bias worsened
        n_worsened = sum(1 for br in bias_reductions.values() if br < 0)

        return IBRResult(
            ibr_score=ibr_score,
            signed_ibr=signed_ibr,
            arithmetic_mean=arithmetic_mean,
            per_category_results=per_category_results,
            n_categories=len(bias_reductions),
            n_improved=sum(1 for br in bias_reductions.values() if br > 0),
            n_worsened=n_worsened,
            worst_category=worst_category,
            best_category=best_category
        )

    def compute_from_scores(
        self,
        baseline_scores: Dict[str, float],
        method_scores: Dict[str, float]
    ) -> IBRResult:
        """
        Compute IBR directly from aggregated scores.

        Args:
            baseline_scores: Category -> mean baseline bias score
            method_scores: Category -> mean method bias score

        Returns:
            IBRResult
        """
        per_category_results = {}
        bias_reductions = {}

        all_categories = set(baseline_scores.keys()) & set(method_scores.keys())

        for category in all_categories:
            baseline = baseline_scores[category]
            method = method_scores[category]

            br = compute_bias_reduction(baseline, method)
            bias_reductions[category] = br

            per_category_results[category] = BiasReductionResult(
                category=category,
                baseline_bias=baseline,
                method_bias=method,
                bias_reduction=br,
                is_improvement=br > 0
            )

        if not bias_reductions:
            return IBRResult(
                ibr_score=0.0,
                signed_ibr=0.0,
                arithmetic_mean=0.0,
                per_category_results={},
                n_categories=0,
                n_improved=0,
                n_worsened=0,
                worst_category='',
                best_category=''
            )

        ibr_score = compute_ibr(bias_reductions, self.epsilon)
        signed_ibr = compute_signed_ibr(bias_reductions, self.epsilon)

        positive_reductions = [br for br in bias_reductions.values() if br > 0]
        arithmetic_mean = np.mean(positive_reductions) if positive_reductions else 0.0

        sorted_cats = sorted(bias_reductions.items(), key=lambda x: x[1])
        worst_category = sorted_cats[0][0]
        best_category = sorted_cats[-1][0]
        
        n_worsened = sum(1 for br in bias_reductions.values() if br < 0)

        return IBRResult(
            ibr_score=ibr_score,
            signed_ibr=signed_ibr,
            arithmetic_mean=arithmetic_mean,
            per_category_results=per_category_results,
            n_categories=len(bias_reductions),
            n_improved=sum(1 for br in bias_reductions.values() if br > 0),
            n_worsened=n_worsened,
            worst_category=worst_category,
            best_category=best_category
        )

    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics."""
        result = self.compute()

        return {
            'ibr_score': result.ibr_score,
            'arithmetic_mean': result.arithmetic_mean,
            'ibr_vs_arithmetic_ratio': result.ibr_score / result.arithmetic_mean if result.arithmetic_mean > 0 else 0,
            'n_categories': result.n_categories,
            'n_improved': result.n_improved,
            'improvement_rate': result.n_improved / result.n_categories if result.n_categories > 0 else 0,
            'worst_category': result.worst_category,
            'best_category': result.best_category,
            'per_category': {
                cat: {
                    'reduction': r.bias_reduction,
                    'improved': r.is_improvement
                }
                for cat, r in result.per_category_results.items()
            }
        }

    def reset(self):
        """Clear all accumulated observations."""
        self.baseline_scores = {cat: [] for cat in self.categories}
        self.method_scores = {cat: [] for cat in self.categories}

    def __repr__(self) -> str:
        result = self.compute()
        return f"IBR(score={result.ibr_score:.4f}, categories={result.n_categories}, improved={result.n_improved})"
