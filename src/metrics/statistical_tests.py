"""
Statistical significance tests for Fair-CB evaluation.

Provides:
- Bootstrap confidence intervals
- Paired t-tests for method comparisons
- Effect size calculations (Cohen's d)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Result of confidence interval computation."""
    mean: float
    lower: float
    upper: float
    confidence_level: float
    n_samples: int


@dataclass
class SignificanceTestResult:
    """Result of a statistical significance test."""
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    effect_interpretation: str


def compute_bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = 'mean'
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Array of observations
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        statistic: 'mean', 'median', or 'std'
        
    Returns:
        ConfidenceInterval with bounds
    """
    if len(data) == 0:
        return ConfidenceInterval(
            mean=0.0, lower=0.0, upper=0.0,
            confidence_level=confidence, n_samples=0
        )
    
    data = np.asarray(data)
    n = len(data)
    
    # Bootstrap sampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        if statistic == 'mean':
            bootstrap_stats.append(np.mean(sample))
        elif statistic == 'median':
            bootstrap_stats.append(np.median(sample))
        elif statistic == 'std':
            bootstrap_stats.append(np.std(sample))
        else:
            bootstrap_stats.append(np.mean(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile method for CI
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ConfidenceInterval(
        mean=np.mean(data),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence,
        n_samples=n
    )


def compute_paired_ttest(
    method1_scores: np.ndarray,
    method2_scores: np.ndarray,
    alpha: float = 0.05
) -> SignificanceTestResult:
    """
    Perform paired t-test between two methods.
    
    Used when the same samples are evaluated by both methods.
    
    Args:
        method1_scores: Scores from method 1
        method2_scores: Scores from method 2
        alpha: Significance level
        
    Returns:
        SignificanceTestResult with p-value and effect size
    """
    if len(method1_scores) != len(method2_scores):
        raise ValueError("Arrays must have same length for paired test")
    
    if len(method1_scores) < 2:
        return SignificanceTestResult(
            statistic=0.0, p_value=1.0, is_significant=False,
            effect_size=0.0, effect_interpretation="insufficient data"
        )
    
    method1_scores = np.asarray(method1_scores)
    method2_scores = np.asarray(method2_scores)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)
    
    # Cohen's d effect size for paired samples
    diff = method1_scores - method2_scores
    effect_size = np.mean(diff) / np.std(diff, ddof=1)
    
    # Interpret effect size
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        interpretation = "negligible"
    elif abs_effect < 0.5:
        interpretation = "small"
    elif abs_effect < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return SignificanceTestResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        effect_size=float(effect_size),
        effect_interpretation=interpretation
    )


def compute_welch_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05
) -> SignificanceTestResult:
    """
    Perform Welch's t-test (unequal variances) between two groups.
    
    Used when comparing independent samples.
    
    Args:
        group1: Scores from group 1
        group2: Scores from group 2
        alpha: Significance level
        
    Returns:
        SignificanceTestResult
    """
    if len(group1) < 2 or len(group2) < 2:
        return SignificanceTestResult(
            statistic=0.0, p_value=1.0, is_significant=False,
            effect_size=0.0, effect_interpretation="insufficient data"
        )
    
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    # Cohen's d for independent samples
    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1, ddof=1) + 
         (len(group2) - 1) * np.var(group2, ddof=1)) /
        (len(group1) + len(group2) - 2)
    )
    
    effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        interpretation = "negligible"
    elif abs_effect < 0.5:
        interpretation = "small"
    elif abs_effect < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return SignificanceTestResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        effect_size=float(effect_size),
        effect_interpretation=interpretation
    )


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for Fair-CB experiments.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        significance_alpha: float = 0.05,
        n_bootstrap: int = 10000
    ):
        self.confidence_level = confidence_level
        self.significance_alpha = significance_alpha
        self.n_bootstrap = n_bootstrap
        
        self.method_results: Dict[str, List[float]] = {}
    
    def add_result(self, method_name: str, score: float):
        """Add a single result for a method."""
        if method_name not in self.method_results:
            self.method_results[method_name] = []
        self.method_results[method_name].append(score)
    
    def add_results(self, method_name: str, scores: List[float]):
        """Add multiple results for a method."""
        if method_name not in self.method_results:
            self.method_results[method_name] = []
        self.method_results[method_name].extend(scores)
    
    def get_confidence_interval(self, method_name: str) -> ConfidenceInterval:
        """Get bootstrap CI for a method's results."""
        if method_name not in self.method_results:
            raise ValueError(f"Unknown method: {method_name}")
        
        data = np.array(self.method_results[method_name])
        return compute_bootstrap_ci(
            data,
            confidence=self.confidence_level,
            n_bootstrap=self.n_bootstrap
        )
    
    def compare_methods(
        self,
        method1: str,
        method2: str,
        paired: bool = True
    ) -> SignificanceTestResult:
        """
        Compare two methods statistically.
        
        Args:
            method1: Name of first method
            method2: Name of second method
            paired: Whether to use paired test (same samples)
            
        Returns:
            SignificanceTestResult
        """
        if method1 not in self.method_results or method2 not in self.method_results:
            raise ValueError("Both methods must have recorded results")
        
        scores1 = np.array(self.method_results[method1])
        scores2 = np.array(self.method_results[method2])
        
        if paired:
            return compute_paired_ttest(
                scores1, scores2,
                alpha=self.significance_alpha
            )
        else:
            return compute_welch_ttest(
                scores1, scores2,
                alpha=self.significance_alpha
            )
    
    def get_summary_table(self) -> Dict[str, Dict]:
        """
        Get summary statistics for all methods.
        
        Returns:
            Dict with method names as keys and stats as values
        """
        summary = {}
        
        for method_name, scores in self.method_results.items():
            ci = self.get_confidence_interval(method_name)
            summary[method_name] = {
                'mean': ci.mean,
                'ci_lower': ci.lower,
                'ci_upper': ci.upper,
                'n_samples': ci.n_samples,
                'std': np.std(scores) if scores else 0.0
            }
        
        return summary
    
    def to_csv_columns(self, method_name: str) -> Dict[str, float]:
        """
        Get CSV-ready columns for a method.
        
        Returns dict with full-form column names.
        """
        ci = self.get_confidence_interval(method_name)
        scores = self.method_results.get(method_name, [])
        
        return {
            'Mean Score': ci.mean,
            'Confidence Interval 95% Lower': ci.lower,
            'Confidence Interval 95% Upper': ci.upper,
            'Standard Deviation': np.std(scores) if scores else 0.0,
            'Sample Count': ci.n_samples
        }
