"""
Novel evaluation metrics for Fair-CB debiasing.

Implements:
1. IBR (Intersectional Bias Reduction) - Harmonic mean of per-category bias reduction
2. FAR (Fairness-Aware Regret) - Combined regret and fairness violation metric
3. Statistical tests - Bootstrap CIs, paired t-tests, effect sizes
4. Comprehensive evaluator for all metrics

These metrics are novel contributions for the TACL publication.
"""

from .ibr import (
    IntersectionalBiasReduction,
    compute_ibr,
    compute_signed_ibr,
    IBRResult,
    BiasReductionResult,
)
from .far import FairnessAwareRegret, compute_far
from .comprehensive_evaluator import ComprehensiveMetricsEvaluator
from .statistical_tests import (
    StatisticalAnalyzer,
    compute_bootstrap_ci,
    compute_paired_ttest,
    ConfidenceInterval,
    SignificanceTestResult,
)

__all__ = [
    'IntersectionalBiasReduction',
    'compute_ibr',
    'compute_signed_ibr',
    'IBRResult',
    'BiasReductionResult',
    'FairnessAwareRegret',
    'compute_far',
    'ComprehensiveMetricsEvaluator',
    'StatisticalAnalyzer',
    'compute_bootstrap_ci',
    'compute_paired_ttest',
    'ConfidenceInterval',
    'SignificanceTestResult',
]
