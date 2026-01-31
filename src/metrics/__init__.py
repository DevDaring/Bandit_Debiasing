"""
Novel evaluation metrics for Fair-CB debiasing.

Implements:
1. IBR (Intersectional Bias Reduction) - Harmonic mean of per-category bias reduction
2. FAR (Fairness-Aware Regret) - Combined regret and fairness violation metric
3. Comprehensive evaluator for all metrics

These metrics are novel contributions for the TACL publication.
"""

from .ibr import IntersectionalBiasReduction, compute_ibr
from .far import FairnessAwareRegret, compute_far
from .comprehensive_evaluator import ComprehensiveMetricsEvaluator

__all__ = [
    'IntersectionalBiasReduction',
    'compute_ibr',
    'FairnessAwareRegret',
    'compute_far',
    'ComprehensiveMetricsEvaluator',
]
