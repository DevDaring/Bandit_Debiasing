"""
Theoretical analysis module for Fair-CB bandit debiasing.

This module implements:
1. Regret tracking and bounds verification
2. Fairness violation monitoring
3. Theoretical bounds computation
4. Adaptive vs static arm comparison
5. Theorem verification via Monte Carlo simulation
"""

from .regret_tracker import RegretTracker
from .fairness_tracker import FairnessViolationTracker
from .bounds import TheoreticalBoundComputer, compute_linucb_regret_bound
from .adaptive_vs_static import AdaptiveStaticComparator
from .theorem_verification import TheoremVerifier

__all__ = [
    'RegretTracker',
    'FairnessViolationTracker',
    'TheoreticalBoundComputer',
    'compute_linucb_regret_bound',
    'AdaptiveStaticComparator',
    'TheoremVerifier',
]
