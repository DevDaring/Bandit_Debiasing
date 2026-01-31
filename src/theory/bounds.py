"""
Theoretical bounds computation for contextual bandits.

Implements:
1. LinUCB Regret Bound: R(T) ≤ O(d√(KT log(T/δ)))
2. Fairness-Aware Bound: R_fair(T) ≤ R(T) + λV(T)
3. Empirical bound verification

Based on:
- Abbasi-Yadkori et al. "Improved Algorithms for Linear Stochastic Bandits" (2011)
- Extended with fairness constraints for Fair-CB
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegretBound:
    """Theoretical regret bound result."""
    bound_type: str
    bound_value: float
    empirical_regret: float
    is_satisfied: bool
    margin: float
    parameters: Dict[str, float]


def compute_linucb_regret_bound(
    T: int,
    d: int,
    K: int,
    delta: float = 0.01,
    lambda_reg: float = 1.0,
    R: float = 1.0,
    S: float = 1.0
) -> float:
    """
    Compute LinUCB regret bound from Abbasi-Yadkori et al. (2011).

    R(T) ≤ O(d * √(KT * log(T/δ)))

    More precisely:
    R(T) ≤ √(T * d * log(1 + TL²/(dλ))) * (R√(d * log(1 + TL²/(dλ)) + log(1/δ)) + √λS)

    Simplified bound used:
    R(T) ≤ d * √(K * T * log(T/δ))

    Args:
        T: Time horizon (number of rounds)
        d: Context dimension
        K: Number of arms
        delta: Confidence parameter (typically 0.01)
        lambda_reg: Regularization parameter
        R: Reward bound (|r| ≤ R)
        S: Parameter bound (||θ|| ≤ S)

    Returns:
        Upper bound on cumulative regret
    """
    if T <= 0:
        return 0.0

    # Simplified bound: d * √(K * T * log(T/δ))
    log_term = np.log(T / delta) if T > 0 and delta > 0 else 1.0
    bound = d * np.sqrt(K * T * log_term)

    return bound


def compute_fairness_aware_bound(
    regret_bound: float,
    violation_bound: float,
    lambda_fairness: float = 0.5
) -> float:
    """
    Compute fairness-aware regret bound.

    FAR_bound(T) = R_bound(T) + λ * V_bound(T)

    For sublinear violations, V(T) ≤ O(√T)

    Args:
        regret_bound: Standard regret bound R(T)
        violation_bound: Fairness violation bound V(T)
        lambda_fairness: Fairness weight parameter

    Returns:
        Fairness-aware regret bound
    """
    return regret_bound + lambda_fairness * violation_bound


class TheoreticalBoundComputer:
    """
    Compute and verify theoretical bounds for Fair-CB.

    Provides:
    1. A priori bounds (before running)
    2. Empirical verification (after running)
    3. Bound tightness analysis
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        delta: float = 0.01,
        lambda_fairness: float = 0.5,
        fairness_threshold: float = 0.3
    ):
        """
        Initialize bound computer.

        Args:
            n_arms: Number of bandit arms (K)
            context_dim: Context vector dimension (d)
            delta: Confidence parameter for high-probability bounds
            lambda_fairness: Weight for fairness violations in FAR
            fairness_threshold: Bias threshold for violations
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.delta = delta
        self.lambda_fairness = lambda_fairness
        self.fairness_threshold = fairness_threshold

    def compute_regret_bound(self, T: int) -> RegretBound:
        """
        Compute theoretical regret bound for horizon T.

        Args:
            T: Time horizon

        Returns:
            RegretBound object with bound details
        """
        bound = compute_linucb_regret_bound(
            T=T,
            d=self.context_dim,
            K=self.n_arms,
            delta=self.delta
        )

        return RegretBound(
            bound_type='LinUCB',
            bound_value=bound,
            empirical_regret=0.0,  # Unknown until we have data
            is_satisfied=True,
            margin=bound,
            parameters={
                'T': T,
                'd': self.context_dim,
                'K': self.n_arms,
                'delta': self.delta
            }
        )

    def compute_far_bound(self, T: int) -> RegretBound:
        """
        Compute Fairness-Aware Regret (FAR) bound.

        FAR(T) = R(T) + λV(T)

        Both R(T) and V(T) should be O(√T) for sublinear rates.

        Args:
            T: Time horizon

        Returns:
            RegretBound object for FAR
        """
        regret_bound = compute_linucb_regret_bound(
            T=T,
            d=self.context_dim,
            K=self.n_arms,
            delta=self.delta
        )

        # Violation bound: assume V(T) ≤ c√T with c=1
        violation_bound = np.sqrt(T)

        far_bound = compute_fairness_aware_bound(
            regret_bound=regret_bound,
            violation_bound=violation_bound,
            lambda_fairness=self.lambda_fairness
        )

        return RegretBound(
            bound_type='FAR (Fairness-Aware Regret)',
            bound_value=far_bound,
            empirical_regret=0.0,
            is_satisfied=True,
            margin=far_bound,
            parameters={
                'T': T,
                'd': self.context_dim,
                'K': self.n_arms,
                'lambda': self.lambda_fairness,
                'regret_component': regret_bound,
                'violation_component': self.lambda_fairness * violation_bound
            }
        )

    def verify_regret_bound(
        self,
        empirical_regret: float,
        T: int
    ) -> Tuple[bool, float]:
        """
        Verify if empirical regret satisfies theoretical bound.

        Args:
            empirical_regret: Observed cumulative regret
            T: Time horizon

        Returns:
            Tuple of (is_satisfied, margin_to_bound)
        """
        bound = compute_linucb_regret_bound(
            T=T,
            d=self.context_dim,
            K=self.n_arms,
            delta=self.delta
        )

        is_satisfied = empirical_regret <= bound
        margin = bound - empirical_regret

        if not is_satisfied:
            logger.warning(
                f"Regret bound violated: R(T)={empirical_regret:.2f} > bound={bound:.2f}"
            )

        return is_satisfied, margin

    def verify_far_bound(
        self,
        empirical_regret: float,
        empirical_violation: float,
        T: int
    ) -> Tuple[bool, float]:
        """
        Verify if empirical FAR satisfies theoretical bound.

        Args:
            empirical_regret: Observed cumulative regret
            empirical_violation: Observed cumulative violation
            T: Time horizon

        Returns:
            Tuple of (is_satisfied, margin_to_bound)
        """
        empirical_far = empirical_regret + self.lambda_fairness * empirical_violation

        far_bound_result = self.compute_far_bound(T)
        bound = far_bound_result.bound_value

        is_satisfied = empirical_far <= bound
        margin = bound - empirical_far

        return is_satisfied, margin

    def compute_bound_trajectory(
        self,
        T_max: int,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bound values over time for plotting.

        Args:
            T_max: Maximum time horizon
            n_points: Number of points to compute

        Returns:
            Tuple of (timesteps, regret_bounds, far_bounds)
        """
        timesteps = np.linspace(1, T_max, n_points).astype(int)

        regret_bounds = np.array([
            compute_linucb_regret_bound(
                T=t, d=self.context_dim, K=self.n_arms, delta=self.delta
            )
            for t in timesteps
        ])

        far_bounds = np.array([
            self.compute_far_bound(t).bound_value
            for t in timesteps
        ])

        return timesteps, regret_bounds, far_bounds

    def compute_regret_per_round_bound(self, T: int) -> float:
        """
        Compute expected regret per round bound.

        For sublinear regret: R(T)/T → 0 as T → ∞

        Bound: R(T)/T ≤ O(d√(K log(T/δ) / T))

        Args:
            T: Time horizon

        Returns:
            Per-round regret bound
        """
        if T <= 0:
            return float('inf')

        bound = compute_linucb_regret_bound(
            T=T, d=self.context_dim, K=self.n_arms, delta=self.delta
        )

        return bound / T

    def get_convergence_rate(self) -> str:
        """Get theoretical convergence rate description."""
        return f"O({self.context_dim}√({self.n_arms}T log(T/{self.delta})))"

    def get_bounds_summary(self, T: int) -> Dict[str, float]:
        """
        Get comprehensive bounds summary.

        Args:
            T: Time horizon

        Returns:
            Dictionary with all bound values
        """
        regret_bound = self.compute_regret_bound(T)
        far_bound = self.compute_far_bound(T)

        return {
            'T': T,
            'd': self.context_dim,
            'K': self.n_arms,
            'delta': self.delta,
            'lambda_fairness': self.lambda_fairness,
            'regret_bound': regret_bound.bound_value,
            'far_bound': far_bound.bound_value,
            'regret_per_round_bound': self.compute_regret_per_round_bound(T),
            'convergence_rate': self.get_convergence_rate(),
            'violation_bound': np.sqrt(T),  # O(√T)
        }

    def __repr__(self) -> str:
        return (f"TheoreticalBoundComputer(K={self.n_arms}, d={self.context_dim}, "
                f"δ={self.delta}, λ={self.lambda_fairness})")
