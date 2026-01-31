"""
Fairness-Aware Regret (FAR) metric.

Novel metric that combines traditional bandit regret with fairness violations.

Formula:
    FAR(T) = R(T) + λ × V(T)

Where:
    R(T) = Cumulative regret = Σ_{t=1}^T [r*(t) - r(a_t)]
    V(T) = Cumulative fairness violation = Σ_{t=1}^T max(0, bias(t) - τ)
    λ = Fairness weight parameter (balances regret vs fairness)
    τ = Bias threshold

Properties:
- Captures trade-off between reward optimization and fairness
- λ = 0: Pure regret minimization (ignores fairness)
- λ → ∞: Pure fairness satisfaction (ignores rewards)
- Reasonable default: λ = 0.5 (equal weight to both objectives)

Theoretical Guarantee:
    For Fair-CB: FAR(T) ≤ O(d√(KT log(T))) + λ × O(√T) = O(d√(KT log(T)))
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FAREntry:
    """Single FAR observation."""
    timestep: int
    regret: float
    violation: float
    far: float
    cumulative_far: float


@dataclass
class FARResult:
    """Complete FAR computation result."""
    far_score: float
    cumulative_regret: float
    cumulative_violation: float
    lambda_weight: float
    regret_component: float
    violation_component: float
    average_far: float
    timesteps: int


def compute_far(
    cumulative_regret: float,
    cumulative_violation: float,
    lambda_weight: float = 0.5
) -> float:
    """
    Compute Fairness-Aware Regret.

    FAR = R(T) + λ × V(T)

    Args:
        cumulative_regret: Total regret R(T)
        cumulative_violation: Total fairness violation V(T)
        lambda_weight: Weight for fairness violations (λ)

    Returns:
        FAR score
    """
    return cumulative_regret + lambda_weight * cumulative_violation


class FairnessAwareRegret:
    """
    Compute and track Fairness-Aware Regret (FAR) over time.

    FAR combines:
    1. Traditional bandit regret (reward loss)
    2. Fairness violations (bias above threshold)

    Higher λ prioritizes fairness over reward optimization.
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        bias_threshold: float = 0.3
    ):
        """
        Initialize FAR calculator.

        Args:
            lambda_weight: Weight for fairness violations (default 0.5)
            bias_threshold: Bias threshold τ for violations
        """
        self.lambda_weight = lambda_weight
        self.bias_threshold = bias_threshold

        # Tracking
        self.history: List[FAREntry] = []
        self.timestep: int = 0

        # Cumulative values
        self.cumulative_regret: float = 0.0
        self.cumulative_violation: float = 0.0
        self.cumulative_far: float = 0.0

    def update(
        self,
        regret: float,
        bias_score: float
    ) -> float:
        """
        Add a new observation.

        Args:
            regret: Instantaneous regret at this timestep
            bias_score: Bias score at this timestep

        Returns:
            Instantaneous FAR value
        """
        self.timestep += 1

        # Compute violation
        violation = max(0.0, bias_score - self.bias_threshold)

        # Update cumulative values
        self.cumulative_regret += regret
        self.cumulative_violation += violation

        # Compute instantaneous FAR
        inst_far = regret + self.lambda_weight * violation
        self.cumulative_far += inst_far

        # Create entry
        entry = FAREntry(
            timestep=self.timestep,
            regret=regret,
            violation=violation,
            far=inst_far,
            cumulative_far=self.cumulative_far
        )

        self.history.append(entry)

        return inst_far

    def update_from_reward(
        self,
        reward: float,
        optimal_reward: float,
        bias_score: float
    ) -> float:
        """
        Update using reward values instead of regret directly.

        Args:
            reward: Actual reward received
            optimal_reward: Reward of optimal arm
            bias_score: Bias score

        Returns:
            Instantaneous FAR value
        """
        regret = max(0.0, optimal_reward - reward)
        return self.update(regret, bias_score)

    def get_far(self) -> float:
        """Get current cumulative FAR score."""
        return self.cumulative_far

    def get_average_far(self) -> float:
        """Get average FAR per timestep."""
        if self.timestep == 0:
            return 0.0
        return self.cumulative_far / self.timestep

    def get_components(self) -> Tuple[float, float]:
        """
        Get FAR component breakdown.

        Returns:
            Tuple of (regret_component, violation_component)
        """
        regret_component = self.cumulative_regret
        violation_component = self.lambda_weight * self.cumulative_violation
        return regret_component, violation_component

    def compute(self) -> FARResult:
        """
        Compute complete FAR result.

        Returns:
            FARResult with full breakdown
        """
        regret_component, violation_component = self.get_components()

        return FARResult(
            far_score=self.cumulative_far,
            cumulative_regret=self.cumulative_regret,
            cumulative_violation=self.cumulative_violation,
            lambda_weight=self.lambda_weight,
            regret_component=regret_component,
            violation_component=violation_component,
            average_far=self.get_average_far(),
            timesteps=self.timestep
        )

    def get_far_over_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get FAR trajectory over time.

        Returns:
            Tuple of (timesteps, cumulative_far)
        """
        if not self.history:
            return np.array([]), np.array([])

        timesteps = np.array([e.timestep for e in self.history])
        cumulative = np.array([e.cumulative_far for e in self.history])

        return timesteps, cumulative

    def get_component_trajectories(self) -> Dict[str, np.ndarray]:
        """
        Get component trajectories for visualization.

        Returns:
            Dict with timesteps, regret, violation, far arrays
        """
        if not self.history:
            return {}

        # Compute cumulative values over time
        regrets = np.cumsum([e.regret for e in self.history])
        violations = np.cumsum([e.violation for e in self.history])

        return {
            'timesteps': np.array([e.timestep for e in self.history]),
            'cumulative_regret': regrets,
            'cumulative_violation': violations,
            'cumulative_far': np.array([e.cumulative_far for e in self.history])
        }

    def compare_lambda_sensitivity(
        self,
        lambda_values: List[float] = None
    ) -> Dict[float, float]:
        """
        Compute FAR for different λ values (sensitivity analysis).

        Args:
            lambda_values: List of λ values to test

        Returns:
            Dict mapping λ -> FAR score
        """
        if lambda_values is None:
            lambda_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]

        results = {}
        for lam in lambda_values:
            far = self.cumulative_regret + lam * self.cumulative_violation
            results[lam] = far

        return results

    def get_pareto_point(self) -> Tuple[float, float]:
        """
        Get (regret, violation) point for Pareto frontier analysis.

        Returns:
            Tuple of (cumulative_regret, cumulative_violation)
        """
        return (self.cumulative_regret, self.cumulative_violation)

    def get_summary(self) -> Dict[str, any]:
        """Get comprehensive FAR summary."""
        result = self.compute()

        return {
            'far_score': result.far_score,
            'average_far': result.average_far,
            'cumulative_regret': result.cumulative_regret,
            'cumulative_violation': result.cumulative_violation,
            'lambda_weight': result.lambda_weight,
            'regret_component': result.regret_component,
            'violation_component': result.violation_component,
            'regret_pct': (result.regret_component / result.far_score * 100) if result.far_score > 0 else 0,
            'violation_pct': (result.violation_component / result.far_score * 100) if result.far_score > 0 else 0,
            'timesteps': result.timesteps,
            'average_regret': self.cumulative_regret / self.timestep if self.timestep > 0 else 0,
            'average_violation': self.cumulative_violation / self.timestep if self.timestep > 0 else 0,
            'violation_rate': len([e for e in self.history if e.violation > 0]) / self.timestep if self.timestep > 0 else 0
        }

    def reset(self):
        """Clear all accumulated observations."""
        self.history = []
        self.timestep = 0
        self.cumulative_regret = 0.0
        self.cumulative_violation = 0.0
        self.cumulative_far = 0.0

    def __repr__(self) -> str:
        result = self.compute()
        return (f"FAR(score={result.far_score:.4f}, λ={self.lambda_weight}, "
                f"regret={result.cumulative_regret:.4f}, violation={result.cumulative_violation:.4f})")
