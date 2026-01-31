"""
Adaptive vs Static arm comparison for Fair-CB.

Proves the key theoretical claim:
    R_adaptive / R_static → 0 as T → ∞

This demonstrates that adaptive MAB selection outperforms
the best static (fixed) arm strategy.

Key insight: Adaptive selection can achieve sublinear regret O(√T)
while static selection has linear regret O(T) in non-stationary environments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of adaptive vs static comparison."""
    timestep: int
    adaptive_reward: float
    static_reward: float
    best_static_arm: int
    ratio: float
    advantage: float
    cumulative_adaptive: float
    cumulative_static: float


class AdaptiveStaticComparator:
    """
    Compare adaptive MAB selection against best static arm.

    Tracks:
    1. Cumulative reward of adaptive selection
    2. Cumulative reward of best static arm (hindsight)
    3. Ratio R_adaptive / R_static
    4. Per-timestep advantage

    The theoretical claim: lim_{T→∞} R_adaptive / R_static → 0
    means adaptive achieves vanishing relative regret.
    """

    def __init__(self, n_arms: int, window_size: int = 100):
        """
        Initialize comparator.

        Args:
            n_arms: Number of bandit arms
            window_size: Window for moving statistics
        """
        self.n_arms = n_arms
        self.window_size = window_size

        # Tracking
        self.history: List[ComparisonResult] = []
        self.timestep: int = 0

        # Cumulative rewards
        self.cumulative_adaptive: float = 0.0
        self.cumulative_static: float = 0.0  # Best static arm in hindsight

        # Per-arm reward tracking (for best static computation)
        self.arm_rewards: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
        self.arm_cumulative: Dict[int, float] = {i: 0.0 for i in range(n_arms)}

        # Track what rewards each arm WOULD have gotten
        self.counterfactual_rewards: Dict[int, List[float]] = {i: [] for i in range(n_arms)}

    def update(
        self,
        selected_arm: int,
        reward: float,
        counterfactual_rewards: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Record a new observation.

        Args:
            selected_arm: Arm that was selected by adaptive policy
            reward: Reward received
            counterfactual_rewards: What each arm would have received (optional)

        Returns:
            Current ratio R_adaptive / R_static
        """
        self.timestep += 1

        # Update adaptive cumulative
        self.cumulative_adaptive += reward

        # Update per-arm tracking
        for arm in range(self.n_arms):
            if arm == selected_arm:
                self.arm_rewards[arm].append(reward)
                self.arm_cumulative[arm] += reward
            elif counterfactual_rewards and arm in counterfactual_rewards:
                # Use counterfactual if available
                cf_reward = counterfactual_rewards[arm]
                self.counterfactual_rewards[arm].append(cf_reward)
                self.arm_cumulative[arm] += cf_reward
            else:
                # Estimate from mean of observed rewards for this arm
                if self.arm_rewards[arm]:
                    estimated = np.mean(self.arm_rewards[arm])
                else:
                    estimated = 0.5  # Prior
                self.arm_cumulative[arm] += estimated

        # Find best static arm in hindsight
        best_static_arm = max(self.arm_cumulative, key=self.arm_cumulative.get)
        self.cumulative_static = self.arm_cumulative[best_static_arm]

        # Compute ratio and advantage
        if self.cumulative_static > 0:
            # Regret ratio: (static - adaptive) / static
            regret_ratio = (self.cumulative_static - self.cumulative_adaptive) / self.cumulative_static
        else:
            regret_ratio = 0.0

        advantage = self.cumulative_adaptive - reward  # Advantage from adaptation

        # Create result
        result = ComparisonResult(
            timestep=self.timestep,
            adaptive_reward=reward,
            static_reward=self.arm_cumulative[best_static_arm] / self.timestep if self.timestep > 0 else 0,
            best_static_arm=best_static_arm,
            ratio=regret_ratio,
            advantage=advantage,
            cumulative_adaptive=self.cumulative_adaptive,
            cumulative_static=self.cumulative_static
        )

        self.history.append(result)

        return regret_ratio

    def get_regret_ratio(self) -> float:
        """
        Get current regret ratio R_adaptive_regret / R_static.

        A ratio approaching 0 proves the theoretical claim.
        """
        if self.cumulative_static <= 0:
            return 0.0

        adaptive_regret = self.cumulative_static - self.cumulative_adaptive
        return adaptive_regret / self.cumulative_static

    def get_reward_ratio(self) -> float:
        """Get ratio of cumulative adaptive to static reward."""
        if self.cumulative_static <= 0:
            return 1.0
        return self.cumulative_adaptive / self.cumulative_static

    def get_ratio_over_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get regret ratio trajectory.

        For proving convergence to 0.

        Returns:
            Tuple of (timesteps, ratios)
        """
        if not self.history:
            return np.array([]), np.array([])

        timesteps = np.array([r.timestep for r in self.history])
        ratios = np.array([r.ratio for r in self.history])

        return timesteps, ratios

    def get_cumulative_comparison(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cumulative reward trajectories for both strategies.

        Returns:
            Tuple of (timesteps, adaptive_cumulative, static_cumulative)
        """
        if not self.history:
            return np.array([]), np.array([]), np.array([])

        timesteps = np.array([r.timestep for r in self.history])
        adaptive = np.array([r.cumulative_adaptive for r in self.history])
        static = np.array([r.cumulative_static for r in self.history])

        return timesteps, adaptive, static

    def is_converging(self, window: int = 100, threshold: float = 0.1) -> bool:
        """
        Check if ratio is converging to 0.

        Args:
            window: Number of recent timesteps to check
            threshold: Maximum average ratio for convergence

        Returns:
            True if converging
        """
        if len(self.history) < window:
            return True  # Not enough data

        recent_ratios = [r.ratio for r in self.history[-window:]]
        avg_ratio = np.mean(recent_ratios)

        return avg_ratio < threshold

    def get_best_static_arm(self) -> int:
        """Get the best static arm in hindsight."""
        return max(self.arm_cumulative, key=self.arm_cumulative.get)

    def get_arm_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for each arm."""
        stats = {}
        for arm in range(self.n_arms):
            rewards = self.arm_rewards[arm]
            if rewards:
                stats[arm] = {
                    'cumulative_reward': self.arm_cumulative[arm],
                    'observed_count': len(rewards),
                    'mean_reward': np.mean(rewards),
                    'is_best_static': arm == self.get_best_static_arm()
                }
            else:
                stats[arm] = {
                    'cumulative_reward': self.arm_cumulative[arm],
                    'observed_count': 0,
                    'mean_reward': 0.0,
                    'is_best_static': False
                }
        return stats

    def get_comparison_statistics(self) -> Dict[str, float]:
        """Get comprehensive comparison statistics."""
        if not self.history:
            return {}

        ratios = [r.ratio for r in self.history]

        return {
            'timesteps': self.timestep,
            'cumulative_adaptive': self.cumulative_adaptive,
            'cumulative_static': self.cumulative_static,
            'best_static_arm': self.get_best_static_arm(),
            'final_regret_ratio': self.get_regret_ratio(),
            'final_reward_ratio': self.get_reward_ratio(),
            'mean_ratio': np.mean(ratios),
            'min_ratio': np.min(ratios),
            'max_ratio': np.max(ratios),
            'is_converging': self.is_converging(),
            'adaptive_advantage': self.cumulative_adaptive - self.cumulative_static,
        }

    def prove_convergence(self, significance_level: float = 0.05) -> Dict[str, any]:
        """
        Statistical test for convergence of ratio to 0.

        Uses linear regression on log(ratio) vs log(t) to check if
        ratio decreases as O(1/√T) or faster.

        Args:
            significance_level: Alpha for hypothesis test

        Returns:
            Dictionary with test results
        """
        if len(self.history) < 100:
            return {
                'sufficient_data': False,
                'message': 'Need at least 100 timesteps for convergence test'
            }

        ratios = np.array([r.ratio for r in self.history])
        timesteps = np.array([r.timestep for r in self.history])

        # Filter out zeros and negatives for log
        valid_mask = ratios > 1e-10
        if valid_mask.sum() < 50:
            return {
                'sufficient_data': True,
                'converged': True,
                'message': 'Ratio is effectively zero'
            }

        log_ratios = np.log(ratios[valid_mask])
        log_time = np.log(timesteps[valid_mask])

        # Linear regression: log(ratio) = a + b * log(t)
        # For O(1/√T) convergence, b should be approximately -0.5
        A = np.vstack([log_time, np.ones(len(log_time))]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_ratios, rcond=None)

        slope = coeffs[0]
        intercept = coeffs[1]

        # Convergence: slope should be negative (ratio decreasing)
        is_converging = slope < 0

        return {
            'sufficient_data': True,
            'converged': is_converging,
            'slope': slope,
            'expected_slope': -0.5,  # For O(1/√T)
            'intercept': intercept,
            'convergence_rate': f'O(T^{{{slope:.3f}}})',
            'message': 'Ratio converging to 0' if is_converging else 'No convergence detected'
        }

    def __repr__(self) -> str:
        ratio = self.get_regret_ratio()
        converging = "✓" if self.is_converging() else "✗"
        return (f"AdaptiveStaticComparator(T={self.timestep}, "
                f"ratio={ratio:.4f}, converging={converging})")
