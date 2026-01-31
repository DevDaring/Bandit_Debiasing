"""
Regret tracker for contextual bandits.

Implements cumulative regret tracking with:
- R(T) = Σ_{t=1}^T [r(a*_t) - r(a_t)]
- Where a*_t is the optimal arm and a_t is the selected arm

For LinUCB, theoretical bound: R(T) ≤ O(d√(KT log(T/δ)))
where d = context dimension, K = number of arms, δ = confidence parameter
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RegretEntry:
    """Single regret observation."""
    timestep: int
    selected_arm: int
    selected_reward: float
    optimal_arm: int
    optimal_reward: float
    instantaneous_regret: float
    cumulative_regret: float
    context_hash: Optional[str] = None


class RegretTracker:
    """
    Track cumulative regret for bandit algorithms.

    Regret is the difference between the reward of the optimal arm
    and the reward of the selected arm over time.

    R(T) = Σ_{t=1}^T [r*(t) - r(a_t)]

    where r*(t) is the reward of the optimal arm at time t.
    """

    def __init__(
        self,
        n_arms: int,
        compute_optimal: bool = True,
        window_size: int = 100
    ):
        """
        Initialize regret tracker.

        Args:
            n_arms: Number of bandit arms
            compute_optimal: Whether to track optimal arm rewards
            window_size: Window for moving average statistics
        """
        self.n_arms = n_arms
        self.compute_optimal = compute_optimal
        self.window_size = window_size

        # Tracking history
        self.history: List[RegretEntry] = []
        self.cumulative_regret: float = 0.0
        self.timestep: int = 0

        # Per-arm statistics
        self.arm_rewards: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
        self.arm_counts: Dict[int, int] = {i: 0 for i in range(n_arms)}

        # Optimal arm tracking (oracle)
        self.optimal_rewards: List[float] = []

    def update(
        self,
        selected_arm: int,
        reward: float,
        optimal_arm: Optional[int] = None,
        optimal_reward: Optional[float] = None,
        context: Optional[np.ndarray] = None
    ) -> float:
        """
        Record a new observation and update cumulative regret.

        Args:
            selected_arm: The arm that was selected
            reward: The reward received
            optimal_arm: The optimal arm (if known, for oracle comparison)
            optimal_reward: The reward the optimal arm would have given
            context: Optional context vector (for logging)

        Returns:
            Current instantaneous regret
        """
        self.timestep += 1

        # Update arm statistics
        self.arm_rewards[selected_arm].append(reward)
        self.arm_counts[selected_arm] += 1

        # Compute instantaneous regret
        if optimal_reward is not None:
            inst_regret = optimal_reward - reward
        elif self.compute_optimal and len(self.arm_rewards[selected_arm]) > 10:
            # Estimate optimal from observed arm means
            arm_means = {
                arm: np.mean(rewards) if rewards else 0
                for arm, rewards in self.arm_rewards.items()
            }
            best_arm = max(arm_means, key=arm_means.get)
            inst_regret = max(0, arm_means[best_arm] - reward)
        else:
            inst_regret = 0.0

        # Update cumulative regret
        self.cumulative_regret += max(0, inst_regret)

        # Create entry
        context_hash = None
        if context is not None:
            context_hash = hash(context.tobytes()) % (10**8)

        entry = RegretEntry(
            timestep=self.timestep,
            selected_arm=selected_arm,
            selected_reward=reward,
            optimal_arm=optimal_arm if optimal_arm is not None else -1,
            optimal_reward=optimal_reward if optimal_reward is not None else 0.0,
            instantaneous_regret=inst_regret,
            cumulative_regret=self.cumulative_regret,
            context_hash=str(context_hash) if context_hash else None
        )

        self.history.append(entry)

        return inst_regret

    def get_cumulative_regret(self) -> float:
        """Get current cumulative regret R(T)."""
        return self.cumulative_regret

    def get_average_regret(self) -> float:
        """Get average regret R(T)/T."""
        if self.timestep == 0:
            return 0.0
        return self.cumulative_regret / self.timestep

    def get_regret_over_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cumulative regret trajectory.

        Returns:
            Tuple of (timesteps, cumulative_regrets)
        """
        if not self.history:
            return np.array([]), np.array([])

        timesteps = np.array([e.timestep for e in self.history])
        cumulative = np.array([e.cumulative_regret for e in self.history])

        return timesteps, cumulative

    def get_moving_average_regret(self) -> np.ndarray:
        """Get moving average of instantaneous regret."""
        if not self.history:
            return np.array([])

        inst_regrets = np.array([e.instantaneous_regret for e in self.history])

        # Compute moving average
        kernel = np.ones(self.window_size) / self.window_size
        ma = np.convolve(inst_regrets, kernel, mode='valid')

        return ma

    def get_arm_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get per-arm statistics."""
        stats = {}
        for arm in range(self.n_arms):
            rewards = self.arm_rewards[arm]
            if rewards:
                stats[arm] = {
                    'count': self.arm_counts[arm],
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'selection_rate': self.arm_counts[arm] / self.timestep
                }
            else:
                stats[arm] = {
                    'count': 0,
                    'mean_reward': 0.0,
                    'std_reward': 0.0,
                    'selection_rate': 0.0
                }
        return stats

    def get_regret_statistics(self) -> Dict[str, float]:
        """Get comprehensive regret statistics."""
        if not self.history:
            return {}

        inst_regrets = np.array([e.instantaneous_regret for e in self.history])

        return {
            'cumulative_regret': self.cumulative_regret,
            'average_regret': self.get_average_regret(),
            'final_instantaneous_regret': inst_regrets[-1] if len(inst_regrets) > 0 else 0,
            'max_instantaneous_regret': float(np.max(inst_regrets)),
            'std_instantaneous_regret': float(np.std(inst_regrets)),
            'timesteps': self.timestep,
            'regret_per_sqrt_t': self.cumulative_regret / np.sqrt(self.timestep) if self.timestep > 0 else 0,
        }

    def is_sublinear(self, threshold: float = 0.1) -> bool:
        """
        Check if regret is sublinear (R(T)/T → 0).

        For sublinear regret, R(T)/T should decrease over time.
        We check if the rate is below threshold.
        """
        if self.timestep < 100:
            return True  # Not enough data

        avg_regret = self.get_average_regret()
        return avg_regret < threshold

    def save(self, filepath: str):
        """Save regret history to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'n_arms': self.n_arms,
            'timestep': self.timestep,
            'cumulative_regret': self.cumulative_regret,
            'statistics': self.get_regret_statistics(),
            'arm_statistics': self.get_arm_statistics(),
            'history': [
                {
                    'timestep': e.timestep,
                    'selected_arm': e.selected_arm,
                    'selected_reward': e.selected_reward,
                    'instantaneous_regret': e.instantaneous_regret,
                    'cumulative_regret': e.cumulative_regret,
                }
                for e in self.history
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved regret history to {path}")

    @classmethod
    def load(cls, filepath: str) -> 'RegretTracker':
        """Load regret tracker from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tracker = cls(n_arms=data['n_arms'])
        tracker.timestep = data['timestep']
        tracker.cumulative_regret = data['cumulative_regret']

        for entry_data in data['history']:
            entry = RegretEntry(
                timestep=entry_data['timestep'],
                selected_arm=entry_data['selected_arm'],
                selected_reward=entry_data['selected_reward'],
                optimal_arm=-1,
                optimal_reward=0.0,
                instantaneous_regret=entry_data['instantaneous_regret'],
                cumulative_regret=entry_data['cumulative_regret']
            )
            tracker.history.append(entry)

        return tracker

    def __repr__(self) -> str:
        return f"RegretTracker(T={self.timestep}, R(T)={self.cumulative_regret:.4f}, R(T)/T={self.get_average_regret():.6f})"
