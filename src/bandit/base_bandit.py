"""
Abstract base class for all bandit algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, List, Optional
import pickle
import time


class BaseBandit(ABC):
    """Abstract base class for contextual bandit algorithms."""

    def __init__(self, n_arms: int, context_dim: int, arm_names: Optional[List[str]] = None):
        """
        Initialize bandit.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of context vector
            arm_names: Optional list of arm names
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.arm_names = arm_names or [f"arm_{i}" for i in range(n_arms)]
        self.history = []  # List of (context, arm, reward, timestamp)
        self.total_rounds = 0

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Select which arm to pull given context.

        Args:
            context: Context vector of shape (context_dim,)

        Returns:
            Tuple of (selected_arm_index, confidence_score)
        """
        pass

    @abstractmethod
    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """
        Update model after observing reward.

        Args:
            context: Context vector used for selection
            arm: Index of arm that was pulled
            reward: Observed reward in [0, 1]
        """
        pass

    def get_arm_values(self, context: np.ndarray) -> np.ndarray:
        """
        Get expected values for all arms.

        Args:
            context: Context vector

        Returns:
            Array of shape (n_arms,) with expected values
        """
        return np.zeros(self.n_arms)

    def save(self, path: str) -> None:
        """Save model state."""
        state = {
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'arm_names': self.arm_names,
            'history': self.history,
            'total_rounds': self.total_rounds,
            'algorithm_state': self._get_algorithm_state()
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load model state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.n_arms = state['n_arms']
        self.context_dim = state['context_dim']
        self.arm_names = state['arm_names']
        self.history = state['history']
        self.total_rounds = state['total_rounds']
        self._set_algorithm_state(state['algorithm_state'])

    def reset(self) -> None:
        """Reset model to initial state."""
        self.history = []
        self.total_rounds = 0
        self._reset_algorithm_state()

    def _get_algorithm_state(self) -> Dict:
        """Get algorithm-specific state for saving."""
        return {}

    def _set_algorithm_state(self, state: Dict) -> None:
        """Set algorithm-specific state when loading."""
        pass

    def _reset_algorithm_state(self) -> None:
        """Reset algorithm-specific state."""
        pass

    def _record_history(self, context: np.ndarray, arm: int, reward: float):
        """Record interaction to history."""
        self.history.append({
            'context': context.copy(),
            'arm': arm,
            'reward': reward,
            'timestamp': time.time(),
            'round': self.total_rounds
        })
        self.total_rounds += 1

    def get_arm_statistics(self) -> Dict:
        """Get statistics per arm."""
        stats = {arm_name: {'count': 0, 'total_reward': 0.0, 'avg_reward': 0.0}
                 for arm_name in self.arm_names}

        for record in self.history:
            arm = record['arm']
            arm_name = self.arm_names[arm]
            stats[arm_name]['count'] += 1
            stats[arm_name]['total_reward'] += record['reward']

        for arm_name in self.arm_names:
            if stats[arm_name]['count'] > 0:
                stats[arm_name]['avg_reward'] = (
                    stats[arm_name]['total_reward'] / stats[arm_name]['count']
                )

        return stats
