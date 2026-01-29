"""
Linear Upper Confidence Bound (LinUCB) algorithm.
"""

import numpy as np
from typing import Tuple, Dict
import logging

from .base_bandit import BaseBandit

logger = logging.getLogger(__name__)


class LinUCB(BaseBandit):
    """
    LinUCB contextual bandit algorithm.

    Uses Sherman-Morrison formula for efficient matrix updates.
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 0.5, **kwargs):
        """
        Initialize LinUCB.

        Args:
            n_arms: Number of arms
            context_dim: Context dimension
            alpha: Exploration parameter
        """
        super().__init__(n_arms, context_dim, **kwargs)
        self.alpha = alpha

        # Initialize A_inv and b for each arm
        self.A_inv = [np.identity(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Select arm using UCB strategy.

        Args:
            context: Context vector

        Returns:
            (arm_index, confidence)
        """
        context = context.reshape(-1)
        assert len(context) == self.context_dim

        ucb_values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            # Compute theta = A_inv @ b
            theta = self.A_inv[arm] @ self.b[arm]

            # Compute UCB
            mean_reward = theta.T @ context
            uncertainty = self.alpha * np.sqrt(context.T @ self.A_inv[arm] @ context)

            ucb_values[arm] = mean_reward + uncertainty

        # Select arm with highest UCB
        selected_arm = int(np.argmax(ucb_values))
        confidence = ucb_values[selected_arm]

        return selected_arm, confidence

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """
        Update LinUCB parameters using Sherman-Morrison formula.

        Args:
            context: Context vector
            arm: Selected arm
            reward: Observed reward
        """
        context = context.reshape(-1)

        # Sherman-Morrison update for A_inv
        # A_new_inv = A_inv - (A_inv @ x @ x.T @ A_inv) / (1 + x.T @ A_inv @ x)
        A_inv_x = self.A_inv[arm] @ context
        denominator = 1.0 + context.T @ A_inv_x

        self.A_inv[arm] -= np.outer(A_inv_x, A_inv_x) / denominator

        # Update b
        self.b[arm] += reward * context

        # Record history
        self._record_history(context, arm, reward)

    def get_arm_values(self, context: np.ndarray) -> np.ndarray:
        """Get expected reward for each arm."""
        context = context.reshape(-1)
        values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            theta = self.A_inv[arm] @ self.b[arm]
            values[arm] = theta.T @ context

        return values

    def _get_algorithm_state(self) -> Dict:
        """Get algorithm-specific state."""
        return {
            'alpha': self.alpha,
            'A_inv': [A.copy() for A in self.A_inv],
            'b': [b.copy() for b in self.b]
        }

    def _set_algorithm_state(self, state: Dict) -> None:
        """Set algorithm-specific state."""
        self.alpha = state['alpha']
        self.A_inv = [A.copy() for A in state['A_inv']]
        self.b = [b.copy() for b in state['b']]

    def _reset_algorithm_state(self) -> None:
        """Reset algorithm state."""
        self.A_inv = [np.identity(self.context_dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
