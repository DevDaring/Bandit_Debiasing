"""Thompson Sampling with linear reward model."""

import numpy as np
from typing import Tuple, Dict
import logging

from .base_bandit import BaseBandit

logger = logging.getLogger(__name__)


class ThompsonSamplingLinear(BaseBandit):
    """Thompson Sampling with Bayesian linear model."""

    def __init__(self, n_arms: int, context_dim: int, noise_std: float = 0.5, **kwargs):
        super().__init__(n_arms, context_dim, **kwargs)
        self.noise_std = noise_std

        # Initialize precision matrices and f vectors
        self.B = [np.identity(context_dim) for _ in range(n_arms)]
        self.f = [np.zeros(context_dim) for _ in range(n_arms)]
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> Tuple[int, float]:
        context = context.reshape(-1)

        sampled_rewards = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            # Compute posterior mean
            try:
                self.mu[arm] = np.linalg.solve(self.B[arm], self.f[arm])
                B_inv = np.linalg.inv(self.B[arm])

                # Sample theta from N(mu, B_inv)
                theta_sample = np.random.multivariate_normal(self.mu[arm], B_inv)

                # Compute expected reward
                sampled_rewards[arm] = theta_sample.T @ context

            except np.linalg.LinAlgError:
                sampled_rewards[arm] = 0.0

        selected_arm = int(np.argmax(sampled_rewards))
        confidence = sampled_rewards[selected_arm]

        return selected_arm, confidence

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        context = context.reshape(-1)

        # Update precision matrix B
        self.B[arm] += np.outer(context, context) / (self.noise_std ** 2)

        # Update f vector
        self.f[arm] += reward * context / (self.noise_std ** 2)

        # Record history
        self._record_history(context, arm, reward)

    def get_arm_values(self, context: np.ndarray) -> np.ndarray:
        context = context.reshape(-1)
        values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            try:
                mu = np.linalg.solve(self.B[arm], self.f[arm])
                values[arm] = mu.T @ context
            except:
                values[arm] = 0.0

        return values

    def _get_algorithm_state(self) -> Dict:
        return {
            'noise_std': self.noise_std,
            'B': [B.copy() for B in self.B],
            'f': [f.copy() for f in self.f],
            'mu': [mu.copy() for mu in self.mu]
        }

    def _set_algorithm_state(self, state: Dict) -> None:
        self.noise_std = state['noise_std']
        self.B = [B.copy() for B in state['B']]
        self.f = [f.copy() for f in state['f']]
        self.mu = [mu.copy() for mu in state['mu']]

    def _reset_algorithm_state(self) -> None:
        self.B = [np.identity(self.context_dim) for _ in range(self.n_arms)]
        self.f = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
        self.mu = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
