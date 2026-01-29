"""Neural network-based contextual bandit with MC Dropout."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
from collections import deque
import logging

from .base_bandit import BaseBandit

logger = logging.getLogger(__name__)


class NeuralBanditNetwork(nn.Module):
    """Neural network for reward prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class NeuralBandit(BaseBandit):
    """Neural contextual bandit with MC Dropout for uncertainty."""

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.1,
        n_mc_samples: int = 10,
        alpha: float = 0.5,
        buffer_size: int = 10000,
        batch_size: int = 32,
        update_frequency: int = 10,
        **kwargs
    ):
        super().__init__(n_arms, context_dim, **kwargs)

        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.n_mc_samples = n_mc_samples
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        # Neural network (runs on CPU to avoid GPU conflict with LLM)
        input_dim = context_dim + n_arms
        self.network = NeuralBanditNetwork(input_dim, hidden_dim, dropout_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)
        self.update_counter = 0

    def select_arm(self, context: np.ndarray) -> Tuple[int, float]:
        context = context.reshape(-1)

        ucb_values = np.zeros(self.n_arms)

        self.network.eval()  # Enable dropout for MC sampling

        for arm in range(self.n_arms):
            # Create input: concat(context, arm_one_hot)
            arm_one_hot = np.zeros(self.n_arms)
            arm_one_hot[arm] = 1.0
            input_vec = np.concatenate([context, arm_one_hot])
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0)

            # MC Dropout sampling
            predictions = []
            for _ in range(self.n_mc_samples):
                with torch.no_grad():
                    pred = self.network(input_tensor).item()
                    predictions.append(pred)

            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            # UCB = mean + alpha * std
            ucb_values[arm] = mean_pred + self.alpha * std_pred

        selected_arm = int(np.argmax(ucb_values))
        confidence = ucb_values[selected_arm]

        return selected_arm, confidence

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        context = context.reshape(-1)

        # Add to replay buffer
        arm_one_hot = np.zeros(self.n_arms)
        arm_one_hot[arm] = 1.0
        input_vec = np.concatenate([context, arm_one_hot])
        self.buffer.append((input_vec, reward))

        # Record history
        self._record_history(context, arm, reward)

        self.update_counter += 1

        # Train network periodically
        if self.update_counter % self.update_frequency == 0 and len(self.buffer) >= self.batch_size:
            self._train_network()

    def _train_network(self):
        """Train network on batch from replay buffer."""
        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        inputs = torch.FloatTensor([item[0] for item in batch])
        targets = torch.FloatTensor([[item[1]] for item in batch])

        # Train
        self.network.train()
        self.optimizer.zero_grad()
        predictions = self.network(inputs)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

    def get_arm_values(self, context: np.ndarray) -> np.ndarray:
        context = context.reshape(-1)
        values = np.zeros(self.n_arms)

        self.network.eval()

        for arm in range(self.n_arms):
            arm_one_hot = np.zeros(self.n_arms)
            arm_one_hot[arm] = 1.0
            input_vec = np.concatenate([context, arm_one_hot])
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0)

            with torch.no_grad():
                values[arm] = self.network(input_tensor).item()

        return values

    def _get_algorithm_state(self) -> Dict:
        return {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'buffer': list(self.buffer),
            'update_counter': self.update_counter,
            'hyperparams': {
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'dropout_rate': self.dropout_rate,
                'n_mc_samples': self.n_mc_samples,
                'alpha': self.alpha
            }
        }

    def _set_algorithm_state(self, state: Dict) -> None:
        self.network.load_state_dict(state['network_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.buffer = deque(state['buffer'], maxlen=self.buffer_size)
        self.update_counter = state['update_counter']
        # Restore hyperparams
        hp = state['hyperparams']
        self.hidden_dim = hp['hidden_dim']
        self.learning_rate = hp['learning_rate']
        self.dropout_rate = hp['dropout_rate']
        self.n_mc_samples = hp['n_mc_samples']
        self.alpha = hp['alpha']

    def _reset_algorithm_state(self) -> None:
        self.network = NeuralBanditNetwork(
            self.context_dim + self.n_arms,
            self.hidden_dim,
            self.dropout_rate
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.buffer.clear()
        self.update_counter = 0
