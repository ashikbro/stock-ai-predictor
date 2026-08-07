"""Simple tabular Q-learning trading agent baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QLearningTradingAgent:
    """A lightweight RL trading baseline with a discrete action space.

    The agent maps coarse market states to Q-values and learns via
    temporal-difference updates. This is a baseline for documentation and
    experimentation, not a production execution engine.
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    n_actions: int = 3
    q_table: dict[int, np.ndarray] = field(default_factory=dict)

    def _ensure_state(self, state: int) -> None:
        """Initialize state-action values if state has not been seen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=float)

    def select_action(self, state: int) -> int:
        """Select an action using epsilon-greedy exploration policy."""
        self._ensure_state(state)

        # Exploration is intentionally explicit so behavior is easier to audit
        # during experiments and backtesting comparisons.
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Apply temporal-difference Q-learning update for one transition."""
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_value = self.q_table[state][action]
        next_max = float(np.max(self.q_table[next_state]))
        td_target = reward + self.discount_factor * next_max
        self.q_table[state][action] = old_value + self.learning_rate * (td_target - old_value)

    def decay_epsilon(self, decay: float = 0.995, floor: float = 0.01) -> None:
        """Decay exploration rate after each episode with a lower bound."""
        self.epsilon = max(floor, self.epsilon * decay)
