from __future__ import annotations

import logging
import random

import numpy as np

from maze_rl.environment import Environment

logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, env: Environment):
        self.state_action_table = [
            [0 for _ in range(env.action_space.n)]
            for _ in range(env.observation_space.n)
        ]

        self.action_space = env.action_space

    def get_action(self, state: int, explore=True, epsilon=0.1):
        if explore and random.uniform(0, 1) < epsilon:
            return self.action_space.sample()
        return np.argmax(self.state_action_table[state])

    def update_policy(self, experiences: list[tuple], weight=0.1, discount=0.9):
        for state, action, reward, next_state in experiences:
            old_value = self.state_action_table[state][action]
            next_max = np.max(self.state_action_table[next_state])
            new_value = (1 - weight) * old_value + weight * (reward + discount * next_max)
            self.state_action_table[state][action] = new_value
