from __future__ import annotations

import logging
import os

from maze_rl.discrete import Discrete

logger = logging.getLogger(__name__)


class Environment:

    seeker, goal = (0, 0), (4, 4)
    info = {
        'seeker': seeker, 'goal': goal
    }

    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(5 * 5)

    def reset(self) -> int:
        self.seeker, self.goal = (0, 0), (4, 4)

        return self.get_observation()

    def get_observation(self) -> int:
        return self.seeker[0] * 5 + self.seeker[1]

    def is_done(self) -> bool:
        return self.seeker == self.goal

    def get_reward(self) -> int:
        return 1 if self.is_done() else 0

    def step(self, action) -> tuple[int, int, bool, dict]:
        if action == 0:  # down
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:  # left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError(f'Invalid action: {action}')

        return self.get_observation(), self.get_reward(), self.is_done(), self.info

    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = [['| ' for _ in range(5)] + ['|\n'] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|S'
        print(''.join([''.join(row) for row in grid]))
