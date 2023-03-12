from __future__ import annotations

import logging
import time

from maze_rl.environment import Environment

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, env: Environment):
        self.env = env

    def rollout(self, policy, render=False, explore=True, epsilon=0.1):
        expreriences = []
        state = self.env.reset()

        done = False

        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_state, reward, done, info = self.env.step(action)
            expreriences.append((state, action, reward, next_state))

            state = next_state

            if render:
                self.env.render()
                time.sleep(0.1)

        return expreriences
