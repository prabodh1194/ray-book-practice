from __future__ import annotations

import logging
import time

from maze_rl.environment import Environment

logger = logging.getLogger(__name__)

env = Environment()

while not env.is_done():
    random_action = env.action_space.sample()
    env.step(random_action)
    env.render()
    time.sleep(0.1)
