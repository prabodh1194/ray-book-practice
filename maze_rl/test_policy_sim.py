from __future__ import annotations

import logging

from maze_rl.environment import Environment
from maze_rl.policy import Policy
from maze_rl.simulation import Simulation

logger = logging.getLogger(__name__)

env = Environment()

policy = Policy(env)
simulation = Simulation(env)

simulation.rollout(policy, render=True)
