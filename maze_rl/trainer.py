from __future__ import annotations

import logging

from maze_rl.environment import Environment
from maze_rl.policy import Policy
from maze_rl.simulation import Simulation

logger = logging.getLogger(__name__)


def train_policy(env, num_episodes=10000, weight=0.1, discount=0.9):
    policy = Policy(env)
    simulation = Simulation(env)

    for episode in range(num_episodes):
        experiences = simulation.rollout(policy)
        policy.update_policy(experiences, weight, discount)

    return policy


trained_policy = train_policy(Environment())


def evaluate_policy(env, policy, num_episodes=10):
    simulation = Simulation(env)
    steps = 0

    for _ in range(num_episodes):
        experiences = simulation.rollout(policy, render=True, explore=False)
        steps += len(experiences)

    return steps / num_episodes


print(evaluate_policy(Environment(), trained_policy))
