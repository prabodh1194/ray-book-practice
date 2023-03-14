from __future__ import annotations

import logging

import ray

from maze_rl.environment import Environment
from maze_rl.policy import Policy
from maze_rl.simulation import Simulation
from maze_rl.trainer import evaluate_policy

logger = logging.getLogger(__name__)


ray.init()
environment = Environment()
env_ref = ray.put(environment)


@ray.remote
def create_policy():
    env = ray.get(env_ref)
    return Policy(env)


@ray.remote
class SimulationActor(Simulation):
    def __init__(self):
        env = ray.get(env_ref)
        super().__init__(env)


@ray.remote
def update_policy_task(policy: Policy, experiences_list):
    for experiences in experiences_list:
        policy.update_policy(ray.get(experiences))

    return policy


def train_policy_parallel(num_episodes=1000, num_simulations=4):
    policy = create_policy.remote()
    sims = [SimulationActor.remote() for _ in range(num_simulations)]

    for _ in range(num_episodes):
        experiences = [sim.rollout.remote(policy) for sim in sims]
        policy = update_policy_task.remote(policy, experiences)

    return ray.get(policy)


parallel_policy = train_policy_parallel()
evaluate_policy(environment, parallel_policy)
