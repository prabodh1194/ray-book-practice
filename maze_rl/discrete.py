from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


class Discrete:
    def __init__(self, num_actions: int):
        self.n = num_actions

    def sample(self) -> int:
        return random.randint(0, self.n - 1)
