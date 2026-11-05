"""GymTosha — 投射呪法, batched PyTorch reinforcement learning environments."""

from gymtosha.base import BatchedEnv
from gymtosha.registry import make, register, registered_envs

__all__ = ["BatchedEnv", "make", "register", "registered_envs"]
__version__ = "0.1.0"
