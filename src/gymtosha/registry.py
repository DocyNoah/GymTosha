"""Public environment registry, separate from Gymnasium's scalar registry."""

from collections.abc import Callable

from gymtosha.base import BatchedEnv
from gymtosha.envs.cartpole import CartPole
from gymtosha.envs.frozen_lake import FrozenLake
from gymtosha.envs.mountain_car import MountainCar

_REGISTRY: dict[str, tuple[Callable[..., BatchedEnv], dict]] = {
    "CartPole-v1": (CartPole, {}),
    "MountainCar-v0": (MountainCar, {}),
    "MountainCarContinuous-v0": (MountainCar, {"continuous": True}),
    "FrozenLake-v1": (FrozenLake, {}),
    "FrozenLake8x8-v1": (FrozenLake, {"map_name": "8x8"}),
}


def register(env_id: str, constructor: Callable[..., BatchedEnv], **defaults) -> None:
    if env_id in _REGISTRY:
        raise ValueError(f"{env_id!r} is already registered")
    _REGISTRY[env_id] = (constructor, defaults)


def registered_envs() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def make(env_id: str, *, num_envs: int = 1, device="cpu", **kwargs) -> BatchedEnv:
    try:
        constructor, defaults = _REGISTRY[env_id]
    except KeyError:
        raise ValueError(
            f"unknown environment {env_id!r}. Choose from {registered_envs()}"
        ) from None
    return constructor(num_envs=num_envs, device=device, **(defaults | kwargs))
