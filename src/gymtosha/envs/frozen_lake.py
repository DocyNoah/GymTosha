"""Batched FrozenLake, including absorbing tiles and slippery transitions."""

import torch
from gymnasium import spaces

from gymtosha.base import BatchedEnv

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


class FrozenLake(BatchedEnv):
    def __init__(
        self,
        *,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        max_episode_steps=None,
        device="cpu",
        **kwargs,
    ):
        if desc is None:
            if map_name not in MAPS:
                raise ValueError(f"unknown map {map_name!r}")
            desc = MAPS[map_name]
        rows = [row.decode() if isinstance(row, bytes) else "".join(row) for row in desc]
        if not rows or not rows[0] or len({len(row) for row in rows}) != 1:
            raise ValueError("desc must be a nonempty rectangular map")
        text = "".join(rows)
        if set(text) - set("SFHG") or "S" not in text or "G" not in text:
            raise ValueError("desc requires S and G tiles and only S, F, H, G characters")
        self.rows, self.columns = len(rows), len(rows[0])
        self.is_slippery = is_slippery
        self.tiles = torch.tensor([ord(char) for char in text], device=device)
        self.starts = torch.tensor([i for i, char in enumerate(text) if char == "S"], device=device)
        self.single_observation_space = spaces.Discrete(len(text))
        self.single_action_space = spaces.Discrete(4)
        if max_episode_steps is None:
            max_episode_steps = 200 if self.rows == 8 else 100
        super().__init__(device=device, max_episode_steps=max_episode_steps, **kwargs)

    def _initial_state(self, mask):
        index = (self._uniform(1, mask).squeeze(-1) * len(self.starts)).long()
        return self.starts[index]

    def _transition(self, actions):
        if self.is_slippery:
            slip = (self._uniform(1).squeeze(-1) * 3).long() - 1
            actions = (actions + slip) % 4
        row, column = self.state // self.columns, self.state % self.columns
        row = (row + (actions == 1).long() - (actions == 3).long()).clamp(0, self.rows - 1)
        column = (column + (actions == 2).long() - (actions == 0).long()).clamp(0, self.columns - 1)
        absorbing = (self.tiles[self.state] == ord("H")) | (self.tiles[self.state] == ord("G"))
        state = torch.where(absorbing, self.state, row * self.columns + column)
        tile = self.tiles[state]
        terminated = (tile == ord("H")) | (tile == ord("G"))
        reward = ((tile == ord("G")) & ~absorbing).float()
        return state, reward, terminated
