# Third-party notices

## Gymnasium

The classic-control equations, FrozenLake rules and maps, and LunarLander
observation/action conventions and reward formulation are adapted from
Farama-Foundation/Gymnasium. Reference tests and measurements use Gymnasium 1.4.0.
LunarLander's contact solver is an independent simplified tensor implementation.

Copyright (c) 2016 OpenAI. Copyright (c) 2022 Farama Foundation.
The full MIT license is included in `licenses/Gymnasium.txt`.

Source: https://github.com/Farama-Foundation/Gymnasium

## CleanRL

`src/gymtosha/benchmarks/algorithms.py` and `train.py` adapt the network designs,
losses and update rules from CleanRL. The tensor replay buffer, batched collection,
backend adapters, logging and truncation bootstrapping are changed for GymTosha.
These are adapted learners, not executions of unmodified upstream scripts.

Pinned source revision: `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`.

- `cleanrl/dqn.py`
- `cleanrl/ppo.py`
- `cleanrl/ppo_continuous_action.py`
- `cleanrl/sac_continuous_action.py`
- `cleanrl/td3_continuous_action.py`

Copyright (c) 2019 CleanRL developers. Upstream includes notices for Scott
Fujimoto, Denis Yarats and the other sources of its continuous-control algorithms.
The complete upstream license and third-party notices are in `licenses/CleanRL.txt`.

Source: https://github.com/vwxyzjn/cleanrl/tree/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb

## Representative image

The README embeds a third-party Naoya Hair Flip GIF hosted by Tenor and links to
its source page. This image is not included in the Python distribution or covered
by the GymTosha source-code license.
