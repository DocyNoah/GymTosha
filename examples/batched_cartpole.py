"""Run a fully batched environment on the best available local device."""

import torch

import gymtosha


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gymtosha.make("CartPole-v1", num_envs=4096, device=device).compile()
    observations, _ = env.reset(seed=42)
    for _ in range(1000):
        actions = torch.randint(2, (env.num_envs,), device=env.device)
        observations, rewards, terminated, truncated, info = env.step(actions)
    print(f"{observations.shape=}, {rewards.device=}")
    env.close()


if __name__ == "__main__":
    main()
