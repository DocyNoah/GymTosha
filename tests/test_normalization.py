import torch

from gymtosha.benchmarks.evaluation import EvaluationConfig
from gymtosha.benchmarks.normalization import RewardNormalizer, RunningMoments
from gymtosha.benchmarks.train import TrainConfig, train


def test_running_moments_and_frozen_normalization():
    moments = RunningMoments(2, "cpu")
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    moments.update(values)
    torch.testing.assert_close(moments.mean, values.mean(0), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        moments.variance, values.var(0, unbiased=False), rtol=1e-3, atol=1e-3
    )
    before = moments.count.clone()
    normalized = moments.normalize(values)
    torch.testing.assert_close(moments.count, before)
    assert normalized.abs().max() <= 10


def test_reward_normalization_resets_episode_return():
    normalizer = RewardNormalizer(2, "cpu", gamma=0.9)
    normalizer.normalize(torch.ones(2), torch.tensor([False, True]))
    torch.testing.assert_close(normalizer.discounted_returns, torch.tensor([1.0, 0.0]))
    reward = normalizer.normalize(torch.tensor([1000.0, 0.0]), torch.tensor([False, False]))
    assert torch.isfinite(reward).all() and reward.abs().max() <= 10


def test_normalized_ppo_runs_with_raw_reward_evaluation():
    result = train(
        TrainConfig(
            env_id="MountainCarContinuous-v0",
            num_envs=4,
            total_timesteps=128,
            rollout_steps=16,
            update_epochs=1,
            num_minibatches=2,
            normalize_observations=True,
            normalize_rewards=True,
        ),
        evaluation=EvaluationConfig(target=1000, interval=64, episodes=4),
    )
    assert result["parameter_delta"] > 0
    assert result["evaluation"]["status"] == "budget_exhausted"
    assert all(r["mean_return"] <= 100 for r in result["evaluation"]["checkpoints"])
