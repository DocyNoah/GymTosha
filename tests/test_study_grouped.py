import pytest

from gymtosha.benchmarks.evaluation import EvaluationConfig
from gymtosha.benchmarks.study import reference_config
from gymtosha.benchmarks.train import TrainConfig, train


def test_group_presets_have_matching_budgets_and_reference_optimizer():
    from gymtosha.benchmarks.study_grouped import group_config

    reference = reference_config(32)
    for count, horizon in [(256, 256), (512, 128), (1024, 64)]:
        config = group_config(reference, count, 65_536, 2048)
        assert config.rollout_steps == horizon
        assert config.ppo_minibatch_size == 2048
        assert config.num_minibatches == 32
        assert config.total_timesteps == 4_194_304
        assert config.update_epochs == reference.update_epochs
        assert config.learning_rate == reference.learning_rate
        assert config.anneal_learning_rate == reference.anneal_learning_rate
    with pytest.raises(ValueError):
        group_config(reference, 1024, 65_535, 2048)


def test_group_candidates_must_qualify_at_both_endpoints():
    from gymtosha.benchmarks.study_grouped import qualified_candidates

    def row(name, reached, score):
        return {
            "value": name,
            "summary": {
                "mean_final_eval_return": score,
                "target_reached": reached,
            },
        }

    small = [row("a", 10, 500), row("b", 10, 491), row("c", 10, 480)]
    large = [row("a", 9, 500), row("b", 10, 489), row("c", 10, 482)]
    assert qualified_candidates([small, large], target=475) == ["b"]


def test_ppo_checkpoints_record_actual_update_and_transition_counts():
    config = TrainConfig(
        env_id="CartPole-v1",
        algorithm="ppo",
        num_envs=2,
        rollout_steps=8,
        total_timesteps=32,
        num_minibatches=2,
        update_epochs=2,
        hidden_sizes=[8],
    )
    result = train(
        config, evaluation=EvaluationConfig(interval=16, episodes=2, stop_on_target=False)
    )
    assert [
        (point["valid_transitions"], point["gradient_updates"])
        for point in result["evaluation"]["checkpoints"]
    ] == [(0, 0), (16, 4), (32, 8)]
