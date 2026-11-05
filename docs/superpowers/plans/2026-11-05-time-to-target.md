# Time-to-target and installation verification

> Execute inline with test-driven-development and verification-before-completion.

## Goal

Complete the user's acceptance criteria with independent held-out evaluation,
measured wall time to a fixed score, a readable README, a public API specification,
and clean installation using env IDs.

## Protocol decided before measurement

Use a common Gymnasium evaluation environment for both training backends,
including LunarLander. Evaluate the deterministic policy on 100 held-out episode
seeds. Require two consecutive evaluation checkpoints at or above the target.
Report confirmation time, transitions, evaluation scores and unsuccessful runs.
Measure wall time from environment creation, including compilation, model setup,
training and evaluation. Also retain training-only timing for diagnosis.

| Environment | Target mean return | Representative learner |
| --- | --- | --- |
| CartPole-v1 | 475 | PPO |
| MountainCar-v0 | -110 | DQN |
| MountainCarContinuous-v0 | 90 | PPO |
| LunarLander-v3 | 200 | PPO |
| FrozenLake-v1 | 0.70 | DQN |

The targets follow Gymnasium's published registry thresholds, including
FrozenLake 0.70 for the slippery 4x4 map. Three training seeds per backend are used in the final suite.
Both backends use identical presets per environment. The original eleven algorithm/task throughput comparisons remain available.
The user clarified that the new target study must use one representative learner
per environment, with hyperparameters selected by actual validation runs.

Never infer a speedup when one backend misses the target. Record such trials as
right-censored at the measured budget. LunarLander transfer failure is a valid
result and cannot be relabeled a matched-physics comparison.

## Review focus

Evaluation cannot alter learner RNG or training state. Terminal observations and
time-limit bootstrapping retain existing semantics. Confirmation time includes
its final evaluation. Compilation overhead is not subtracted from wall time.
Incomplete runs and heterogeneous configs cannot be aggregated as successes.

## Tasks

### 1. Evaluation and early stopping

- [ ] Write failing tests for policy evaluation, RNG preservation, consecutive
  target confirmation, early stop and unchanged throughput mode.
- [ ] Add `EvaluationConfig` and a held-out Gymnasium evaluator.
- [ ] Integrate optional evaluation into existing training loops and expose
  effective transition counts and target status.
- [ ] Verify and commit.

### 2. Repeatable target benchmark

- [ ] Add predefined environment/learner budgets and the `time-to-target` CLI.
- [ ] Persist checkpoint progress, final JSON/CSV and target-time figures.
- [ ] Verify comparison config equality and censored result handling.
- [ ] Run on the RTX 3090 server using push/pull and inspect all outcomes.

### 3. Public documentation and packaging

- [ ] Publish API and benchmark specifications.
- [ ] Redesign README hierarchy, installation, examples and evidence sections.
- [ ] Verify an external-project Git/wheel installation and every public env ID.
- [ ] Run Ruff lint/format, tests, build and independent code review.
- [ ] Commit, push and synchronize the server checkout.
