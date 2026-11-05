# CartPole PPO study implementation plan

The user supplied an eight-step experiment and requires staged execution.
Specification lives in `docs/CARTPOLE_STUDY.md`.

- [ ] Preserve old learner defaults while supporting fixed 256×256×256 actor and critic networks.
- [ ] Add the Gymnasium CartPole vector entry point to benchmark adapters with correct NEXT_STEP sample masking.
- [ ] Implement isolated capacity probes and a doubling/halving search. Async is capped at 32 and system limits remain unchanged.
- [ ] Save probe outcomes, failure bounds, memory and hardware metadata. Show the first-stage results before launching parameter sweeps.
- [ ] Implement restartable single-parameter searches with exactly 20 training seeds per candidate and frozen selection history.
- [ ] Separate selection seeds from final comparison seeds and preserve unsuccessful trials.
- [ ] Carry selected parameters to doubled environment counts and record their scaling trends.
- [ ] Add candidate plots, environment-count heatmaps and per-backend final comparisons.
- [ ] Verify tests, lint, formatting and reproducible execution, and commit each completed stage.
