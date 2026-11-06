# CartPole PPO study implementation plan

The user supplied an eight-step experiment and requires staged execution.
Specification lives in `docs/CARTPOLE_STUDY.md`.

- [x] Preserve old learner defaults while supporting fixed 256×256×256 actor and critic networks.
- [x] Add the Gymnasium CartPole vector entry point to benchmark adapters with correct NEXT_STEP sample masking.
- [x] Implement isolated capacity probes and a doubling/halving search. Async is capped at 32 and system limits remain unchanged.
- [x] Save probe outcomes, failure bounds, memory and hardware metadata. Show the first-stage results before launching parameter sweeps.
- [x] Implement restartable single-parameter searches with 10 training seeds per candidate and frozen selection history.
- [x] Separate selection seeds from the 20 final comparison seeds and preserve unsuccessful trials.
- [x] Finish the four anchor searches and derive the other environment-count presets.
- [x] Add candidate plots, environment-count heatmaps and per-backend comparison plots.
- [ ] Execute all backend comparisons with 20 independent seeds per condition.
- [ ] Verify and publish the completed study tables and graphs.
