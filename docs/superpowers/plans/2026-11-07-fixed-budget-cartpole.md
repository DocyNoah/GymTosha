# Fixed-budget CartPole PPO implementation plan

> **For agentic workers:** Use executing-plans and test-driven development.

**Goal:** Re-select the CartPole rollout horizon at 1,024 environments and compare all supported backends and environment counts at the same experience budget.

**Spec:** [PPO benchmark pipeline](../../PPO_BENCHMARK_PIPELINE.md)

- [ ] Assert that horizon candidates and final presets share one budget and aligned evaluation slots while using whole PPO rollouts.
- [ ] Assert that the exact-budget native CartPole adapter collects a real transition in every slot, including immediately after episode termination.
- [ ] Add the fixed-budget study controller, retaining the grouped controller for historical results. Reuse the existing worker, capacity checks, and reporting.
- [ ] Verify tests and formatting. Stop the old controllers while retaining their files. Start the fixed-budget horizon search on RTX 3090, then run all supported comparisons on RTX 3090 and Apple MPS with the selected preset.
- [ ] Verify every final seed and publish learning curves, stable time-to-target results, and the final configuration table.
