# Grouped CartPole PPO Implementation Plan

> **For agentic workers:** Implement task by task using superpowers:executing-plans and test-driven development.

**Goal:** Run the grouped CartPole benchmark on RTX 3090 with equal rollout and optimizer budgets within each group.

**Architecture:** Add a grouped controller beside the existing study. Reuse the worker, search scheduler, evaluator, capacity subprocess, and report renderer. Store grouped artifacts in a new output directory.

**Tech Stack:** Python 3.12, uv, PyTorch, Gymnasium, pytest, Ruff.

**Spec:** [PPO benchmark pipeline](../../PPO_BENCHMARK_PIPELINE.md)

- [ ] Verify that grouped presets preserve the reference PPO parameters and derive (T=B/N), (K=B/M), and a common (U B) for each group. Reject invalid batches. Test that a group selection must qualify at both endpoints.
- [ ] Implement resumable endpoint search, midpoint validation, isolated timing, and one-update capacity checks with the actual presets. Reuse the archived 32-environment selection and keep the legacy CLI reproducible.
- [ ] Record valid transitions and actual optimizer calls at PPO evaluation checkpoints. Use them in study exports and learning curves. Test compatibility with previous result files.
- [ ] Verify formatting and tests locally. Commit and push the work branch, pull on RTX 3090, and run grouped CartPole through capacity, tuning, and isolated comparisons. Review the recorded results and curves.
