# 4,096-environment comparison

The user requires at least 4,096 simultaneous GymTosha environments.
Retain the existing branch, native tensor API and one representative learner per task.
Run GymTosha tuning with 4,096 environments and compare time to the same held-out
Gymnasium score against the tuned small-batch Gymnasium baseline.

Independent tuning changes environment counts, update schedules and budgets.
Record this explicitly as `independently-tuned` comparison mode. Preserve strict
matched-config validation for old results and the default CLI. Evaluation seed,
episode count, target and confirmation count must remain equal. Evaluation interval
may follow each backend's complete training batches and is saved in every record.
Require unchanged per-backend settings across final training seeds.

- [ ] Add failing regression tests for 4,096-environment candidate validity,
  independent per-backend presets, and strict evaluation/config validation.
- [ ] Extend the existing tuner and target CLI, preserving old result loading.
- [ ] Test, commit, push and pull on the RTX 3090 server.
- [ ] Run four candidate settings per task, confirm the best two on a second seed.
- [ ] Freeze the selected settings and run three final seeds per backend.
- [ ] Publish new raw data, 4,096-environment plots and explicit configuration tables.
- [ ] Run lint, format, full tests and build, commit and synchronize the server.

Incomplete targets remain visible. Approximate LunarLander transfer failure remains
a failure in the common evaluation. The finite search selects among measured
candidates and does not establish a global optimum.
