# GymTosha initial implementation

## Requirements

Python 3.12, uv, Ruff. Environments execute whole batches as PyTorch tensors on
CPU, CUDA or MPS without Python loops over environments. Support torch.compile.
Expose an env-ID factory and Gymnasium-style five-value step results.
Initial environments are CartPole-v1, MountainCar-v0, MountainCarContinuous-v0,
FrozenLake-v1 and LunarLander-v3. Train with CleanRL-derived DQN, PPO, SAC and TD3
and generate measured training-speed comparisons against Gymnasium.
Use a feature branch and incremental commits. Project name derives from
投射呪法 (Tōsha Juhō). Desired representative image is Naoya Hair Flip GIF.

## Implementation decisions

The user delegated implementation decisions. Use a src-layout package and
keep the existing checkout on a feature branch. Native batched tensor kernels
are the chosen architecture. Wrapping SyncVectorEnv/AsyncVectorEnv would retain
the per-environment Python/IPC overhead the user wants to remove.

Classic-control equations and FrozenLake transition probabilities will follow
Gymnasium and be tested against it. LunarLander requires a separate contact
solver. A simplified tensor solver must be explicitly labeled experimental,
with its physics difference carried into benchmark figures. Full Box2D parity
is an open user clarification and must not be claimed.

Reset is explicit, with optional same-step autoreset. Terminal observations,
termination versus truncation, episode returns and episode lengths are tensors.
Random state is local to each environment batch, seeded independently of the
policy RNG. Compilation covers the complete batched step including reset.

Benchmark both simulation throughput and actual learner throughput. Use the
same learner, network, transition budget, update cadence and seed for both
backends. Save machine metadata, configuration, raw measurements and figures.
Keep compile warm-up separate and report it. Return-versus-time curves accompany
speed figures so simulation speed cannot be confused with time to solve.
GPU results must be measured on an available GPU, never extrapolated.

## Deliverables

- Installable package, public registry and examples.
- Environment correctness, vectorization, reset, seed and compile tests.
- CleanRL-derived trainers and a repeatable comparison CLI.
- Actual local benchmark data and graphs, labeled with hardware and limitations.
- README with supported IDs, device/compile usage and physics compatibility.
- Naoya image integration if a suitable source asset can be obtained.
