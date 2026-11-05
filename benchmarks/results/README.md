# 초기 처리량 측정

**목표 성능 도달 시간의 후속 검증은 [별도 보고서](time-to-target/README.md)에 있습니다.**

2026-11-05에 다음 두 시스템에서 측정했습니다. 모든 학습은 CPU 스레드 하나를 사용했습니다.

- CPU·MPS는 Apple M3 Max, macOS arm64, Python 3.12.10, PyTorch 2.15.0입니다.
- CUDA는 AMD Ryzen 5 5600X + NVIDIA RTX 3090 24GB, Linux x86_64, Python 3.12.12, PyTorch 2.15.0+cu132입니다.
- NVIDIA 드라이버는 580.178.04, CUDA 런타임은 13.2입니다. Triton FP fusion은 1로 고정했습니다.
- Gymnasium 1.4.0과 동일한 GymTosha 학습기를 사용했습니다. 하드웨어 상세와 소스 커밋은 `cuda/metadata.json`에 있습니다.

## 학습 조건

- 환경 64개, 실행당 65,536 transitions, seed 1·2·3을 사용했습니다.
- 다섯 환경의 PPO, 이산 환경 네 개의 DQN, MountainCarContinuous의 SAC·TD3를 실행했습니다.
- 세 디바이스 × 두 백엔드 × 11개 조합 × 세 seed로 198회, 총 12,976,128 transitions를 측정했습니다.
- GymTosha에는 torch.compile을 적용했습니다. Gymnasium에는 SyncVectorEnv와 동일한 텐서 학습 어댑터를 사용했습니다.
- 같은 환경·알고리즘의 두 백엔드는 네트워크, 학습률, 샘플 예산, 업데이트 횟수와 배치 크기가 같습니다.
- PPO는 rollout 128스텝, epoch 4회, minibatch 4개입니다. Off-policy는 replay batch 256, 학습 시작 1,024 transitions, 256 transitions당 critic update 한 번입니다.
- 시간에는 정책 추론, 환경 전이, 텐서 변환, 학습 업데이트와 구간 로그 집계를 포함했습니다. 환경 생성과 환경 커널 워밍업은 별도 기록했습니다.
- GPU 구간은 시작 전·종료 후 동기화했습니다. 각 실행의 첫 역전파·optimizer 초기화 비용은 학습 시간에 포함됩니다.

## 학습 처리량

표의 처리량은 seed 세 개의 중앙값이며 단위는 초당 천 transitions입니다.
배율은 같은 디바이스의 Gymnasium SyncVectorEnv 대비 값입니다.

| 환경 | 알고리즘 | CPU Gymnasium | CPU GymTosha | CPU 배율 | MPS 배율 | CUDA 배율 |
| --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | PPO | 77.0 | 111.6 | 1.45× | 3.72× | 1.49× |
| CartPole-v1 | DQN | 141.7 | 381.3 | 2.69× | 6.46× | 1.94× |
| MountainCar-v0 | PPO | 69.6 | 113.5 | 1.63× | 3.88× | 1.66× |
| MountainCar-v0 | DQN | 115.8 | 365.8 | 3.16× | 6.73× | 2.19× |
| MountainCarContinuous-v0 | PPO | 100.7 | 122.9 | 1.22× | 2.18× | 1.26× |
| MountainCarContinuous-v0 | SAC | 83.5 | 96.5 | 1.16× | 3.40× | 1.15× |
| MountainCarContinuous-v0 | TD3 | 111.0 | 151.8 | 1.37× | 4.01× | 1.30× |
| LunarLander-v3 * | PPO | 34.1 | 112.8 | 3.31× | 4.84× | 2.52× |
| LunarLander-v3 * | DQN | 43.2 | 343.5 | 7.95× | 8.18× | 3.54× |
| FrozenLake-v1 | PPO | 75.4 | 115.9 | 1.54× | 3.78× | 1.50× |
| FrozenLake-v1 | DQN | 141.2 | 374.4 | 2.65× | 6.15× | 1.80× |

이 배치 크기와 작은 MLP 네트워크에서는 CPU의 GymTosha 절대 처리량이 MPS보다 높았습니다.
GPU 사용 여부와 성능 향상은 배치 크기·네트워크·전이 계산량에 따라 달라집니다.

\* LunarLander는 근사 텐서 물리와 Box2D를 비교한 수치입니다. 접촉·관절 계산량과
에피소드 길이가 다르므로 동일 물리의 가속 배율로 해석할 수 없습니다.

학습 곡선은 기록 구간에서 끝난 에피소드들의 평균 리턴입니다. 이 짧은 고정 예산
실험은 처리량 비교이며 모든 환경을 해결하거나 동일한 최종 성능에 도달했음을 입증하지 않습니다.
일부 구간에는 완료된 에피소드가 없어 리턴 값이 비어 있습니다.

## 환경 처리량

정책 추론과 역전파를 제외하고 배치 크기 1·64·1,024에서 200 vector steps를 세 번 반복했습니다.
총 315회 측정했습니다. CPU는 Gymnasium, GymTosha eager, GymTosha compiled를 비교했습니다.
MPS와 CUDA는 Gymnasium의 CPU 환경+GPU 텐서 전송과 GymTosha compiled를 비교했습니다.
Gymnasium의 텐서 어댑터와 자동 초기화 비용도 측정 구간에 포함되어 있습니다.

## 원시 데이터와 그림

- `cpu/training.json`, `mps/training.json`, `cuda/training.json`에 설정, 구간별 리턴·시간, 업데이트 수, 파라미터 변화량과 환경 정보가 있습니다.
- `cpu/throughput.json`, `mps/throughput.json`, `cuda/throughput.json`에 개별 반복 결과와 워밍업 시간이 있습니다.
- `combined/`는 세 디바이스의 결과를 합친 JSON·CSV·PNG·SVG입니다.
- `training_speed.png`는 학습 처리량, `learning_curves.png`는 리턴-시간 곡선, `environment_throughput.png`는 배치 크기별 환경 처리량입니다.

## CUDA 검증

RTX 3090 서버의 `~/source/GymTosha`에서 전체 테스트 59개가 통과했습니다.
MPS 전용 테스트 10개는 해당 장치가 없어 건너뛰었습니다. 여섯 환경 ID의 실제 CUDA
Inductor 컴파일, 자동 초기화, 최종 관측과 episode 통계의 eager/compiled 일치성을 검사했습니다.
전체 테스트 출력은 `cuda/pytest.txt`에 있습니다.

## 재현 명령

```bash
uv sync --extra benchmark
uv run --extra benchmark gymtosha-benchmark train --compile --output outputs/cpu
uv run --extra benchmark gymtosha-benchmark train --device mps --compile --output outputs/mps
uv run --extra benchmark gymtosha-benchmark throughput --compile --output outputs/cpu
uv run --extra benchmark gymtosha-benchmark throughput --backends gymtosha --output outputs/cpu --append
uv run --extra benchmark gymtosha-benchmark throughput --device mps --compile --output outputs/mps
TRITON_DEFAULT_FP_FUSION=1 uv run --extra benchmark gymtosha-benchmark train --device cuda --compile --output outputs/cuda
TRITON_DEFAULT_FP_FUSION=1 uv run --extra benchmark gymtosha-benchmark throughput --device cuda --compile --output outputs/cuda
uv run --extra benchmark gymtosha-benchmark plot --inputs outputs/cpu outputs/mps outputs/cuda --output outputs/combined
```

다른 하드웨어에서 비교할 때는 전체 조합을 다시 실행합니다. 각 배율은 같은 디바이스와
호스트의 Gymnasium을 기준으로 계산합니다. Apple 시스템과 RTX 3090 서버의 절대 처리량은
CPU·GPU·운영체제 차이도 포함합니다.
