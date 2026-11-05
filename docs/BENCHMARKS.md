# 성능 검증 명세

## 비교 종류

1. `throughput`은 정책 추론과 역전파를 제외한 전이 처리량을 측정합니다. Gymnasium 텐서 어댑터 비용을 포함합니다.
2. `train`은 동일 스텝 예산의 학습 처리량과 학습 에피소드 리턴을 기록합니다.
3. `time-to-target`은 동일 평가 성능을 확인할 때까지의 전체 경과 시간을 측정합니다.

## 목표 성능 도달 시간

| 환경 | 목표 평균 리턴 | 기본 학습기 |
| --- | --- | --- |
| CartPole-v1 | 475 | PPO |
| MountainCar-v0 | -110 | DQN |
| MountainCarContinuous-v0 | 90 | PPO |
| LunarLander-v3 | 200 | PPO |
| FrozenLake-v1 | 0.70 | DQN |

목표값은 Gymnasium registry의 reward threshold를 사용합니다. FrozenLake의 리턴은 성공 1,
실패 0이므로 평균 리턴 0.70은 성공률 70%입니다. 기본 slippery 4×4 맵을 사용합니다.

두 학습 백엔드 모두 **별도의 Gymnasium 환경**에서 결정론적 정책을 평가합니다.
학습과 독립적인 seed 100000부터 100개 에피소드를 실행하며, 같은 평가 seed를 모든 백엔드와
체크포인트에 사용합니다. 정책 가중치를 CPU로 복사하므로 평가 디바이스도 동일합니다.
평가는 학습 환경과 학습 난수 상태를 변경하지 않습니다.

100개 에피소드의 평균이 목표 이상인 평가가 연속 두 번 확인되어야 달성입니다.
목표 도달 시점은 두 번째 확인 평가가 끝난 시각이며 첫 평가 시각으로 소급하지 않습니다.
보고하는 wall time은 환경 생성 시작부터 모델 생성, 환경 컴파일·워밍업, 학습과 평가를
포함합니다. Python import 시간은 제외합니다. 디스크 컴파일 캐시는 재사용할 수 있으며
실제 환경 워밍업 시간은 원시 데이터에 기록합니다.

MountainCar DQN preset은 탐험 행동 후보를 16개 vector step 동안 유지합니다.
이는 성공 경험이 드문 과제에서 시간적으로 연관된 탐험을 만들기 위한 학습 설정입니다.
환경의 전이·보상·평가 정책은 바꾸지 않으며 두 백엔드에 똑같이 적용합니다.
일반 학습의 기본값은 1스텝이고 모든 설정은 원시 JSON에 기록합니다.

## 동일 조건과 실패 처리

같은 환경·학습기는 두 백엔드에서 네트워크, 학습률, 배치 크기, 탐험 일정, 최대 transition 예산,
평가 간격과 평가 seed가 같습니다. 설정이 다른 실행을 하나의 비교로 집계하면 오류를 반환합니다.
본 실험의 학습 seed는 11·22·33입니다. 사전 실행의 학습 seed 1과 평가 seed 900000은 본 실험에서 사용하지 않습니다. 사전 실행에서 설정을 바꾸면 본 실험 전체를 같은 설정으로 다시 실행합니다.

예산 안에 목표를 달성하지 못하면 `budget_exhausted`로 기록하고 도달 시간은 `null`입니다.
그래프의 `×`는 해당 예산까지 관찰했으나 목표를 확인하지 못한 실행입니다.
모든 대응 seed가 양쪽에서 목표를 달성한 비교에만 paired time ratio의 중앙값을 표시합니다.
실패한 실행을 빼고 가속 배율을 계산하지 않습니다.

LunarLander는 두 학습 환경의 물리가 다릅니다. 공통 Gymnasium 평가를 통과해야 같은 성능으로
인정하며, GymTosha 정책의 전이 실패도 실험 결과에 포함합니다.

## 실행

```bash
uv sync --frozen --extra benchmark
TRITON_DEFAULT_FP_FUSION=1 uv run --extra benchmark gymtosha-benchmark time-to-target \
  --device cuda --seeds 11 22 33 --output outputs/time-to-target
```

환경별 기본 preset과 최대 예산은 `src/gymtosha/benchmarks/target.py`에 고정합니다.
`--algorithms ppo dqn sac td3`는 호환되는 행동 공간 조합을 실행합니다.
`--target`, `--eval-interval`, `--eval-episodes`, `--confirmations`, `--eval-seed`로 평가 조건을
명시할 수 있습니다. `--resume`는 설정까지 일치하는 완료 실행을 건너뜁니다.

- `time_to_target.json`에는 전체 설정, 평가별 100개 리턴, 완료 상태와 시간이 있습니다.
- `time_to_target.csv`에는 실행별 요약이 있습니다.
- `target_summary.json`과 `.csv`에는 대응 seed의 달성률과 시간 비교가 있습니다.
- `*.progress.json`은 진행 중인 체크포인트 기록입니다.
- `time_to_target.png/.svg`는 목표 도달 또는 예산 소진 시각입니다.
- `evaluation_curves.png/.svg`는 공통 평가 리턴과 목표선을 표시합니다.

단위 테스트는 타임아웃 부트스트랩, replay 최종 관측, 평가 RNG 보존, 연속 확인,
목표 미달성 처리, 서로 다른 설정의 집계 거부와 실제 CLI 출력을 검사합니다.

## 하이퍼파라미터 선택

각 환경은 위 표의 대표 알고리즘 하나로 검증합니다. `benchmarks/tune_targets.py`에는
환경별 후보 4개가 고정되어 있습니다. 학습률, 병렬 환경 수, rollout 길이, epoch 수,
탐험 행동 유지 길이와 할인율을 비교합니다.

1. Gymnasium 백엔드, 학습 seed 101, 평가 seed 900000에서 네 후보를 실행합니다.
2. 목표 달성 여부와 도달 시간을 우선하여 상위 두 후보를 선택합니다.
3. 학습 seed 202에서 두 후보를 재검증합니다.
4. 두 시드에서의 달성률을 먼저 비교하고, 모두 달성했다면 도달 시간 중앙값이 가장 짧은 후보를 선택합니다.
5. 선택한 설정을 고정하고 학습 seed 11·22·33, 평가 seed 100000으로 양쪽 백엔드를 비교합니다.

탐색 기록에는 실패한 후보도 포함됩니다. 어떤 후보도 목표를 확인하지 못했다면
가장 높은 최종 점수의 설정으로 본 실험을 진행하되, 효율적인 목표 도달 설정을 찾았다고
판정하지 않습니다. 이는 명시한 후보 집합 안의 탐색이며 전역 최적성을 주장하지 않습니다.

```bash
TRITON_DEFAULT_FP_FUSION=1 uv run --extra benchmark python benchmarks/tune_targets.py \
  --device cuda --output outputs/tuning
TRITON_DEFAULT_FP_FUSION=1 uv run --extra benchmark gymtosha-benchmark time-to-target \
  --device cuda --presets outputs/tuning/selected.json --seeds 11 22 33 \
  --output outputs/time-to-target
```

연속 PPO의 선택 후보는 CleanRL 원본과 같이 관측의 running mean/variance 및 discounted-return
기반 보상 정규화, ±10 clipping을 적용합니다. 학습 중 관측 통계를 동결해 평가 정책에 복사하며,
평가 점수는 정규화 이전의 실제 환경 리턴입니다. 이전 처리량 모드는 정규화를 기본으로 켜지 않습니다.
