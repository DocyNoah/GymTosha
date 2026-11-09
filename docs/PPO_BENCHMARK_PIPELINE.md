# PPO 환경 벤치마크 파이프라인

GymTosha와 Gymnasium에 같은 양의 학습 경험을 주고, 평가 목표에 처음 도달하는 데 필요한 트랜지션과 학습 시간을 비교합니다. 현재 CartPole 최종 비교는 rollout 길이 $T=64$를 사용합니다.

## 1. 과제와 평가 기준

- 환경 ID·버전, 관측·행동, 보상과 종료 조건을 기록합니다. 환경 구현이 다르면 차이를 명시합니다.
- PPO 학습기는 공통으로 사용하고 정책망과 가치망은 각각 `256–256–256`으로 구성합니다. CartPole은 GAE λ `0.99`와 기존 학습률·스케줄을 사용합니다.
- PPO 업데이트 직후에 별도 Gymnasium 평가 환경에서 100개 에피소드를 실행하고 개별 리턴을 TensorBoard에 기록합니다. 실행이 끝나면 후처리가 평가 평균이 **처음 475점 이상**인 체크포인트의 유효 트랜지션 수와 학습 월타임을 계산합니다. 평가와 이벤트 기록에 걸린 시간은 학습 월타임에서 제외합니다.
- CartPole 평가는 모든 백엔드에서 같은 CPU NumPy 벡터 환경을 사용합니다. 각 평가 행은 Gymnasium SyncVectorEnv와 동일한 `seed + 행 번호`로 초기화하고 첫 에피소드만 집계합니다.

## 2. 공통 학습 예산

환경 수 $N$, 환경당 rollout 길이 $T$, rollout 횟수 $U$, 미니배치 크기 $M$, PPO epoch 수 $E$로 표기합니다.

| 단계 | 고정 | 변경 |
| --- | --- | --- |
| rollout 길이 탐색 | $N,S_{\rm tune},M,E$, 학습률과 평가 시점 | $T$, $U=S_{\rm tune}/(NT)$ |
| 환경 수와 백엔드 비교 | $T,S_{\rm final},M,E$, 학습률, 매 업데이트 후 평가 | $N$, $U=S_{\rm final}/(NT)$, 평가 간격 $NT$ |

각 단계에서 총 수집 예산은 $S=UNT$로 고정합니다. $NT$는 $M$으로 나누어떨어져야 하며, 전체 optimizer 호출 수는 $ES/M$입니다. 최종 비교는 학습 종료 예산과 학습률 감쇠 기준을 분리합니다. 초기 학습률은 0.003이며, optimizer 호출 순서에 따라 기존 8,388,608 트랜지션에 해당하는 65,536회 기준으로 선형 감쇠합니다.

기존 탐색은 같은 누적 수집량에서 평가했습니다. 최종 비교는 업데이트마다 평가하므로 평가 시점의 트랜지션 간격이 $NT$입니다. 목표에 도달한 뒤에도 공통 예산까지 학습합니다.

Gymnasium의 CartPole 전용 NumPy 백엔드에서는 종료한 행을 같은 스텝에 리셋합니다. 원래 벡터 물리 계산과 리셋 분포를 사용하면서 종료 관측을 보존하므로, 다음 수집 슬롯도 학습에 쓸 수 있습니다. 모든 비교에서 수집 슬롯, 유효 트랜지션, 실제 optimizer 호출 수를 따로 기록합니다.

## 3. rollout 길이 탐색

CartPole의 기존 탐색은 RTX 3090에서 $N=1{,}024$를 고정하고 $T=16,32,64,128,256,512$를 같은 10개 시드와 공통 경험 예산으로 비교했습니다. 최종 비교의 rollout 길이는 64로 고정합니다.

| 항목 | CartPole 설정 |
| --- | ---: |
| 총 트랜지션 $S_{\rm tune}$ | 4,194,304 |
| 미니배치 크기 $M$ | 512 |
| PPO epoch 수 $E$ | 4 |
| 탐색의 공통 평가 간격 | 524,288 트랜지션 |
| 탐색의 예정 optimizer 호출 수 | 32,768 |

$S_{\rm tune}$은 탐색에서 가장 큰 배치인 $1{,}024\times512$의 rollout 8회분입니다. $M=512$는 가장 작은 최종 후보인 $32\times16$ 수집 배치까지 나눌 수 있습니다. 탐색은 장치 메모리와 호스트 메모리 한도 안에서 여러 시드를 병렬 실행합니다. 탐색 시간은 실행 기록에 남기고, 속도 비교는 최종 단독 측정에서 수행합니다.

기존 탐색은 마지막 5개 평가의 시드별 평균과 시드 간 표준편차를 사용했습니다. 당시 도달 조건은 평균 475점 이상, 100개 중 95개 이상이 475점 이상인 평가를 5회 연속 통과하는 것이었습니다. 이 조건은 기존 탐색 결과 해석에 적용합니다.

## 4. 백엔드 용량과 최종 비교

- CPU·CUDA·MPS 후보는 **32, 256, 512, 1,024개**이고, Gymnasium Async는 **32개**만 비교합니다. 기존 1,024개 초과 결과와 이전 실행 매트릭스는 보존합니다.
- 시드당 예산은 **2,097,152 유효 트랜지션**입니다. $T=64$, $M=512$, $E=4$를 사용하므로 시드마다 optimizer를 16,384회 호출합니다. 가장 큰 환경 수인 1,024개에서는 PPO rollout과 평가를 각각 32회 수행합니다.
- 동일한 호스트·런타임·학습 설정에서 완료된 T=64 용량 검사를 재사용합니다. 추가 검사는 환경 초기화부터 PPO 업데이트 한 번까지 60초 이내에 완료해야 통과입니다.
- 통과한 조건마다 첫 비교에 **시드 1001–1005**를 사용합니다. 후속 실행은 `--seed-start`와 새 출력 경로를 사용합니다. 서로 다른 시드 묶음은 별도 보고서로 생성합니다. Gymnasium과 GymTosha CPU·CUDA는 RTX 3090 호스트에서, GymTosha MPS는 Mac에서 측정합니다.
- RTX 3090의 실행 순서는 GymTosha CUDA compile, CUDA eager, GymTosha CPU compile과 eager, Gymnasium NumPy, Sync, Async입니다. Mac은 MPS compile, MPS eager 순입니다. 같은 구성에서는 작은 환경 수부터 실행하며, 완료된 시드는 재사용합니다.
- 목표 도달 월타임은 환경 생성부터 컴파일·워밍업과 최초로 475점을 달성한 정책의 PPO 업데이트 종료까지입니다. 도달률과 함께 도달한 시드의 트랜지션·월타임 평균 및 표본 표준편차를 표시합니다.

## 5. 결과 기록과 실행

TensorBoard 로그는 조건과 시드별로 분리합니다. 업데이트마다 평가 리턴 100개, 유효 트랜지션, 누적 학습 시간, 초기화·워밍업과 누적 수집·업데이트 시간, 업데이트 번호, optimizer 호출과 완료 에피소드 수를 기록합니다. PPO 진단은 학습률, 정책 엔트로피, policy loss, value loss, approx KL, clip fraction을 남깁니다. 실행 메타데이터에는 설정, 호스트, 장치, 코드 버전, 상태가 들어갑니다.

기존 T=64 실행의 학습 설정·평가 방식과 학습률 스케줄을 확인하고, 호환되면 2,097,152 트랜지션까지의 이벤트를 새 출력에 복사합니다. 기존 JSON은 저장된 값을 이벤트로 변환하고, 당시 측정하지 않은 항목은 메타데이터에 표시합니다. 재사용한 시드는 대시보드와 시드 CSV에 출처를 남깁니다. 기존 8,388,608 트랜지션의 원시 이벤트와 T=256 결과는 기존 경로에 보존합니다.

```bash
uv sync --frozen --extra benchmark
uv run --no-sync python -m gymtosha.benchmarks.cartpole_final \
  --baseline benchmarks/results/cartpole-study/scaling/selected.json \
  --output outputs/cartpole-t64-2m-3090 --prefix-from outputs/cartpole-t64-tb-3090 \
  --legacy outputs/cartpole-t64-3090 --devices cpu cuda

uv run --no-sync python -m gymtosha.benchmarks.cartpole_final \
  --baseline benchmarks/results/cartpole-study/scaling/selected.json \
  --output outputs/cartpole-t64-2m-mps --prefix-from outputs/cartpole-t64-tb-mps \
  --legacy outputs/cartpole-t64-mps --devices mps

uv run --no-sync tensorboard --logdir outputs/cartpole-t64-2m-mps/events
```

두 호스트의 `matrix.json`과 `events/`를 한곳에 모아 후처리합니다. 스크립트는 이벤트에서 첫 도달, 조건별 통계, 트랜지션·학습 시간 기준 곡선, 환경 수별 평가 간격, 같은 환경 수의 비교와 백엔드별 최적 환경 수를 재생성합니다.

```bash
uv run --no-sync python -m gymtosha.benchmarks.cartpole_report \
  --inputs outputs/cartpole-t64-2m-3090 outputs/cartpole-t64-2m-mps \
  --output outputs/cartpole-t64-tb-report
```

생성된 `index.html`을 브라우저에서 열면 호스트·장치·환경 수 필터, 같은 환경 수와 백엔드별 최적 환경 수 비교, 시드 상세를 사용할 수 있습니다. 학습 곡선은 유효 트랜지션과 학습 시간 축을 전환할 수 있고, 실행 매트릭스는 완료·도달률·도달 시간·도달 트랜지션을 표시합니다. `static.html`과 CSV도 함께 생성합니다.

특정 조건의 시드를 교체할 때는 호스트 출력 경로의 `seed-replacements.json`에 조건별 매핑을 기록합니다. `{"gymtosha-cuda-1-256": {"1005": 1006}}`은 해당 조건에서 1005를 제외하고 `replacements/gymtosha-cuda-1-256/seed-1006/`의 원시 이벤트를 집계합니다. 다른 조건은 매트릭스의 시드를 그대로 사용합니다. 매핑 파일과 원시 이벤트 디렉터리를 수동으로 복사한 뒤 보고서를 다시 생성하면 시드 상세와 CSV에 교체 관계가 표시됩니다.

## 6. 실행 중 대시보드 갱신

Mac에서 감시기를 실행하면 로컬 또는 RTX에서 시드가 종료될 때마다 TensorBoard 기록을 읽어 대시보드를 갱신합니다. RTX 호스트와 Mac 모두 감시 모듈이 포함된 브랜치를 사용합니다. 진행률은 종료 시드 수로, 평균·표준편차와 학습 곡선의 표본 수는 검증된 완료 시드 수로 표시합니다. 조건 통계는 다섯 시드를 모두 검증한 뒤 확정합니다.

```bash
uv run --no-sync python -m gymtosha.benchmarks.cartpole_watch \
  --inputs outputs/cartpole-t64-2m-mps \
  --ssh-host link-3090v2-x --remote-root outputs/cartpole-t64-2m-3090 \
  --mirror outputs/cartpole-t64-2m-3090 --output outputs/cartpole-t64-tb-report

uv run --no-sync python -m http.server 8782 --bind 127.0.0.1 \
  --directory outputs/cartpole-t64-tb-report
```

브라우저에서 `http://127.0.0.1:8782/`를 열면 현재 필터를 유지한 채 새 스냅샷을 표시합니다. 모든 시드가 완료되면 원시 이벤트를 다시 읽어 최종 CSV와 학습 곡선을 생성합니다. 수동 중간 집계에는 보고서 명령의 `--partial` 옵션을 사용합니다. 이전 그룹별 비교는 [기존 CartPole 실험 명세](CARTPOLE_STUDY.md)에 있습니다.

## 7. Mac 후속 시드 1006–1010

```bash
uv run --no-sync python -m gymtosha.benchmarks.cartpole_final \
  --baseline benchmarks/results/cartpole-study/scaling/selected.json \
  --output outputs/cartpole-t64-2m-mps-6-10 --prefix-from outputs/cartpole-t64-tb-mps \
  --devices mps --seed-start 1006

uv run --no-sync python -m gymtosha.benchmarks.cartpole_watch \
  --inputs outputs/cartpole-t64-2m-mps-6-10 --output outputs/cartpole-t64-mps-6-10-report
```

메인 대시보드의 매트릭스와 통계에 Mac 후속 결과를 합치려면 보고서 생성기 또는 감시기에 `--follow-up outputs/cartpole-t64-2m-mps-6-10`을 추가합니다. Mac은 조건당 10개, RTX는 5개 시드를 표시합니다. 두 보고서 폴더를 연결하면 Mac 후속 5개 시드만 따로 확인할 수도 있습니다.

```bash
ln -s ../cartpole-t64-mps-6-10-report outputs/cartpole-t64-tb-report/mps-6-10
```
