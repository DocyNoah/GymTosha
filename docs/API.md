# GymTosha API 명세

## 설치 계약

Python 3.12 이상과 PyTorch 2.15 이상을 사용합니다. `gymtosha`를 설치하면 PyTorch와
Gymnasium 메타데이터 의존성이 함께 설치됩니다. NVIDIA 드라이버와 PyTorch CUDA 런타임의
호환 여부는 실행 머신에서 확인합니다. 학습·그래프 도구는 `benchmark` extra로 설치합니다.

## 환경 생성

```python
gymtosha.make(env_id: str, *, num_envs: int = 1, device="cpu", **kwargs) -> BatchedEnv
```

`num_envs`는 양의 정수입니다. `device`는 PyTorch device 또는 `"cpu"`, `"cuda"`,
`"cuda:0"`, `"mps"` 등의 문자열입니다. 사용할 수 없는 디바이스는 PyTorch 오류를 반환합니다.
알 수 없는 ID는 지원 ID 목록을 포함한 `ValueError`를 반환합니다.

| ID | 단일 관측 | 단일 행동 | 기본 시간 제한 |
| --- | --- | --- | --- |
| `CartPole-v1` | Box `(4,)` | Discrete 2 | 500 |
| `MountainCar-v0` | Box `(2,)` | Discrete 3 | 200 |
| `MountainCarContinuous-v0` | Box `(2,)` | Box `(1,)`, -1~1 | 999 |
| `LunarLander-v3` | Box `(8,)` | Discrete 4 | 1000 |
| `LunarLanderContinuous-v3` | Box `(8,)` | Box `(2,)`, -1~1 | 1000 |
| `FrozenLake-v1` | Discrete 16 | Discrete 4 | 100 |
| `FrozenLake8x8-v1` | Discrete 64 | Discrete 4 | 200 |

모든 환경은 `max_episode_steps`와 `autoreset=True`를 받습니다.

- MountainCar 두 종류는 `goal_velocity=0.0`을 받습니다.
- LunarLander는 `gravity=-10.0`과 `continuous`를 받습니다. 중력 범위는 `-12 < gravity < 0`입니다.
- FrozenLake는 `is_slippery=True`, `map_name`, `desc`를 받습니다. `desc`는 `S`, `F`, `H`, `G`로 구성한 직사각형 문자열 행 목록입니다. 시작 타일과 목표 타일이 필요합니다.

## 공통 속성

- `num_envs`, `device`는 실제 배치 크기와 디바이스입니다.
- `single_observation_space`, `single_action_space`는 단일 환경의 Gymnasium space입니다.
- `observation_space`, `action_space`는 배치 차원을 포함한 Gymnasium space입니다.
- `metadata["autoreset_mode"]`는 `SAME_STEP` 또는 `DISABLED`입니다.

Space 객체는 메타데이터로 사용합니다. GPU 행동 샘플링은 `torch.randint` 또는
`torch.rand`로 수행하여 행동을 환경과 같은 디바이스에 둡니다.

## 초기화

```python
observation, info = env.reset(*, seed=None, options=None)
```

`seed`를 지정하면 해당 환경 배치의 난수 상태를 재설정합니다. 같은 설정·디바이스·seed·행동
순서에서 재현됩니다. 전역 PyTorch 난수 상태를 바꾸지 않으며 Gymnasium의 NumPy 난수열과는
다릅니다. `seed=None`은 현재 환경 난수열을 계속 사용합니다.

`options={"reset_mask": mask}`는 선택한 행만 초기화합니다. `mask`는 환경과 같은 디바이스의
`torch.bool` 텐서이며 shape은 `(num_envs,)`입니다. 선택되지 않은 행의 물리 상태, 진행 스텝,
누적 보상과 난수 상태는 유지됩니다. 반환 `info`는 빈 dict입니다.

## 전이

```python
observation, reward, terminated, truncated, info = env.step(actions)
```

| 출력 | shape | dtype |
| --- | --- | --- |
| Box 관측 | `(N, *single_observation_space.shape)` | float32 |
| Discrete 관측 | `(N,)` | int64 |
| reward | `(N,)` | float32 |
| terminated, truncated | `(N,)` | bool |
| info의 final_observation | 관측과 동일 | 관측과 동일 |
| info의 _final_observation | `(N,)` | bool |
| info의 episode_return | `(N,)` | float32 |
| info의 episode_length | `(N,)` | int64 |

모든 출력 텐서는 환경 디바이스에 있습니다. 이산 행동은 `(N,)`의 int32 또는 int64,
연속 행동은 `(N, *action_shape)`의 부동소수점 텐서입니다. 연속 행동은 float32로 계산합니다.
입력 shape·dtype·device가 맞지 않으면 `ValueError` 또는 `TypeError`를 반환합니다.
이산 행동의 값 범위는 호출자가 지켜야 합니다. 값 범위를 매 스텝 CPU에서 검사하지 않습니다.

`terminated`는 과제의 종료 조건, `truncated`는 시간 제한입니다. 기본 자동 초기화에서는
종료 행의 반환 관측이 새 에피소드의 시작 관측입니다. `info["final_observation"]`에서
초기화 전 관측을 얻고 `info["_final_observation"]` 마스크로 종료 행을 선택합니다.
`episode_return`, `episode_length`는 초기화 전 누적값이며 진행 중인 행에도 값이 있습니다.

`autoreset=False`이면 종료 이후 해당 행을 다음 `step` 전에 수동으로 초기화합니다.
전이는 `torch.no_grad()`에서 실행됩니다. 미분 가능한 시뮬레이터 API는 제공하지 않습니다.

## 컴파일

```python
env.compile(**torch_compile_kwargs) -> BatchedEnv
```

호출한 환경 자체를 반환합니다. 물리 전이·보상·종료·마스크 초기화를 함께 컴파일합니다.
기본값은 `fullgraph=True`이며 graph break는 오류가 됩니다. 환경마다 컴파일 캐시를
분리합니다. MPS는 Metal 버퍼 한도에 맞춰 기본 fusion 크기를 8로 설정합니다.
`backend`, `mode`, `options` 등 PyTorch 컴파일 인자를 전달할 수 있습니다.
첫 전이에는 컴파일 비용이 발생할 수 있습니다. 디스크 캐시 상태에 따라 비용이 달라집니다.

## 등록과 정리

```python
gymtosha.register(env_id, constructor, **defaults) -> None
gymtosha.registered_envs() -> tuple[str, ...]
env.close() -> None
```

등록 constructor는 `BatchedEnv`를 생성하는 callable입니다. 중복 ID 등록은 `ValueError`입니다.
등록된 기본 인자는 `make`의 명시적 kwargs로 덮어씁니다. 이 등록부는 `gymtosha.make`에서
사용합니다. 프로세스 전체에서 등록이 유지됩니다.

## 물리 호환성

CartPole과 MountainCar 두 종류는 Gymnasium의 전이 수식을 float32로 계산합니다.
FrozenLake는 같은 맵·전이 확률·종료 규칙을 사용합니다. 테스트는 경계 상태와 전이 확률도 검사합니다.

LunarLander는 고정된 두 발을 가진 근사 텐서 물리입니다. Box2D의 관절·스프링·접촉 해석과
다르며 동일 관측 형식이 동일 궤적을 의미하지 않습니다. 렌더링과 바람은 현재 지원 범위 밖입니다.
