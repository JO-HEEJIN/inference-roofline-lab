# CONTEXT

이 문서는 저장소의 현재 상태만 기술한다. 계획과 지시서는 이슈 트래커에,
결정의 역사는 [`ADR.md`](ADR.md)에 있다.

## 현재 상태 요약

**NPU에서 측정한 결과는 없다.** latency, throughput, correctness 어느 것도
실제 Ascend 장치에서 실행된 적이 없다. `runs/` 디렉터리는 존재하지 않는다.

실행 가능한 Ascend 장치를 확보하지 못해 G1(실행 환경 확보)에서 대기 중이다.
이것은 외부 blocker이며 harness의 결함이 아니다. 로컬 검증(G0)은 통과 상태다.

| 항목 | 상태 |
|---|---|
| Stage 0 correctness harness | 구현 완료, Ascend 실행 대기 |
| Stage 1 baseline timing harness | 구현 완료, Ascend 실행 대기 |
| Stage 2 core-transition (opt-in) | 구현 완료, 실행 전제 미충족 |
| 로컬 정적 검증 | 통과 (테스트 15개) |
| upstream working-tree checkout | 미완성. `.git`만 있고 소스 파일 없음 |
| 실측 결과 | 없음 |

## 대상과 고정 지점

측정 대상은 SGLang Ascend NPU 커널 `torch.ops.npu.cache_loc_assign`이며
upstream은 [`sgl-project/sgl-kernel-npu`](https://github.com/sgl-project/sgl-kernel-npu)
commit `cdcb9d8b719a6100dadc3544c8cd33ff53bf668a`에 고정되어 있다.
정확한 소스 경로, blob ID, 빌드 target 예시는
[`experiments/kv-cache-assign/upstream.json`](experiments/kv-cache-assign/upstream.json)에 있다.

upstream checkout은 의도적으로 저장소에서 제외한다(`.gitignore`의 `/upstream/`).
현재 `upstream/sgl-kernel-npu/`에는 Git 메타데이터만 있고 working tree 파일이
없다. 지금까지의 소스 관찰은 GitHub 읽기 API로 수행했으며 로컬 checkout이나
NPU 빌드가 완료된 상태가 아니다.

커널과 host tiling은 변경하지 않는다. 이 저장소는 reference, 입력 사례,
측정 도구, 결과만 관리한다.

## harness가 하는 일

`experiments/kv-cache-assign/benchmark_cache_location_assign.py`가 단일
진입점이다. 사용법과 출력 계약은
[`experiments/kv-cache-assign/README.md`](experiments/kv-cache-assign/README.md)에 있다.

- **`--stage stage01`** (기본): Stage 0은 batch 8/32/128 × update 1/2/8/16 ×
  request index int32/int64의 fresh fixture 24개를 검사한다. Stage 1은 그중
  correctness가 통과한 int64 case 12개만 logical capacity 2048에서 측정하며,
  case마다 isolated와 sustained 두 가지 timing record를 남긴다.
- **`--stage stage2`** (opt-in): 실제 assign blockDim/vector-core 수 `C`를
  기준으로 `B = C/2, C, 2C`만 생성한다. `C±1`과 capacity sweep은 범위 밖이다.
  생성된 batch가 모두 8의 배수이고 128 이하일 때만 허용하며, 아니면
  `ValueError`로 거부한다.
- **`--dry-run`**: NPU runtime을 import하지 않고 case 정의와 manifest만
  생성한다. manifest status는 `dry-run-static-validation`이다.

### fixture 보호

token pool의 physical row width는 2064이다(2048 logical + 16 guard). start는
int32 8개 단위로 정렬되며, upstream의 고정 16-element 전송이 guard에 닿을 수
없는 내부 위치에 둔다. packed cache view는 `batch × 16` + guard이고, useful
`L = batch × update_length` 뒤쪽은 sentinel로 채운 뒤 매 호출마다 검증한다.
metadata tensor는 32-byte 정렬 복사에 필요한 backing을 유지하면서 logical
batch-length view만 operator에 전달한다.

useful cache 값은 모두 음수로 생성한다. logical pool 값은 양수이므로 저장이
하나라도 누락되면 checker가 반드시 관측한다. 검사 항목은 변경 cell의 값과
변경 mask 자체, untouched cell 보존, token guard, cache padding/guard,
metadata 무손상이다.

### 실패 처리

correctness 또는 timing 실패가 하나라도 있으면 exit 1과 manifest status
`completed-with-failures`를 기록한다. 예외로 중단되면 `aborted`를 남기고
예외를 전파한다. correctness가 실패한 case는 timing을 실행하지 않는다.
정적 환경 수집은 설치된 distribution metadata만 읽으며 `torch`나 `torch_npu`를
import하지 않는다.

| exit code | 의미 |
|---|---|
| 0 | 요청한 run이 기록된 실패 없이 완료, 또는 dry run 완료 |
| 1 | correctness 또는 timing 실패 |
| 2 | 잘못된 설정 또는 NPU runtime 사용 불가 |

### 출력

`results.jsonl`과 `results.csv`는 같은 schema를 쓴다. raw timing 배열과
sustained block 측정값은 `raw-samples/`에 보존하고, record가 상대 경로로
참조한다. `run-manifest.json`에는 source commit과 dirty 상태, 실제 checkout
HEAD와 pin 일치 여부, 미검증 binary/source 연결 상태, harness hash, 환경,
case 정의, 저장 정책, 측정 parameter가 들어간다.

raw 측정값과 profiler trace는 로컬에만 둔다(`.gitignore`의 `/runs/`).

## 측정 용어

구분을 흐리면 결론이 틀리므로 다음 용어를 섞어 쓰지 않는다.

- **useful bytes**: 선택된 int32 값의 읽기+쓰기 `8 × sum(lengths)`.
  `8L`로 표기한다.
- **source-copy bytes**: 소스에서 정적으로 계산한 GM↔UB copy payload 모델.
  실측 HBM traffic이 아니다.
- **main-memory bytes**: profiler가 보고하는 값. 위 두 항목과 별개 필드다.
- **isolated API latency**: operator 호출부터 completion synchronize까지의
  wall time. reset과 비교는 timing 구간 밖에서 수행한다. kernel time이 아니다.
- **sustained throughput**: warmup 뒤 200 call을 한 block으로 측정한다.
  `median_us`/`p95_us`는 block 평균 call 시간의 통계이며 per-request tail
  latency가 아니다. `sustained_calls_per_second`는 block rate의 median이다.
- **useful effective bandwidth**: `8L / latency` 또는 `8L × calls/sec`.
  모든 bandwidth 값에는 timing 범위를 붙인다.

guard 보존은 out-of-bounds read가 없다는 증거가 아니다. wall-clock timing
단독으로는 kernel bottleneck을 입증하지 못하며, API latency를 kernel task나
특정 pipeline에 귀속시키려면 profiler trace가 필요하다.

## 로컬 검증

NPU 없는 환경에서 harness 자체를 검증한다. NPU import, 설치, 네트워크 요청,
유료 자원 생성이 없다.

```bash
bash scripts/validate-local.sh
```

테스트 15개 통과, compileall 및 diff check 성공, dry-run case 24개,
timing record 0개가 통과 조건이다. 로그와 dry-run manifest는 스크립트가
출력하는 임시 디렉터리에 남는다.

CPU 테스트(`test_fixture_cpu.py`)는 독립적인 scalar reference로 fixture와
checker를 검증하고 손상을 주입해 검출되는지 본다. **checker 검증이며 NPU
kernel 실행 증거가 아니다.** 정적 검증 통과를 하드웨어 검증 성공으로 읽지
않는다.

## 실행 환경 요건

Ascend 장치와 CANN, `torch_npu`, 고정 revision에 맞게 빌드·설치한
`sgl_kernel_npu`가 필요하다. 현재 개발 호스트는 Darwin arm64이며 `npu-smi`,
`torch_npu`, `sgl_kernel_npu`가 없다.

실측 전 인수 조건은 [`docs/ascend-runbook.md`](docs/ascend-runbook.md),
확인된 접근 경로 비교는
[`docs/ascend-access-options.md`](docs/ascend-access-options.md),
공급자 인수표는
[`docs/templates/ascend-offer.md`](docs/templates/ascend-offer.md)에 있다.
공급자 항목은 전부 UNVERIFIED 상태이고, 결제·서버 생성·지원 티켓 발송을 한
적이 없다. 계정 식별자, 전화번호, 카드 정보, SSH 비밀키는 저장소에 기록하지
않는다.

## 문서 규칙

- **CONTEXT.md** (이 문서): 현재 동작·구조·용어. 현재 시제. 동작을 바꾸는
  커밋과 같은 커밋에서 갱신한다.
- **ADR.md**: 결정의 역사. 과거 시제, 불변. 채택된 ADR은 수정하지 않고
  후속 기록을 추가한다. 과거 공급자 추천을 현재 승인으로 해석하지 않는다.
- **이슈 트래커**: 계획과 실행 지시서. 미래 시제 문서는 저장소에 두지 않는다.
  [#1](https://github.com/JO-HEEJIN/inference-roofline-lab/issues/1) 조사 계획과
  가설, [#2](https://github.com/JO-HEEJIN/inference-roofline-lab/issues/2)
  G0~G4 실행 지시서와 중단 조건.

추측을 측정 결과로 기록하지 않는다.
