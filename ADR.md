# Architecture Decision Records

이 문서는 Inference Roofline Lab의 설계·구현·실험 결정을 기록한다. 결정 당시 확인한 근거, 대안, 선택 이유, 검증 방법과 결과를 남긴다. 코드 관찰과 실제 하드웨어 측정을 구분하며, 새로운 증거로 결정을 바꿀 때는 이전 기록을 지우지 않고 후속 기록을 추가한다.

## ADR-001 — 독립 실험 저장소와 고정된 upstream

- 날짜: 2026-09-14
- 상태: 채택
- 배경: 사용자가 `Inference Roofline Lab` 초기화와 첫 `kv-cache-assign` 실험을 승인했다. 실제 연산자는 `sgl-project/sgl-kernel-npu`에 있다. 기존 작업 폴더에는 초기화된 Git 저장소만 있으며 commit은 없다.
- 결정: 이 저장소에는 reference, 입력 사례, 측정 도구와 결과를 관리한다. upstream은 무시되는 `upstream/sgl-kernel-npu/`에 별도로 checkout하고, 실험의 `upstream.json`에 정확한 commit과 소스 경로를 고정한다.
- 기준 commit: `cdcb9d8b719a6100dadc3544c8cd33ff53bf668a`.
- 대안: upstream 전체를 이 저장소에 복사하거나 항상 최신 main을 사용한다.
- 이유: 소스 출처와 기준 버전을 보존하고, 실험 코드와 upstream 수정의 차이를 명확히 하기 위해서다. 아직 공유할 두 번째 실험이 없으므로 공통 framework와 빈 backend 디렉터리는 만들지 않는다.
- 검증: checkout의 HEAD와 manifest를 대조한다. 향후 수정은 기준 commit 대비 patch로 기록한다.
- 결과: 부분 clone의 HEAD와 기준 commit이 일치하고 주요 파일의 Git blob ID를 확인했다. 소스 checkout에 필요한 추가 네트워크 실행은 사용자에게 거절되어 working tree는 미완성이다. 아래 ADR-003의 명시적인 소스 검토 요청에 따라 GitHub 읽기 API로 내용을 검토했다. 전체 로컬 checkout이나 NPU 빌드가 완료됐다는 뜻은 아니다.

## ADR-002 — Mac의 논리 검증과 NPU 측정을 분리

- 날짜: 2026-09-14
- 상태: 채택
- 관찰: upstream package는 import 시 `torch_npu`와 NPU shared library를 로드한다. 커널은 AscendC와 NPU 실행 환경에 의존한다.
- 결정: 로컬 reference·사례 생성·테스트는 Python 표준 라이브러리로 실행한다. `torch`, `torch_npu`, upstream package는 NPU 실행 경로에서만 import한다.
- 대안: 모든 로컬 작업에 PyTorch/NPU package를 요구하거나 Mac에 CANN VM부터 구성한다.
- 이유: 첫 목표는 연산 계약과 측정 준비이며, 로컬 검증이 원격 하드웨어 확보에 종속될 필요가 없다.
- 검증: NPU package 없이 테스트와 CPU smoke 실행이 가능해야 한다. NPU가 없으면 측정은 명시적으로 실패해야 하며 CPU 결과로 대체하지 않는다.
- 결과: `reference.py`, `cases.py`, `contract.py` 초안을 작성한 뒤 사용자의 ADR-003 지시에 따라 구현을 중단했다. 테스트와 benchmark는 아직 구현·실행하지 않았다. 초안과 source-level 모델은 Ascend 커널의 정확성 또는 성능 증거가 아니다.

## ADR-003 — 코드 변경을 멈추고 가설 중심 조사 계획부터 확정

- 날짜: 2026-09-14
- 상태: 채택; 조사 계획 작성 완료, 실측 대기
- 계기: 사용자가 kernel, host tiling, tests, benchmark 경로를 읽고 성능 조사 계획을 만들되 "Do not change code yet"라고 명시했다.
- 결정: 추가 코드 변경을 중단한다. [조사 계획](experiments/kv-cache-assign/investigation-plan.md)에 source 관찰, copy payload 모델, 반증 가능한 가설, 입력 검증 선행 조건, 측정 경계와 순서를 기록한다. 이번 단계의 변경은 문서뿐이다.
- 관찰: 각 active core가 batch 전체의 metadata와 `16 × batch` cache buffer를 읽는다. assign은 짧은 update에서도 row의 16개 int32를 읽고 다시 쓴다. packed offset 계산은 각 core가 앞선 request 길이를 반복 순회한다. retrieve는 한 core로 실행한다. 기존 timing은 correctness test 안에 있다.
- 대안: 먼저 kernel을 최적화하거나, 기존 test의 평균값을 그대로 baseline으로 사용한다.
- 이유: 코드에 보이는 중복 작업이 실제 critical path인지는 알 수 없다. host 비용, L2 reuse, MTE/Scalar 병목, core별 작업 편차를 분리해야 한다. 입력 할당과 고정 copy 크기의 불일치도 성능 측정보다 먼저 검증해야 한다.
- 검증 방법: 기준 commit 소스와 helper·CI 실행 경로를 읽고 copy-byte 및 prefix-loop 수식을 산술 검산했다. 코드 수정 없이 가설마다 예상 signature와 반증 조건을 설정했다.
- 결과: source 검토 완료. 실제 HBM traffic, pipeline overlap, NPU latency, 최적화 효과는 미측정이다. 이전 Python 초안은 수정·실행하지 않고 보존한다.

## ADR-004 — 유효 데이터, GM↔UB copy payload, HBM 실측을 구분

- 날짜: 2026-09-14
- 상태: 채택
- 질문: 이 커널의 "effective bandwidth"를 무엇으로 정의해야 하는가?
- 결정: 선택된 int32 값의 읽기+쓰기 `8 × sum(lengths)`를 useful bytes로 명시한다. 정적으로 계산한 GM↔UB copy payload와 profiler의 main-memory bytes는 별도 필드로 둔다. 모든 bandwidth 값에 timing 범위를 붙인다.
- 이유: 같은 metadata를 여러 core가 읽어도 L2에서 공급될 수 있다. source-copy 비율이 크다고 HBM 대역폭 포화나 동일 비율의 latency 낭비를 의미하지 않는다. row capacity 증가는 이 커널의 고정 copy 길이를 늘리지 않지만 Torch reference의 전체 row gather 비용은 늘린다.
- 검증/반증: GM↔UB bytes, main-memory bytes, L2 통계, pipe 절대 시간과 kernel duration을 함께 측정한다. 모델상의 bytes가 줄어도 지연이 줄지 않으면 해당 변경을 성능 개선으로 인정하지 않는다.
- 결과: 측정 정의와 source 수식만 확정했다. 실측값은 없다.

## ADR-005 — 재현 가능한 timing과 memory-access gate를 먼저 설계

- 날짜: 2026-09-14
- 상태: 채택; 실행 전
- 관찰: 기존 test는 반복 index 1부터 시계를 시작하지만 20으로 나누며 해당 경계에서 device 작업을 동기화하지 않는다. 두 dtype의 correctness 실행에서 수정된 출력이 재사용되고, row별 초기 내용도 동일하다. host는 cache buffer 길이를 `16 × batch`로 가정하지만 test는 실제 길이 합으로 할당한다.
- 결정: fresh state, row-distinct 값, 전체 buffer/guard 검증을 계획에 포함한다. API wall time, sustained throughput, stream-event interval, profiler kernel task duration을 구분한다. profile run과 latency run은 분리한다.
- 이유: 정답 출력만으로 접근 범위를 검증할 수 없고, 비동기 실행에서 측정 구간이 모호하면 어떤 비용이 개선됐는지 판단할 수 없다. batch-mean 분포를 per-call p95로 표시하지 않는다.
- 대안: 원본 workload와 timing을 수정 없이 성능 근거로 채택한다. source 관찰상 제약이 해결되기 전에는 채택하지 않는다.
- 검증: 이후 NPU 환경에서 메모리 검사와 정확성 gate를 통과한 workload만 timing 대상으로 삼는다. padding 등 fixture adaptation은 원본 geometry와 구분해 기록한다.
- 결과: 계획에 반영했다. 안전한 입력 범위, 실제 allocator rounding, 특정 batch의 지원 여부는 hardware 검증 전이며 확정하지 않는다.

## ADR-006 — Stage 0/1 controlled baseline harness

- 날짜: 2026-09-14
- 상태: 구현 완료; Ascend 실행 대기
- 계기: 사용자가 Stage 0 correctness와 Stage 1 baseline instrumentation만 구현하도록 명시했다. kernel, host tiling, 이후 stage의 capacity/core/alignment/retrieve 실험은 범위 밖이다.
- 결정: `experiments/kv-cache-assign/benchmark_cache_location_assign.py`에 독립 harness를 추가했다. Stage 0은 batch 8/32/128, update 1/2/8/16, request-index int32/int64의 fresh fixture를 검사한다. Stage 1은 그중 정확성이 통과한 int64 assign case만 2048 logical capacity에서 측정한다.
- 안전한 backing: token pool의 physical row width는 2064(2048 logical + 16 guard)이며, starts는 8-element aligned interior로 고정했다. packed cache view는 `batch × 16` plus guard이며 useful suffix는 sentinel이다. 요청 index와 offsets는 source의 32-byte alignment count를 만족하는 물리 backing을 사용하되 logical batch-length view를 전달한다.
- timing: isolated samples는 reset/synchronize를 interval 밖에서 수행하고 operator invocation부터 completion synchronize까지 잰다. sustained blocks는 fresh state와 warmup 뒤 200 calls를 한 block으로 재며, raw samples를 별도 JSON에 보존한다. event timing을 kernel time으로 명명하지 않는다.
- 결과 기록: JSONL, CSV, raw sample JSON, run manifest에 source commit/dirty status, loaded library hash, runtime/device/CANN metadata, fixture layout과 measurement parameters를 기록한다. correctness failure는 raw diagnostic과 함께 기록하고 같은 int64 case의 timing을 생략한다.
- 검증: Python compile, 표준 라이브러리 unittest 6개, Stage 0/1 dry run, 그리고 no-NPU fail-closed path를 실행했다. 현재 Mac에는 `torch_npu`와 Ascend device가 없으므로 real correctness/latency 결과는 생성하지 않았다.

## ADR-007 — Stage 2는 실제 core 수 기준의 coarse transition만 허용

- 날짜: 2026-09-16
- 상태: 구현 완료; Ascend 실행 대기
- 배경: Stage 0/1 구현 뒤 다음 계획 단계를 진행한다. 계획서의 순서는 용량 sweep이 아니라 core-transition/prefix-work 관찰이며, 이후 Stage 4가 capacity 128/2048/16384과 alignment/reuse를 다룬다.
- 결정: harness에 opt-in `--stage stage2`를 추가한다. 실제 assign blockDim/vector-core 수 `C`를 device properties에서 조회하거나 `--assign-active-cores`로 명시하고, int64 indices·capacity 2048·aligned start·update length 1/16에 대해 `B = floor(C/2), C, 2C`만 만든다. 각 case는 기존 guarded fresh fixture correctness gate를 통과해야 timing을 기록한다. 결과와 manifest에는 `below/equal/above active cores` 관계를 남긴다.
- 대안: 지금 바로 `C-1/C+1` 또는 sequence capacity 128/16384을 포함한다.
- 이유: `C±1`은 source의 metadata/scratch access와 runtime allocation을 별도로 검증해야 하는 case이며, capacity sweep은 core-count 및 prefix scalar work와 다른 가설을 시험한다. 한 번에 추가하면 latency 변화의 원인을 구분할 수 없다.
- 검증: Stage 2 matrix는 `C=32` dry run에서 batch 16/32/64와 길이 1/16으로 생성되고 C±1을 포함하지 않는 단위 테스트를 추가한다. 실제 device의 `C`, correctness, latency, profiler per-core data는 Ascend 하드웨어에서만 기록한다.
- 결과: Mac에서 static validation만 가능하다. `torch_npu`가 없으므로 performance 또는 core-transition 결과는 생성하지 않았다.

## 기록 형식

각 후속 ADR에는 날짜·상태, 관찰 또는 질문, 결정, 대안과 이유, 검증 방법, 실제 결과와 남은 제약을 적는다. 성능 가설에는 반증 조건을 포함한다. 추측을 측정 결과로 기록하지 않는다.
