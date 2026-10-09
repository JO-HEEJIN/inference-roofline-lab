# 다음 실행자 작업 지시서

## 목표와 고정 범위

목표는 **변경하지 않은 pinned upstream의 Stage 0/1 실측 증거 확보**다.
실행 가능한 서버는 아직 없다. 신용카드도 없다. Huawei 가입 진행은 서버
확보가 아니다. NPU 성능 결과는 아직 없으며 Mac 테스트는 harness 검증이다.

커널·host tiling 수정, 최적화, Stage 2 실행, capacity/alignment sweep,
retrieve benchmark를 하지 않는다. 기존 Stage 2 옵션은 그대로 둔다.
실행자는 설계를 다시 만들지 말고 아래 gate 순서대로 진행한다.
사용자 최종 검토 전 성능 결론이나 유료 계약을 확정하지 않는다.

## 읽을 파일과 우선순위

1. 이 문서: 현재 실행 순서와 중단 조건.
2. `experiments/kv-cache-assign/upstream.json`: revision과 source 경로.
3. `experiments/kv-cache-assign/README.md`: fixture/timing/output 계약.
4. `docs/ascend-runbook.md`: 장치 실측과 증거 인수 조건.
5. `experiments/kv-cache-assign/investigation-plan.md`: 가설의 근거.
6. `ADR.md`: 결정의 역사. 과거 공급자 추천을 현재 승인으로 해석하지 않는다.

## G0 — 지금 로컬에서 실행

저장소 루트에서:

```bash
git status --short
bash scripts/validate-local.sh
```

스크립트가 출력하는 임시 디렉터리에 테스트 로그와 dry-run manifest가 남는다.
통과 조건: 테스트 실패 0, compileall 및 diff check 성공, dry-run case 24개,
timing record/raw timing 0개. torch가 없어서 CPU 테스트가 skip되면 그 사실을
기록한다. skip을 pass로 보고하지 않는다. 실패 시 실패한 로컬 문제만 수정하고
재검증한다. 환경 설치를 위해 시스템 Python이나 CANN을 임의 변경하지 않는다.

현재 작업 트리에는 이전에 작성한 미커밋 harness/test/docs 변경이 있다.
reset/clean/stash로 없애지 않는다. 공개 upstream clone에는 이 변경이 없을 수
있으므로 원격 전송 전에 현재 revision과 diff를 함께 보존한다.

## G1 — 실행 환경 확보 (현재 외부 병목)

`docs/templates/ascend-offer.md`를 복사해 계정 비밀정보 없이 채운다.
Huawei 국제 계정의 **읽기 전용** 콘솔에서 사용 가능한 NPU SKU, 견적,
결제 방식, 권한을 확인한다. 생성/구매/충전 버튼은 누르지 않는다.
문서상 지원과 해당 계정의 실제 가용성을 별도 기록한다.

결정 규칙:

| 관찰 | 다음 행동 |
|---|---|
| 단일 호환 NPU, custom kernel 실행, 카드 없는 결제, 총견적 확인 | 인수표와 제한된 파일럿 비용을 사용자에게 최종 승인 요청 |
| 체크카드 별도 승인 또는 송금 자격 문의 필요 | 템플릿의 지원 문의를 구체화하고 발송 승인 요청; 임의 발송 금지 |
| 로그인/SMS/실명 인증이 필요 | 그 인증만 사용자에게 요청; 새 공급자 가입을 반복하지 않음 |
| 월정액/다중 카드만 가능 | 짧은 파일럿 조건 불충족으로 기록; 구매하지 않음 |
| API만 제공, custom operator 불가 | 탈락. 저렴해도 대체 환경으로 사용하지 않음 |
| 가격/재고 미확인 | UNVERIFIED 유지. 최저가 또는 설치 가능으로 보고하지 않음 |

CANNLab은 한국 전화 인증에서 막혔다. 같은 절차를 다시 권하지 않는다.
CloudGPU는 SSH custom kernel 환경으로 확인되지 않았다. Yissou 광고는 견적이
아니다. 새로운 공급자를 찾더라도 위 표를 채울 수 있을 때만 후보로 추가한다.
현재 조건에서 적격 후보가 없다면 G1에서 멈춘다. G0 작업은 독립적으로 완료한다.

## G2 — 승인받은 장치에서 빌드 환경 고정

기존 서버 정보를 받으면 결제를 다시 요구하지 않고 승인 범위 내에서 진행한다.
SSH host fingerprint 확인 후 접속하고 `npu-smi info`, OS/architecture,
driver/firmware, 설치된 CANN/compiler/Python/torch/torch_npu 버전을 보존한다.
필요한 항목이 없으면 unknown으로 기록하고 이유를 남긴다. 시스템 driver 변경 금지.

`upstream.json`의 repository를 전용 새 디렉터리에 clone하고 해당 commit을
checkout한다. 기존 checkout을 덮어쓰지 않는다. 그 revision의 README와
build.sh를 읽어 실제 칩 및 SDK 조합의 지원을 확인한다.
`Ascend910_9382`는 예시일 뿐이다. 모든 910B에서 된다고 추정하지 않는다.
지원 조합을 확인하지 못하면 빌드 gate 실패로 기록하고 중단한다.

지원되는 기존 이미지/격리 환경을 사용해 해당 revision의 절차로 build/install한다.
장치가 정해지지 않아 build 명령은 여기서 미리 고정하지 않는다. 다음 항목은
실제 실행값으로 `runs/<session>/build/`에 저장한다:

- exact source HEAD + dirty diff, 실행한 build/install 명령, 전체 build log;
- compiler 및 package 버전, wheel 또는 빌드 산출물 hash;
- 실제 import된 `sgl_kernel_npu` 경로와 library hash;
- lab HEAD + dirty diff 및 untracked 소스 파일 사본 (비밀정보 제외).

import가 성공하고 `torch.npu.is_available()`가 true이며
`torch.ops.npu.cache_loc_assign`가 등록되어야 G3로 간다. 미등록이면 다른
operator로 대체하거나 최신 upstream으로 pin을 바꾸지 않는다.

## G3 — Stage 0/1만 실행

장치 공유 여부와 host가 사용하는 vector-core/blockDim 근거를 기록한다.
generic compute core 수를 vector-core 수로 대입하지 않는다. 런타임 discovery가
안 되면 소스/장치 근거로 확인한 값만 `--assign-active-cores`에 전달한다.

아래 명령을 각각 **별도 프로세스**로 실행한다. 첫 실패 후 후속 run을 실행하지
않는다. 기존 디렉터리는 지우지 않고 새 session 경로를 사용한다.

```bash
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py --stage stage01 --seed 20260914 --warmup 20 --isolated-samples 200 --sustained-blocks 5 --sustained-calls 200 --output-dir runs/session-001/stage01-run1
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py --stage stage01 --seed 20260914 --warmup 20 --isolated-samples 200 --sustained-blocks 5 --sustained-calls 200 --output-dir runs/session-001/stage01-run2
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py --stage stage01 --seed 20260914 --warmup 20 --isolated-samples 200 --sustained-blocks 5 --sustained-calls 200 --output-dir runs/session-001/stage01-run3
```

장치의 메모리 검사 도구가 있으면 동일 fixture를 별도 검사 실행으로 확인한다.
도구 이름/버전/명령/전체 로그를 남기고 검사 실행을 timing 결과와 섞지 않는다.
검사 도구가 없으면 memory-access validation pending으로 표시한다. guard 보존은
OOB read가 없다는 증거가 아니다. 검사에서 오류가 나오면 성능 해석을 중단한다.

## G4 — 인수 검사와 사람에게 전달

각 run의 exit code뿐 아니라 다음 전부를 확인한다:

- manifest status=`completed`, failure_count=0, stage=`stage01`;
- B=8/32/128 × update=1/2/8/16 × dtype=int32/int64의 correctness 24개 passed;
- timed dtype=int64만 12개 case × 2 timing modes = 24개 timing record;
- 각 isolated raw `samples_us` 길이 200, sustained raw `blocks` 길이 5,
  `calls_per_block`=200; CSV/JSONL에서 같은 raw 파일을 가리킴;
- capacity=2048, seed=20260914, warmup=20, 모든 raw 경로 존재;
- 실패 correctness case에는 timing 없음. timing_failure가 있으면 불완전 run;
- 실제 upstream HEAD와 pin 일치, 실제 로드 binary와 build 산출물 연결 근거 존재.

run별 median/p95 및 sustained calls/sec를 나란히 비교한다. 서로 다른 run을
하나로 합쳐 변동을 숨기지 않는다. sustained p95는 block 평균의 percentile이다.
API 시간을 kernel time이라고 쓰지 않는다. `8L` bandwidth와 modeled copy bytes를
실측 HBM traffic이라고 쓰지 않는다. 변동이 크면 원인을 모르는 상태라고 기록한다.

전체 session 폴더를 로컬로 반출하고 파일 크기/hash로 확인한 뒤 공급자의 정확한
과금 종료 조작을 수행한다. 승인된 임대의 종료와 기존 데이터 삭제를 구별한다.
기존 공유 자원은 삭제하지 않는다. 남은 디스크/리소스의 과금 상태를 확인한다.

최종 전달물은 환경 인수표, build evidence, 3개 완전한 run, 메모리 검사 상태,
run별 비교표, 남은 불확실성이다. 이 지점에서 사람이 다음 실험을 판단한다.
원시 결과는 `runs/`에 보존하고 공개 저장소에 계정/SSH/결제 정보를 올리지 않는다.

## 다음 모델에 그대로 전달할 프롬프트

> docs/NEXT_AGENT.md를 실행 지시서로 사용하라. G0부터 현재 증거로 완료 여부를
> 확인하고, 완료된 일을 반복하지 말고 다음 미완료 gate를 진행하라. 커널 및
> host tiling 변경, Stage 2 실행, 실측 없는 성능 주장 금지. G1의 미확인 항목을
> 확인된 것처럼 채우지 말라. 사용자에게는 실제 필요한 인증/발송 승인/구체적인
> 비용 승인만 요청하라. 같은 장애에서 계정 가입을 반복하지 말라. 작업마다
> ADR.md에 이유·검증·결과를 기록하라. 현재 gate, 새로 만든 파일, 다음 실행
> 명령과 정확한 외부 blocker를 짧게 남겨 인계하라.
