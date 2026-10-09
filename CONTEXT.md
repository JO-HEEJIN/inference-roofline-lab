# CONTEXT

This document describes only the current state of the repository. Plans and
execution instructions live in the issue tracker; the history of decisions
lives in [`ADR.md`](ADR.md).

## Current state

**There are no NPU measurements.** No latency, throughput, or correctness
result has ever been produced on a real Ascend device. The `runs/` directory
does not exist.

Work is waiting at G1 (obtaining an execution environment) because no usable
Ascend device has been secured. This is an external blocker, not a defect in
the harness. Local validation (G0) passes.

| Item | Status |
|---|---|
| Stage 0 correctness harness | Implemented; awaiting Ascend execution |
| Stage 1 baseline timing harness | Implemented; awaiting Ascend execution |
| Stage 2 core transition (opt-in) | Implemented; preconditions unmet |
| Local static validation | Passing (15 tests) |
| Upstream working-tree checkout | Incomplete: `.git` only, no source files |
| Measured results | None |

## Target and pinned revision

The measurement target is the SGLang Ascend NPU kernel
`torch.ops.npu.cache_loc_assign`. Upstream is pinned to
[`sgl-project/sgl-kernel-npu`](https://github.com/sgl-project/sgl-kernel-npu)
at commit `cdcb9d8b719a6100dadc3544c8cd33ff53bf668a`. Exact source paths, blob
IDs, and the build-target example are recorded in
[`experiments/kv-cache-assign/upstream.json`](experiments/kv-cache-assign/upstream.json).

The upstream checkout is deliberately excluded from this repository
(`/upstream/` in `.gitignore`). Right now `upstream/sgl-kernel-npu/` holds only
Git metadata, with no working-tree files. Source observations so far were made
through the GitHub read API; neither a local checkout nor an NPU build has been
completed.

The kernel and host tiling are not modified. This repository holds only the
reference implementation, input cases, measurement tooling, and results.

## What the harness does

`experiments/kv-cache-assign/benchmark_cache_location_assign.py` is the single
entry point. Its usage and output contract are documented in
[`experiments/kv-cache-assign/README.md`](experiments/kv-cache-assign/README.md).

- **`--stage stage01`** (default): Stage 0 checks 24 fresh fixtures covering
  batch 8/32/128 × update 1/2/8/16 × request index int32/int64. Stage 1 then
  measures only the 12 int64 cases whose correctness record passed, at logical
  capacity 2048, emitting one isolated and one sustained timing record per case.
- **`--stage stage2`** (opt-in): derives the actual assign blockDim / active
  vector-core count `C` and generates only `B = C/2, C, 2C`. `C±1` and capacity
  sweeps are out of scope. It is admitted only when every generated batch is a
  multiple of eight and at most 128; otherwise it raises `ValueError`.
- **`--dry-run`**: generates case definitions and a manifest without importing
  any NPU runtime. The manifest status is `dry-run-static-validation`.

### Fixture protection

The token pool's physical row width is 2064 (2048 logical cells plus a 16-cell
guard). Starts are aligned to eight int32 elements and placed far enough inside
the logical row that upstream's fixed 16-element transfer cannot reach the
guard. The packed cache view is `batch × 16` plus a guard; the region after the
useful `L = batch × update_length` entries is filled with sentinels and
verified after every call. Metadata tensors retain enough backing storage for
the inspected 32-byte-aligned copies while exposing only logical batch-length
views to the operator.

Useful cache values are all generated negative. Logical pool values are
positive, so the checker is guaranteed to observe even a single missing store.
The checks cover changed-cell values and the change mask itself, preservation of
untouched cells, the token guard, cache padding and guard, and metadata
integrity.

### Failure handling

A single correctness or timing failure produces exit 1 and manifest status
`completed-with-failures`. An abort through an exception records `aborted` and
re-raises. A case whose correctness failed is never timed. Static environment
collection reads installed distribution metadata only; it does not import
`torch` or `torch_npu`.

| Exit code | Meaning |
|---|---|
| 0 | The requested run completed with no recorded failures, or a dry run completed |
| 1 | A correctness or timing failure occurred |
| 2 | Invalid configuration, or no usable NPU runtime |

### Output

`results.jsonl` and `results.csv` share one schema. Raw timing arrays and
sustained block measurements are retained under `raw-samples/` and referenced
by relative path from each record. `run-manifest.json` captures the source
commit and dirty state, the actual checkout HEAD and whether it matches the
pin, the unverified binary/source linkage, the harness hash, the environment,
case definitions, the storage policy, and the measurement parameters.

Raw measurements and profiler traces stay local (`/runs/` in `.gitignore`).

## Measurement vocabulary

Blurring these distinctions produces wrong conclusions, so the following terms
are never used interchangeably.

- **useful bytes**: the read plus write of the selected int32 values,
  `8 × sum(lengths)`, written as `8L`.
- **source-copy bytes**: the GM↔UB copy-payload model computed statically from
  the source. Not measured HBM traffic.
- **main-memory bytes**: what the profiler reports. A field separate from both
  of the above.
- **isolated API latency**: wall time from operator invocation to completion
  synchronize. Reset and comparison happen outside the timed interval. This is
  not kernel time.
- **sustained throughput**: 200 calls measured as one block after warmup.
  `median_us` and `p95_us` summarize block-average call times, not per-request
  tail latency. `sustained_calls_per_second` is the median of block rates.
- **useful effective bandwidth**: `8L / latency` or `8L × calls/sec`. Every
  bandwidth figure carries its timing scope.

A preserved guard is not evidence that no out-of-bounds read occurred.
Wall-clock timing alone does not establish a kernel bottleneck; attributing API
latency to kernel tasks or particular pipelines requires a profiler trace.

## Local validation

This validates the harness itself on a machine without an NPU. It performs no
NPU imports, installs, network requests, or paid-resource creation.

```bash
bash scripts/validate-local.sh
```

The pass conditions are 15 tests passing, successful `compileall` and diff
check, 24 dry-run cases, and zero timing records. Logs and the dry-run manifest
are left in the temporary directory the script prints.

The CPU tests (`test_fixture_cpu.py`) exercise the fixtures and the checker
against an independent scalar reference and inject corruption to confirm it is
detected. **They validate the checker; they are not evidence that the NPU
kernel ran.** Passing static validation is not read as a successful hardware
validation.

## Execution environment requirements

An Ascend device is required, with CANN, `torch_npu`, and an `sgl_kernel_npu`
built and installed for the pinned revision. The current development host is
Darwin arm64 and has no `npu-smi`, `torch_npu`, or `sgl_kernel_npu`.

Acceptance conditions before taking measurements are in
[`docs/ascend-runbook.md`](docs/ascend-runbook.md), the comparison of confirmed
access paths is in
[`docs/ascend-access-options.md`](docs/ascend-access-options.md), and the
provider acceptance table is in
[`docs/templates/ascend-offer.md`](docs/templates/ascend-offer.md). Every
provider entry is UNVERIFIED, and no payment, server creation, or support
ticket has ever been made. Account identifiers, phone numbers, card details,
and SSH private keys are not recorded in this repository.

## Documentation rules

- **CONTEXT.md** (this file): current behavior, structure, and vocabulary, in
  present tense. Updated in the same commit as the behavior change it
  describes.
- **ADR.md**: the history of decisions, in past tense, immutable. An accepted
  ADR is never edited; a follow-up entry is appended instead. A past provider
  recommendation is never read as a current approval.
- **Issue tracker**: plans and execution instructions. Future-tense documents
  are not kept in this repository.
  [#1](https://github.com/JO-HEEJIN/inference-roofline-lab/issues/1) holds the
  investigation plan and its hypotheses;
  [#2](https://github.com/JO-HEEJIN/inference-roofline-lab/issues/2) holds the
  G0–G4 execution instructions and stop conditions.

Conjecture is never recorded as a measurement result.
