# Ascend baseline execution and evidence gate

No real Ascend correctness or performance result has been collected in this lab.
The local tests validate harness control flow, not device memory safety.

## Prepare the target

Use the pinned upstream commit in `experiments/kv-cache-assign/upstream.json`.
Build and install its package using the instructions at that revision and the
target's compatible CANN / PyTorch / torch_npu environment. Retain the exact
build command, flags, full build log, compiler version and resulting shared
library hash. A checkout hash alone does not identify a loaded binary's source.
Record device model, driver/firmware versions, actual host assign blockDim and
whether another process shares the device. Do not infer vector-core count from
a generic compute-core count.

## Execute Stage 0/1 first

From the repository root, run three independent processes into separate empty
directories. Keep seed, device, build, and all measurement parameters constant.

```bash
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py --output-dir runs/stage01-run1
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py --output-dir runs/stage01-run2
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py --output-dir runs/stage01-run3
```

If discovery fails, append `--assign-active-cores` with the verified host
blockDim. The defaults are 20 warmups, 200 isolated samples, and five sustained
blocks of 200 calls. Tensors/reset and comparisons are outside timing.

Exit 0 means the requested run completed without recorded failures (or an
explicit dry run completed). Exit 1 means correctness or timing failed. Exit 2
means invalid configuration or unavailable NPU runtime. Always inspect manifest
status; `dry-run-static-validation` is not hardware evidence. Unexpected errors
leave an `aborted` manifest and propagate the error.

Preserve each entire run directory, including raw samples. Check all 24
correctness records and the 12 pairs of timing records. A failed case must have
no timing record. Check source checkout versus pinned commit and retain the
build evidence separately. Use the target's available memory-access checking
tool before admitting performance evidence: preserved guards cannot detect
out-of-bounds reads. Keep checker/profiler runs separate from latency runs.

## Decide whether Stage 2 is justified

Compare per-process medians and isolated p95 across the fixed matrix. Sustained
p95 describes block means, not request tail latency. Do not pool processes to
hide drift. Useful bandwidth and modeled GM/UB bytes are not HBM measurements.

The prepared Stage 2 option is not a prerequisite for establishing the baseline.
Run it only after the Stage 0/1 evidence supports a core-transition investigation.
It independently checks its own cases; it does not load or enforce approval
from a prior Stage 0/1 manifest.

Stage 2 currently refuses C/2, C, 2C matrices containing nonmultiples of eight or
batches above 128. These bounds prevent newly generated shapes from silently
bypassing the initial storage contract; they do not certify UB capacity.

## Human review packet

Provide run manifests, JSONL/CSV, retained raw files, build log and library hash,
memory-check results, device-sharing conditions, and an assessment of variation
across processes. Identify which hypothesis is supported, which alternatives
remain, and the smallest next measurement. A profiler trace is needed to
attribute API latency to kernel tasks or particular pipelines. Neither Stage 1
nor Stage 2 wall-clock timing alone establishes a kernel bottleneck.
