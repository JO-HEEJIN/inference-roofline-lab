# Inference Roofline Lab

Reproducible experiments on LLM inference performance, memory traffic, and accelerator kernels.

The first experiment investigates SGLang's Ascend NPU `cache_location_assign` path. It separates correctness validation, source-level copy-payload modeling, API latency, sustained throughput, and profiler-derived kernel evidence. The current implementation provides guarded Stage 0/1 fixtures and baseline instrumentation; it does not claim NPU performance results without an Ascend runtime.

## Layout

- [`ADR.md`](ADR.md): decisions, evidence, and changed assumptions.
- [`docs/`](docs/): experiment-method pointers.
- [`experiments/kv-cache-assign/`](experiments/kv-cache-assign/): pinned-upstream provenance, plan, guarded fixture harness, tests, and result schema.

## First benchmark

The harness requires an Ascend environment with CANN, `torch_npu`, and a compatible `sgl_kernel_npu` build. Its command and output contract are documented in [`experiments/kv-cache-assign/README.md`](experiments/kv-cache-assign/README.md).

For static validation on a machine without an NPU:

```bash
python3 -m unittest experiments/kv-cache-assign/test_benchmark_cache_location_assign.py -v
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py \
  --output-dir /tmp/kv-cache-assign-dry-run --dry-run
```

## Upstream

The experiment pins [SGLang Kernel NPU](https://github.com/sgl-project/sgl-kernel-npu) at commit `cdcb9d8b719a6100dadc3544c8cd33ff53bf668a`. The checkout itself is intentionally excluded from this repository; [`upstream.json`](experiments/kv-cache-assign/upstream.json) records the source paths and build target.
