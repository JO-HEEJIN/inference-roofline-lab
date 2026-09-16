# Controlled `cache_location_assign` baseline

`benchmark_cache_location_assign.py` implements only Stage 0 and Stage 1 of the [investigation plan](investigation-plan.md). It measures the unchanged upstream `torch.ops.npu.cache_loc_assign` operator. It does not benchmark retrieve, vary sequence capacity, sweep alignment residue or core transitions, reorder ragged work, profile a kernel task, or modify upstream code.

The logical sequence capacity is fixed at 2048. Each fixture uses a physical row width of 2064: 2048 logical cells followed by a 16-cell guard. Starts are aligned to eight int32 elements and lie far enough inside the logical row that the upstream fixed 16-element transfer cannot touch the guard. The packed cache view has `batch × 16` elements, followed by a guard; its logical suffix after the useful `L = batch × update_length` entries is initialized with sentinels and verified after every correctness call. Metadata tensors similarly retain enough backing storage for the inspected 32-byte-aligned copies while exposing the logical batch-length views to the operator.

The harness runs fresh fixtures for all Stage 0 combinations: batches 8, 32, and 128; updates 1, 2, 8, and 16; request indices `int32` and `int64`. It records a correctness result for each. Timed Stage 1 cases are the matching `int64` cases only, and only when their own correctness record passed.

Run on an Ascend environment with the pinned upstream package built and installed:

```bash
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py \
  --output-dir runs/kv-cache-assign-stage01 \
  --seed 20260914 \
  --warmup 20 \
  --isolated-samples 200 \
  --sustained-blocks 5 \
  --sustained-calls 200
```

If device properties do not expose the vector-core count used by upstream host tiling, supply the real value explicitly. The harness refuses to label the total source-copy model without it.

```bash
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py \
  --output-dir runs/kv-cache-assign-stage01 \
  --assign-active-cores ACTUAL_BLOCK_DIM
```

For static validation without importing `torch_npu` or executing NPU code:

```bash
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py \
  --output-dir /tmp/kv-cache-assign-dry-run --dry-run
```

`results.jsonl` and `results.csv` use the same schema. There is one `correctness` record per admitted Stage 0 case, then `isolated_synchronized_api_latency` and `sustained_synchronized_blocks` records for each eligible Stage 1 case. Raw timing arrays and sustained block measurements are retained under `raw-samples/`; CSV/JSONL records reference their relative paths. `run-manifest.json` captures source, dirty state, harness hash, environment, case definitions, physical-storage policy, and timing parameters.

`median_us` and `p95_us` for isolated records summarize individual synchronized API samples. For sustained records they summarize block-average call times; `sustained_calls_per_second` is the median of block rates. Neither is called kernel time. Useful effective bandwidth is `8L / latency` or `8L × calls/sec`; source-copy bytes are the static AscendC copy-payload model, not measured HBM traffic.
