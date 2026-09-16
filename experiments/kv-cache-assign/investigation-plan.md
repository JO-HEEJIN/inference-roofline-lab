# cache_location_assign: performance investigation plan

Date: 2026-09-14. Status: source review complete; hardware hypotheses untested.

Upstream: `sgl-project/sgl-kernel-npu`, commit `cdcb9d8b719a6100dadc3544c8cd33ff53bf668a`.

The immediate objective is to establish a trustworthy baseline and distinguish host overhead, redundant transfers, scalar prefix work, and per-core imbalance. No performance improvement is claimed. No upstream code has been changed. Local Python drafts created before the user requested a read-only investigation remain paused and unvalidated; this document does not treat those drafts as evidence.

The first experiment should characterize **assign**, then examine **retrieve** separately. Both use the same device entry point, but the host launches them differently. Correctness and memory-access validation precede performance measurement. The strongest initial candidates are repeated full-batch preprocessing, fixed 16-element row transfers, and redundant prefix scanning. Double buffering is a question to measure, not an established optimization.

## 1. Source and measurement paths inspected

All repository links below are pinned to the commit above.

| Path | Relevant observations |
| --- | --- |
| [Device kernel][kernel] | Core-to-row mapping; per-core UB allocation; preprocessing; assign/retrieve loops; queue operations; explicit events; scalar packed-offset scan. |
| [Host tiling and launch][host] | Assign launches all reported vector cores; retrieve launches one. Tiling depends on batch, row width, pool size, and request-index dtype. Each invocation allocates a CPU tiling tensor and copies it to device. |
| [Tiling structure][tiling] | `MAX_STEP = 16`; metadata fields passed to the kernel. |
| [Alignment helpers][common] | int32 counts round to multiples of eight; int64 counts round to multiples of four: 32-byte copy-size rounding. |
| [Torch launch helpers][helper] | Tiling transfer pins host memory, copies to the current device, and launches through the current NPU stream. |
| [Bindings][bindings] and [package loader][loader] | `torch.ops.npu.cache_loc_assign` / `cache_loc_update` dispatch to the host functions. Import loads `torch_npu` and the shared library. |
| [Assignment test][assign-test] | Embedded Python/golden, Torch-native, and AscendC timings; batch 300, pool 2000 × 16384, updates of one or two entries. |
| [Retrieval test][retrieve-test] | Embedded Python/reference and AscendC timings; batch 300, pool 2000 × 8192, updates of one or two entries. |
| [CI runner][runner] | Cache tests run as scripts because they have `__main__`; this is not an independent benchmark harness. |
| [Other assignment test][other-test] | Exercises `assign_cache_op`, a distinct operator. Its long-update cases do not establish support for this kernel. |
| [Slot lookup benchmark][slot-bench] / [unidex copy benchmark][copy-bench] | Different operators. Their warmup, event, and synchronized wall-time patterns are useful references, not cache-location measurements. |

The pinned `benchmark/` tree contains the two unrelated benchmark scripts above and an empty `bench_deepep.py`. Repository symbol search found the cache-location invocation in its embedded test, not a dedicated benchmark script. The relevant reproduction paths are `tests/python/sgl_kernel_npu/test_cache_assign.py` and `test_cache_update.py`, subject to the input checks below. CI invocation is not evidence that the current revision passed on a particular device.

## 2. Operation and execution model

For request-list position `i`, define `l_i = end_i - start_i` and packed offset `p_i = sum(l_j for j < i)`.

- **Assign:** write packed `cache_locations[p_i : p_i + l_i]` into the selected token-pool row interval. Preserve everything outside those intervals.
- **Retrieve:** copy the selected row intervals into a packed output. In upstream naming this is `cache_loc_update`. Preserve the pool and any explicitly allocated output padding.

The payload is int32 cache-location metadata, not K/V activation tensors. Request indices support int32 or int64; offsets are int64. The kernel also narrows offsets to int32 for length calculations. Initial experiments therefore use small, nonnegative offsets and unique request indices; duplicates, aliasing, arbitrary strides, and oversized offsets need a separately established contract.

The assign path is:

1. Python dispatch → host parameter check → CPU tiling allocation → pinned-memory/device transfer → kernel launch.
2. All launched vector cores execute initialization. Cores assigned no rows skip the processing body, but initialization occurs before that guard.
3. Each active core loads the **whole batch's** request indices, start/end offsets, and `16 × batch` cache entries into its own UB.
4. An explicit MTE2-to-vector event precedes offset casts and length calculation.
5. For each owned row: read 16 int32 entries, compute the packed offset, update `l_i` entries with scalar `GetValue`/`SetValue`, and write 16 entries back.

Retrieve uses one core, the same initial full-cache read, and 16-entry row reads. It fills the packed prefix of its UB cache buffer, then writes the entire `16 × batch` buffer after an explicit vector-to-MTE3 event. The initial cache read preserves entries that are not replaced. Any proposed removal of that read must first specify which output entries need preservation. See [processing and transfers][kernel].

## 3. Static GM↔UB copy model

Let:

- `B` = batch size; `L = sum(l_i)` = useful entries; `C` = launched vector cores.
- `P = min(B, C)` = active assign cores, for nonempty valid input. Retrieve has `P = 1`.
- `s` = request-index size, four or eight bytes.
- `R32(x) = 32 × ceil(x / 32)`.
- `A = R32(sB) + 2 × R32(8B) + 64B`: preprocessing bytes read **per active core**.

The source issues the following explicit copy payloads:

| Direction | Assign | Retrieve |
| --- | --- | --- |
| GM → UB | `P × A + 64B` | `A + 64B` |
| UB → GM | `64B` | `64B` |
| Sum | `P × A + 128B` | `A + 128B` |
| Useful selected-value read + write bytes | `8L` | `8L` |

For uniform updates of length `l`, assign's row transfers alone have a `16/l` payload-to-useful ratio: 16× at one entry, 8× at two, and 1× at sixteen. This excludes the repeated preprocessing, so it is not the full amplification.

For an **illustrative, unmeasured** int64-index case with `B = 300` and **assumed** `P = 32`, `A = 26,400` bytes. Assign requests 883,200 copy bytes; retrieve requests 64,800. A one-entry update has 2,400 useful read/write bytes, giving source-copy ratios of 368× and 27×. The actual device's core count must be queried; 32 is not a hardware claim.

These are copy payloads derived from [host counts][host], [alignment helpers][common], and [device operations][kernel]. They exclude tiling metadata access, host-to-device setup, transaction rounding, cache-line effects, and UB-internal scalar/vector accesses. They are **not measured HBM traffic**. Shared cache hits may greatly reduce main-memory reads despite redundant GM→UB transfers. Do not infer memory saturation from these ratios.

A sequence-capacity sweep does not, by itself, increase this kernel's per-row copy length: that remains 16 entries. Capacity changes row stride and address locality. In contrast, the existing Torch references gather full selected rows, making their work proportional to `B × sequence_capacity`. Their speedup comparison must be described as a whole-operator comparison between different implementations, not as evidence of superior per-byte kernel efficiency.

## 4. Prerequisites: validate the accessed storage

The current tests are useful starting points, but the following source mismatches must be resolved before timing. These are static observations and risks, not hardware-confirmed failure reports.

| Observation | Required verification before performance measurement |
| --- | --- |
| Host sets cache extent to `16B`, while tests allocate only `L` packed entries. | Inspect logical size and backing allocation separately. Use a deliberately sized `16B` physical cache buffer for a diagnostic fixture; preserve and check its unused suffix. Label this fixture adaptation explicitly. Do not assume allocator over-allocation makes the original test valid. |
| Every pool access covers 16 entries from `start_i`, even when `l_i` is one. | Check row ends, last allocation boundary, and concurrent neighboring-row accesses. Start with intervals wholly inside a row. Later test guards/padded physical rows and separately record changed row stride; a padded run does not reproduce the original geometry. |
| Request/offset copy lengths are rounded to 32 bytes. | Verify backing storage for rounded loads. Distinguish length rounding from pointer alignment. Validate target-specific `DataCopy` address rules before sweeping offset residues. |
| Offset casts process `4 × ceil(B/4)` entries, while three int32 scratch requests use `B` entries. | Inspect CANN's actual UB allocation rounding and generated accesses. This is not proof of an overflow: allocator padding may cover it. Keep nonmultiples of four out of the initial timing set until checked. |
| Host validation checks dtype, not the full shape/layout/range contract. | Verify bounds, monotonic intervals, update lengths ≤16, contiguous storage, same device, no unintended aliases, and unique selected rows in the lab's initial contract. Empty inputs and invalid cases belong in correctness investigation first. |
| Repeated rows are initially identical in the upstream tests. | Use row-distinct values, request permutations, and full-buffer comparison so wrong-row accesses cannot be hidden. |
| The second request-index dtype run reuses already modified outputs. | Reset all mutable buffers before each independent correctness case and dtype transition. Verify a fresh state transition; an idempotent/no-op second run must not pass by accident. |

For UB sizing, let `R = R32(sB)` and `O = R32(8B)`. The explicit buffer requests total approximately `R + 2O + 76B + 128` bytes per launched core: metadata, three int32 scratch arrays, full cache, and two 64-byte queue slots. The host check uses `3O + 76B + 64`. Their difference is `R - O + 64`, before runtime allocation rounding. It undercounts by 64 bytes for int64 indices at the request-size level; the int32 case can instead be conservative. Reconcile actual allocation, dtype, and rounding before probing batch sizes near the UB limit. Do not report a universal undercount or derive a safe maximum batch from this estimate alone. [Allocation sites][kernel], [host bound][host].

The acceptance gate is exact integer agreement for changed cells, preservation of untouched cells and guard regions, and clean execution under the target's available memory-checking facilities. Correct output alone cannot establish absence of an out-of-bounds read. If a case fails, exclude it from performance comparisons and keep its failure artifact.

## 5. Hypotheses and distinguishing experiments

Every entry below is a hypothesis about the **critical path**, not merely a count of extra instructions. Experiments describe future measurement work. No kernel or host variant is implemented or authorized by this document.

| ID / priority | Source evidence and hypothesis | Observation on the unchanged implementation | Evidence against the hypothesis / later discriminating change |
| --- | --- | --- | --- |
| H1 — High: host setup floor | Every API call creates and transfers tiling data. Small updates may be dominated by host allocation, pinning, transfer, and dispatch. | Capture host/API and device timelines; compare synchronized call latency, sustained call throughput, and the matching kernel task duration as `B` and `l` increase. | If kernel time dominates and API cost tracks it, deprioritize host caching. A later cached-tiling variant must preserve device/stream lifetime and demonstrate lower whole-call latency, not just a shorter trace gap. |
| H2 — High: repeated preprocessing | Each active core copies `A` bytes and computes whole-batch lengths, even when owning one row. Replication may consume MTE2, UB, or scalar/vector resources. | Compare measured GM→UB payload with `P × A + 64B`; sweep `B` below/above `C`; compare repeated-address and rotating-address runs, main-memory traffic, and L2 hit data. | High GM→UB bytes with low memory/pipe time rejects a transfer bottleneck, though redundant computation may remain. Core-count sweeps or per-core metadata slicing require a later host/kernel variant because blockDim is not an input knob here. |
| H3 — High: short-update fixed transfers | Assign reads 64 bytes and writes 64 bytes per row regardless of useful length, preserving the unused tail by read-modify-write. | At fixed `B`, row addresses, dtype, and physical layout, sweep `l = 1, 2, 4, 8, 16`. Expect nearly fixed row-copy volume while useful bytes rise. Relate latency to MTE time and scalar work. | Flat bytes but scalar/host-dominated latency would reject fixed transfers as the immediate limit. A later exact-length write must satisfy alignment and preservation requirements and improve measured latency; fewer modeled bytes alone is insufficient. |
| H4 — High: prefix scan and scalar start delay | Every core starts its packed-offset scan at request zero. Later cores scan more preceding lengths before their first row; each token then uses scalar UB access. | Plot per-core scalar cycles and finish times against owned rows, first request-list position, and assigned token count. Use equal-length, equal-row-count cases first, then the same lengths in different orders. | Similar per-core scalar cost and finish times would weaken this explanation. A later shared/passed prefix representation must include its construction and communication costs in end-to-end comparisons. |
| H5 — Medium: incomplete pipeline overlap | Queue depth is two, but each loop calls copy-in and immediate dequeue/compute before advancing. There is no explicit next-row prefetch in the source. Explicit events also fence preprocessing and retrieve's final copy. | Examine MTE2/MTE3, scalar/vector activity and queue/event stalls with several rows per core. Inspect compiler output or supported instruction traces for actual cross-row overlap and implicit synchronization. | Visible useful overlap and little waiting reject the assumption that depth-two buffering is ineffective. Do not infer absence of overlap from source ordering, or synchronization cost by adding overlapping pipe ratios. Later pipelining changes require buffer-lifetime and output-preservation checks. |
| H6 — Medium: equal rows, unequal work | Tiling divides contiguous request-list ranges by row count, not `sum(l_i)`. A tail core may have one extra row; clustered long updates may increase scalar cost on one core. | Hold the same `(request, offset, length, values)` tuples fixed and permute their list order: interleaved short/long versus clustered long-last. Keep total useful work constant. Correlate per-core duration with both token sum and prefix start. | If ordering does not affect finish skew beyond baseline variation, deprioritize token-balanced tiling. Fixed 64-byte row copies can make row-count balancing adequate; distinguish this from H4 using equal-length controls. |
| H7 — Medium, retrieve only: serial execution | Host deliberately sets retrieve to one core, followed by a full output-buffer write. | Confirm launched/active core count; measure latency growth with `B` and `L`, scalar cycles, and full-cache transfer cost. Compare assign/retrieve trends without treating their difference as a causal core-count experiment. | If host cost dominates or the one core is underutilized, multicore execution may not pay. A later multicore retrieve must write disjoint output slices; copying the whole output from multiple cores would introduce races. |
| H8 — Medium: alignment/locality and UB limits | Fixed copy sizes, rounded metadata, arbitrary pool addresses, and full-batch UB allocation may cause transaction amplification, cache conflicts, or batch-size cliffs. | After the storage gate, vary start residues modulo eight, aligned versus perturbed row strides, request locality, index dtype, and batch near observed boundaries. Compare actual traffic, memory conflicts, and per-core timing. | If traffic and duration remain stable, alignment is not the current limit. Hold useful work and allocated layout explicit; do not conflate a larger pool's address pattern with additional copied entries or extend beyond verified UB capacity. |

For H4, the scalar prefix-add count is derivable. If a core owns `n_c > 0` consecutive request-list positions beginning at `r_c`, its total `GetCacheIdx` additions are `r_c + n_c - 1`. When `B ≤ C`, total additions are `B(B-1)/2`; when `B` is divisible by `C`, the count is `B(C+1)/2 - C`. These are source-loop counts, not cycle predictions. For an illustrative `B = C = 32`, there are 496 prefix additions across cores, with the last core doing 31 before handling its only row. This gives a concrete per-core prediction independent of ragged lengths. [Packed-offset scan][kernel].

## 6. Measurement contract

The embedded timings are not suitable as final performance evidence without a separate controlled harness. Their clocks start at loop index one, but totals are divided by twenty; the first iteration is not fenced at that boundary. With asynchronous execution, the exact measured work is ambiguous rather than simply nineteen completed device operations. They also reuse the same addresses and already updated data, lack a latency distribution, and combine Python/framework work with device execution. The Torch-native reference allocates intermediates and gathers/scatters entire rows. Retrieval's reference output uses int64 before the NPU comparison converts its copy to int32, adding another implementation difference. [Assignment timing][assign-test], [retrieval timing][retrieve-test].

Future measurements will report these distinct quantities:

| Quantity | Boundary and interpretation |
| --- | --- |
| Synchronized API latency | Drain prior work; start monotonic wall clock; invoke the operator; synchronize completion; stop clock. Includes the call, tiling setup, launch, and completion-wait overhead. Establish the synchronization/timer floor separately. |
| Sustained operator throughput | Warm up, synchronize, time exactly `N` calls, synchronize, divide by `N`. Measures an amortized invocation cost; may overlap host/device work and is not isolated request latency. |
| Device-stream interval | Events around calls, when supported. Includes stream-visible tiling copies, ordering, and possible dispatch gaps; do not label it kernel-only time. |
| Kernel task duration | Identify the exact matching device task in a profiler trace. Both API modes share the `cache_loc_assign` entry point; profile assign and retrieve in separate runs and record mode. |
| Useful effective bandwidth | `8L / duration`, with the duration type in the column name. This excludes metadata and is not HBM bandwidth. |
| Copy and memory metrics | Keep source-predicted copy bytes, measured GM↔UB bytes, and measured main-memory bytes in separate columns, with counters and units preserved. |

Start with 20 warmup invocations and extend if timing has not stabilized; record the actual count. For each selected case, collect approximately 200 isolated API samples and five sustained blocks of 200 calls, across at least three independent process runs. Adjust block size if the timer resolution or run duration requires it and record the adjustment. Isolated per-call samples can support median/p95; a percentile of block averages must not be presented as per-call tail latency.

Construct tensors, initialize values, and compare outputs outside the timed region. Reset before independent correctness checks. A repeated idempotent fixture is valid only as an explicitly labeled steady-state workload. Also measure a preallocated rotating-address workload; use the footprint of **touched** cache lines, not total tensor allocation, when interpreting cache reuse. A 2000-row pool does not imply its entire contents are accessed by this kernel. Verify cache behavior through counters before describing anything as cold-cache.

Randomize case order, repeat a sentinel baseline periodically to detect drift, and keep profiler collection separate from headline latency runs. A profiled run may be perturbed or replayed. Capture dtype, shape, physical stride/padding, actual interval lengths, request ordering, seed, mutation/reset policy, timing boundary, warmup, samples, and failures.

For provenance, record source commit and dirty diff, build command/flags, submodule revisions, shared-library hash, installed package path/version, NPU model and actual vector-core count, driver/firmware/CANN/compiler/PyTorch/torch_npu/profiler versions, and device-sharing/power conditions where observable. A checked-out source commit does not prove the loaded binary was built from it; retain the build log and binary hash together.

## 7. Focused workload sequence

Do not begin with a full Cartesian sweep. Use these stages, increasing scope only when a preceding result motivates it.

| Stage | Controlled workload | Question / output |
| --- | --- | --- |
| 0 — Input contract | Fresh row-distinct fixtures; dtype variants; lengths 1, 2, 8, 16; row/packed-buffer guards; request permutation. Start with batch divisible by eight and comfortably within UB limits. | Exact outputs and access validation. Boundary, empty, duplicate, length>16, and unusual stride cases are separate contract investigations, not performance data. |
| 1 — Baseline and host floor | Assign, `B = 8, 32, 128`, sequence capacity 2048, aligned interior starts, unique rows, int64 indices, `l = 1, 2, 8, 16`. Retain only admitted cases. | Twelve primary points for API/kernel separation, source bytes, and update-length sensitivity. No assumption that 32 equals the device core count. |
| 2 — Core transition and prefix work | Query `C`; choose valid batches below, at, and above `C`, including multiples of `C` where possible. Keep `l = 1` and 16. | Per-core scalar work, prefix-start correlation, initialization of inactive cores, and row-count stair steps. Exact `C±1` cases wait for padded metadata/scratch validation. |
| 3 — Load imbalance | Same multiset of short/long request tuples, interleaved versus clustered; at least two rows/core; identical total tokens and index set. | Separate token imbalance from prefix-position bias and row-locality effects. |
| 4 — Alignment and reuse | Lengths 1 and 16; admitted start residues; row-stride and index-locality controls; repeated versus rotating buffers. Then capacities 128, 2048, 16384 at fixed `B` and `L`. | Measured copy/transaction amplification, cache behavior, and attribution of capacity sensitivity. |
| 5 — Retrieve | Repeat a small representative subset, initially `B = 8, 32, 128`, `l = 1` and 16. | One-core scaling, preserved cache tail, final-write cost. Add ragged cases only after correctness checks. |
| 6 — Original test geometry | Batch 300 and original capacities, only after UB and backing-store checks. | Reconcile the upstream example with the controlled baseline. Preserve original and adapted-layout labels separately. |

No core-count sweep is available through the current API: the host fixes blockDim by mode. No cached-tiling, exact-length copy, prefix-sum, or pipelined variant belongs in the unchanged-code baseline. Such variants would be proposed after the strongest hypothesis has measurement support.

## 8. Profiling evidence and attribution

Use the installed tool's supported equivalents of operator basic information, per-vector-core duration/scalar/vector/MTE activity, GM↔UB/main-memory traffic, L2 hit data, UB traffic, and resource-conflict metrics. Store raw files and tool version; use N/A for unavailable metrics. Current Ascend documentation describes these separately, which supports keeping memory levels distinct. [Profiler data reference][prof-data].

Inspect instruction/pipeline timelines only when the target and tool support them. Some timeline modes are hardware-specific; otherwise compiler output or simulator traces can test ordering hypotheses. Simulator timing is not measured NPU latency. Code-region instrumentation itself would be a later code change, not part of this phase. [Profiler usage and supported modes][prof-guide].

A high scalar or MTE percentage is evidence to investigate, not proof of the limiting resource. Pipe activities can overlap, ratios can use different denominators, and low bandwidth can mean insufficient work rather than an inefficient memory system. Relate absolute per-core times, task duration, traffic, and the controlled input change before identifying a bottleneck. If a mechanism cannot be isolated without a variant, mark it unresolved rather than causal.

The planned plots are: latency versus batch and update length; modeled/measured/useful byte comparison; per-core finish/scalar time versus prefix start and token sum; repeated versus rotating-address latency; alignment residue versus transaction volume. A FLOP roofline is not the primary view for this integer metadata kernel.

## 9. Decision gates and deliverables

1. **Contract established:** exact-output and memory-access checks, documented admitted/rejected cases, and any layout adaptation.
2. **Baseline credible:** reproducible unprofiled timings, precise measurement boundaries, environment/build linkage, and raw sample retention.
3. **Hypothesis assessed:** for every tested hypothesis, record the expected signature, observed result, counterevidence, confounders, and confidence. A flat result is a valid outcome.
4. **One follow-up proposed:** choose the smallest variant that distinguishes or addresses the strongest measured limit. Include expected regressions and acceptance criteria before changing code.
5. **Later variant accepted only with evidence:** exact correctness, repeated latency shift beyond baseline variability across independent runs, mechanism-consistent profile changes, and a complete regression table. Fewer instructions or fewer modeled bytes alone is not success.

Results should contain a run manifest, admitted-case definitions, raw timing samples, profiler artifact references/checksums, modeled traffic, measured traffic or N/A, per-core summaries, and a findings note linked from `ADR.md`. Keep large traces outside Git. Do not extrapolate this microbenchmark's speedup to end-to-end serving throughput without measuring its share of that execution path.

At the time of writing, only source review and arithmetic checks have been performed. The Git commit/tree metadata is available locally, but the partial clone's working-tree checkout is incomplete; approved GitHub read APIs supplied the reviewed source text. No NPU test, simulator run, benchmark, or optimization has been performed.

[kernel]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/csrc/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp
[host]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/csrc/cache_location_assign/op_host/cache_loc_assign.cpp
[tiling]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/csrc/cache_location_assign/op_host/tiling/cache_loc_assign.h
[common]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/csrc/utils/common.h
[helper]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/csrc/utils/torch_helper.h
[bindings]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/csrc/pytorch_extensions.cpp
[loader]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/python/sgl_kernel_npu/sgl_kernel_npu/__init__.py
[assign-test]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/tests/python/sgl_kernel_npu/test_cache_assign.py
[retrieve-test]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/tests/python/sgl_kernel_npu/test_cache_update.py
[runner]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/scripts/run_kernel_tests.sh
[other-test]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/tests/python/sgl_kernel_npu/test_inplace_assign_cache.py
[slot-bench]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/benchmark/sparsity_driven_kv_offload/bench_slot_map_lookup.py
[copy-bench]: https://github.com/sgl-project/sgl-kernel-npu/blob/cdcb9d8b719a6100dadc3544c8cd33ff53bf668a/benchmark/sparsity_driven_kv_offload/bench_unidex_copy.py
[prof-data]: https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md
[prof-guide]: https://www.hiascend.com/document/detail/en/mindstudio/2610/optools/Operatordevelopmenttools/docs/en/user_guide/msopprof_user_guide.md
