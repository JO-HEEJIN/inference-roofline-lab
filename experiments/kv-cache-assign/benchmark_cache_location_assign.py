#!/usr/bin/env python3
"""Controlled baseline and core-transition harness for ``cache_location_assign``.

This file deliberately measures the unchanged upstream operator only.  It does
not implement a replacement kernel, cache tiling, or an optimization variant.

The physical fixture is wider than the logical sequence capacity and has an
oversized packed-cache backing store.  The upstream kernel copies fixed 16-value
segments and batch-aligned metadata; the backing storage makes those reads safe
without relying on allocator over-allocation.  Timed workload metadata keeps the
logical sequence capacity separate from physical row width.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from shutil import which
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MAX_STEP = 16
LOGICAL_SEQUENCE_CAPACITY = 2048
STAGE01 = "stage01"
STAGE2 = "stage2"
DEFAULT_BATCH_SIZES = (8, 32, 128)
DEFAULT_UPDATE_LENGTHS = (1, 2, 8, 16)
STAGE2_UPDATE_LENGTHS = (1, 16)
OUTPUT_FIELDS = (
    "run_id",
    "record_type",
    "experiment_stage",
    "core_transition_relation",
    "operation",
    "batch_size",
    "sequence_capacity",
    "physical_row_width",
    "update_length",
    "request_index_dtype",
    "seed",
    "warmup_count",
    "sample_count",
    "timing_mode",
    "raw_samples_path",
    "median_us",
    "p95_us",
    "sustained_calls_per_second",
    "total_useful_entries_L",
    "useful_bytes_8L",
    "modeled_source_copy_bytes",
    "modeled_active_assign_cores",
    "useful_effective_bandwidth_gbps",
    "correctness_status",
    "failure_reason",
    "source_commit",
    "source_dirty_tree_status",
    "lab_dirty_tree_status",
    "device_metadata",
    "cann_metadata",
    "torch_version",
    "torch_npu_version",
)


class NpuUnavailable(RuntimeError):
    """Raised before measurement when this process cannot use the Ascend runtime."""


@dataclass(frozen=True)
class CaseSpec:
    operation: str
    batch_size: int
    sequence_capacity: int
    update_length: int
    request_index_dtype: str
    seed: int
    guard_width: int = MAX_STEP
    stage: str = STAGE01

    def validate(self) -> None:
        if self.operation != "assign":
            raise ValueError("the controlled harness permits only operation='assign'")
        if self.stage not in (STAGE01, STAGE2):
            raise ValueError(f"unknown experiment stage: {self.stage}")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.stage == STAGE01 and self.batch_size not in DEFAULT_BATCH_SIZES:
            raise ValueError(f"Stage 0/1 batch must be one of {DEFAULT_BATCH_SIZES}")
        if self.sequence_capacity != LOGICAL_SEQUENCE_CAPACITY:
            raise ValueError("Stage 1 permits only sequence capacity 2048")
        allowed_updates = DEFAULT_UPDATE_LENGTHS if self.stage == STAGE01 else STAGE2_UPDATE_LENGTHS
        if self.update_length not in allowed_updates:
            raise ValueError(f"{self.stage} update length must be one of {allowed_updates}")
        if self.request_index_dtype not in ("int32", "int64"):
            raise ValueError("request-index dtype must be int32 or int64")
        if self.guard_width < MAX_STEP:
            raise ValueError("guard width must cover the fixed 16-element transfer")
        if self.sequence_capacity < 2 * MAX_STEP:
            raise ValueError("sequence capacity is too small for an interior transfer")

    @property
    def total_useful_entries(self) -> int:
        return self.batch_size * self.update_length

    @property
    def useful_bytes(self) -> int:
        # Each selected int32 location is read once and written once.
        return 8 * self.total_useful_entries

    @property
    def physical_row_width(self) -> int:
        return self.sequence_capacity + self.guard_width

    @property
    def case_id(self) -> str:
        return (
            f"assign-{self.stage}-b{self.batch_size}-s{self.sequence_capacity}"
            f"-u{self.update_length}-{self.request_index_dtype}-seed{self.seed}"
        )


@dataclass(frozen=True)
class SafeStoragePlan:
    """Logical views and backing sizes needed by static source accesses."""

    logical_rows: int
    logical_sequence_capacity: int
    physical_row_width: int
    request_logical_count: int
    request_backing_count: int
    offset_logical_count: int
    offset_backing_count: int
    cache_logical_count: int
    cache_padding_count: int
    cache_guard_count: int


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def storage_plan(spec: CaseSpec) -> SafeStoragePlan:
    """Plan safe backing for the currently inspected upstream implementation."""
    spec.validate()
    request_alignment = 8 if spec.request_index_dtype == "int32" else 4
    cache_logical_count = spec.batch_size * MAX_STEP
    return SafeStoragePlan(
        logical_rows=spec.batch_size + 3,
        logical_sequence_capacity=spec.sequence_capacity,
        physical_row_width=spec.physical_row_width,
        request_logical_count=spec.batch_size,
        request_backing_count=round_up(spec.batch_size, request_alignment),
        offset_logical_count=spec.batch_size,
        offset_backing_count=round_up(spec.batch_size, 4),
        cache_logical_count=cache_logical_count,
        cache_padding_count=cache_logical_count - spec.total_useful_entries,
        cache_guard_count=MAX_STEP,
    )


def aligned_interior_starts(spec: CaseSpec) -> tuple[int, ...]:
    """Return deterministic, nonnegative, 8-element-aligned safe starts."""
    spec.validate()
    # Leave a full fixed transfer at both logical row ends.  The set of starts
    # may repeat, but requests always select distinct rows.
    lower = MAX_STEP
    upper = spec.sequence_capacity - MAX_STEP
    positions = tuple(range(lower, upper + 1, 8))
    if not positions:
        raise ValueError("no aligned interior starts exist")
    return tuple(positions[(spec.seed + 17 * index) % len(positions)] for index in range(spec.batch_size))


def modeled_source_copy_bytes(spec: CaseSpec, active_assign_cores: int) -> int:
    """Source-level copy-payload model, not a measured memory-traffic counter."""
    spec.validate()
    if not 1 <= active_assign_cores <= spec.batch_size:
        raise ValueError("active assign cores must be between 1 and batch size")
    request_bytes = 4 if spec.request_index_dtype == "int32" else 8
    metadata_and_cache_per_core = (
        round_up(spec.batch_size * request_bytes, 32)
        + 2 * round_up(spec.batch_size * 8, 32)
        + spec.batch_size * MAX_STEP * 4
    )
    row_transfers = 2 * spec.batch_size * MAX_STEP * 4
    return active_assign_cores * metadata_and_cache_per_core + row_transfers


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires a nonempty sequence")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def git_output(args: Sequence[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            list(args), cwd=cwd, capture_output=True, check=False, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"
    if completed.returncode != 0:
        return f"unavailable: {completed.stderr.strip() or completed.returncode}"
    return completed.stdout.strip()


def git_dirty_status(cwd: Path) -> str:
    status = git_output(("git", "status", "--porcelain"), cwd)
    if status == "":
        return "clean"
    return status


def read_upstream_metadata(workspace: Path) -> dict[str, Any]:
    manifest_path = workspace / "experiments" / "kv-cache-assign" / "upstream.json"
    source = json.loads(manifest_path.read_text())
    source_root = workspace / source["checkout"]
    return {
        "source_commit": source["commit"],
        "source_repository": source["repository"],
        "source_build_target_example": source.get("build_target_example"),
        "source_dirty_tree_status": git_dirty_status(source_root),
        "lab_dirty_tree_status": git_dirty_status(workspace),
    }


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_metadata(workspace: Path) -> dict[str, Any]:
    cann_paths = {
        key: os.environ.get(key)
        for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH")
        if os.environ.get(key)
    }
    cann_versions = []
    for location in cann_paths.values():
        for relative in ("version.info", "latest/version.info"):
            version_path = Path(location) / relative
            if version_path.is_file():
                cann_versions.append({"path": str(version_path), "contents": version_path.read_text(errors="replace").strip()})
    npu_smi = which("npu-smi")
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": None,
        "torch_npu_version": None,
        "device": None,
        "cann": {"environment_paths": cann_paths, "version_files": cann_versions},
        "harness_sha256": sha256_file(Path(__file__)),
        "upstream_manifest_sha256": sha256_file(
            workspace / "experiments" / "kv-cache-assign" / "upstream.json"
        ),
    }
    if npu_smi:
        metadata["npu_smi_path"] = npu_smi
        metadata["npu_smi_version"] = git_output((npu_smi, "--version"), workspace)
    try:
        import torch  # type: ignore

        metadata["torch_version"] = torch.__version__
    except Exception as error:  # pragma: no cover - environment dependent
        metadata["torch_import_error"] = repr(error)
    try:
        import torch_npu  # type: ignore

        metadata["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
    except Exception as error:
        metadata["torch_npu_import_error"] = repr(error)
    return metadata


class ResultWriter:
    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.raw_dir = output_dir / "raw-samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "results.jsonl"
        self.csv_path = output_dir / "results.csv"
        self.jsonl_path.touch()
        self._csv = self.csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._csv, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        self._csv_writer.writeheader()

    def close(self) -> None:
        self._csv.close()

    def write_raw(self, case_id: str, timing_mode: str, payload: dict[str, Any]) -> str:
        path = self.raw_dir / f"{case_id}-{timing_mode}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return str(path.relative_to(self.output_dir))

    def record(self, record: dict[str, Any]) -> None:
        normalized = {field: record.get(field) for field in OUTPUT_FIELDS}
        normalized["run_id"] = self.run_id
        with self.jsonl_path.open("a") as handle:
            handle.write(json.dumps(normalized, sort_keys=True) + "\n")
        csv_record = dict(normalized)
        for key in ("device_metadata", "cann_metadata"):
            if isinstance(csv_record.get(key), (dict, list)):
                csv_record[key] = json.dumps(csv_record[key], sort_keys=True)
        self._csv_writer.writerow(csv_record)
        self._csv.flush()


class NpuContext:
    def __init__(self, static: dict[str, Any], active_core_override: int | None) -> None:
        try:
            import torch  # type: ignore
            import torch_npu  # type: ignore  # noqa: F401
            import sgl_kernel_npu  # type: ignore
        except Exception as error:
            raise NpuUnavailable(f"Ascend runtime/import unavailable: {error!r}") from error
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise NpuUnavailable("torch_npu is installed but no available NPU was reported")
        if not hasattr(torch.ops.npu, "cache_loc_assign"):
            raise NpuUnavailable("torch.ops.npu.cache_loc_assign is not registered")
        self.torch = torch
        self.device = torch.device("npu")
        self.metadata = static
        package_path = Path(sgl_kernel_npu.__file__).resolve()
        library_path = package_path.parent / "lib" / "libsgl_kernel_npu.so"
        self.metadata["loaded_sgl_kernel_npu"] = {
            "package_path": str(package_path),
            "library_path": str(library_path),
            "library_sha256": sha256_file(library_path),
        }
        self.metadata["device"] = self._device_metadata()
        self.active_assign_cores = active_core_override or self._discover_active_cores()
        if active_core_override is not None:
            self.metadata["assign_active_cores_source"] = "command_line.--assign-active-cores"
        if self.active_assign_cores is None:
            raise NpuUnavailable(
                "could not discover Ascend vector-core count; pass --assign-active-cores "
                "with the actual host-tiling blockDim"
            )
        if self.active_assign_cores <= 0:
            raise NpuUnavailable("assign active core count must be positive")

    def _device_metadata(self) -> dict[str, Any]:
        torch = self.torch
        details: dict[str, Any] = {"device_index": torch.npu.current_device()}
        for method_name in ("get_device_name", "get_device_properties"):
            method = getattr(torch.npu, method_name, None)
            if method is None:
                continue
            try:
                value = method(torch.npu.current_device())
                if method_name == "get_device_properties":
                    attributes = {}
                    for name in dir(value):
                        if name.startswith("_"):
                            continue
                        candidate = getattr(value, name)
                        if isinstance(candidate, (str, int, float, bool)):
                            attributes[name] = candidate
                    details[method_name] = attributes
                else:
                    details[method_name] = str(value)
            except Exception as error:  # pragma: no cover - target dependent
                details[f"{method_name}_error"] = repr(error)
        return details

    def _discover_active_cores(self) -> int | None:
        properties = self.metadata.get("device", {}).get("get_device_properties", {})
        for field in ("aivcore_num", "aiv_core_num", "vector_core_num"):
            value = properties.get(field)
            if isinstance(value, int) and value > 0:
                self.metadata["assign_active_cores_source"] = f"device_properties.{field}"
                return value
        return None

    def synchronize(self) -> None:
        self.torch.npu.synchronize()


class AssignFixture:
    """Fresh, guarded fixture for a single assign case."""

    def __init__(self, context: NpuContext, spec: CaseSpec) -> None:
        spec.validate()
        self.context = context
        self.spec = spec
        self.plan = storage_plan(spec)
        torch = context.torch
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        self.starts = aligned_interior_starts(spec)
        self.ends = tuple(start + spec.update_length for start in self.starts)
        self.request_rows_cpu = torch.randperm(
            self.plan.logical_rows, generator=generator, dtype=torch.int64
        )[: spec.batch_size]
        if torch.unique(self.request_rows_cpu).numel() != spec.batch_size:
            raise AssertionError("fixture generation did not create unique request rows")
        if any(start < 0 or end > spec.sequence_capacity for start, end in zip(self.starts, self.ends)):
            raise AssertionError("fixture start/end is not a nonnegative interior range")

        # Every physical row has distinct values. The final 16 columns are the
        # row guard: they are included in the physical row stride but excluded
        # from the logical 2048-element capacity.
        row_ids = torch.arange(self.plan.logical_rows, dtype=torch.int32).unsqueeze(1)
        columns = torch.arange(self.plan.physical_row_width, dtype=torch.int32).unsqueeze(0)
        pool_cpu = row_ids * self.plan.physical_row_width + columns + 1
        pool_cpu[:, spec.sequence_capacity :] = -2_000_000 - (
            row_ids * spec.guard_width + torch.arange(spec.guard_width, dtype=torch.int32))

        cache_values = torch.full(
            (self.plan.cache_logical_count + self.plan.cache_guard_count,), -3_000_000, dtype=torch.int32
        )
        cache_values[: spec.total_useful_entries] = torch.arange(
            100_000, 100_000 + spec.total_useful_entries, dtype=torch.int32
        )
        cache_values[self.plan.cache_logical_count :] = -4_000_000 - torch.arange(
            self.plan.cache_guard_count, dtype=torch.int32
        )

        request_dtype = torch.int32 if spec.request_index_dtype == "int32" else torch.int64
        request_physical = torch.full((self.plan.request_backing_count,), -5, dtype=request_dtype)
        request_physical[: spec.batch_size] = self.request_rows_cpu.to(request_dtype)
        start_physical = torch.full((self.plan.offset_backing_count,), -6, dtype=torch.int64)
        end_physical = torch.full((self.plan.offset_backing_count,), -7, dtype=torch.int64)
        start_physical[: spec.batch_size] = torch.tensor(self.starts, dtype=torch.int64)
        end_physical[: spec.batch_size] = torch.tensor(self.ends, dtype=torch.int64)

        self.initial_pool_cpu = pool_cpu.clone()
        self.initial_cache_cpu = cache_values.clone()
        self.initial_request_cpu = request_physical.clone()
        self.initial_start_cpu = start_physical.clone()
        self.initial_end_cpu = end_physical.clone()

        self.pool_physical = pool_cpu.to(context.device)
        self.cache_physical = cache_values.to(context.device)
        self.request_physical = request_physical.to(context.device)
        self.start_physical = start_physical.to(context.device)
        self.end_physical = end_physical.to(context.device)
        self.request_indices = self.request_physical.narrow(0, 0, spec.batch_size)
        self.start_offsets = self.start_physical.narrow(0, 0, spec.batch_size)
        self.end_offsets = self.end_physical.narrow(0, 0, spec.batch_size)
        self.cache_locations = self.cache_physical.narrow(0, 0, self.plan.cache_logical_count)
        self.context.synchronize()

    def reset(self) -> None:
        """Reset all mutable/device-visible backing outside a timed interval."""
        self.pool_physical.copy_(self.initial_pool_cpu.to(self.context.device))
        self.cache_physical.copy_(self.initial_cache_cpu.to(self.context.device))
        self.request_physical.copy_(self.initial_request_cpu.to(self.context.device))
        self.start_physical.copy_(self.initial_start_cpu.to(self.context.device))
        self.end_physical.copy_(self.initial_end_cpu.to(self.context.device))
        self.context.synchronize()

    def invoke(self) -> None:
        self.context.torch.ops.npu.cache_loc_assign(
            self.request_indices,
            self.pool_physical,
            self.start_offsets,
            self.end_offsets,
            self.cache_locations,
        )

    def verify(self) -> dict[str, Any]:
        self.reset()
        self.invoke()
        self.context.synchronize()
        torch = self.context.torch
        actual_pool = self.pool_physical.cpu()
        actual_cache = self.cache_physical.cpu()
        actual_requests = self.request_physical.cpu()
        actual_starts = self.start_physical.cpu()
        actual_ends = self.end_physical.cpu()
        expected_pool = self.initial_pool_cpu.clone()
        changed_mask = torch.zeros_like(expected_pool, dtype=torch.bool)
        cursor = 0
        for row, start, end in zip(self.request_rows_cpu.tolist(), self.starts, self.ends):
            expected_pool[row, start:end] = self.initial_cache_cpu[cursor : cursor + (end - start)]
            changed_mask[row, start:end] = True
            cursor += end - start
        changed_expected = expected_pool[changed_mask]
        changed_actual = actual_pool[changed_mask]
        exact_changed = torch.equal(changed_actual, changed_expected)
        untouched = torch.equal(actual_pool[~changed_mask], self.initial_pool_cpu[~changed_mask])
        token_guards = torch.equal(
            actual_pool[:, self.spec.sequence_capacity :],
            self.initial_pool_cpu[:, self.spec.sequence_capacity :],
        )
        # The logical packed-cache padding starts immediately after L.  It is
        # deliberately part of the view passed to upstream because that source
        # copies B * MAX_STEP entries before applying the useful updates.
        cache_padding = torch.equal(
            actual_cache[self.spec.total_useful_entries : self.plan.cache_logical_count],
            self.initial_cache_cpu[self.spec.total_useful_entries : self.plan.cache_logical_count],
        )
        cache_input_preserved = torch.equal(actual_cache, self.initial_cache_cpu)
        cache_guard = torch.equal(
            actual_cache[self.plan.cache_logical_count :], self.initial_cache_cpu[self.plan.cache_logical_count :]
        )
        metadata_guards = (
            torch.equal(actual_requests, self.initial_request_cpu)
            and torch.equal(actual_starts, self.initial_start_cpu)
            and torch.equal(actual_ends, self.initial_end_cpu)
        )
        checks = {
            "exact_changed_cells": bool(exact_changed),
            "untouched_cells_preserved": bool(untouched),
            "token_guard_preserved": bool(token_guards),
            "packed_cache_padding_preserved": bool(cache_padding),
            "packed_cache_guard_preserved": bool(cache_guard),
            "packed_cache_input_preserved": bool(cache_input_preserved),
            "metadata_and_alignment_guards_preserved": bool(metadata_guards),
            "unique_request_rows": bool(torch.unique(self.request_rows_cpu).numel() == self.spec.batch_size),
            "nonnegative_interior_offsets": all(
                start >= 0 and end <= self.spec.sequence_capacity for start, end in zip(self.starts, self.ends)
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "status": "passed" if not failed else "failed",
            "failure_reason": None if not failed else "; ".join(failed),
            "checks": checks,
            "physical_storage": asdict(self.plan),
            "starts": list(self.starts),
            "ends": list(self.ends),
            "request_rows": self.request_rows_cpu.tolist(),
        }


def benchmark_isolated_api_latency(fixture: AssignFixture, warmup: int, samples: int) -> list[float]:
    """Synchronized per-call API latency in microseconds; resets happen outside timing."""
    fixture.reset()
    for _ in range(warmup):
        fixture.invoke()
    fixture.context.synchronize()
    timings_us = []
    for _ in range(samples):
        fixture.reset()
        start_ns = time.perf_counter_ns()
        fixture.invoke()
        fixture.context.synchronize()
        timings_us.append((time.perf_counter_ns() - start_ns) / 1_000.0)
    return timings_us


def benchmark_sustained_blocks(
    fixture: AssignFixture, warmup: int, blocks: int, calls_per_block: int
) -> list[dict[str, float]]:
    """Amortized throughput blocks; values become idempotent after the first call."""
    results = []
    for block in range(blocks):
        fixture.reset()
        for _ in range(warmup):
            fixture.invoke()
        fixture.context.synchronize()
        start_ns = time.perf_counter_ns()
        for _ in range(calls_per_block):
            fixture.invoke()
        fixture.context.synchronize()
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        results.append(
            {
                "block": float(block),
                "elapsed_us": elapsed_us,
                "mean_call_us": elapsed_us / calls_per_block,
                "calls_per_second": calls_per_block * 1_000_000.0 / elapsed_us,
            }
        )
    return results


def base_record(
    spec: CaseSpec,
    provenance: dict[str, Any],
    metadata: dict[str, Any],
    warmup: int,
    correctness_status: str,
    failure_reason: str | None,
    active_assign_cores: int | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "measurement",
        "experiment_stage": spec.stage,
        "core_transition_relation": core_transition_relation(spec, active_assign_cores),
        "operation": spec.operation,
        "batch_size": spec.batch_size,
        "sequence_capacity": spec.sequence_capacity,
        "physical_row_width": spec.physical_row_width,
        "update_length": spec.update_length,
        "request_index_dtype": spec.request_index_dtype,
        "seed": spec.seed,
        "warmup_count": warmup,
        "sample_count": None,
        "timing_mode": None,
        "raw_samples_path": None,
        "median_us": None,
        "p95_us": None,
        "sustained_calls_per_second": None,
        "total_useful_entries_L": spec.total_useful_entries,
        "useful_bytes_8L": spec.useful_bytes,
        "modeled_source_copy_bytes": None,
        "modeled_active_assign_cores": None,
        "useful_effective_bandwidth_gbps": None,
        "correctness_status": correctness_status,
        "failure_reason": failure_reason,
        "source_commit": provenance["source_commit"],
        "source_dirty_tree_status": provenance["source_dirty_tree_status"],
        "lab_dirty_tree_status": provenance["lab_dirty_tree_status"],
        "device_metadata": metadata.get("device"),
        "cann_metadata": metadata.get("cann"),
        "torch_version": metadata.get("torch_version"),
        "torch_npu_version": metadata.get("torch_npu_version"),
    }


def case_specs(seed: int, dtypes: Iterable[str]) -> list[CaseSpec]:
    """Return the fixed Stage 0/1 matrix (kept as the public baseline helper)."""
    return [
        CaseSpec("assign", batch, LOGICAL_SEQUENCE_CAPACITY, update, dtype, seed, stage=STAGE01)
        for dtype in dtypes
        for batch in DEFAULT_BATCH_SIZES
        for update in DEFAULT_UPDATE_LENGTHS
    ]


def stage2_batch_sizes(active_assign_cores: int) -> tuple[int, ...]:
    """Choose below/at/above-core batch points without probing C±1 yet.

    Stage 2 is intentionally coarse.  The plan reserves C±1 for a later
    metadata/scratch-access investigation, so this emits only multiples or
    fractions of the discovered host blockDim.
    """
    if active_assign_cores < 2:
        raise ValueError("Stage 2 needs at least two active assign cores")
    return tuple(sorted({max(1, active_assign_cores // 2), active_assign_cores, 2 * active_assign_cores}))


def stage2_case_specs(seed: int, active_assign_cores: int) -> list[CaseSpec]:
    return [
        CaseSpec("assign", batch, LOGICAL_SEQUENCE_CAPACITY, update, "int64", seed, stage=STAGE2)
        for batch in stage2_batch_sizes(active_assign_cores)
        for update in STAGE2_UPDATE_LENGTHS
    ]


def selected_case_specs(stage: str, seed: int, active_assign_cores: int | None = None) -> list[CaseSpec]:
    if stage == STAGE01:
        return case_specs(seed, ("int32", "int64"))
    if stage == STAGE2:
        if active_assign_cores is None:
            raise ValueError("Stage 2 needs the discovered core count or --assign-active-cores")
        return stage2_case_specs(seed, active_assign_cores)
    raise ValueError(f"unknown stage: {stage}")


def core_transition_relation(spec: CaseSpec, active_assign_cores: int | None) -> str | None:
    if spec.stage != STAGE2 or active_assign_cores is None:
        return None
    if spec.batch_size < active_assign_cores:
        return "below_active_cores"
    if spec.batch_size == active_assign_cores:
        return "equal_active_cores"
    return "above_active_cores"


def manifest_case_definitions(
    specs: Iterable[CaseSpec], active_assign_cores: int | None
) -> list[dict[str, Any]]:
    return [
        asdict(spec)
        | {
            "case_id": spec.case_id,
            "core_transition_relation": core_transition_relation(spec, active_assign_cores),
            "storage": asdict(storage_plan(spec)),
        }
        for spec in specs
    ]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty directory for one benchmark run")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--isolated-samples", type=int, default=200)
    parser.add_argument("--sustained-blocks", type=int, default=5)
    parser.add_argument("--sustained-calls", type=int, default=200)
    parser.add_argument(
        "--stage",
        choices=(STAGE01, STAGE2),
        default=STAGE01,
        help="stage01: fixed baseline; stage2: C/2, C, and 2C core-transition cases",
    )
    parser.add_argument(
        "--assign-active-cores",
        type=int,
        help="actual blockDim for assign when torch.npu device properties do not expose vector-core count",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the selected matrix and write a manifest without importing or executing NPU code",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.isolated_samples <= 0:
        raise ValueError("--isolated-samples must be positive")
    if args.sustained_blocks <= 0 or args.sustained_calls <= 0:
        raise ValueError("sustained block and call counts must be positive")
    if args.assign_active_cores is not None and args.assign_active_cores <= 0:
        raise ValueError("--assign-active-cores must be positive")
    if args.dry_run and getattr(args, "stage", STAGE01) == STAGE2 and args.assign_active_cores is None:
        raise ValueError("Stage 2 dry run requires --assign-active-cores to construct the C-relative matrix")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    stage = getattr(args, "stage", STAGE01)
    workspace = Path(__file__).resolve().parents[2]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("--output-dir must be new or empty so artifacts cannot be mixed across runs")
    provenance = read_upstream_metadata(workspace)
    metadata = static_metadata(workspace)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    writer = ResultWriter(args.output_dir, run_id)
    manifest_path = args.output_dir / "run-manifest.json"
    static_cores = args.assign_active_cores if stage == STAGE2 else None
    all_specs = selected_case_specs(stage, args.seed, static_cores) if (stage == STAGE01 or static_cores) else []
    scope = (
        "Stage 0 correctness plus Stage 1 int64 assign baseline only"
        if stage == STAGE01
        else "Stage 2 int64 assign core-transition baseline: C/2, C, and 2C; lengths 1 and 16"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "scope": scope,
        "benchmark_parameters": {
            "operation": "assign",
            "experiment_stage": stage,
            "batch_sizes": [spec.batch_size for spec in all_specs],
            "sequence_capacity": LOGICAL_SEQUENCE_CAPACITY,
            "update_lengths": list(DEFAULT_UPDATE_LENGTHS if stage == STAGE01 else STAGE2_UPDATE_LENGTHS),
            "correctness_request_index_dtypes": ["int32", "int64"] if stage == STAGE01 else ["int64"],
            "timed_request_index_dtype": "int64",
            "warmup_count": args.warmup,
            "isolated_samples": args.isolated_samples,
            "sustained_blocks": args.sustained_blocks,
            "sustained_calls_per_block": args.sustained_calls,
            "seed": args.seed,
            "physical_backing_policy": "guarded rows and aligned metadata backing; see storage_plan",
            "stage2_batch_policy": "C/2, C, 2C; excludes C-1 and C+1" if stage == STAGE2 else None,
        },
        "source": provenance,
        "environment": metadata,
        "reproduction": {
            "invoked_argv": sys.argv,
            "upstream_build_target_example": provenance["source_build_target_example"],
            "benchmark_script": str(Path(__file__).resolve()),
        },
        "case_definitions": manifest_case_definitions(all_specs, static_cores),
        "result_files": {"jsonl": "results.jsonl", "csv": "results.csv", "raw_samples": "raw-samples/"},
    }
    try:
        if args.dry_run:
            manifest["status"] = "dry-run-static-validation"
            manifest["note"] = "No NPU imports, correctness calls, or timings were attempted."
            return 0
        try:
            context = NpuContext(metadata, args.assign_active_cores)
        except NpuUnavailable as error:
            manifest["status"] = "unavailable-no-npu-results"
            manifest["failure_reason"] = str(error)
            print(f"NPU benchmark not run: {error}", file=sys.stderr)
            return 2
        manifest["environment"] = context.metadata
        manifest["assign_active_cores"] = context.active_assign_cores
        all_specs = selected_case_specs(stage, args.seed, context.active_assign_cores)
        manifest["benchmark_parameters"]["batch_sizes"] = [spec.batch_size for spec in all_specs]
        manifest["case_definitions"] = manifest_case_definitions(all_specs, context.active_assign_cores)
        manifest["status"] = "running"
        write_manifest(manifest_path, manifest)
        correctness: dict[tuple[int, int, str], dict[str, Any]] = {}
        for spec in all_specs:
            base = base_record(
                spec, provenance, context.metadata, args.warmup, "failed", None, context.active_assign_cores
            )
            base["record_type"] = "correctness"
            base["timing_mode"] = "correctness"
            try:
                fixture = AssignFixture(context, spec)
                verification = fixture.verify()
                status = verification["status"]
                base["correctness_status"] = status
                base["failure_reason"] = verification["failure_reason"]
                raw_path = writer.write_raw(spec.case_id, "correctness", verification)
                base["raw_samples_path"] = raw_path
            except Exception as error:
                base["failure_reason"] = f"{type(error).__name__}: {error}"
                raw_path = writer.write_raw(
                    spec.case_id,
                    "correctness-failure",
                    {"exception": traceback.format_exc(), "spec": asdict(spec)},
                )
                base["raw_samples_path"] = raw_path
            writer.record(base)
            correctness[(spec.batch_size, spec.update_length, spec.request_index_dtype)] = base

        for spec in (candidate for candidate in all_specs if candidate.request_index_dtype == "int64"):
            status = correctness[(spec.batch_size, spec.update_length, spec.request_index_dtype)]
            if status["correctness_status"] != "passed":
                continue
            active_cores = min(context.active_assign_cores, spec.batch_size)
            modeled_bytes = modeled_source_copy_bytes(spec, active_cores)
            try:
                isolated_fixture = AssignFixture(context, spec)
                isolated = benchmark_isolated_api_latency(
                    isolated_fixture, args.warmup, args.isolated_samples
                )
                isolated_median = statistics.median(isolated)
                isolated_record = base_record(
                    spec, provenance, context.metadata, args.warmup, "passed", None, context.active_assign_cores
                )
                isolated_record.update(
                    {
                        "timing_mode": "isolated_synchronized_api_latency",
                        "sample_count": len(isolated),
                        "raw_samples_path": writer.write_raw(
                            spec.case_id,
                            "isolated-synchronized-api-latency",
                            {
                                "unit": "microseconds",
                                "timing_boundary": "reset/synchronize outside interval; invoke through completion synchronize",
                                "samples_us": isolated,
                            },
                        ),
                        "median_us": isolated_median,
                        "p95_us": percentile(isolated, 0.95),
                        "modeled_source_copy_bytes": modeled_bytes,
                        "modeled_active_assign_cores": active_cores,
                        "useful_effective_bandwidth_gbps": spec.useful_bytes / isolated_median / 1_000.0,
                    }
                )
                writer.record(isolated_record)

                sustained_fixture = AssignFixture(context, spec)
                blocks = benchmark_sustained_blocks(
                    sustained_fixture, args.warmup, args.sustained_blocks, args.sustained_calls
                )
                mean_call_us = [block["mean_call_us"] for block in blocks]
                calls_per_second = [block["calls_per_second"] for block in blocks]
                sustained_rate = statistics.median(calls_per_second)
                sustained_record = base_record(
                    spec, provenance, context.metadata, args.warmup, "passed", None, context.active_assign_cores
                )
                sustained_record.update(
                    {
                        "timing_mode": "sustained_synchronized_blocks",
                        "sample_count": len(blocks),
                        "raw_samples_path": writer.write_raw(
                            spec.case_id,
                            "sustained-synchronized-blocks",
                            {
                                "calls_per_block": args.sustained_calls,
                                "blocks": blocks,
                                "timing_boundary": "warmup/reset outside block; N invokes through completion synchronize",
                            },
                        ),
                        "median_us": statistics.median(mean_call_us),
                        "p95_us": percentile(mean_call_us, 0.95),
                        "sustained_calls_per_second": sustained_rate,
                        "modeled_source_copy_bytes": modeled_bytes,
                        "modeled_active_assign_cores": active_cores,
                        "useful_effective_bandwidth_gbps": spec.useful_bytes * sustained_rate / 1_000_000_000.0,
                    }
                )
                writer.record(sustained_record)
            except Exception as error:
                failure = base_record(
                    spec, provenance, context.metadata, args.warmup, "passed", str(error), context.active_assign_cores
                )
                failure.update(
                    {
                        "record_type": "timing_failure",
                        "timing_mode": "not_measured",
                        "raw_samples_path": writer.write_raw(
                            spec.case_id,
                            "timing-failure",
                            {"exception": traceback.format_exc(), "spec": asdict(spec)},
                        ),
                        "modeled_source_copy_bytes": modeled_bytes,
                        "modeled_active_assign_cores": active_cores,
                    }
                )
                writer.record(failure)
        manifest["status"] = "completed"
        return 0
    finally:
        write_manifest(manifest_path, manifest)
        writer.close()


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ValueError, FileNotFoundError) as error:
        print(f"benchmark configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
