"""Static traffic model and deliberately conservative NPU fixture admission.

These are lab restrictions inferred from the pinned implementation, not claims that
all admitted inputs have been validated on hardware or that all others are invalid.
"""

from cases import Case

MAX_STEP = 16


def validate_npu_case(case: Case) -> None:
    case.validate()
    if not 8 <= case.batch_size <= 128 or case.batch_size % 8:
        raise ValueError("NPU baseline requires batch 8..128 divisible by 8; use npu-baseline suite")
    if case.seq_capacity % 8 or any(start % 8 for start in case.start_offsets):
        raise ValueError("NPU baseline requires capacity and starts aligned to 8 int32 elements")
    if any(not 1 <= length <= MAX_STEP for length in case.lengths):
        raise ValueError("NPU baseline requires each update length in 1..16")


def traffic_model(case: Case, operation: str, active_cores: int | None = None) -> dict:
    case.validate()
    if operation not in ("assign", "retrieve"):
        raise ValueError("operation must be assign or retrieve")
    batch = case.batch_size
    useful = 2 * sum(case.lengths) * 4  # Selected int32 values, one read plus one write.
    req_bytes = 4 if case.req_index_dtype == "int32" else 8
    align32 = lambda size: ((size + 31) // 32) * 32
    metadata = align32(batch * req_bytes) + 2 * align32(batch * 8)
    full_cache = batch * MAX_STEP * 4
    per_active_core = metadata + full_cache
    # Kernel-visible copy payloads, NOT DRAM transactions. Cache hierarchy is unmodeled.
    row_copy_bytes = batch * MAX_STEP * 4 * (2 if operation == "assign" else 1)
    total = None
    if operation == "retrieve":
        active_cores = 1 if batch else 0  # Pinned host explicitly sets blockDim = 1.
    if active_cores is not None:
        if type(active_cores) is not int or not 0 <= active_cores <= batch:
            raise ValueError("active core count must be an integer between 0 and batch size")
        if batch and active_cores == 0:
            raise ValueError("nonempty case requires at least one active core")
        total = per_active_core * active_cores + row_copy_bytes
        if operation == "retrieve" and batch:
            total += full_cache
    return {
        "model": "pinned-source-copy-payload-v1",
        "useful_value_bytes": useful,
        "metadata_bytes_per_active_core": metadata,
        "cache_copy_bytes_per_active_core": full_cache,
        "row_copy_bytes": row_copy_bytes,
        "active_cores_assumed": active_cores,
        "estimated_copy_payload_bytes": total,
        "hardware_traffic_bytes": None,
        "scope": "AscendC explicit copy payloads only; excludes host tiling transfer and cache effects",
    }


def layout_audit(case: Case) -> dict:
    case.validate()
    risks = []
    if any(length > MAX_STEP for length in case.lengths):
        risks.append("update exceeds fixed 16-element row buffer")
    if case.batch_size % 4:
        risks.append("aligned offset cast count exceeds batch-sized int32 scratch tensors")
    if any(start + MAX_STEP > case.seq_capacity for start in case.start_offsets):
        risks.append("fixed row copy extends beyond logical row width")
    if len(case.cache_locations) < case.batch_size * MAX_STEP:
        risks.append("packed logical cache is shorter than fixed full-cache copy extent")
    return {
        "logical_cache_elements": len(case.cache_locations),
        "upstream_cache_copy_elements": case.batch_size * MAX_STEP,
        "static_risks_for_unpadded_input": risks,
        "hardware_verified": False,
    }
