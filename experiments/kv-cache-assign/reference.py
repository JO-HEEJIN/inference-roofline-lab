"""Independent scalar specification for packed cache-location mapping.

No NPU imports, alignment assumptions, fixed-size transfers, or upstream code copies.
Validation finishes before mutation so invalid inputs cannot partially update the pool.
"""

from collections.abc import Sequence

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def _integer(value: int, label: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{label} must be a Python integer")


def validate_mapping(
    token_pool: list[list[int]],
    req_indices: Sequence[int],
    start_offsets: Sequence[int],
    end_offsets: Sequence[int],
) -> int:
    """Return packed length; this experiment deliberately excludes duplicate requests."""
    batch = len(req_indices)
    if len(start_offsets) != batch or len(end_offsets) != batch:
        raise ValueError("request and offset arrays must have equal lengths")
    width = len(token_pool[0]) if token_pool else 0
    if any(len(row) != width for row in token_pool):
        raise ValueError("token_pool must be rectangular")
    for row in token_pool:
        for value in row:
            _integer(value, "token_pool value")
            if not INT32_MIN <= value <= INT32_MAX:
                raise ValueError("token_pool values must fit int32")
    seen = set()
    total = 0
    for req, start, end in zip(req_indices, start_offsets, end_offsets):
        for value in (req, start, end):
            _integer(value, "request/offset")
        if req in seen:
            raise ValueError("duplicate request indices are outside this experiment's contract")
        seen.add(req)
        if not 0 <= req < len(token_pool):
            raise ValueError("request index is outside token_pool")
        if not 0 <= start <= end <= width:
            raise ValueError("offsets must satisfy 0 <= start <= end <= row width")
        total += end - start
    return total


def assign(
    token_pool: list[list[int]],
    req_indices: Sequence[int],
    start_offsets: Sequence[int],
    end_offsets: Sequence[int],
    cache_locations: Sequence[int],
) -> None:
    """Mutate only selected pool cells, consuming locations in request-list order."""
    count = validate_mapping(token_pool, req_indices, start_offsets, end_offsets)
    if len(cache_locations) != count:
        raise ValueError("cache_locations length must equal sum(end - start)")
    for value in cache_locations:
        _integer(value, "cache location")
        if not INT32_MIN <= value <= INT32_MAX:
            raise ValueError("cache locations must fit int32")
    cursor = 0
    for req, start, end in zip(req_indices, start_offsets, end_offsets):
        for column in range(start, end):
            token_pool[req][column] = cache_locations[cursor]
            cursor += 1


def retrieve(
    token_pool: list[list[int]],
    req_indices: Sequence[int],
    start_offsets: Sequence[int],
    end_offsets: Sequence[int],
) -> list[int]:
    """Read packed locations without changing the pool (upstream: cache_loc_update)."""
    validate_mapping(token_pool, req_indices, start_offsets, end_offsets)
    output = []
    for req, start, end in zip(req_indices, start_offsets, end_offsets):
        for column in range(start, end):
            output.append(token_pool[req][column])
    return output
