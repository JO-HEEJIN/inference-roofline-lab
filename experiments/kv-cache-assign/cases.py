"""Deterministic logical inputs; physical NPU padding is a separate adapter concern."""

import random
from dataclasses import asdict, dataclass

from reference import INT32_MAX, INT32_MIN

MAX_POOL_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class Case:
    name: str
    pool_rows: int
    seq_capacity: int
    req_indices: tuple[int, ...]
    start_offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    cache_locations: tuple[int, ...]
    seed: int = 0
    req_index_dtype: str = "int64"

    @property
    def batch_size(self) -> int:
        return len(self.req_indices)

    @property
    def end_offsets(self) -> tuple[int, ...]:
        return tuple(start + length for start, length in zip(self.start_offsets, self.lengths))

    def validate(self) -> None:
        for value in (self.pool_rows, self.seq_capacity, self.seed):
            if type(value) is not int:
                raise ValueError("dimensions and seed must be integers")
        if self.pool_rows < 0 or self.seq_capacity < 0:
            raise ValueError("dimensions must be nonnegative")
        if self.pool_rows * (self.seq_capacity + 16) > MAX_POOL_ELEMENTS:
            raise ValueError("case exceeds the local fixture memory budget")
        if self.req_index_dtype not in ("int32", "int64"):
            raise ValueError("request index dtype must be int32 or int64")
        if len(self.start_offsets) != self.batch_size or len(self.lengths) != self.batch_size:
            raise ValueError("case vectors must have equal lengths")
        for values in (self.req_indices, self.start_offsets, self.lengths, self.cache_locations):
            if any(type(value) is not int for value in values):
                raise ValueError("case vectors must contain integers")
        if len(set(self.req_indices)) != self.batch_size:
            raise ValueError("duplicate request indices are unsupported")
        for req, start, length in zip(self.req_indices, self.start_offsets, self.lengths):
            if not 0 <= req < self.pool_rows:
                raise ValueError("request index outside pool")
            if length < 0 or not 0 <= start <= start + length <= self.seq_capacity:
                raise ValueError("case offsets exceed logical sequence capacity")
        if len(self.cache_locations) != sum(self.lengths):
            raise ValueError("packed cache length does not match request lengths")
        if any(not INT32_MIN <= value <= INT32_MAX for value in self.cache_locations):
            raise ValueError("cache values must fit int32")

    def to_dict(self) -> dict:
        return asdict(self)


def make_pool(case: Case, *, padding: int = 0) -> list[list[int]]:
    case.validate()
    if not 0 <= padding <= 16:
        raise ValueError("padding must be between 0 and 16")
    width = case.seq_capacity + padding
    # Distinct negative sentinels expose wrong-row accesses and unintended writes.
    return [[-(1 + row * width + col) for col in range(width)] for row in range(case.pool_rows)]


def generated_case(
    name: str,
    batch: int,
    capacity: int,
    lengths: tuple[int, ...],
    seed: int,
    *,
    aligned: bool = False,
    dtype: str = "int64",
) -> Case:
    if batch < 0 or len(lengths) != batch or capacity <= 0:
        raise ValueError("invalid generated case dimensions")
    if any(length < 0 or length > capacity for length in lengths):
        raise ValueError("invalid generated update length")
    rng = random.Random(seed)
    rows = batch + 2
    requests = tuple(rng.sample(range(rows), batch))
    step = 8 if aligned else 1
    starts = tuple(rng.randrange((capacity - length) // step + 1) * step for length in lengths)
    values = tuple(rng.randint(1, 1_000_000) for _ in range(sum(lengths)))
    case = Case(name, rows, capacity, requests, starts, lengths, values, seed, dtype)
    case.validate()
    return case


def smoke_cases(seed: int = 20260914) -> list[Case]:
    return [
        Case("hand-worked", 4, 6, (2, 0), (1, 4), (2, 1), (101, 102, 201), seed),
        Case("row-boundaries", 3, 17, (2, 0), (16, 0), (1, 16), tuple(range(1, 18)), seed),
        Case("empty-batch", 2, 8, (), (), (), (), seed),
        Case("zero-length", 3, 8, (2, 0), (8, 0), (0, 0), (), seed),
        Case("int32-extremes", 2, 4, (1,), (1,), (2,), (INT32_MIN, INT32_MAX), seed),
        generated_case("ragged-permuted", 9, 33, (0, 1, 2, 3, 8, 16, 1, 7, 4), seed),
    ]


def npu_baseline_cases(seed: int = 20260914) -> list[Case]:
    cases = []
    for batch in (8, 32, 128):
        for capacity in (128, 2048):
            for update in (1, 2, 8, 16):
                name = f"npu-b{batch}-s{capacity}-u{update}"
                cases.append(generated_case(name, batch, capacity, (update,) * batch, seed, aligned=True))
    cases.append(generated_case("npu-ragged-int32", 8, 128, (1, 2, 4, 8, 16, 1, 2, 8), seed,
                                aligned=True, dtype="int32"))
    # End-of-logical-row cases exercise the extra physical guard columns.
    cases.append(Case("npu-row-tail", 10, 128, tuple(range(8)), (120,) * 8, (1,) * 8,
                      tuple(range(101, 109)), seed))
    return cases
