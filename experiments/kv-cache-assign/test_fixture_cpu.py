"""Fault injection for the fixture/checker only; never executes an NPU kernel."""

import unittest

from test_benchmark_cache_location_assign import benchmark

try:
    import torch
except ImportError:
    torch = None


class CpuContext:
    def __init__(self):
        self.torch = torch
        self.device = "cpu"

    def synchronize(self):
        pass


class CpuReferenceFixture(benchmark.AssignFixture):
    def invoke(self):
        cursor = 0
        for request in range(self.spec.batch_size):
            row = int(self.request_indices[request])
            for column in range(int(self.start_offsets[request]), int(self.end_offsets[request])):
                self.pool_physical[row, column] = self.cache_locations[cursor]
                cursor += 1


@unittest.skipIf(torch is None, "optional CPU PyTorch is not installed")
class FixtureFaultTests(unittest.TestCase):
    def fixture(self, dtype="int64", update=1):
        return CpuReferenceFixture(CpuContext(), benchmark.CaseSpec("assign", 8, 2048, update, dtype, 7))

    def test_all_baseline_shapes_pass_independent_cpu_reference(self):
        for spec in benchmark.case_specs(7, ("int32", "int64")):
            with self.subTest(case=spec.case_id):
                fixture = CpuReferenceFixture(CpuContext(), spec)
                self.assertEqual(fixture.verify()["status"], "passed")

    def test_fresh_cases_and_reset_do_not_share_mutable_state(self):
        first, second = self.fixture(), self.fixture()
        for name in ("pool_physical", "cache_physical", "request_physical", "start_physical", "end_physical"):
            left, right = getattr(first, name), getattr(second, name)
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
            original = right.clone()
            left.fill_(42)
            self.assertTrue(torch.equal(right, original))
        first.reset()
        self.assertTrue(torch.equal(first.pool_physical, first.initial_pool_cpu))
        self.assertEqual(first.verify()["status"], "passed")

    def test_noop_and_wrong_row_writes_are_rejected(self):
        for mutation in ("noop", "wrong_row"):
            fixture = self.fixture()
            reference = fixture.invoke

            def corrupt():
                if mutation == "wrong_row":
                    reference()
                    fixture.pool_physical.copy_(fixture.pool_physical.roll(1, dims=0))

            fixture.invoke = corrupt
            result = fixture.verify()
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["checks"]["exact_changed_cells"])

    def test_each_preservation_check_detects_corruption(self):
        corruptions = {
            "untouched_cells_preserved": ("pool_physical", (0, 0)),
            "token_guard_preserved": ("pool_physical", (0, -1)),
            "packed_cache_padding_preserved": ("cache_physical", 8),
            "packed_cache_guard_preserved": ("cache_physical", -1),
            "metadata_and_alignment_guards_preserved": ("request_physical", 0),
        }
        for check, (name, index) in corruptions.items():
            with self.subTest(check=check):
                fixture = self.fixture()
                reference = fixture.invoke

                def corrupt():
                    reference()
                    getattr(fixture, name)[index] = 42

                fixture.invoke = corrupt
                result = fixture.verify()
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["checks"][check])


if __name__ == "__main__":
    unittest.main()
