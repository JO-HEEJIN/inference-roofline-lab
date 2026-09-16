"""Static tests for the Stage 0/1 harness; no torch_npu or NPU required."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("benchmark_cache_location_assign.py")
SPEC = importlib.util.spec_from_file_location("cache_location_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class BenchmarkPlanTests(unittest.TestCase):
    def test_stage_matrix_is_exactly_stage_zero_and_one(self) -> None:
        specs = benchmark.case_specs(7, ("int32", "int64"))
        self.assertEqual(len(specs), 24)
        self.assertEqual({case.batch_size for case in specs}, {8, 32, 128})
        self.assertEqual({case.sequence_capacity for case in specs}, {2048})
        self.assertEqual({case.update_length for case in specs}, {1, 2, 8, 16})
        self.assertEqual({case.request_index_dtype for case in specs}, {"int32", "int64"})

    def test_stage_two_matrix_is_core_relative_and_excludes_c_plus_or_minus_one(self) -> None:
        specs = benchmark.stage2_case_specs(7, 32)
        self.assertEqual({case.batch_size for case in specs}, {16, 32, 64})
        self.assertEqual({case.update_length for case in specs}, {1, 16})
        self.assertEqual({case.request_index_dtype for case in specs}, {"int64"})
        self.assertEqual({case.stage for case in specs}, {benchmark.STAGE2})
        self.assertNotIn(31, {case.batch_size for case in specs})
        self.assertNotIn(33, {case.batch_size for case in specs})

    def test_guarded_storage_covers_static_alignment_reads(self) -> None:
        for dtype in ("int32", "int64"):
            for batch in (8, 32, 128):
                spec = benchmark.CaseSpec("assign", batch, 2048, 1, dtype, 3)
                plan = benchmark.storage_plan(spec)
                request_alignment = 8 if dtype == "int32" else 4
                self.assertEqual(plan.request_backing_count % request_alignment, 0)
                self.assertEqual(plan.offset_backing_count % 4, 0)
                self.assertEqual(plan.cache_logical_count, batch * 16)
                self.assertGreaterEqual(plan.cache_guard_count, 16)
                self.assertEqual(plan.physical_row_width, 2048 + 16)

    def test_starts_are_nonnegative_aligned_and_interior(self) -> None:
        for update in (1, 2, 8, 16):
            spec = benchmark.CaseSpec("assign", 128, 2048, update, "int64", 11)
            for start in benchmark.aligned_interior_starts(spec):
                self.assertGreaterEqual(start, 16)
                self.assertEqual(start % 8, 0)
                self.assertLessEqual(start + update, 2048)
                self.assertLessEqual(start + 16, 2048)

    def test_source_copy_model_and_percentile_are_explicit(self) -> None:
        spec = benchmark.CaseSpec("assign", 8, 2048, 1, "int64", 1)
        self.assertEqual(benchmark.modeled_source_copy_bytes(spec, 1), 1_728)
        self.assertAlmostEqual(benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)

    def test_output_schema_keeps_raw_and_aggregate_fields_separate(self) -> None:
        required = {
            "operation",
            "batch_size",
            "sequence_capacity",
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
            "useful_effective_bandwidth_gbps",
            "correctness_status",
            "failure_reason",
            "source_commit",
            "source_dirty_tree_status",
            "device_metadata",
            "cann_metadata",
            "torch_version",
            "torch_npu_version",
        }
        self.assertTrue(required <= set(benchmark.OUTPUT_FIELDS))

    def test_dry_run_writes_manifest_without_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            code = benchmark.run(
                type(
                    "Arguments",
                    (),
                    {
                        "output_dir": output,
                        "seed": 1,
                        "warmup": 20,
                        "isolated_samples": 200,
                        "sustained_blocks": 5,
                        "sustained_calls": 200,
                        "assign_active_cores": None,
                        "dry_run": True,
                        "stage": benchmark.STAGE01,
                    },
                )()
            )
            self.assertEqual(code, 0)
            manifest = json.loads((output / "run-manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry-run-static-validation")
            self.assertFalse((output / "results.jsonl").read_text())
            self.assertIn("timing_mode", (output / "results.csv").read_text().splitlines()[0])

    def test_stage_two_dry_run_requires_and_records_core_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            code = benchmark.run(
                type(
                    "Arguments",
                    (),
                    {
                        "output_dir": output,
                        "seed": 1,
                        "warmup": 20,
                        "isolated_samples": 200,
                        "sustained_blocks": 5,
                        "sustained_calls": 200,
                        "assign_active_cores": 32,
                        "dry_run": True,
                        "stage": benchmark.STAGE2,
                    },
                )()
            )
            self.assertEqual(code, 0)
            manifest = json.loads((output / "run-manifest.json").read_text())
            self.assertEqual(manifest["benchmark_parameters"]["experiment_stage"], benchmark.STAGE2)
            self.assertEqual(
                {case["core_transition_relation"] for case in manifest["case_definitions"]},
                {"below_active_cores", "equal_active_cores", "above_active_cores"},
            )


if __name__ == "__main__":
    unittest.main()
