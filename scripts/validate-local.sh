#!/usr/bin/env bash
# No NPU imports, installs, network requests, or paid resources.
set -euo pipefail
cd "$(dirname "$0")/.."
validation_dir=$(mktemp -d "${TMPDIR:-/tmp}/inference-roofline-validation.XXXXXX")
python3 -m unittest discover -s experiments/kv-cache-assign -p 'test_*.py' -v > "$validation_dir/tests.log" 2>&1 || {
  cat "$validation_dir/tests.log"
  exit 1
}
cat "$validation_dir/tests.log"
python3 -m compileall -q experiments/kv-cache-assign
git diff --check
python3 experiments/kv-cache-assign/benchmark_cache_location_assign.py \
  --stage stage01 --seed 20260914 --warmup 20 --isolated-samples 200 \
  --sustained-blocks 5 --sustained-calls 200 \
  --output-dir "$validation_dir/dry-run" --dry-run
python3 - "$validation_dir/dry-run" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'run-manifest.json').read_text())
assert manifest['status'] == 'dry-run-static-validation'
assert len(manifest['case_definitions']) == 24
assert not (root / 'results.jsonl').read_text().strip()
assert not list((root / 'raw-samples').glob('*.json'))
print('Static validation passed; no NPU execution or timing evidence.')
print('Retained local evidence:', root.parent)
PY
