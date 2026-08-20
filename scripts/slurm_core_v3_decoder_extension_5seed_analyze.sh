#!/usr/bin/env bash
#SBATCH --job-name=baa-dec-ext-5s-analysis
#SBATCH --partition=cu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
DATA_DIR="$PROJECT_ROOT/data/core_v3_decoder_extension_5seed"
OUTPUT_DIR="$PROJECT_ROOT/experiments/core_v3_decoder_extension_5seed"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$DATA_DIR" "$OUTPUT_DIR"
export OMP_NUM_THREADS=4

"$PYTHON" scripts/analyze_core_v3_decoder_extension.py \
  --runs-root "$PROJECT_ROOT/runs" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON" - "$DATA_DIR" "$OUTPUT_DIR" <<'PY'
import csv
import json
from pathlib import Path
import sys

data_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
audit = json.loads((output_dir / "gate_audit.json").read_text(encoding="utf-8"))
if audit.get("status") != "pass" or audit.get("confirmatory_gate_pass") is not True:
    raise SystemExit("FAIL: five-seed confirmatory gate")
if audit.get("model_seeds") != [17, 29, 43, 59, 71]:
    raise SystemExit("FAIL: five-seed audit seed set")
expected_rows = {
    "core_v3_decoder_extension.csv": 122880,
    "core_v3_decoder_extension_summary.csv": 24,
    "core_v3_decoder_extension_per_image.csv": 46080,
}
for name, expected in expected_rows.items():
    with (data_dir / name).open(encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    if rows != expected:
        raise SystemExit(f"FAIL: {name} rows={rows} expected={expected}")
print("PASS: five-seed decoder-extension output contract")
PY
