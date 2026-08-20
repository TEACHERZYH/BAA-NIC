#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-e123-analysis
#SBATCH --partition=cu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm experiments/core_v3_analysis tables
export OMP_NUM_THREADS=4

"$PYTHON" scripts/analyze_core_v3_quality.py \
  --runs-root "$PROJECT_ROOT/runs" \
  --data-dir "$PROJECT_ROOT/data" \
  --tables-dir "$PROJECT_ROOT/tables" \
  --output-dir "$PROJECT_ROOT/experiments/core_v3_analysis"

"$PYTHON" scripts/validate_core_v3_results.py \
  data/core_v3_bitstream_roundtrip.csv \
  data/core_v3_operating_points.csv \
  data/core_v3_complexity_breakdown.csv \
  data/core_v3_decoder_asymmetry.csv \
  data/core_v3_progressive_ordering.csv
