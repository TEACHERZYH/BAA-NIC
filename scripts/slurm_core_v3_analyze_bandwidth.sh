#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-e4-analysis
#SBATCH --partition=cu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm experiments/core_v3_analysis
export OMP_NUM_THREADS=4

"$PYTHON" scripts/analyze_core_v3_bandwidth.py \
  --runs-root "$PROJECT_ROOT/runs" \
  --data-dir "$PROJECT_ROOT/data" \
  --output-dir "$PROJECT_ROOT/experiments/core_v3_analysis"

"$PYTHON" scripts/validate_core_v3_results.py \
  data/core_v3_bandwidth_detail.csv \
  data/core_v3_bandwidth_statistics.csv
