#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-e4
#SBATCH --partition=cu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
SEEDS=(17 29 43)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
BAA_CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
OPENJPEG_CACHE="$PROJECT_ROOT/runs/core_v3_openjpeg/seed_${SEED}/openjpeg_action_cache.sqlite"
TRACE_DIR="$PROJECT_ROOT/runs/core_v3_traces"
OUTPUT_DIR="$PROJECT_ROOT/runs/core_v3_bandwidth/seed_${SEED}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$OUTPUT_DIR"
export OMP_NUM_THREADS=1
test -f "$BAA_CACHE" && test -f "$OPENJPEG_CACHE"
test -f "$TRACE_DIR/development_traces.npz" && test -f "$TRACE_DIR/test_traces.npz"

"$PYTHON" scripts/evaluate_core_v3_bandwidth.py \
  --seed "$SEED" \
  --cache "$BAA_CACHE" \
  --openjpeg-cache "$OPENJPEG_CACHE" \
  --traces-dir "$TRACE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mode formal

"$PYTHON" scripts/validate_core_v3_results.py \
  "$OUTPUT_DIR/core_v3_bandwidth_detail.csv"
