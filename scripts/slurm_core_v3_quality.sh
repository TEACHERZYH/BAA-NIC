#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-quality
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
VISDRONE_ROOT="${VISDRONE_ROOT:-$HOME/datasets/VisDrone}"
UAVID_ROOT="${UAVID_ROOT:-$HOME/datasets/UAVid}"
SEEDS=(17 29 43)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
HEAVY="$PROJECT_ROOT/runs/core_v3_decoder/seed_${SEED}/heavy/best.pt"
LIGHT="$PROJECT_ROOT/runs/core_v3_decoder/seed_${SEED}/light/best.pt"
OUTPUT_DIR="$PROJECT_ROOT/runs/core_v3_quality/seed_${SEED}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$OUTPUT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CACHE" && test -f "$HEAVY" && test -f "$LIGHT"

"$PYTHON" scripts/evaluate_core_v3_quality.py \
  --seed "$SEED" \
  --cache "$CACHE" \
  --heavy-checkpoint "$HEAVY" \
  --light-checkpoint "$LIGHT" \
  --output-dir "$OUTPUT_DIR" \
  --visdrone-root "$VISDRONE_ROOT" \
  --uavid-root "$UAVID_ROOT" \
  --device cuda

"$PYTHON" scripts/validate_core_v3_results.py \
  "$OUTPUT_DIR/core_v3_decoder_asymmetry.csv" \
  "$OUTPUT_DIR/core_v3_progressive_ordering.csv"
