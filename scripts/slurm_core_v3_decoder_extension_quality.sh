#!/usr/bin/env bash
#SBATCH --job-name=baa-dec-ext-quality
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
VISDRONE_ROOT="${VISDRONE_ROOT:-$HOME/datasets/VisDrone}"
UAVID_ROOT="${UAVID_ROOT:-$HOME/datasets/UAVid}"
read -r -a SEEDS <<< "${CORE_V3_SEEDS:-17 29 43}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
ENHANCED="$PROJECT_ROOT/runs/core_v3_decoder_extension/seed_${SEED}/enhanced_r1/best.pt"
HEAVY="$PROJECT_ROOT/runs/core_v3_decoder/seed_${SEED}/heavy/best.pt"
OUTPUT_DIR="$PROJECT_ROOT/runs/core_v3_decoder_extension_quality/seed_${SEED}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$OUTPUT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CACHE" && test -f "$ENHANCED" && test -f "$HEAVY"

"$PYTHON" scripts/evaluate_core_v3_decoder_extension.py \
  --seed "$SEED" \
  --cache "$CACHE" \
  --enhanced-checkpoint "$ENHANCED" \
  --heavy-checkpoint "$HEAVY" \
  --output-dir "$OUTPUT_DIR" \
  --visdrone-root "$VISDRONE_ROOT" \
  --uavid-root "$UAVID_ROOT" \
  --prefix-batch-size 2 \
  --device cuda

"$PYTHON" scripts/measure_core_v3_decoder_extension.py \
  --seed "$SEED" \
  --enhanced-checkpoint "$ENHANCED" \
  --heavy-checkpoint "$HEAVY" \
  --output "$OUTPUT_DIR/core_v3_decoder_extension_complexity.csv" \
  --device cuda
