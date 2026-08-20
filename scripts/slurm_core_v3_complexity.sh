#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-e1
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
VISDRONE_ROOT="${VISDRONE_ROOT:-$HOME/datasets/VisDrone}"
SEEDS=(17 29 43)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SENDER="$PROJECT_ROOT/runs/core_v3_joint/seed_${SEED}/best.pt"
HEAVY="$PROJECT_ROOT/runs/core_v3_decoder/seed_${SEED}/heavy/best.pt"
LIGHT="$PROJECT_ROOT/runs/core_v3_decoder/seed_${SEED}/light/best.pt"
OUTPUT_DIR="$PROJECT_ROOT/runs/core_v3_complexity/seed_${SEED}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$OUTPUT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$SENDER" && test -f "$HEAVY" && test -f "$LIGHT"

"$PYTHON" scripts/evaluate_core_v3_complexity.py \
  --seed "$SEED" \
  --checkpoint "$SENDER" \
  --heavy-decoder-checkpoint "$HEAVY" \
  --light-decoder-checkpoint "$LIGHT" \
  --output-dir "$OUTPUT_DIR" \
  --visdrone-root "$VISDRONE_ROOT" \
  --devices cuda,cpu \
  --threads 4 \
  --timing-images 100 \
  --repeats 5 \
  --warmups 50 \
  --development-limit 256

"$PYTHON" scripts/validate_core_v3_results.py \
  "$OUTPUT_DIR/core_v3_complexity_breakdown.csv"
