#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-g0b
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
VISDRONE_ROOT="${VISDRONE_ROOT:-$HOME/datasets/VisDrone}"
UAVID_ROOT="${UAVID_ROOT:-$HOME/datasets/UAVid}"
SEEDS=(17 29 43)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="$PROJECT_ROOT/runs/core_v3_joint/seed_${SEED}/best.pt"
OUTPUT_DIR="$PROJECT_ROOT/runs/core_v3_g0b/seed_${SEED}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$OUTPUT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CHECKPOINT"

"$PYTHON" scripts/evaluate_core_v3_g0b.py \
  --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --visdrone-root "$VISDRONE_ROOT" \
  --uavid-root "$UAVID_ROOT" \
  --device cuda \
  --images-per-dataset 32

"$PYTHON" scripts/validate_core_v3_results.py \
  "$OUTPUT_DIR/core_v3_bitstream_roundtrip.csv" \
  "$OUTPUT_DIR/core_v3_operating_points.csv"
