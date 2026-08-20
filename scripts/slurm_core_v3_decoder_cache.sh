#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-cache
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
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/VisDrone}"
UAVID_ROOT="${UAVID_ROOT:-$HOME/datasets/UAVid}"
read -r -a SEEDS <<< "${CORE_V3_SEEDS:-17 29 43}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="$PROJECT_ROOT/runs/core_v3_joint/seed_${SEED}/best.pt"
CACHE_DIR="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}"
CACHE="$CACHE_DIR/sender_cache.sqlite"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$CACHE_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CHECKPOINT"

"$PYTHON" scripts/build_core_v3_decoder_cache.py \
  --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --cache "$CACHE" \
  --data-root "$DATA_ROOT" \
  --device cuda \
  --mode train \
  --commit-interval 200

"$PYTHON" scripts/build_core_v3_evaluation_cache.py \
  --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --cache "$CACHE" \
  --visdrone-root "$DATA_ROOT" \
  --uavid-root "$UAVID_ROOT" \
  --device cuda \
  --commit-interval 100
