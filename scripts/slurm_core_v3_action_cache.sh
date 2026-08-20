#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-actions
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
VISDRONE_ROOT="${VISDRONE_ROOT:-$HOME/datasets/VisDrone}"
SEEDS=(17 29 43)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
SENDER="$PROJECT_ROOT/runs/core_v3_joint/seed_${SEED}/best.pt"
HEAVY="$PROJECT_ROOT/runs/core_v3_decoder/seed_${SEED}/heavy/best.pt"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CACHE" && test -f "$SENDER" && test -f "$HEAVY"

"$PYTHON" scripts/build_core_v3_action_cache.py \
  --seed "$SEED" \
  --cache "$CACHE" \
  --checkpoint "$SENDER" \
  --heavy-decoder-checkpoint "$HEAVY" \
  --visdrone-root "$VISDRONE_ROOT" \
  --device cuda
