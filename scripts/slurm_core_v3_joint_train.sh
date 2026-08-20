#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-joint
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/VisDrone}"
read -r -a SEEDS <<< "${CORE_V3_SEEDS:-17 29 43}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$PROJECT_ROOT/runs/core_v3_joint/seed_${SEED}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$RUN_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu.csv"
resume_args=()
if [[ -f "$RUN_DIR/last.pt" ]]; then
  resume_args=(--resume "$RUN_DIR/last.pt")
fi

"$PYTHON" scripts/train_core_v3.py \
  --seed "$SEED" \
  --run-dir "$RUN_DIR" \
  --data-root "$DATA_ROOT" \
  --micro-batch-size 4 \
  --workers 8 \
  --device cuda \
  --mode train \
  --log-interval 100 \
  "${resume_args[@]}"
