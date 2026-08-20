#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-joint-bench
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/VisDrone}"
RUN_DIR="$PROJECT_ROOT/runs/core_v3_benchmarks/joint_seed17_${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$RUN_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RUN_DIR/gpu.csv"
"$PYTHON" -m pip freeze > "$RUN_DIR/pip_freeze.txt"
"$PYTHON" -c 'import torch, compressai; print(torch.__version__, torch.version.cuda, compressai.__version__, torch.cuda.is_available())' \
  > "$RUN_DIR/import_check.txt"
"$PYTHON" scripts/train_core_v3.py \
  --seed 17 \
  --run-dir "$RUN_DIR" \
  --data-root "$DATA_ROOT" \
  --micro-batch-size 4 \
  --workers 8 \
  --device cuda \
  --mode benchmark \
  --steps 500 \
  --log-interval 25

printf '%s\n' "$RUN_DIR" > runs/core_v3_benchmarks/latest_joint_run.txt
