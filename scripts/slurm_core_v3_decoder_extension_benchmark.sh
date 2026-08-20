#!/usr/bin/env bash
#SBATCH --job-name=baa-dec-ext-bench
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/VisDrone}"
SEED=17
CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
RUN_DIR="$PROJECT_ROOT/runs/core_v3_decoder_extension_benchmark/seed_${SEED}_${SLURM_JOB_ID}"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$RUN_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CACHE"

"$PYTHON" scripts/train_core_v3_decoder_extension.py \
  --seed "$SEED" \
  --cache "$CACHE" \
  --run-dir "$RUN_DIR" \
  --data-root "$DATA_ROOT" \
  --micro-batch-size 1 \
  --device cuda \
  --mode benchmark \
  --steps 500 \
  --log-interval 50
