#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-dec-bench
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/VisDrone}"
: "${SENDER_CHECKPOINT:?submit with --export=ALL,SENDER_CHECKPOINT=/absolute/path/to/last.pt}"
ROOT_RUN="$PROJECT_ROOT/runs/core_v3_benchmarks/decoder_seed17_${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}"
CACHE="$ROOT_RUN/sender_cache.sqlite"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$ROOT_RUN"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$ROOT_RUN/gpu.csv"
"$PYTHON" scripts/build_core_v3_decoder_cache.py \
  --seed 17 \
  --checkpoint "$SENDER_CHECKPOINT" \
  --cache "$CACHE" \
  --data-root "$DATA_ROOT" \
  --device cuda \
  --mode benchmark \
  --steps 500 \
  --commit-interval 100

for decoder in heavy light; do
  "$PYTHON" scripts/train_core_v3_decoder.py \
    --seed 17 \
    --decoder "$decoder" \
    --cache "$CACHE" \
    --run-dir "$ROOT_RUN/$decoder" \
    --data-root "$DATA_ROOT" \
    --micro-batch-size 4 \
    --device cuda \
    --mode benchmark \
    --steps 500 \
    --log-interval 25
done

printf '%s\n' "$ROOT_RUN" > runs/core_v3_benchmarks/latest_decoder_run.txt
