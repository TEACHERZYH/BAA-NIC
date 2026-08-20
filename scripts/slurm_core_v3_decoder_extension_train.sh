#!/usr/bin/env bash
#SBATCH --job-name=baa-dec-ext-train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/VisDrone}"
read -r -a SEEDS <<< "${CORE_V3_SEEDS:-17 29 43}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
RUN_DIR="$PROJECT_ROOT/runs/core_v3_decoder_extension/seed_${SEED}/enhanced_r1"
PILOT_GATE="$PROJECT_ROOT/runs/core_v3_decoder_extension/seed_17/enhanced_r1/pilot_gate.json"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$RUN_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CACHE"
test -f "$PILOT_GATE"
grep -q '"status": "pass"' "$PILOT_GATE"
resume_args=()
if [[ -f "$RUN_DIR/last.pt" ]]; then
  resume_args=(--resume "$RUN_DIR/last.pt")
fi

"$PYTHON" scripts/train_core_v3_decoder_extension.py \
  --seed "$SEED" \
  --cache "$CACHE" \
  --run-dir "$RUN_DIR" \
  --data-root "$DATA_ROOT" \
  --micro-batch-size 1 \
  --device cuda \
  --mode train \
  --log-interval 100 \
  "${resume_args[@]}"
