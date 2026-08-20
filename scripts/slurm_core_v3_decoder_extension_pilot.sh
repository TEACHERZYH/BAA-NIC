#!/usr/bin/env bash
#SBATCH --job-name=baa-dec-ext-pilot
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/VisDrone}"
SEED=17
CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
RUN_DIR="$PROJECT_ROOT/runs/core_v3_decoder_extension/seed_${SEED}/enhanced_r1"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$RUN_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
test -f "$CACHE"
if [[ -f "$RUN_DIR/pilot_gate.json" ]] && grep -q '"status": "pass"' "$RUN_DIR/pilot_gate.json"; then
  echo "PASS: frozen pilot gate already completed"
  exit 0
fi
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
  --mode pilot \
  --log-interval 100 \
  "${resume_args[@]}"
