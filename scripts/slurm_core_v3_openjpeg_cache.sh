#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-opj-cache
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/slurm/%x_%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
PYTHON="${PYTHON:-$HOME/envs/baa_nic_core_v3/bin/python}"
VISDRONE_ROOT="${VISDRONE_ROOT:-$HOME/datasets/VisDrone}"
OPJ_ROOT="${OPJ_ROOT:-$PROJECT_ROOT/tools/codecs/openjpeg-2.5.4-linux}"
SEEDS=(17 29 43)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
BAA_CACHE="$PROJECT_ROOT/runs/core_v3_decoder_cache/seed_${SEED}/sender_cache.sqlite"
OUTPUT_DIR="$PROJECT_ROOT/runs/core_v3_openjpeg/seed_${SEED}"
CACHE="$OUTPUT_DIR/openjpeg_action_cache.sqlite"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$OUTPUT_DIR"
export OMP_NUM_THREADS=4
test -f "$BAA_CACHE"
test -x "$OPJ_ROOT/bin/opj_compress" && test -x "$OPJ_ROOT/bin/opj_decompress"

"$PYTHON" scripts/build_core_v3_openjpeg_cache.py \
  --seed "$SEED" \
  --baa-cache "$BAA_CACHE" \
  --cache "$CACHE" \
  --opj-compress "$OPJ_ROOT/bin/opj_compress" \
  --opj-decompress "$OPJ_ROOT/bin/opj_decompress" \
  --visdrone-root "$VISDRONE_ROOT" \
  --device cuda \
  --calibration-iterations 3 \
  --mode formal
