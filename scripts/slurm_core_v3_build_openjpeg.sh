#!/usr/bin/env bash
#SBATCH --job-name=baa-corev3-opj-build
#SBATCH --partition=cu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/BAA-NIC_RemoteSensing_Paper_core_v3}"
ARCHIVE="$PROJECT_ROOT/tools/codecs/downloads/openjpeg-v2.5.4-source.tar.gz"
SOURCE_ROOT="$PROJECT_ROOT/tools/codecs/src/openjpeg-2.5.4"
BUILD_ROOT="$PROJECT_ROOT/tools/codecs/build/openjpeg-2.5.4"
INSTALL_ROOT="$PROJECT_ROOT/tools/codecs/openjpeg-2.5.4-linux"

cd "$PROJECT_ROOT"
mkdir -p runs/slurm "$PROJECT_ROOT/tools/codecs/src" "$PROJECT_ROOT/tools/codecs/build"
test -f "$ARCHIVE"
if [[ ! -f "$SOURCE_ROOT/CMakeLists.txt" ]]; then
  mkdir -p "$SOURCE_ROOT"
  tar -xzf "$ARCHIVE" --strip-components=1 -C "$SOURCE_ROOT"
fi
cmake -S "$SOURCE_ROOT" -B "$BUILD_ROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
  -DBUILD_CODEC=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF
cmake --build "$BUILD_ROOT" --parallel "${SLURM_CPUS_PER_TASK:-8}"
cmake --install "$BUILD_ROOT"
test -x "$INSTALL_ROOT/bin/opj_compress" && test -x "$INSTALL_ROOT/bin/opj_decompress"
set +e
"$INSTALL_ROOT/bin/opj_compress" -h >/dev/null 2>&1
HELP_STATUS=$?
set -e
if [[ "$HELP_STATUS" -ne 0 && "$HELP_STATUS" -ne 1 ]]; then
  exit "$HELP_STATUS"
fi
sha256sum "$INSTALL_ROOT/bin/opj_compress" "$INSTALL_ROOT/bin/opj_decompress"
