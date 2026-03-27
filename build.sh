#!/bin/bash
# Build vllm-multiquant container
#
# Usage:
#   ./build.sh                     # DGX Spark (SM121a + SM120a)
#   ./build.sh --rtx               # RTX PRO 6000 (SM120a only)
#   ./build.sh --jobs 8            # Limit parallel jobs
#   ./build.sh --no-cache          # Force full rebuild

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="vllm-multiquant"
BASE="nvcr.io/nvidia/vllm:26.02-py3"
ARCHS="12.0a;12.1a"
JOBS=16
EXTRA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rtx)       ARCHS="12.0a"; shift ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        --no-cache)  EXTRA="--no-cache"; shift ;;
        --tag)       IMAGE="$2"; shift 2 ;;
        *)           echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "================================================"
echo "  Building ${IMAGE}"
echo "  Base:  ${BASE}"
echo "  Archs: ${ARCHS}"
echo "  Jobs:  ${JOBS}"
echo "================================================"

# Copy FlashInfer wheels if available
if [ ! -f build/wheels/flashinfer_python*.whl ]; then
    MARLIN_WHEELS="$HOME/vllm-marlin-sm12x/build/wheels"
    if [ -d "$MARLIN_WHEELS" ]; then
        echo "Copying FlashInfer wheels from vllm-marlin-sm12x..."
        mkdir -p build/wheels
        cp "$MARLIN_WHEELS"/flashinfer*.whl build/wheels/
    else
        echo "No local FlashInfer wheels — will install from PyPI (slower)"
    fi
fi

podman build \
    -f Dockerfile.multiquant \
    -t "$IMAGE" \
    --build-arg BASE_IMAGE="$BASE" \
    --build-arg CUDA_ARCHS="$ARCHS" \
    --build-arg MAX_JOBS="$JOBS" \
    $EXTRA \
    .

echo ""
echo "================================================"
echo "  Done: ${IMAGE}"
echo "  Size: $(podman image inspect "$IMAGE" --format '{{.Size}}' | numfmt --to=iec)"
echo "================================================"
