#!/bin/bash
# Build vllm-multiquant-v020 container (vLLM 0.20.1 base)
#
# Usage:
#   ./build.v020.sh                     # DGX Spark (SM121a + SM120a)
#   ./build.v020.sh --rtx               # RTX PRO 6000 (SM120a only)
#   ./build.v020.sh --jobs 8            # Limit parallel jobs
#   ./build.v020.sh --use-layer-cache   # Use podman layer cache
#
# Default: --no-cache (fresh git clone for branch updates).

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="vllm-multiquant-v020"
DOCKERFILE="Dockerfile.multiquant-v020"
BASE="nvcr.io/nvidia/vllm:26.03-py3"
BRANCH="multiquant-vllm-0.20"
ARCHS="12.0a;12.1a"
JOBS=16
EXTRA="--no-cache"

while [[ $# -gt 0 ]]; do
    case $1 in
        --rtx)              ARCHS="12.0a"; shift ;;
        --jobs)             JOBS="$2"; shift 2 ;;
        --use-layer-cache)  EXTRA=""; shift ;;
        --tag)              IMAGE="$2"; shift 2 ;;
        --branch)           BRANCH="$2"; shift 2 ;;
        *)                  echo "Unknown: $1"; exit 1 ;;
    esac
done

CACHE_BASE="/data/sources"
PIP_CACHE="$CACHE_BASE/pip-cache"
CCACHE_DIR="$CACHE_BASE/ccache"
TORCH_EXT="$CACHE_BASE/torch-extensions"
CUTLASS_DIR="$CACHE_BASE/cutlass"

mkdir -p "$PIP_CACHE" "$CCACHE_DIR" "$TORCH_EXT"

echo "================================================"
echo "  Building $IMAGE"
echo "  Dockerfile: $DOCKERFILE"
echo "  Base:    $BASE"
echo "  Branch:  $BRANCH"
echo "  Archs:   $ARCHS"
echo "  Jobs:    $JOBS"
echo "  Caches:  $CACHE_BASE/"
echo "================================================"

if [ ! -d "$CUTLASS_DIR/include" ]; then
    echo "WARN: $CUTLASS_DIR not populated — Dockerfile will git-clone CUTLASS"
fi

podman build \
    -f "$DOCKERFILE" \
    -t "$IMAGE" \
    --build-arg BASE_IMAGE="$BASE" \
    --build-arg CUDA_ARCHS="$ARCHS" \
    --build-arg MAX_JOBS="$JOBS" \
    --build-arg VLLM_BRANCH="$BRANCH" \
    -v "$PIP_CACHE:/pip-cache:rw" \
    -v "$CCACHE_DIR:/ccache:rw" \
    -v "$TORCH_EXT:/root/.cache/torch_extensions:rw" \
    -v "$CUTLASS_DIR:/opt/cutlass-local:ro" \
    $EXTRA \
    .

echo ""
echo "================================================"
echo "  Build done: localhost/$IMAGE"
podman images "localhost/$IMAGE" --format '  size: {{.Size}}'
echo "  Branch: $BRANCH"
echo "  ccache: $(du -sh "$CCACHE_DIR" 2>/dev/null | cut -f1)"
echo "  pip:    $(du -sh "$PIP_CACHE" 2>/dev/null | cut -f1)"
echo "================================================"
