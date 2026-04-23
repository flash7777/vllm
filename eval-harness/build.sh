#!/bin/bash
# Build + tag the MultiQuant eval-harness container.
# Usage:  eval-harness/build.sh  [additional podman build args]
set -euo pipefail

cd "$(dirname "$0")"

TAG="localhost/vllm-eval-harness:lm-0.4.11"
podman build -t "$TAG" "$@" .

echo ""
echo "Built $TAG"
podman images "$TAG" --format "{{.Repository}}:{{.Tag}}  {{.Size}}"
