#!/bin/bash
# Bench all KV cache variants sequentially
# Usage: ./bench_all_kv.sh [model_name]
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-GLM-4.7-Flash-int4-AutoRound}"
RESULTS=()

test_kv() {
    local KV=$1
    echo ""
    echo "══════════ $KV ══════════"

    # Stop any running serve containers
    podman stop mq-serve 2>/dev/null || true
    podman rm mq-serve 2>/dev/null || true

    # Check for competing GPU processes
    local GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
    if [ "$GPU_PROCS" -gt 0 ]; then
        echo "  WARNING: $GPU_PROCS GPU process(es) still running:"
        nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        echo "  Waiting 10s for cleanup..."
        sleep 10
    fi
    sleep 2

    ./start.multiquant --model "$MODEL" --kv "$KV"

    echo -n "  Warte... "
    for i in $(seq 1 100); do
        if curl -s http://localhost:8011/v1/models 2>/dev/null | grep -q "glm\|qwen\|llama"; then
            echo "ready (${i}x5s)"
            break
        fi
        if [ $i -gt 10 ] && podman logs mq-serve 2>&1 | tail -5 | grep -q "RuntimeError"; then
            if podman logs mq-serve 2>&1 | grep -q "out of memory\|OOM\|less than desired.*memory"; then
                echo "OOM"
                RESULTS+=("$KV: OOM")
            else
                echo "CRASHED"
                podman logs mq-serve 2>&1 | grep "Error:" | tail -1
                RESULTS+=("$KV: CRASHED")
            fi
            return
        fi
        if [ $i -eq 100 ]; then
            echo "TIMEOUT"
            RESULTS+=("$KV: TIMEOUT")
            return
        fi
        sleep 5
    done

    # Run bench.py (perf + math + optional context)
    echo "  Benchmarking..."
    python3 "$(dirname "$0")/bench.py" \
        --url http://localhost:8011 \
        --model glm-4.7-flash \
        --label "$KV" \
        --perf-rounds 3 \
        --math-count 10 2>&1 | tee /tmp/bench_${KV}.log | grep -E "^\s+(short|medium|long|Math):"

    # Extract summary for results table
    local TPS=$(grep "short" /tmp/bench_${KV}.log 2>/dev/null | awk '{print $2}' | head -1)
    local MATH=$(grep "Math:" /tmp/bench_${KV}.log 2>/dev/null | head -1 | sed 's/.*Math: //')
    RESULTS+=("$KV: short=${TPS:-?} tok/s  $MATH")
}

echo "MultiQuant KV Bench — Model: $MODEL"
echo "════════════════════════════════════════"

# Baselines first, then MultiQuant variants
for KV in auto fp8 tq3 tq4 rq3 rq4 rq2; do
    test_kv "$KV"
done

echo ""
echo "════════════════════════════════════════"
echo "  ERGEBNISSE"
echo "════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    printf "  %-6s %s\n" "$(echo "$r" | cut -d: -f1):" "$(echo "$r" | cut -d: -f2-)"
done
echo ""
echo "  auto=BF16 KV, fp8=FP8 KV (baselines)"
echo "════════════════════════════════════════"

podman stop mq-serve 2>/dev/null || true
