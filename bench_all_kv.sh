#!/bin/bash
# Bench all KV cache variants sequentially
# Usage: ./bench_all_kv.sh [model_name]
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-GLM-4.7-Flash-int4-AutoRound}"
MAX_LEN=40000
RESULTS=()

test_kv() {
    local KV=$1
    echo ""
    echo "══════════ $KV ══════════"
    podman stop mq-serve 2>/dev/null || true
    podman rm mq-serve 2>/dev/null || true
    sleep 2

    ./start.multiquant --model "$MODEL" --kv "$KV" --max-model-len "$MAX_LEN"

    echo -n "  Warte... "
    for i in $(seq 1 100); do
        if curl -s http://localhost:8011/v1/models 2>/dev/null | grep -q "glm\|qwen\|llama"; then
            echo "ready (${i}x5s)"
            break
        fi
        if [ $i -gt 10 ] && podman logs mq-serve 2>&1 | tail -3 | grep -q "RuntimeError"; then
            echo "CRASHED"
            podman logs mq-serve 2>&1 | grep "Error:" | tail -1
            RESULTS+=("$KV: CRASHED")
            return
        fi
        sleep 5
    done

    # Warmup
    curl -s http://localhost:8011/v1/completions -H "Content-Type: application/json" \
        -d '{"model":"glm-4.7-flash","prompt":"hi","max_tokens":5}' > /dev/null 2>&1
    sleep 1

    # Bench: 100 tokens
    local START=$(date +%s%N)
    local RESP=$(curl -s http://localhost:8011/v1/completions -H "Content-Type: application/json" \
        -d '{"model":"glm-4.7-flash","prompt":"Explain attention in transformers in detail:\n\n","max_tokens":100,"temperature":0}')
    local END=$(date +%s%N)
    local TOKS=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
    local MS=$(( (END - START) / 1000000 ))

    # Memory
    local MEM=""
    local CONTAINER=$(podman ps --filter "name=mq-serve" --format "{{.Names}}" | head -1)
    if [ -n "$CONTAINER" ]; then
        MEM=$(podman exec "$CONTAINER" python3 -c "
import torch
a=torch.cuda.memory_allocated()/1024**3
p=torch.cuda.max_memory_allocated()/1024**3
print(f'{a:.1f}/{p:.1f} GiB')
" 2>/dev/null || echo "?")
    fi

    if [ "$TOKS" -gt 0 ]; then
        local TPS=$(python3 -c "print(f'{$TOKS / ($MS / 1000):.1f}')")
        echo "  → $KV: $TOKS tok / ${MS}ms = $TPS tok/s  [mem: $MEM]"
        RESULTS+=("$KV: $TPS tok/s  mem=$MEM")
    else
        echo "  → $KV: FEHLER"
        RESULTS+=("$KV: FEHLER")
    fi
}

echo "MultiQuant KV Bench — Model: $MODEL"
echo "════════════════════════════════════════"

for KV in tq3 tq4 rq3 rq4 rq2; do
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
echo "  FP8 Baseline: 37-40 tok/s"
echo "════════════════════════════════════════"

podman stop mq-serve 2>/dev/null || true
