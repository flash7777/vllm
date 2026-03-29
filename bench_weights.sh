#!/bin/bash
# Bench weight quantization variants
# Tests: native weights (INT4/FP8/BF16) vs Archer online-quant (TQ3/RQ3)
# Usage: ./bench_weights.sh
set -euo pipefail
cd "$(dirname "$0")"

MAX_LEN=40000
KV=tq3
RESULTS=()

test_variant() {
    local LABEL=$1
    local MODEL=$2
    local WEIGHTS=$3  # "" = native, "tq3"/"rq3" = Archer

    echo ""
    echo "══════════ $LABEL ══════════"
    podman stop mq-serve 2>/dev/null || true
    podman rm mq-serve 2>/dev/null || true
    sleep 2

    local ARGS=(--model "$MODEL" --kv "$KV" --max-model-len "$MAX_LEN")
    if [ -n "$WEIGHTS" ]; then
        ARGS+=(--weights "$WEIGHTS")
    fi
    ./start.multiquant "${ARGS[@]}"

    echo -n "  Warte... "
    for i in $(seq 1 120); do
        if curl -s http://localhost:8011/v1/models 2>/dev/null | grep -q "glm\|qwen\|llama"; then
            echo "ready (${i}x5s)"
            break
        fi
        if [ $i -gt 15 ] && podman logs mq-serve 2>&1 | tail -3 | grep -q "RuntimeError"; then
            echo "CRASHED"
            podman logs mq-serve 2>&1 | grep "Error:" | tail -1
            RESULTS+=("$LABEL: CRASHED")
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
        echo "  → $LABEL: $TOKS tok / ${MS}ms = $TPS tok/s  [mem: $MEM]"
        RESULTS+=("$LABEL: $TPS tok/s  mem=$MEM")
    else
        echo "  → $LABEL: FEHLER"
        RESULTS+=("$LABEL: FEHLER")
    fi
}

echo "MultiQuant Weight Bench — KV=$KV"
echo "════════════════════════════════════════"

# Native weights (no Archer)
test_variant "INT4 Marlin"  "GLM-4.7-Flash-int4-AutoRound"  ""
test_variant "FP8 Dynamic"  "GLM-4.7-Flash-FP8"             ""

# Archer online-quant (BF16 → compressed at load)
test_variant "BF16→Archer TQ3"  "GLM-4.7-Flash"  "tq3"
test_variant "BF16→Archer RQ3"  "GLM-4.7-Flash"  "rq3"

# FP8 model + Archer (FP8→compressed, double quant)
test_variant "FP8→Archer TQ3"   "GLM-4.7-Flash-FP8"  "tq3"

echo ""
echo "════════════════════════════════════════"
echo "  ERGEBNISSE (KV=$KV)"
echo "════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    printf "  %-20s %s\n" "$(echo "$r" | cut -d: -f1):" "$(echo "$r" | cut -d: -f2-)"
done
echo ""
echo "  FP8 Baseline (ohne MQ): 37-40 tok/s"
echo "════════════════════════════════════════"

podman stop mq-serve 2>/dev/null || true
