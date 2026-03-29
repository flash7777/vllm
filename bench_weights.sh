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

    local MTP=${4:-false}

    # KV dtype: use Archer method if set, else default
    local KV_USE="$KV"
    if [ -n "$WEIGHTS" ]; then
        KV_USE="$WEIGHTS"  # Archer rq3 → KV rq3
    fi

    local ARGS=(--model "$MODEL" --kv "$KV_USE" --max-model-len "$MAX_LEN")
    if [ -n "$WEIGHTS" ]; then
        ARGS+=(--weights "$WEIGHTS")
    fi
    if [ "$MTP" = "true" ]; then
        ARGS+=(--mtp)
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

echo "MultiQuant Weight Bench"
echo "═══════════════════════════════════════════════════════════════════"
printf "  %-4s  %-12s  %-10s  %-6s  %-3s\n" "#" "Modell-Quant" "Archer" "KV" "MTP"
echo "───────────────────────────────────────────────────────────────────"

#                    Label                Model                            Archer  MTP
#       Modell-Quant | Archer-Gewichte | KV-Cache | MTP
test_variant "INT4  / —    / tq3 / —"    "GLM-4.7-Flash-int4-AutoRound"  ""      ""
test_variant "FP8   / —    / tq3 / —"    "GLM-4.7-Flash-FP8"            ""      ""
test_variant "BF16  / tq3  / tq3 / —"    "GLM-4.7-Flash"               "tq3"   ""
test_variant "BF16  / rq3  / rq3 / —"    "GLM-4.7-Flash"               "rq3"   ""
test_variant "FP8   / tq3  / tq3 / —"    "GLM-4.7-Flash-FP8"           "tq3"   ""
test_variant "INT4  / —    / tq3 / mtp"  "GLM-4.7-Flash-int4-AutoRound" ""      "true"
test_variant "FP8   / —    / tq3 / mtp"  "GLM-4.7-Flash-FP8"           ""      "true"
test_variant "BF16  / tq3  / tq3 / mtp"  "GLM-4.7-Flash"               "tq3"   "true"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ERGEBNISSE"
echo "───────────────────────────────────────────────────────────────────"
printf "  %-30s  %s\n" "Modell / Archer / KV / MTP" "tok/s     mem"
echo "───────────────────────────────────────────────────────────────────"
for r in "${RESULTS[@]}"; do
    printf "  %-30s  %s\n" "$(echo "$r" | cut -d: -f1)" "$(echo "$r" | cut -d: -f2-)"
done
echo "───────────────────────────────────────────────────────────────────"
echo "  FP8 Baseline (ohne MQ):       37-40 tok/s"
echo "═══════════════════════════════════════════════════════════════════"

podman stop mq-serve 2>/dev/null || true
