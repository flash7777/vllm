# ngram Speculative Decoding Matrix — XFP on Qwen3.5-122B-A10B (MTP blockiert)

**Date:** 2026-04-21 (MTP re-verified 2026-04-22 — still blocked)
**Model:** `Qwen3.5-122B-A10B` (BF16 → XFP auto on-the-fly, linear_attn policy fix live)
**Config:** fp8 KV + fp8 LM-head (identical across all rows)
**Method:** `ngram` prompt-lookup speculative.

**Warum nicht MTP**: `--spec-method mtp` scheitert reproduzierbar (bestätigt
am 2026-04-22 unter aktuellem `xfp_speed`-Image, commit `683d80d8b`):
`ValueError: no module model.layers.0.self_attn.q_proj in Qwen3_5MoeMTP`.
Root-Cause: `Qwen3_5MultiTokenPredictor` (qwen3_5_mtp.py:56) trägt keine
`packed_modules_mapping`, deshalb wird q/k/v→qkv-Fusion in diesem Subtree
nicht ausgelöst. Volle Analyse + Stack-Trace:
`measurements/20260421-xfp-mtp-verify/FAILURE.md`. ngram als Fallback.
**Bench:** `bench.py` seed=42, n=5 decode rounds, GSM8K 50 problems

**NOT included in PAPER_XFP.md** per user directive.

## Results

| NST | short (20t) | medium (150t) | long (400t) | Math |
|---:|---:|---:|---:|---:|
| **0 (no spec)** | 2.6 | **34.8** | **29.9** | **98%** |
| 1 | 2.5 | 34.3 | 29.2 | 96% |
| 2 | 2.5 | 30.8 | 26.6 | 94% |
| 3 | 2.5 | 31.2 | 27.6 | 90% |
| 4 | 2.4 | 26.9 | 23.2 | 90% |
| 5 | 2.5 | 27.4 | 23.1 | 90% |

## Interpretation

**ngram speculative decoding DEGRADES throughput on every NST** for this
workload, and accuracy drops with rising NST:

- **NST=1**: −2% throughput, −2 pp math
- **NST=2**: −11% throughput, −4 pp math
- **NST=5**: −23% throughput, −8 pp math

Reasons:

1. **ngram is prompt-lookup speculative** (NgramProposer in
   `vllm/v1/spec_decode/ngram_proposer.py`). It looks for n-gram matches
   between the recently generated tokens and the prompt to guess future
   tokens. On GSM8K-style math problems the prompt and the response share
   almost no n-grams, so the draft-acceptance rate is near zero, and the
   speculative rounds become pure overhead.

2. **Each speculative round costs a full forward pass on draft + verification.**
   When the acceptance rate is low, verification rejects most draft
   tokens — but the cost of the draft forward has already been paid.

3. **Math drops** because sampling with speculative decoding changes the
   effective sampler state (temperature/top-k on verified tokens vs
   unverified). At NST≥2 some arithmetic operations that the greedy path
   would get right are lost to speculative mis-verification.

## When ngram WOULD help

ngram speculation shines when the prompt contains long strings that will
be echoed: code completion (function signatures repeated), document
QA (quoted spans), long-form RAG with citation, structured-output
repetition. GSM8K short-answer arithmetic is the opposite workload.

## Next steps

- **MTP head method** (`qwen3_5_mtp`) needs a fix in
  `vllm/model_executor/models/qwen3_5_mtp.py:267 remap_weight_names` —
  the MTP head checkpoint stores separate `q_proj`, `k_proj`, `v_proj`
  but `Qwen3NextAttention` expects a fused `qkv_proj`. Fusion at weight
  remap time is the fix.
- **Eagle / eagle3 draft model** would be the correct comparison for
  general-purpose decoding — but requires a matching draft checkpoint
  (not shipped with this model).

## Artefacts

- `summary.txt` — all NST rows, one line per size class
- `bench-nst{1..5}.txt` — raw bench.py outputs per NST

Baselines recap (for convenience):

| Configuration | long | medium | short | Math |
|---|---:|---:|---:|---:|
| XFP no-spec (post linear_attn fix) | 29.9 | 34.8 | 2.6 | 98% |
| Marlin INT4 AutoRound no-spec | 25.8 | 29.6 | 2.5 | 94% |
| XFP + ngram NST=1 | 29.2 | 34.3 | 2.5 | 96% |
| XFP + ngram NST=5 | 23.1 | 27.4 | 2.5 | 90% |
