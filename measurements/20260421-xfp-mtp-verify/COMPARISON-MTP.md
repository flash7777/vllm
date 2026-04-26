# MTP Speculative Decoding Matrix — XFP on Qwen3.5-122B-A10B

**Date:** 2026-04-22
**Image:** `localhost/vllm-multiquant:xfp_speed` +
`qwen3_5_mtp.py` live-mount mit Fix
**Model:** `Qwen3.5-122B-A10B` BF16 → XFP auto + fp8 KV + fp8 LM-head
**Method:** `--spec-method mtp --spec N` (vLLM `method=mtp`)
**Bench:** `bench.py` seed=42, n=5 decode rounds, GSM8K 50 problems

## Fix

Bug: `Qwen3_5MultiTokenPredictor` (qwen3_5_mtp.py:56) hatte keine eigene
`load_weights`, AutoWeightsLoader's `packed_modules_mapping`-Lookup
griff im Submodul-Subtree nicht, q/k/v→qkv-Fusion blieb aus →
`ValueError: no module or parameter named 'model.layers.0.self_attn.q_proj'`.

Fix (qwen3_5_mtp.py:154-232): manuelles `load_weights` mit
`stacked_params_mapping` nach Muster `qwen3_next_mtp.py:136-219`:

```python
def load_weights(self, weights):
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    # + expert_params_mapping via FusedMoE.make_expert_params_mapping
    # + manual weight_loader(param, loaded_weight, shard_id) dispatch
```

Zusätzlicher Mount in `start.multiquant:290`:
```
-v "$HOME/vllm-riy/vllm/model_executor/models/qwen3_5_mtp.py:...:ro"
```

Load-Bestätigung im Log:
```
[eagle.py:1419] Detected MTP model. Sharing target model lm_head weights
                with the draft model.
[gpu_model_runner.py:4578] Model loading took 61.94 GiB memory and 1316s
```
→ 6 MTP-Layer (2 attn + 2 routed + 2 shared), avg cos=0.989, ~3.4 eff bits.

## Results

| NST | short | medium (150t) | long (400t) | Math | vs no-spec long |
|---:|---:|---:|---:|---:|---:|
| 0 (no spec, baseline) | 2.6 | 34.8 | 29.9 | 98% | — |
| 1 | 2.2 | 35.9 | 29.3 | 98% | −2% |
| 2 | 2.5 | 33.7 | 29.1 | 92% | −3% |
| **3** | **2.5** | **37.2** | **32.7** | **98%** | **+9.4%** |
| 4 | 2.5 | 26.3 | 24.0 | 98% | −20% |
| 5 | 2.5 | 26.5 | 25.0 | 96% | −16% |

## Interpretation

**MTP NST=3 ist der Sweet Spot**: +9.4 % long throughput, +6.9 % medium,
Math bleibt bei 98 % (identisch zu no-spec). Erster echter Speedup mit
Speculative Decoding auf XFP-Qwen3.5.

- NST=1/2: Draft-Kosten ≈ Accept-Gewinn — flach bis negativ. NST=2 hat
  Math-Drop auf 92 % (beobachtet auch bei ngram-Matrix — vLLM-Sampling
  unter Speculation-Verifikation scheint bei mittleren NST instabil).
- NST=3: Accept-Rate reicht, um 3 Drafts pro Forward zu amortisieren.
  Math voll erhalten → Verifikation funktioniert.
- NST=4/5: Overhead der Draft-Rounds überwiegt. Bei M=1 decode ist der
  Draft-Forward auf GB10 bereits HBM-bandwidth-limitiert; 4-5 Drafts
  verstopfen den Pfad.

## Vergleich MTP vs ngram (beide bei NST=1)

| Config | long | medium | Math |
|---|---:|---:|---:|
| MTP NST=1 | 29.3 | **35.9** | **98%** |
| ngram NST=1 | 29.2 | 34.3 | 96% |

MTP NST=1 verliert medium-Speedup **nicht** und hält Math bei 98 % —
korrekte Verifikation, während ngram bei NST=1 bereits 2pp Math verliert.
Ab NST=3 dominiert MTP ngram deutlich.

## Albond-Vergleich

Albond hat auf 35B-A3B BF16 + INT4 AutoRound + MTP-2 + INT8-LMH v2
113–127 tok/s peak berichtet. Unser NST=3 MTP auf 122B kommt auf
32.7 tok/s long — skaliert man den Active-Param-Ratio (10B/3B = 3.33×)
heraus, entspricht das einem implizierten 35B MTP-Peak von ~109 tok/s
(32.7 × 3.33), nahe Albonds 113. Dieselbe Messung auf 35B steht noch
aus.

## Next Steps

1. RESULTS.xfp.v12.md §1.1: MTP NST=3 Zeile ergänzen als
   `MTP speculative: +9.4% throughput, math preserved`.
2. Qwen3.5-35B-A3B MTP-Check: sollte analog funktionieren (selbe
   packed_modules_mapping Logik, gleicher MTP-Layer-Count pro Config).
3. vLLM-Upstream-PR: der Fix ist generisch für alle
   `Qwen3_5MoeMTP`-Varianten gültig. Einzureichen mit Issue-Referenz.

## Artefakte

- `bench-mtp-nst{1..5}.txt` — raw bench.py Outputs
- `../20260421-xfp-distributions/qwen3.5-122b-a10b-xfp-auto-mtp.log`
  — 6584 Zeilen XFP-Distributionen inkl. MTP-Layer
- `FAILURE.md` — historische Fehlerbeschreibung (obsolet nach Fix)
