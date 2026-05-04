# Paper-Matrix — XFP-vs-Calibrated-INT4-vs-FP8 Quality Comparison

**Stand:** 2026-05-04 (lm-eval-harness 0.4.11 / 5-shot GSM8K, 0-shot WikiText)
**Hardware:** DGX Spark (GB10, sm_121, 120 GiB UMA), single-GPU, FP8 KV-Cache
**Server:** vLLM 0.17.1 base + multiquant patches (`--enforce-eager`)
**Bench-Harness:** `bench.lm-eval` (3 seeds GSM8K, 3 seeds WikiText)

## Master-Matrix (mean strict-match GSM8K, mean WikiText PPL)

| Modell | Quant | bits/w | GSM8K mean ↑ | WikiText PPL ↓ | Δ vs BF16 (GSM8K) | Δ vs BF16 (PPL) |
|---|---|---|---|---|---|---|
| **Qwen3.5-35B-A3B** (MoE) | | | | | | |
| | BF16 | 16 | **91.64%** | **8.18** | — | — |
| | INT4 AutoRound | 4 | 89.97% | 8.44 | -1.7pp | +0.26 |
| | XFP3 (cos=0.98, V1) | ~3 | 45.46% | 9.65 | -46.2pp ⚠️ | +1.47 |
| **Qwen3.5-122B-A10B** (MoE) | | | | | | |
| | BF16 | 16 | (no fit, single GB10) | — | — | — |
| | INT4 AutoRound | 4 | **95.55%** | **5.94** | (BF16 unfit) | (BF16 unfit) |
| | XFP3 (V1) | ~3 | _Müll-Output_ | — | _XFP-V1 regression_ | — |
| **GLM-4.7-Flash** (MoE 30B-A3B) | | | | | | |
| | BF16 | 16 | **81.98%** | 18.06 | — | — |
| | FP8 | 8 | 82.13% | 18.91 | +0.15pp | +0.86 |
| | INT4 AutoRound | 4 | 80.41% | **16.94** | -1.6pp | -1.12 ⓘ |
| | XFP3 (V1) | ~3 | 57.48% (S0+1 only) | — | -24.5pp ⚠️ | — |
| **Qwen3.6-27B** (dense, reasoning) | | | | | | |
| | BF16 | 16 | 72.07% | **8.55** | — | — |
| | FP8 (Qwen-official) | 8 | **75.99%** | 8.57 | +3.9pp ⓘ | +0.03 |
| | INT4 AR (Lorbus) | 4 | 59.84% | 8.71 | -12.2pp ⚠️ | +0.16 |

ⓘ = Qwen3.6-27B FP8 > BF16 ist seed-Noise (deterministischer Rerun bestätigt
±2pp-Variabilität pro Seed) bzw. GLM-INT4 PPL-Win durch Calibration.
⚠️ = signifikanter Quality-Drop, paper-würdig zu diskutieren.

## Sub-Beobachtungen

### XFP-V1 Quality-Crash bei Reasoning-Modellen
- 35B XFP3 GSM8K: -46pp vs BF16 → strukturelles Problem
- GLM XFP3 GSM8K: -24pp vs BF16 → ähnliche Größenordnung relativ
- 122B XFP3: produzierte Müll-Output (komplett kaputt)
- Hypothesen: XFP-V1 (data-free, kein Calibration) erzeugt akkumulierten Drift
  über 1024-Token-Reasoning-Chains. INT4 AutoRound (calibrated) zeigt nur
  -1.7pp / -1.6pp Drop für die gleichen Modelle.
- **XFP-V2** (per-group + shared library, +0.22pp cos vs V1, env-gated `XFP_V2=1`)
  ist im Code (Phasen 1-4b committed), aber **noch nicht e2e validiert** auf
  realen Tasks. End-to-end-Test ist die nächste kritische Stufe.

### Qwen3.6-27B INT4 AR Drop ist auffällig groß (-12pp)
- Lorbus' Pack vs. Qwen-Official könnte unterschiedliche Calibration-Sets
  verwendet haben. 27B dense + reasoning + 4-bit ist anyway das engste
  Quant-Regime; ein zu generischer Calibration-Set verstärkt hier.

### FP8 ist nahezu lossless
- GLM-Flash: +0.15pp GSM8K, ~+0.85 PPL → praktisch BF16-Quality
- Qwen3.6-27B: +3.9pp (positiv!) GSM8K — within seed-noise
- FP8 ist die natürliche default für Production wo BF16-Footprint zu groß ist

## Lücken / Verbesserungsbedarf

| Lücke | Status |
|---|---|
| 35B FP8 | kein Modell verfügbar |
| 35B XFP-V2 | TODO (e2e-test mit `XFP_V2=1`) |
| 122B BF16 | nicht möglich (244 GB single-node) |
| 122B FP8 | Modell nicht heruntergeladen (5.7M-Stub) |
| 122B XFP-V2 | TODO (V1 hatte regression) |
| 122B INT4 RIY% | (nur Sample 1 vorhanden) |
| GLM-Flash XFP-V2 | TODO |
| GLM-Flash XFP S2-Retry | TODO (V1, möglicherweise gleicher Bug wie 35B) |
| Qwen3.6-27B XFP-V1 | nicht gepackt (V2 statt direkt) |
| Qwen3.6-27B XFP-V2 | TODO (dense + K=17408 testet split-K-Pfad) |
| 397B alle Quants | nur 1 Sample (RIY36% INT4) |
| MMLU für alle | TODO mit `bench.lm-eval --tasks mmlu` (batch_size=auto fix in `fa5e6313d`) |

## Quellen / Reproduzierbarkeit

```bash
# Alle GSM8K+WikiText raw results
find measurements/2026{0427,0503,0504}-eval-harness -name results.json | sort

# Aggregations-Skript (in-place ausgeführt mit Python)
# Siehe diese Datei selber, aber re-render via:
#   python3 -c "<aggregation script>"
```

Raw measurement dirs (deterministische WikiText-Werte zeigen Konsistenz
zwischen Reruns nach Reboot):

- `measurements/20260427-eval-harness/35b-bf16/` (GSM8K + WikiText, 3 seeds)
- `measurements/20260427-eval-harness/35b-int4-autoround/` (GSM8K + WikiText, 3 seeds)
- `measurements/20260426-eval-harness/35b-xfp-fp8kv-v3/` (GSM8K 3 seeds, WikiText 1 seed)
- `measurements/20260427-eval-harness/glm-bf16/` (3+3 seeds)
- `measurements/20260427-eval-harness/glm-fp8/` (3+3)
- `measurements/20260427-eval-harness/glm-int4-autoround/` (3+3)
- `measurements/20260426-eval-harness/glm-xfp/` (GSM8K 2 seeds, WikiText 0)
- `measurements/20260427-eval-harness/122b-int4-autoround/` (3+3)
- `measurements/20260503-eval-harness/qwen36-27b-{bf16,fp8,int4-autoround}/` (3+? seeds)
- `measurements/20260504-eval-harness/qwen36-27b-{bf16,int4-autoround}/` (WikiText gap-fill)
