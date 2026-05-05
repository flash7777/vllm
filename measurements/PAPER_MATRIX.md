# Paper-Matrix — XFP-vs-Calibrated-INT4-vs-FP8 Quality Comparison

**Stand:** 2026-05-05 (lm-eval-harness 0.4.11 / 5-shot GSM8K, 0-shot WikiText)
**Hardware:** **RTX PRO 6000 (sm_120, 96 GB)** für V2a-Bench-Reihe
**Hardware (alt):** DGX Spark (GB10, sm_121, 120 GiB UMA) für DGX-Daten
**Server:** vLLM 0.17.1 base + multiquant patches
**Bench-Harness:** `bench.lm-eval` (3 seeds GSM8K, 3 seeds WikiText) bzw. n=50-Probe für V2a

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

## RTX V2a-Bench-Reihe (2026-05-05)

**Hardware:** RTX PRO 6000 (sm_120, 96 GB), GPU 1 isoliert (GPU 0 = mq-serve PreV2a-Referenz)
**Image:** localhost/vllm-multiquant:latest (re-built 2026-05-04 mit V2a + V3 splitk + custom_op fixes)
**Bench-Größe:** GSM8K **n=50, seed=0** (Probe — Δ BF16↔XFP-V2a relevant, nicht Absolutwert)

| Modell | Quant | GSM8K strict (n=50) | Δ vs BF16 | bench.py tok/s (medium) | Math 50 |
|---|---|---|---|---|---|
| **Qwen3.5-35B-A3B** (MoE) | BF16 baseline (full, 02.05.) | 76.02% | — | — | — |
| | XFP-V2 (full, 02.05.) | 77.18% | **+1.16 pp** | — | — |
| | **XFP-V2a (n=50)** | **76.0% ±6.1** | innerhalb stderr | **200.4** | 92% |
| **Qwen3.5-122B-A10B** (MoE) | BF16 | nicht möglich (>1 GPU) | — | — | — |
| | INT4 Marlin (full, 02.05.) | 95.27% | — | — | — |
| | XFP-V2 (full, 02.05.) | 94.62% | -0.65 pp vs Marlin | — | — |
| | **XFP-V2a (n=50)** | **98.0% ±2.0** | (n=50, plausibel innerhalb stderr) | **106.7** | 96% |
| **Qwen3.6-27B** (dense, MM) | **BF16 (n=50)** | **64.0% ±6.86** | — | **28.6** | 96% |
| | **XFP-V2a (n=50)** | **58.0% ±7.05** | -6 pp (within stderr) | **36.7** (+28%) | 94% |
| **GLM-4.7-Flash** (MoE Lite) | **BF16 (n=50)** | **68.0% ±6.66** | — | **116.8** | 70% |
| | **XFP-V2a (n=50)** ✅ | **76.0% ±6.1** | +8 pp (n=50 noise) | **116.8** | 68% |

### V2 → V2a Pfad-Validation
- **35B** (K ≤ 4096): V2a == V2 algorithmisch, n=50 Probe innerhalb stderr ✅
- **122B** (K ≤ 4096): V2a == V2 algorithmisch, n=50 Probe innerhalb stderr ✅
- **Q3.6-27B** (K=17408): V2a-Pfad funktional ✅ (-6 pp innerhalb stderr, +28% Throughput vs BF16)
- **GLM** (K=10240): V2a-Pfad funktional ✅ via XFP_SKIP_LAYERS=kv_b_proj (47 kv_b_proj layers stay BF16, ~24 MB overhead). MLA-Absorption findet weight wieder, kein AttributeError.

### Throughput-Beobachtung
122B XFP-V2a misst 106.7 tok/s medium / 68.3 long auf RTX, vs alte Erinnerung "138 tok/s" (PreV2a). Möglicherweise Mikro-Regression durch cudaFuncSetAttribute oder custom_op-overhead. Nicht qualitäts-relevant.

### GLM-MLA Inkompatibilität (gefixed, commit 6a3fea816)
Glm4MoeLiteForCausalLM nutzt MLA — `mla_attention.py:766` ruft `get_and_maybe_dequant_weights(kv_b_proj)`. XFP-V2 hatte `del layer.weight` gemacht. **Fix:** env `XFP_SKIP_LAYERS` (default `kv_b_proj`), substring-Match auf layer_prefix. Skipped layers behalten `weight` BF16 + `_xfp_skipped=True`. `apply()` short-circuit auf `F.linear`. Memory-overhead: 47 × ~0.5 MB = ~24 MB BF16, vs ~14 GB XFP-Total → vernachlässigbar.

## Lücken / Verbesserungsbedarf

| Lücke | Status |
|---|---|
| 35B XFP-V2a full GSM8K 3 seeds | n=50 ✅, full ausstehend |
| 122B XFP-V2a full GSM8K 3 seeds | n=50 ✅ (98.0%), full ausstehend |
| Q3.6-27B XFP-V2a full GSM8K 3 seeds | n=50 ✅ (58.0%), full ausstehend |
| GLM XFP-V2a full GSM8K 3 seeds | n=50 ✅ (76.0%), full ausstehend |
| 122B FP8 | Modell nicht heruntergeladen (5.7M-Stub) |
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
