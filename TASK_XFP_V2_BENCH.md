# TASK: XFP V1 / V2 / V2a — Bench-Matrix für Paper

## Begriffs-Klarstellung (Stand 2026-05-04)

| Stufe | Algorithmik | Kernel | Cache-Schema | K-Limit | Quality |
|---|---|---|---|---|---|
| **V1** | per-channel Lloyd-Codebook (data-free), keine library | `xfp_gemm_v12.cu` (top-perf), `v11.cu` (K-fallback) | schema 1/2 | hard 8192 in v12 | mean cos 0.992; 122B Müll-Output |
| **V2** | per-group Lloyd + shared library across groups | `xfp_gemm_v17_lib.cu` + `_splitm.cu` (M≥8) | schema 3 (env XFP_V2=1) | hard 8192 in v17_lib (vor 04.05.) | mean cos 0.995 (+0.22pp), bits 4.31/param |
| **V2a** | V2 + K_SMEM_MAX 8192→32768 + cudaFuncSetAttribute 96 KB | gleicher v17_lib + splitm, lift in core_v2.cuh | schema 3 (gleich wie V2) | dynamic SMEM bis K=32768 | identisch zu V2 für K≤8192 |
| **V3** | echter split-K oder split-M Algorithmus | `xfp_gemm_*_splitk.cu` | TBD | beliebig | nicht vorhanden, nicht in Test |

**Kritische Punkte:**
- V2 = V2a auf 35B/122B (alle Linear-K ≤ 4096), unterschiedlich nur auf
  Q3.6-27B/GLM (Linear-down_proj K = 17408 / 10240).
- V1-Cache und V2-Cache **kohabitieren** unter unterschiedlichen
  cache_keys (env XFP_V2 wird in den Hash eingerechnet).
- V2 produziert das Pack-Manifest mit zusätzlichen Sidecars (library,
  group_lib_id, group_scale, group_mid). V1-Reader kann V2-Cache NICHT
  lesen und vice versa — kollision-frei.

## Stand der Messungen (vor diesem Plan)

| Modell | V1 GSM8K | V2 GSM8K | V2a GSM8K | V1 throughput | V2 throughput |
|---|---|---|---|---|---|
| 35B-A3B | 44/45/47% (27.04.) | ❌ | ❌ | bench.py 90% Math (22.04.) | ❌ |
| 122B-A10B | _Müll-Output_ (Smoke-Crash) | ❌ | ❌ | bench.py 98% Math (22.04.) | ❌ |
| Q3.6-27B (dense) | _nicht möglich (K=17408 > 8192 v12)_ | ❌ | ✅ Server bringt Smoke "Paris", GSM8K offen | — | ❌ |
| GLM-4.7-Flash | 60%/55%/❌ (S0-2, 26.04.) | ❌ | ❌ | bench.py 52% Math (22.04.) | ❌ |

**Lücken: 11 Bench-Runs offen** (8 GSM8K + 3 throughput).

## Kernel-Mikro-Bench (vorhanden)

- **Phase 3.0b** (`tests/xfp/bench_v17_vs_v12.py`): v17_lib (V2) vs v12 (V1)
  median **+37% latency** (worst case +42%, best +21%) auf RTX PRO 6000.
- **Phase 3.0c** (per-warp SMEM cache for group metadata) committed —
  nicht neu gemessen, sollte overhead verkleinern.
- **K=17408 / K=10240 correctness** (`test_kernel_v17_correctness.py`)
  cos=1.00000 (V2a Probe heute ✅).

## Bench-Plan

### A) Throughput-Probe (bench.py, ~30s pro Run)

Sehr schnell, gute Sanity. **bench.py in `~/vllm-riy/bench.py`** macht:
- 50 deterministische Math-Aufgaben + 3 prompt-types (short/medium/long)
- seed=42, temperature=0
- Output: tok/s long, math accuracy 50/50

```bash
# Pro Server-Konfig (Server up + bench.py call)
python3 bench.py --url http://localhost:8011 --model glm-4.7-flash \
    --label "<MODELL> <V1|V2|V2a>"
```

### B) Quality-Probe (lm-eval-harness GSM8K --limit 50, ~10 min)

Quick V2-quality-Sanity. Wenn ähnlich V1 oder besser → grün für full bench.

```bash
./bench.lm-eval --label "<modell>-xfp-<v1|v2|v2a>-probe" \
    --tasks gsm8k --limit 50 --seeds 1
```

### C) Full GSM8K (3 seeds, ~1.5h pro Run)

Nur bei OK-quality-Probe.

```bash
./bench.lm-eval --label "<modell>-xfp-<v1|v2|v2a>" --tasks gsm8k --seeds 3
```

## Reihenfolge / Plan

| # | Modell | Quant | Aktion | ETA |
|---|---|---|---|---|
| 1 | Q3.6-27B | V2a | Server up (Cache-Hit) + Smoke + bench.py + GSM8K --limit 50 | ~15 min |
| 2 | Q3.6-27B | V2a | Full GSM8K 3 seeds (nur falls Probe OK) | ~1.5h |
| 3 | 35B-A3B | V2 | Server start (V2-cache TBD, vermutlich miss → re-pack) + bench.py | ~25 min |
| 4 | 35B-A3B | V2 | GSM8K --limit 50 + full 3 seeds | ~1.5h |
| 5 | 35B-A3B | V2a | Identisch zu V2 (K klein), evtl. ein Sample zur Konsistenz | ~30 min |
| 6 | 122B-A10B | V2 | Server start + V2 fresh-pack + bench.py + GSM8K --limit 50 | ~45 min |
| 7 | 122B-A10B | V2 | Full GSM8K 3 seeds | ~3h |
| 8 | 122B-A10B | V2a | Sample (K klein, sollte == V2) | ~30 min |

**Total Wallclock:** ~8-10h sequentiell auf DGX.

## Erwartung / Hypothesen

| Modell × Quant | Hypothese GSM8K | Begründung |
|---|---|---|
| Q3.6-27B V2a | 50-65% strict | Reasoning-Modell, V2 (calibration-free) sollte deutlich besser als V1 sein, aber noch unter INT4 AR (Lorbus 60%) |
| 35B V2 | 55-75% strict | V1 war 44-47%; Phase 1 cos +0.22pp sollte ein paar pp helfen |
| 122B V2 | 70-85% strict | V1 produzierte Müll = strukturelles V1-Problem; V2-Library sollte das fixen |
| Throughput V2 vs V1 | -30% bis -10% | Phase 3.0b zeigt -37%; Phase 3.0c sollte mitigieren |

Falls 122B V2 GSM8K < 50% → V2 hat das V1-Problem nicht gelöst, deeper
analysis nötig (NaN-Layer-Check, group_size adjustment).

## Documentation Outputs

- **PAPER_MATRIX.md** updated mit V1/V2/V2a-Spalten
- **TAGEBUCH.20260504.md** Tageseintrag mit V2a-lift + V2-bench-Start
- **TASK_XFP_V2_BENCH.md** (diese Datei) — Plan-of-record

## Out-of-Scope

- V3 (echter split-K Kernel) — nur falls K > 32768 oder Throughput
  bei 1 Block/SM untragbar
- bits=2/3 V2-Pfade — Phase 3.4 deferred
- TP=2 RTX 397B — separater Workstream, siehe TASK_XFP_RTX.md
