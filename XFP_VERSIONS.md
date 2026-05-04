# XFP Versions — V1 / V2 / V2a / V3

> Stand: 2026-05-04. Lebende Übersicht. Bei neuer Stufe hier ergänzen,
> nicht in Bench-Files duplizieren.

## Algorithmus-Versionen (V1 / V2 / V2a)

V-Stufen unterscheiden den **Quantisierungs-Algorithmus** und das
**Cache-Schema**. Innerhalb V2/V2a gibt es mehrere GEMM-Kernel
(split-N, splitm, splitk=V3) — siehe nächster Abschnitt.

| Version | Algorithmik | Cache-Schema | K-Limit | Default? | Status |
|---|---|---|---|---|---|
| **V1** | per-channel Lloyd-Codebook (data-free, kein Library) | schema 1/2 | hard 8192 in v12 | nur ohne `XFP_V2=1` | legacy — Müll-Output auf 122B, -46pp Quality auf Q3.6-27B |
| **V2** | per-group Lloyd + shared Library (≤64 Codebooks/Layer) | schema 3 | hard 8192 in v17_lib (vor 04.05.) | `XFP_V2=1` | aktiv; +0.22 pp cos vs V1, bits 4.31/param |
| **V2a** | V2 + K_SMEM_MAX 8192 → 32768 + cudaFuncSetAttribute 96 KB | schema 3 (gleich V2) | dynamic SMEM bis K = 32768 | `XFP_V2=1` (transparent ab 04.05.) | aktiv; **funktional bewiesen auf Q3.6-27B (K=17408)** Smoke + Math-Distributive-Property |

## Kernel-Varianten (split-N / splitm / splitk)

Innerhalb von V2/V2a gibt es **drei orthogonale GEMM-Kernels**, die je
nach (M, K)-Tile gewählt werden. **Sie unterscheiden sich darin, welche
Achse über mehrere Akkumulatoren bzw. Chunks zerteilt wird:**

| Kernel | Datei | Was wird zerteilt? | M-Range | K-Range | Zweck |
|---|---|---|---|---|---|
| **split-N** | `xfp_gemm_v17_lib.cu` | N-Achse (Output) auf Warps verteilt; **A-Row vollständig im SMEM** (`s_A` = K · bf16) | beliebig (top für M=1) | ≤ 32768 (mit V2a Lift) | Decode-Pfad, M=1 Prefill |
| **splitm** | `xfp_gemm_v17_lib_splitm.cu` | **M-Achse intern** (M_CHUNK=2/4/8 Akkumulatoren pro Block) | ≥ 16 | ≤ 8192 | Prefill-Tiles, amortisiert per-Group-Codebook-Rebuild |
| **splitk** (V3) | `xfp_gemm_v17_lib_splitk.cu` | **K-Achse in Chunks** (K_CHUNK ∈ {2048, 4096}); nur 1 Chunk im SMEM | beliebig (M_CHUNK ∈ {1,2,4}) | beliebig (>8192) | K-skalierender Pfad bei großem K |

**Wichtig:** splitm und splitk sind **nicht** verschiedene V-Stufen. Sie
sind alternative Kernels innerhalb V2/V2a für unterschiedliche (M, K)-
Profile. V3 ist die Bezeichnung **nur für splitk**, nicht für splitm.

## Wann welcher Pfad

### Default-Dispatcher (`dispatch_v2_linear_gemm`, `XFP_V3=0`)

| K | M | Pfad | Begründung |
|---|---|---|---|
| ≤ 8192 | < 16 | split-N v17_lib (V2) | top throughput für decode (M=1) und kleine prefill-Tiles |
| ≤ 8192 | ≥ 16 | splitm v17_lib_splitm (V2) | M_CHUNK=2/4/8 amortisiert Codebook-Rebuild über M-Akkumulatoren |
| > 8192 | beliebig | split-N v17_lib (V2a) | K_SMEM_MAX 32768 + 96 KB Carveout, **funktional auf Q3.6-27B** |

### Mit `XFP_V3=1`

| K | M | Pfad |
|---|---|---|
| ≤ 8192 | < 16 | split-N v17_lib (V2) |
| ≤ 8192 | ≥ 16 | splitm v17_lib_splitm (V2) |
| > 8192 | beliebig | **splitk v17_lib_splitk (V3)** — M_CHUNK aus M, K_CHUNK=4096 default |

V3 ist nicht "Pflicht für K > 8192" — V2a deckt diesen Fall funktional ab.
V3 ist eine Throughput-**Alternative** mit der Hypothese:
- V2a bei K=17408: SMEM s_A = 35 KB → ~2 Blocks/SM
- V3 bei K=17408 mit K_CHUNK=4096: SMEM s_A_chunk = 8-32 KB → 3+ Blocks/SM

Per Phase-3.0b RTX-Bench ist V3 bei K=8192 ~10-20% **langsamer** als
splitm — der Crossover-Punkt zugunsten V3 liegt bei K > 12-16K.

## Kernel-Files

```
kernels/multiquant/
├── xfp_gemm_v11.cu                  # V1 legacy (synchronous per-warp-global)
├── xfp_gemm_v12.cu                  # V1 top-perf (split-N M=1, K ≤ 8192)
├── xfp_gemm_core_v2.cuh             # V2 + V2a Linear-Policy (K_SMEM_MAX = 32768)
├── xfp_gemm_v17_lib.cu              # V2 split-N Linear (M=1)
├── xfp_gemm_core_v2_splitm.cuh      # V2 split-M Policy
├── xfp_gemm_v17_lib_splitm.cu       # V2 split-M Linear (M ≥ 16, K ≤ 8192)
├── xfp_gemm_core_v2_splitk.cuh      # V3 split-K core (K-chunk loop)
└── xfp_gemm_v17_lib_splitk.cu       # V3 split-K Linear (K > 8192)
```

## Env-Variablen

| Env | Wirkung | Default |
|---|---|---|
| `XFP_V2` | aktiviert V2-Algorithmus (per-group + Library, schema-3 cache) | `0` (V1) |
| `XFP_V2_LOG=verbose` | log jeden V2-dispatch (sehr chatty) | bucketed first-seen |
| `XFP_V2_LIBRARY_SIZE` | Codebook-Library-Größe pro Layer | 64 |
| `XFP_V2_GROUP_SIZE` | per-Group-Blocksize | 128 |
| `XFP_V3` | aktiviert V3 split-K Pfad für K > 8192 (statt V2a fallback) | `0` (V2a) |
| `XFP_MIN_COS` | minimaler cos-similarity threshold pro Layer | 0.99 |
| `XFP_AUTO_BITS` | dynamische Bit-Wahl bei Quality-Einbuße | `0` |

## Modell × K-Achse

Welche Modelle betrifft welcher K-Pfad?

| Modell | Layer | K | Pfad |
|---|---|---|---|
| Qwen3.5-35B-A3B | mlp.experts (MoE) | ~1024-2048 | V2 splitm/split-N |
| Qwen3.5-35B-A3B | qkv/o_proj | 4096 | V2 split-N |
| Qwen3.5-122B-A10B | mlp.experts (MoE) | ~1024-3072 | V2 splitm/split-N |
| Qwen3.5-122B-A10B | qkv/o_proj | 3072 | V2 split-N |
| **Qwen3.6-27B** (dense) | gate_up_proj | 5120 | V2 split-N |
| **Qwen3.6-27B** (dense) | **down_proj** | **17408** | **V2a default / V3 opt-in** |
| **GLM-4.7-Flash** | gate_up_proj | 4096 | V2 split-N |
| **GLM-4.7-Flash** | **down_proj** | **10240** | **V2a default / V3 opt-in** |

## Bench-Status

### Throughput (RTX PRO 6000 = Referenz-Hardware)

| Modell | V1 | V2 | V2a | V3 |
|---|---|---|---|---|
| 35B-A3B | bench.py 90% Math (22.04.) | ❌ pending | == V2 (K ≤ 4096) | n/a (K ≤ 4096) |
| **122B-A10B** | bench.py 98% Math (22.04.) | **138 tok/s** ✅ | == V2 (K ≤ 3072) | n/a (K ≤ 3072) |
| Q3.6-27B | n/a (K=17408 > v12-Limit) | n/a (K=17408 > v17_lib pre-V2a) | ❌ pending (RTX) | ❌ pending (in devel) |
| GLM-4.7-Flash | bench.py 52% Math (22.04.) | ❌ pending | ❌ pending (RTX) | ❌ pending (in devel) |

**Performance-Referenz = RTX**, nicht DGX. DGX-Messungen werden
weiter geführt (eigenes Profil, Quality-OK), aber für Throughput-
Vergleiche **nicht aussagekräftig** — DGX und RTX sind weit
auseinander. Die 138 tok/s 122B V2 kamen von RTX.

### GSM8K (Quality)

Stand der GSM8K-Messungen (vgl. `TASK_XFP_V2_BENCH.md` für Details):

| Modell | V1 | V2 | V2a | V3 |
|---|---|---|---|---|
| 35B-A3B | 44/45/47% (27.04.) | ❌ pending | == V2 (K ≤ 4096) | n/a (K ≤ 4096) |
| 122B-A10B | Müll-Output | ❌ pending | == V2 (K ≤ 3072) | n/a (K ≤ 3072) |
| Q3.6-27B | n/a (K=17408 > 8192 v12) | n/a (K=17408 > pre-V2a v17_lib) | ✅ Smoke ok, GSM8K offen | ❌ pending (in devel) |
| GLM-4.7-Flash | 60/55/❌ (S0-2, 26.04.) | ❌ pending | ❌ pending | ❌ pending (in devel) |

## Migrationspfad / Empfehlungen

- **Neue Modelle aufsetzen:** `XFP_V2=1` setzen, V3 nicht aktivieren (V2a deckt alle K-Werte ≤ 32768)
- **Throughput-Tuning K > 8192:** V3 als Alternative ausprobieren (`XFP_V3=1`), gegen V2a benchen
- **V1 cache vorhanden, V2 fresh-pack nicht möglich:** V1 lädt weiter mit `XFP_V2=0`, kollidiert nicht mit V2-cache (separate cache-keys)

## Out-of-Scope (deferred)

- bits=2/3 V2-Pfade (Phase 3.4)
- 2D-Tile (M+K-split kombiniert)
- group_size != 128 (Phase 3.4)
