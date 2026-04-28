# TASK: XFP-V2 — Codebook-Library + per-Group Quantization

**Status:** Konzept-Phase, Daten validiert.
**Datum:** 2026-04-28

## Motivation

XFP heute: per-channel learned codebook (16 Centroids pro Row).
Test ergab nur **+0.47pp cos** gegenüber naivem int4-RTN-per-channel,
und **−0.4pp cos** gegenüber int4 per-group g=32. Der angekündigte
"per-Expert/per-Channel-Codebook"-USP zündet strukturell nicht für
Qwen3.5-A3B-Verteilungen (gaussian-symmetric, std≈0.006).

Konsequenz auf GSM8K (35B-A3B, TP=1, full-set 1319 problems):
- BF16 Baseline: 0.912
- int4-AutoRound iter=0: 0.894 (−2pp, per-group RTN)
- **XFP-4 (heute): 0.679 (−23pp)** ← strukturelles Quality-Defizit

## Zwei-Dimensionen-Diagnose

### Dim 1 — Quantisierungs-Granularität (Quality)

| Methode | cos (avg über 8 Layer-Klassen) | bits | per-row Overhead |
|---|---|---|---|
| int4 per-channel (1 scale/row) | 0.984 | 4 | 2 B |
| **XFP-4 per-channel codebook (16 cents/row)** | **0.992** | 4 | 32 B |
| int4 per-group g=128 (16 scales/row) | 0.991 | 4 | 32 B |
| **int4 per-group g=32 (64 scales/row)** | **0.995** | 4 | 128 B |

→ **Group-size schlägt Codebook-Lernen.** Aber XFP-V2 könnte BEIDES
kombinieren: gelernte Centroids in fein-granulärer Group-Anordnung.

### Dim 2 — SMEM-Realität (Hardware)

GPU shared-memory pro SM ~ 100 KB nutzbar. Davon braucht der GEMM
Tile-Loading der Input-Activation A (z.B. 32 KB) und KV-Cache-Bereich.
Codebook konkurriert um den Rest.

| Variante | SMEM/Tile (64 rows) | passt? |
|---|---|---|
| XFP heute (per-row codebook) | 2 KB | ✓ reichlich |
| int4 per-group g=128 (16 scales) | 2 KB | ✓ |
| int4 per-group g=32 (64 scales) | 8 KB | ✓ |
| **XFP per-group g=32 (64 codebooks/row)** | **128 KB** | ✗ **zu groß** |

→ Naive per-group XFP-codebook scheitert an SMEM. **Codebook-Library
ist der Ausweg**: feste prototype codebooks sharen sich Library-weit,
pro Group nur ein 4-8-Bit Index.

## Datenbasierte Library-Größenwahl

`tests/xfp/test_codebook_library_size.py` auf 35B-A3B Layer-0,
42048 per-row codebooks aus 8 Linear-Klassen + 16 routed-experts:

| Library-Größe | p5 cos | p50 cos | min cos |
|---|---|---|---|
| **8** | **0.998** | **0.999** | 0.983 |
| **16** | **0.998** | **1.000** | 0.984 |
| 32 | 0.999 | 1.000 | 0.986 |
| 256 | 0.999 | 1.000 | 0.989 |

→ **16 Prototyp-Codebooks reichen** für quasi-verlustfreie Approximation
aller per-row codebooks (gaussian-symmetric Verteilungen → wenige
Form-Cluster). 42048 codebooks komprimieren auf 16 Prototypen ist
**2628× Reduktion** des Codebook-Overheads.

## Vorgeschlagenes XFP-V2-Design

### Daten-Layout

Pro Layer (post-pack):
```
library:        [L, 16] fp16    # L ∈ {16, 32}, normalisiert auf [-1, +1]
indices_packed: [N, K/2] uint8  # 4-bit weight indices into per-group codebook (heute)
group_lib_id:   [N, G]   uint4  # G = K // group_size, je 4 Bit Library-Index
group_scale:    [N, G]   fp16    # per-group magnitude (factor out norm)
group_mid:      [N, G]   fp16    # per-group midpoint (für asymmetrische Distributionen)
```

Pro group_size=32, K=2048, N=1024:
- group_lib_id: 1024 × 64 × 4 bit = 32 KB
- group_scale + mid: 1024 × 64 × 4 B = 256 KB  ← der Hauptoverhead
- library: 1024 B (resident in SMEM!)

→ **Realistic group_size ist g=128 oder g=256**, nicht g=32 (zu viel
scale/mid overhead). Mit g=128: scale+mid = 64 KB pro 1024-row Layer
≈ 0.25 bits/param Overhead. Acceptable.

### Forward-Kernel-Path

Pro Group eines Threads-Tile:
1. Load `lib_id` für die Group → in Register
2. Load entsprechende 16 Library-Centroids aus SMEM (Library voll
   resident in SMEM, je Layer ~512B−1KB)
3. Pro Weight im Group: 4-bit-Index → centroid_norm
4. Dequant: `weight = group_mid + group_scale × centroid_norm`
5. Multiply mit Activation-A, accumulate

Vergleich zu heute: **+1 Indirection** (Library-Lookup), aber
Library ist L1/SMEM-resident → effektiv kein Latenz-Hit.

### Pack-Pipeline (offline / load-time)

1. **Pro Group fitten**: Lloyd auf den `group_size` Werten der Row →
   16 normalisierte centroids + scale + midpoint.
2. **Library aufbauen** (einmalig pro Layer-Class oder pro Modell):
   - Sammle alle normalisierten Group-Codebooks
   - K-means mit `library_size` (z.B. 32) auf den 16-D Codebook-Vektoren
   - Sortiere Library-Entries für deterministische Reihenfolge
3. **Pro Group**: nearest-library-id zuweisen → speichere als 4-Bit-Index.
4. **Per-Group Re-Assignment der Weight-Indices** anhand des
   gewählten Library-Codebooks.

## Reuse-Map: was bleibt, was kommt dazu

V2 ist explizit ein **Delta** auf der bestehenden Pack/Kernel-Infrastruktur,
nicht ein Rewrite. Die Engineering-Investition in Lloyd, repack, outlier-
scatter, SMEM-A-cache, MoE-Marlin-pattern wird vollständig wiederverwendet.

### Pack-Pipeline (`xfp_pack.py`)

| Heute (V1) | V2 (Delta) |
|---|---|
| `_lloyd_per_channel(W, 16, iters)` für `[N, K]` | **wiederverwendet** — über reshape `[N, K] → [N*G, group_size]` |
| `_pack_indices(idx, bits)` | **wiederverwendet** unverändert |
| `xfp_repack` (warp-interleaved 1D) | **wiederverwendet** unverändert |
| Outlier-Extraktion (`outlier_sigma`, `outlier_max_fraction`) | **wiederverwendet** — wirkt orthogonal zur Library |
| `xfp_pack` Top-Level | erweitert: ruft Lloyd auf den Group-Reshape, fügt Library-K-means+Re-Assignment dazu |

Konkret: `xfp_pack_v2(W, bits=4, group_size=128, library_size=32)` ist ein
Wrapper, der `_lloyd_per_channel` mit `[N*G, group_size]`-Reshape ruft (jede
Group ist eine virtuelle Row), die resultierenden Codebooks in der
Library clustert (k-means), und neue group-relative Indices schreibt.

### Kernel (`kernels/multiquant/`)

| Heute (V1) | V2 (Delta) |
|---|---|
| `xfp_gemm_v12.cu` (production) | **Basis** — das wird in `xfp_gemm_v17_lib.cu` per Copy + Patch erweitert |
| Static SMEM A-row cache (K_SMEM_MAX=8192 linear, 4096 MoE) | **unverändert** — derselbe Tile-Pattern |
| Warp-interleaved index-loading | **unverändert** |
| Fused outlier-scatter (v9-Style) | **unverändert** — outlier-Pfad bleibt orthogonal |
| `xfp_gemm_core.cuh` (shared dequant loops) | **erweitert** um eine Library-Lookup-Variante |
| MoE-Marlin-pattern (`xfp_moe_gemm_v12`) | **Basis für** `xfp_moe_gemm_v17_lib` analog |

Patch-Größe: erwarte **~30-50 LOC pro Kernel-Variante**. Der Kernel
macht ein einziges zusätzliches `lib_id = group_lib_id[row, group]`
und einen extra Index-Hop in der Library statt direkt in den per-row
Codebook. Die Library wird einmal pro Kernel-Start in SMEM geladen
(512 Bytes-1 KB pro Layer) und ist dann frei zugänglich.

### Cache (`weight_cache.py`, `xfp_weight_cache.py`)

| Heute (V1) | V2 (Delta) |
|---|---|
| `save/load` mit `tensors` dict + `metadata` | **wiederverwendet** — V2 fügt zusätzliche Tensoren hinzu |
| Schema-version 3 | erweitert auf 4 (V2-marker) |
| `tensor_meta` für TP-slicing | **wiederverwendet** — neue Tensoren bekommen passende `tp_role`-Tags |
| `cache_key` (model + policy + env) | erweitert um `group_size`, `library_size` |

V1-Caches bleiben loadbar; V2 läuft parallel bis stable.

### Online (`online_linear.py`, `online_moe.py`)

`process_weights_after_loading` bekommt einen V2-Branch (gegated auf
env `XFP_V2=1`). `apply()` ruft `xfp_gemm_v17_lib` statt `v12` wenn
Library-Tensoren vorhanden. Branch-Patch ~20 LOC pro File.

## Implementierungs-Plan (Phasen)

### Phase 1 — Pack/Decode (CPU-only, kein Kernel)

Ziel: `xfp_pack_v2()` + `dequant_xfp_v2()` reine Python/PyTorch.
Verifizierbar ohne CUDA.

Files:
- `vllm/multiquant/xfp/xfp_pack.py`:
  - Neue Funktion `xfp_pack_v2(W, bits=4, group_size=128, library_size=32)`
    - Reshape `W[N,K] → W_groups[N*G, group_size]` mit `G = K // group_size`
    - Existierender `_lloyd_per_channel(W_groups, 16, iters=20)` produziert
      `[N*G, 16]` Codebooks (UNGEÄNDERT)
    - Neue `_build_library(codebooks, library_size)`: k-means → `[L, 16]` lib
    - Re-quantize: jede Group → nearest-library + 4-bit weight indices
    - Repack via existierendem `xfp_repack` (UNGEÄNDERT)
    - Outlier-Pfad via existierender Logik (UNGEÄNDERT, optional disable)
  - Output-Tuple: `(packed, library, group_lib_id, group_scale, group_mid,
    o_idx, o_val, stats)`
- `dequant_xfp_v2(packed, library, group_lib_id, group_scale, group_mid, K, bits)`:
  - Zerlegt packed in groups, lookup `library[group_lib_id[n,g]]`, gather
    centroids per index, multiply by `group_scale`, add `group_mid`.

Verifikation:
- `tests/xfp/test_pack_v2_quality.py`: 8 Layer-Klassen, vs V1 + vs int4-g32.
- Pass: avg cos ≥ 0.997 (besser als int4-g=32's 0.995); worst-row ≥ 0.99.

### Phase 2 — Cache-Format

Files:
- `vllm/multiquant/xfp/xfp_weight_cache.py`:
  - `save_linear_v2/save_moe_v2`: existierendes `cache.save()` mit
    erweitertem `tensors`-dict (zusätzlich `library`, `group_lib_id`,
    `group_scale`, `group_mid`). Outlier-Tensoren bleiben optional.
  - `load_*_v2`: spiegelbildlich.
  - `tensor_meta` für TP-slicing: `library` ist replicated, `group_lib_id`
    folgt der weight-row tp_role (column oder row parallel), `group_scale`
    und `group_mid` ebenso.
- `vllm/multiquant/weight_cache.py`:
  - `_MANIFEST_SCHEMA_VERSION = 4` (oder eigener Schema-bump nur für XFP)
  - `compute_cache_key` integriert `XFP_GROUP_SIZE` und `XFP_LIBRARY_SIZE` envs.

V1-Caches bleiben gültig; V2 erkennt am Method-string (`xfp_linear_v2`).

### Phase 3 — Forward-Kernel (Delta auf v12)

Files:
- `kernels/multiquant/xfp_gemm_v17_lib.cu` (Copy von `v12.cu`):
  - Im SMEM-Setup: zusätzlich Library laden (klein, 16 × 16 × 2 Byte = 512B)
  - Im inner-loop dequant: ersetze `cb[row, idx]` durch
    `lib[group_lib_id[row, group], idx] * group_scale[row, group] + group_mid[row, group]`
  - SMEM-A-cache, warp-interleave, outlier-scatter UNVERÄNDERT
- `kernels/multiquant/xfp_moe_gemm_v17_lib.cu` (Copy von `v12.cu`-MoE-Version):
  - Gleicher Patch im inner-loop. Marlin-pattern (sorted_token_ids) UNVERÄNDERT.
- `vllm/multiquant/xfp/xfp_kernel.py`:
  - V17-Pfad in `_load_xfp_gemm` einfügen, gegated auf `_xfp_v2_active`.
- `vllm/multiquant/xfp/online_linear.py:apply()`:
  - V2-Branch: gleicher Custom-Op-Pfad (wie V1), nur erweiterte Args.

Verifikation:
- Output-cos zwischen V2-Kernel und Phase-1 Python-Reference ≥ 0.999.
- Wenn niedriger: kernel-mathematischer Bug. Bisect via `xfp_gemm_v9`-Style
  diagnostic-Mode (kernel mit print).

### Phase 4 — MoE + E2E Validation

Files:
- `vllm/multiquant/xfp/online_moe.py`:
  - V2-Branch im `process_weights_after_loading` (gleiche Logik wie Linear,
    aber pro Expert).
- E2E auf 35B-A3B + 122B + 397B mit den existierenden start.multiquant-Aufrufen
  und Bench-Scripten.

Pass-Kriterien:
- Phase 1: cos vs BF16 ≥ 0.997, besser als int4-g32.
- Phase 3: kernel cos vs Phase-1 ≥ 0.999 (kein Math-Bug im Kernel).
- Phase 4: 35B-A3B GSM8K flex ≥ 0.85 (closing 80%+ der Lücke zu BF16).
  Bei ≥ 0.89 → schlägt int4-AutoRound iter=0.

## Was NICHT in V2 ist (separater Task)

- **Calibration** (activation-aware codebook tuning, wie AutoRound):
  optionale Verfeinerung in V3 wenn V2 nicht ausreicht.
- **Outlier-Extraktion**: bleibt wie heute (4σ replicated). V2-Group-Codebooks
  könnten outliers besser absorbieren → outlier-Pfad evtl. obsolet.
- **Dynamic group_size pro Layer-Klasse**: auto-Wahl von g=64/128/256
  basierend auf Verteilungs-Heavy-Tailedness.

## Verifikation / Acceptance-Tests

```bash
# Phase 1 standalone (CPU, schnell):
podman run --rm -v $HOME/vllm-riy/tests:/tests:ro \
    -v $HOME/vllm-riy/vllm/multiquant:/usr/local/lib/python3.12/dist-packages/vllm/multiquant:ro \
    -v /data/tensordata:/data/tensordata:ro \
    localhost/vllm-multiquant:latest \
    python3 /tests/xfp/test_pack_v2_quality.py

# Phase 4 E2E (TP=1 35B):
ssh -p 2020 root@10.249.0.99
cd /root/vllm-riy && \
    XFP_V2=1 XFP_GROUP_SIZE=128 XFP_LIBRARY_SIZE=32 \
    ./start.multiquant --model Qwen3.5-35B-A3B-BF16 \
        --tp 1 --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8 \
        --max-model-len 32768 --gpu-memory-utilization 0.5 \
        --eager --max-num-batched-tokens 4096
# wait API ready, then:
podman exec -e OPENAI_API_KEY=dummy mq-serve lm_eval \
    --model local-completions \
    --model_args base_url=http://localhost:8011/v1/completions,model=glm-4.7-flash,num_concurrent=8,max_retries=3,tokenized_requests=False,tokenizer_backend=None \
    --tasks gsm8k --num_fewshot 5 \
    --gen_kwargs max_gen_toks=1024,temperature=0,seed=0 \
    --output_path /tmp/eval_xfp_v2
```

Pass-Kriterien:
- Phase 1: cos vs BF16 ≥ 0.997 für alle Layer-Klassen → besser als g=32.
- Phase 3: Forward-Output cos vs Phase-1-dequant ≥ 0.999 → kernel-mathematisch korrekt.
- Phase 4: GSM8K flex ≥ 0.85 (closing 80%+ der Lücke zu BF16). Wenn
  ≥ 0.89 erreicht: V2 schlägt int4-AutoRound iter=0 — Innovation.

## Daten-Anhang

### Layer-Class-Statistiken (35B-A3B Layer 0)

| Klasse | cos XFP-V1 | cos int4-g128 | cos int4-g32 | norm | std |
|---|---|---|---|---|---|
| ATTN in_proj_qkv | 0.9950 | 0.9928 | 0.9955 | 63.6 | 0.0155 |
| ATTN out_proj | 0.9919 | 0.9914 | 0.9946 | 45.4 | 0.0157 |
| ATTN in_proj_z | 0.9939 | 0.9929 | 0.9951 | 49.7 | 0.0172 |
| SHARED gate_proj | 0.9911 | 0.9891 | 0.9938 | 12.8 | 0.0125 |
| SHARED down_proj | 0.9925 | 0.9900 | 0.9940 | 11.3 | 0.0110 |
| ROUTED expert 0 gate_up | 0.9926 | 0.9916 | 0.9947 | 9.12 | 0.0063 |

### Library-Coverage bei lib_size=16 (per-class)

Aus `test_codebook_library_size.py` Output (42048 codebooks):
- p5 cos = 0.998
- p50 cos = 1.000
- min cos = 0.984

→ Praktisch verlustfreie Library-Approximation bei 16 Prototypen.
