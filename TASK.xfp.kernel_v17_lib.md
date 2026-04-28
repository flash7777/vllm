# TASK: XFP-V2 Kernel `xfp_gemm_v17_lib` (Phase 3)

**Status:** Plan-Phase. Phase 1, 2, 4a, 4b sind committed (Pack/Cache/Online).
**Datum:** 2026-04-28
**Ziel:** Forward-Kernel der V2-Logik, der die existierenden v12-Optimierungen (SMEM A-cache, Warp-interleave, fused Outlier-Scatter, Marlin-pattern für MoE) **vollständig erhält**.

## Engineering-Investment, der zwingend bleibt

`xfp_gemm_v12` schlägt Marlin um ~17%. Die Optimierungen die das ermöglichen:

| Optimierung | Ort im Code | Was sie tut |
|---|---|---|
| **Static SMEM A-row cache** | `xfp_gemm_core.cuh:55-86` | A-Row einmal coop in `s_A[K_SMEM_MAX]` → alle 8 Warps lesen aus SMEM |
| **Warp-interleaved B layout** | `_repack.py` + ctx.B_packed indexing | jede Warp-iter lädt N×WS=256 packed words coalesced |
| **Codebook in SHFL** | core:104-107 | lane `l` hält cb[lane], `__shfl_sync(idx)` macht Lookup ohne SMEM |
| **Hot-loop FMA** | core:240-251 | bf162 vector load + 2× fmaf per slot, kein Branch |
| **Tail-Loop separate** | core:255-271 | Fast-path hat KEINE Bounds-Checks |
| **Marlin pattern (MoE)** | MoEPolicy.prologue | sorted_token_ids/expert_ids — eine Block pro (token, expert) Paar |
| **Block-uniform A_row** | `block_A_row` policies | `nullptr` early-exit ohne `__syncthreads`-Mismatch |
| **Templated by BITS** | core:39 | `BITS` constexpr → Compiler unrollt VALS_PER_WORD slot loop |

**Strategie:** v17_lib ist **kein Rewrite** — es ist `v12 + 30-50 LOC Patch im inner dequant loop`, und ein **neuer Policy-Slot** für die V2-Per-Group-Metadata. Das Kernel-File (xfp_gemm_v17_lib.cu) ruft den selben `xfp_gemm_core<BITS, LinearPolicyV2>` Template; die Policy macht den Unterschied.

## Warp-Layout vs Group-Boundary — der Knackpunkt

In v12 verteilt eine Warp ihre 32 Lanes über `XFP_WARP_SIZE` packed words pro outer iter. Pro packed word werden `VALS_PER_WORD` weights dekodiert. Also:

| BITS | VALS/WORD | Warp covered K per outer iter |
|---|---|---|
| 2 | 16 | 32 × 16 = **512** |
| 3 | 10 | 32 × 10 = **320** |
| **4** | **8** | **32 × 8 = 256** |

In v12 ist die Codebook konstant über alle K der Row → kein Conflict. In V2 wechselt die effektive Codebook **jeder `group_size` Weights**. Damit ein outer iter SAUBER innerhalb einer einzigen Group bleibt, muss:

```
group_size  ≡  0  (mod warp_K_stride)
```

Für bits=4 also: **group_size ∈ {256, 512, 1024, 2048, ...}**.

Aus den Phase-1 Daten:
| g | cos | bits/param |
|---|---|---|
| 64  | 0.99606 | 4.62 |
| 128 | 0.99464 | 4.31 |
| **256** | **0.99332** | **4.16** ← v17_lib v1 target |
| 512 | (nicht gemessen, vermutlich schlechter)| 4.07 |

**v17_lib v1 wählt group_size=256 als default**: aligned mit warp K-stride für bits=4, beste Kombi aus quality + bit-budget bei Phase-3-Komplexität-Minimum. Verliert ~0.13pp gegen g=128, gewinnt aber Kernel-Einfachheit.

**v17_lib v2 (Phase 3.2 später):** g=128 oder g=64 via per-Lane-Library-Lookup (16 Register/Lane statt SHFL). Höhere Quality, höhere Komplexität. Zurückgestellt.

## Hot-Path-Diff (kommentiert)

Aktueller v12 Code (`xfp_gemm_core.cuh:104-107`):

```cuda
// Codebook in register, SHFL lookup
float my_cb_val = (lane < LUT_SIZE)
    ? __half2float(ctx.codebook_slice[lane])
    : 0.0f;
```

V2 wird:

```cuda
// Library in SMEM (loaded once per block, before the K-loop)
extern __shared__ half s_library[/* LIBRARY_SIZE * LUT_SIZE */];
{
    int total = LIBRARY_SIZE * LUT_SIZE;
    for (int i = threadIdx.x; i < total; i += XFP_BLOCK_SIZE) {
        s_library[i] = ctx.library_global[i];
    }
}
__syncthreads();

// Codebook per group — ALL 32 lanes load DIFFERENT centroid for the SAME group.
// Reload in inner loop (once per outer iter for g=256, bits=4).
float my_cb_val = 0.0f;
```

Dann im outer loop:

```cuda
for (int gi = 0; gi < n_full_groups; gi++) {
    // ─── V2 PATCH: per-group codebook reload ──
    // For g=256, every outer iter is exactly one group.
    int group_idx = gi;  // (gi * XFP_WARP_SIZE * VALS_PER_WORD) / GROUP_SIZE
    // Lane 0 reads metadata; broadcast scalars via SHFL.
    int lib_id;
    half scale_h, mid_h;
    if (lane == 0) {
        lib_id  = (int)ctx.group_lib_id[ctx.n * G + group_idx];
        scale_h = ctx.group_scale[ctx.n * G + group_idx];
        mid_h   = ctx.group_mid[ctx.n * G + group_idx];
    }
    lib_id = __shfl_sync(0xffffffff, lib_id, 0);
    float scale_f = __half2float(__shfl_sync(0xffffffff, scale_h, 0));
    float mid_f   = __half2float(__shfl_sync(0xffffffff, mid_h, 0));
    // Each lane reloads ITS centroid from library + applies (scale, mid).
    my_cb_val = (lane < LUT_SIZE)
        ? __half2float(s_library[lib_id * LUT_SIZE + lane]) * scale_f + mid_f
        : 0.0f;

    // ─── EXISTING HOT LOOP UNCHANGED ──
    int kw = lane + gi * XFP_WARP_SIZE;
    uint32_t packed = ctx.B_packed[gi * N * XFP_WARP_SIZE + n_offset];
    int k_base = kw * VALS_PER_WORD;
    #pragma unroll
    for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
        __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(A_src + k_base + slot);
        float a0 = __bfloat162float(__low2bfloat16(a2));
        float a1 = __bfloat162float(__high2bfloat16(a2));
        int idx0 = (int)((packed >> (slot * BITS)) & MASK);
        int idx1 = (int)((packed >> ((slot + 1) * BITS)) & MASK);
        float w0 = __shfl_sync(0xffffffff, my_cb_val, idx0);
        float w1 = __shfl_sync(0xffffffff, my_cb_val, idx1);
        acc = fmaf(w0, a0, acc);
        acc = fmaf(w1, a1, acc);
    }
}
```

**Hot-Loop-Cost-Delta:** vor jedem outer iter zusätzlich:
- 3× SHFL broadcast (lib_id, scale, mid) — ~6 Cycles
- 1× SMEM load für library entry — ~10 Cycles uncontended
- 1× FMA (scale × cb + mid) — 1 Cycle

Pro outer iter: ~17 Zusatz-Cycles vs ursprünglich 0. Aber outer iter macht `VALS_PER_WORD * 2 = 16` Slots × 2 FMA = 32 FMA + 16 SHFL = ~48 Cycles. **Overhead: ~35% pro outer iter, aber SMEM-A-cache + B coalesced bleiben — der absolute Latenz-Anteil wird kleiner durch K_SMEM_MAX-Reuse.**

Erwartung: **5-15% Throughput-Verlust gegen v12 V1.** Annehmbar als Tradeoff für Quality-Win.

## SMEM-Budget v17_lib

Linear (K_SMEM_MAX=8192):
| Ressource | v12 V1 | v17_lib V2 | Δ |
|---|---|---|---|
| `s_A` (A-row) | 8192 × 2 B = 16 KB | 16 KB | 0 |
| `s_library` | — | 32 × 16 × 2 B = **1 KB** | +1 KB |
| **Total** | 16 KB | **17 KB** | +6% |

Plus die globalen Loads für lib_id/scale/mid sind 3 × G × 1-4 B per row. Werden NICHT in SMEM gepuffert (zu groß), sondern direkt aus L2 über die outer iter gelesen.

MoE (K_SMEM_MAX=4096) analog, library jetzt per Stack (eine pro w13 / w2 — wir haben ZWEI Libraries pro layer).

## Files-Liste (Phase 3.1: g=256-aligned)

Neue Files:
- `kernels/multiquant/xfp_gemm_v17_lib.cu` (Copy von v12 + LinearPolicyV2)
- `kernels/multiquant/xfp_moe_gemm_v17_lib.cu` (Copy von xfp_moe_gemm_v12 + MoEPolicyV2)

Modifizierte Files:
- `kernels/multiquant/xfp_gemm_core.cuh`:
  - Neuer Policy-Slot: `Policy::has_v2_metadata` constexpr, plus optional Methods `Policy::group_lib_id_for(n, group)`, `Policy::group_scale_for`, `Policy::group_mid_for`, `Policy::library_ptr`.
  - Hot-Loop-Patch (`#ifdef XFP_CORE_V2_GROUP_LIB`).
- `kernels/multiquant/xfp_gemm.cu` (top-level dispatch): erweitern um v17_lib Pfad wenn V2-Tensoren vorhanden.

Neue Linear-Policy `LinearPolicyV2` :
```cuda
struct LinearPolicyV2 : LinearPolicy {
    // Identical to LinearPolicy except Ctx carries V2 pointers.
    struct Ctx {
        // ... base fields from LinearPolicy::Ctx ...
        const half* library;        // [LUT * LIBRARY_SIZE] fp16
        const uint8_t* group_lib_id;// [N, G] (or [N*G])
        const half* group_scale;    // [N, G]
        const half* group_mid;      // [N, G]
        int G;
    };
    // group_size and library_size as compile-time templates
    template <int BITS, int LUT>
    __device__ static Ctx prologue(...);
};
```

MoEPolicyV2 analog mit per-Expert-Offset für `group_lib_id`, `scale`, `mid`.

## Python-API-Patch

`vllm/multiquant/xfp/xfp_kernel.py`:
- `_load_xfp_gemm_v2()` JIT-kompiliert `xfp_gemm_v17_lib.cu`
- Dispatch in `online_linear.apply()` V2-Branch: wenn Kernel-Lib geladen, calls `xfp_gemm_v17_lib(...)` statt Python-Reference.

`vllm/multiquant/xfp/xfp_moe_kernel.py` analog.

## Test-Strategie

### Stage 3.0 — Korrektness vs Python-Reference

`tests/xfp/test_kernel_v17_correctness.py`:
- 32 Layer × {bits=4, group_size=256, lib=32}
- Pack via `xfp_pack_v2`, attach to a layer
- Python: `dequant_xfp_v2_packed → x @ W_rec.T`
- Kernel: `xfp_gemm_v17_lib(x, packed, library, group_lib_id, scale, mid, ...)`
- Pass: `cos(y_kernel, y_python) ≥ 0.999` und `max_abs_err ≤ 1e-3 * |y_python|.max()`.

### Stage 3.1 — Throughput vs v12 V1

`tests/xfp/bench_v17_vs_v12.py`:
- Same matrix shapes (z.B. 4096×4096, batch sizes 1, 8, 64)
- Messung: tok/s, ns/output-element
- Pass: v17_lib Latenz innerhalb 1.20× von v12 (max +20% slowdown akzeptabel).

### Stage 3.2 — End-to-End

35B-A3B XFP_V2=1 + Kernel: full GSM8K via lm-eval-harness.
Pass-Kriterium: ≥ 0.85 GSM8K flex (closing 80%+ der BF16-Lücke). Schon Phase 4a Python-Reference erreicht das mathematisch — Kernel reproduziert die Math, also gleiches Ergebnis erwartet. Hier wird Geschwindigkeit der Bottleneck: ≥ V1 latency × 1.20 ist OK.

### Stage 3.3 — MoE

Analog mit xfp_moe_gemm_v17_lib + bench_moe_v17_vs_v12.py.
Plus E2E mit 122B-A10B (oder 35B-A3B mit MoE-Layer aktiviert).

## Risiko-Register

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|---|---|---|---|
| Register pressure → spill bei lib_id+scale+mid Variables | mittel | groß (50% slowdown) | Profile mit `nvcc --ptxas-options=-v`; ggf. `__launch_bounds__` reduzieren oder Werte in SMEM cachen |
| SHFL broadcast Latenz dominiert bei kleinem K | niedrig | mittel | Prefetch der nächsten Group's metadata 1 outer iter ahead (cp.async-style) |
| Library-SMEM 1KB führt zu weniger Block-Concurrency auf SM | niedrig | niedrig | Library ist klein genug; CUDA scheduler kann viele Blocks pro SM stacken |
| Per-(N,G)-Metadata schlechte L2-cache-Hit-Rate | niedrig | mittel | Layout `[N, G]` ist bereits N-major (matches warp's n stride); G-Achse ist sequenziell durch outer-loop |
| Floating-point Genauigkeit weicht von Python-Ref ab (fp16 codebook vs fp32) | mittel | niedrig | Existierender v12 hat gleiche fp16-Codebook-Struktur, Toleranz ≥ 0.999 cos sollte halten |

## Phasen-Reihenfolge (ich empfehle strikt sequenziell)

1. **3.0 — LinearPolicyV2 + xfp_gemm_v17_lib.cu** (Linear only, g=256, bits=4)
2. **3.0a — Kernel-Korrekt­heits­test** vs Python-Reference (Stage 3.0)
3. **3.0b — Bench Linear v17 vs v12** (Stage 3.1)
4. **3.1 — MoEPolicyV2 + xfp_moe_gemm_v17_lib.cu** (analog für MoE)
5. **3.1a — MoE-Korrekt­heits­test + Bench**
6. **3.2 — Python-API Dispatch** (online_linear.apply + online_moe.apply nutzen Kernel statt Python-Reference)
7. **3.3 — E2E GSM8K Bench** auf 35B-A3B; Pass = quality entspricht Python-Reference + Latenz ≤ 1.20× v12

Erst nach 3.3-Pass: optional Phase 3.4 = group_size flexibilisieren (g=128 via per-Lane-Library-Lookup).

## Out-of-Scope für Phase 3.1

- Calibration (gradient-aware codebook tuning) — Phase 5+
- Outlier-Pfad in V2 — orthogonal, kann mit existing v9-Outlier-Code-Pfad reaktiviert werden falls Quality-Bedarf
- bits=2 oder bits=3 V2-Support — erst nach bits=4 stable
