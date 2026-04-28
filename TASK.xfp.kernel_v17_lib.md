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

## Warp-Layout vs Group-Boundary — Re-design ohne Quality-Verlust

**Initial Annahme (verworfen):** group_size muss Vielfaches der Warp-K-stride (256 für bits=4) sein.

**Korrekte Constraint:** `lanes_per_group ≥ LUT_SIZE`. Jede Lane hält EINE Codebook-Entry; mehrere Codebooks pro outer iter sind möglich, solange jeder eine vollständige Lane-Subgroup zugewiesen wird.

In v12 hält Lane `l` den Wert `cb[l]` (für `l < LUT_SIZE`), und `__shfl_sync(my_cb_val, idx)` broadcastet Lane `idx`'s Wert an alle Lanes. Für V2 wenn 2 Codebooks pro Warp-Iter aktiv sind:

- Lanes 0-15 sind in Group A, halten `library[lib_id_A][lane]` × scale_A + mid_A
- Lanes 16-31 sind in Group B, halten `library[lib_id_B][lane-16]` × scale_B + mid_B
- Lane in Group A dekodiert via `__shfl_sync(my_cb_val, 0 + idx)` → liest Lanes 0-15 → Codebook A ✓
- Lane in Group B dekodiert via `__shfl_sync(my_cb_val, 16 + idx)` → liest Lanes 16-31 → Codebook B ✓

**Same SHFL instruction, nur per-lane verschiedene src-lane**. Funktioniert.

Generalisierte Constraint:

| BITS | LUT_SIZE | max Codebooks/iter | **min group_size** | Aligned mit Phase-1 sweet spot? |
|---|---|---|---|---|
| 2 | 4  | 32 / 4  = 8 | 512 / 8 = **64** | (nicht im Sweep) |
| 3 | 8  | 32 / 8  = 4 | 320 / 4 = **80** | (awkward) |
| **4** | **16** | **32 / 16 = 2** | **256 / 2 = 128** | **✓ Sweet spot 0.99464 cos!** |

**v17_lib v1 wählt group_size=128 als default** (für bits=4). Damit verliert XFP-V2 **kein Quality** gegenüber Phase-1's optimaler Konfiguration. v17_lib v2 (Phase 3.2 später) könnte g=64 ergänzen via per-Lane-Library-Lookup mit 16 Registern/Lane (höhere Komplexität).

Der hot-loop hat dann pro Iter **2 Codebook-Reloads** (eine pro Lane-Subgroup):
- Lane-Subgroup-Index: `lane_group = lane / LUT_SIZE` (0 oder 1)
- `cb_lane_offset = lane_group * LUT_SIZE` (0 oder 16)
- Per Lane: `lib_id`, `scale`, `mid` für IHRE Group direkt aus Global Memory (lane-coalesced)

Konkret pro outer iter (bits=4, g=128):
```
int my_cb_idx = lane % LUT_SIZE;        // 0..15
int lane_group = lane / LUT_SIZE;        // 0 oder 1
int cb_lane_offset = lane_group * LUT_SIZE;  // 0 oder 16
int my_group_idx = gi * 2 + lane_group;
int lib_id = (int) ctx.group_lib_id[ctx.n * G + my_group_idx];
float scale_f = __half2float(ctx.group_scale[ctx.n * G + my_group_idx]);
float mid_f   = __half2float(ctx.group_mid  [ctx.n * G + my_group_idx]);
my_cb_val = __half2float(s_library[lib_id * LUT_SIZE + my_cb_idx]) * scale_f + mid_f;
```

Metadata reads sind über die 32 Lanes naturally coalesced (32 × 1 B lib_id + 32 × 2 B scale + 32 × 2 B mid = 160 B per outer iter), keine SHFL-Broadcasts nötig.

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

Dann im outer loop (bits=4, g=128 → 2 Codebooks pro outer iter, je 16 Lanes):

```cuda
// Outside the loop (constants):
constexpr int CB_PER_ITER = 2;                       // for g=128, bits=4
const int my_cb_idx       = lane % LUT_SIZE;          // 0..15
const int lane_group      = lane / LUT_SIZE;          // 0 or 1
const int cb_lane_offset  = lane_group * LUT_SIZE;    // 0 or 16

for (int gi = 0; gi < n_full_groups; gi++) {
    // ─── V2 PATCH: per-lane-group codebook reload ──
    // Each outer iter covers CB_PER_ITER groups; this lane belongs to
    // group (gi*CB_PER_ITER + lane_group).
    int my_group_idx = gi * CB_PER_ITER + lane_group;
    // Each lane reads ITS group's metadata directly (coalesced 32-byte
    // chunks across the warp — natural cache-line alignment).
    int   lib_id  = (int)   ctx.group_lib_id[ctx.n * G + my_group_idx];
    float scale_f = __half2float(ctx.group_scale[ctx.n * G + my_group_idx]);
    float mid_f   = __half2float(ctx.group_mid  [ctx.n * G + my_group_idx]);
    my_cb_val = __half2float(s_library[lib_id * LUT_SIZE + my_cb_idx])
                * scale_f + mid_f;

    // ─── EXISTING HOT LOOP, only SHFL src lane gets cb_lane_offset ──
    int kw = lane + gi * XFP_WARP_SIZE;
    uint32_t packed = ctx.B_packed[gi * N * XFP_WARP_SIZE + n_offset];
    int k_base = kw * VALS_PER_WORD;
    #pragma unroll
    for (int slot = 0; slot < VALS_PER_WORD; slot += 2) {
        __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(
            A_src + k_base + slot);
        float a0 = __bfloat162float(__low2bfloat16(a2));
        float a1 = __bfloat162float(__high2bfloat16(a2));
        int idx0 = (int)((packed >> (slot * BITS)) & MASK);
        int idx1 = (int)((packed >> ((slot + 1) * BITS)) & MASK);
        // V2: SHFL src lane is cb_lane_offset + idx (16 or 0 + idx)
        float w0 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx0);
        float w1 = __shfl_sync(0xffffffff, my_cb_val, cb_lane_offset + idx1);
        acc = fmaf(w0, a0, acc);
        acc = fmaf(w1, a1, acc);
    }
}
```

**Hot-Loop-Cost-Delta** pro outer iter:
- 3× per-lane Global-Loads (lib_id 1B, scale 2B, mid 2B) — coalesced, in L2-Cache, ~5-8 Cycles uncontended
- 1× SMEM load library entry — ~5 Cycles
- 1× FMA (`scale × cb + mid`) — 1 Cycle
- SHFL pattern unverändert (gleicher Throughput, nur src-lane = base + idx)

Pro outer iter: **~12 Zusatz-Cycles** vs ursprünglich 0. Outer iter macht `VALS_PER_WORD * 2 / 2 = 8` Slots × 2 FMA = 16 FMA + 16 SHFL = ~48 Cycles ohne V2. **Overhead: ~25% pro outer iter.** Da SMEM A-cache K_SMEM_MAX-Reuse den absoluten Wall-Clock dominiert, expect **5-12% Throughput-Verlust** gegen v12 V1 — knapper als die ursprüngliche 5-15% Schätzung weil keine Broadcasts nötig sind.

## SMEM-Budget v17_lib

Linear (K_SMEM_MAX=8192):
| Ressource | v12 V1 | v17_lib V2 | Δ |
|---|---|---|---|
| `s_A` (A-row) | 8192 × 2 B = 16 KB | 16 KB | 0 |
| `s_library` | — | 32 × 16 × 2 B = **1 KB** | +1 KB |
| **Total** | 16 KB | **17 KB** | +6% |

Plus die globalen Loads für lib_id/scale/mid sind 3 × G × 1-4 B per row. Werden NICHT in SMEM gepuffert (zu groß), sondern direkt aus L2 über die outer iter gelesen.

MoE (K_SMEM_MAX=4096) analog, library jetzt per Stack (eine pro w13 / w2 — wir haben ZWEI Libraries pro layer).

## Files-Liste (Phase 3.1: g=128, bits=4, kein Quality-Verlust)

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
