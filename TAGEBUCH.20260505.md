# TAGEBUCH 2026-05-05

## Tagesziel

V2a-Bench-Reihe auf **RTX PRO 6000** (Referenz-Hardware) für 4 Modelle:
- Qwen 3.5 35B-A3B
- Qwen 3.5 122B-A10B
- Qwen 3.6 27B (dense, multimodal, K=17408)
- GLM-4.7-Flash (MoE Lite, K=10240)

User-Vorgabe: nur GPU 1 verwenden (GPU 0 = mq-serve Pre-V2a 122B-Referenz, läuft seit 47h). Bench immer auf der Modell-Maschine (nicht remote).

## Verlauf

### 1. RTX-Bestandsaufnahme + Cleanup

- mq-serve auf GPU 0: ✅ läuft 122B XFP-V2 mit XFP_V2=1, alter Image (21.04.), Pre-V2a-Referenz
- xfp35b-bf16-serve + nifty_aryabhata (lm_eval) gestoppt — User-Freigabe für GPU 1

**Existing measurements gefunden:**
- `paper-35b-bf16-bf16-baseline` (03.05.): 76.02% strict (full, 3 seeds)
- `paper-35b-xfp-v2-tp1` (02.05.): **77.18%** strict, +1.16 pp über BF16
- `paper-122b-marlin-tp1` (02.05.): 95.27% strict
- `paper-122b-tp1-128k-final` (02.05.): **94.62%** strict — bestätigt als XFP-V2 (mq-serve env XFP_V2=1)
- `paper-122b-tp2-fresh` (02.05.): 94.49% TP=2

### 2. Q3.6-27B BF16 Baseline ✅

- Container `q36-27b-bf16` (Image vllm-multiquant:latest, alter), GPU 1, Port 8013
- `Qwen3_5ForConditionalGeneration` multimodal, BF16, KV bf16
- bench.py: short 28.3 / medium 28.6 / long 24.7 tok/s, Math 96%
- GSM8K --limit 50: **64.0%** strict ±6.86

### 3. GLM-4.7-Flash BF16 Baseline ✅

- Container `glm47-bf16`, parallel zu vllm-multiquant Image-Build (CPU-bound)
- `Glm4MoeLiteForCausalLM`, hidden=2048, intermediate=10240, 47 layers, topk=4
- bench.py: short 103.5 / medium 116.8 / long 115.4 tok/s, Math 70% (überraschend schwach)
- GSM8K --limit 50: **68.0%** strict ±6.66

### 4. RTX → DGX Repo-Sync (verlustreich)

**Lehre:** Bei RTX-lokal-Modifikationen + hard-reset auf origin/multiquant nur die unique-Files (`xfp_moe_gemm_v17_lib.cu`, `test_kernel_v17_splitm_correctness.py`, `start.multiquant2.sh`) salvaged. Die 9 modified-File-Hunks (`online_linear.py`, `online_moe.py`, `xfp_kernel.py`) NICHT gesichert. Diese enthielten u.a. die V2-aot_compile-Fixes.

→ ~2-3h Lebenszeit für Re-Discovery der gleichen Fixes.

### 5. Image-Rebuild auf RTX

- `./build.sh --rtx --jobs 8` 1. Versuch: TLS-Fehler bei `git clone` (Spiegel-2 NIC bug, Memory)
- 2. Versuch mit `--use-layer-cache`: erfolgreich, 42.5 GB image
- ABER: Layer-Cache hat alten git-clone wiederverwendet → Image enthielt **alten Code** trotz Rebuild
  - Workaround: file-overlay via volume mount auf laufende Container — kein Rebuild nötig

### 6. V2-aot_compile Crash-Cascade auf Q3.6-27B XFP-V2a

Q3.6-27B mit linear_attn K=6144 → splitm-Pfad → torch.compile fullgraph_capture crasht.

| # | Crash | Fix | Commit |
|---|---|---|---|
| 1 | `os.path.exists` in lazy `_load_xfp_v2_kernels` (gb0291) | Eager-Load in `_process_v2` | `0199a1fc7` |
| 2 | `logger.info` in `_xfp_v2_log_dispatch` (gb0291) | `@torch.compiler.disable` (verboten in fullgraph!) → entfernt | `33dbb4394` |
| 3 | `v17_splitm.xfp_gemm_v17_lib_splitm` C++ extension call (gb0007) | `xfp_v2_apply` als torch custom_op via `direct_register_custom_op` | `f3d5cc2fb` |
| 4 | `_load_xfp_v2_kernels()` 3-tuple unpack in `online_moe.py` | unpack entfernt | `a0aa1c92d` |
| 5 | `torch.cuda.synchronize()` mid-stream-capture | DIAG-calls aus MoE V2 entfernt | `629066cfa` |

**5 sequentielle Crashes**. Alle mit file-overlay live gefixt (kein image-rebuild dazwischen).

### 7. V2a Bench-Resultate auf RTX

**35B XFP-V2a** (K ≤ 4096, V2a == V2 algorithmisch):
- 6. Container-Versuch: ✅ READY
- bench.py: 161.6 / **200.4** / 96.2 tok/s, Math 92%
- GSM8K --limit 50: **76.0%** strict ±6.1 (BF16 76.02% — innerhalb stderr ✅)

**122B XFP-V2a** (K ≤ 4096):
- bench.py: 80.0 / 106.7 / 68.3 tok/s, Math 96%
- GSM8K --limit 50: **98.0%** strict ±2.0 (V2 full 94.62% — n=50 plausibel innerhalb stderr)
- Throughput 106.7 tok/s medium ist niedriger als alte 138 tok/s — Mikro-Regression durch custom_op-overhead oder cudaFuncSetAttribute? Quality unbeeinträchtigt.

**GLM-4.7-Flash XFP-V2a**: ❌ **MLA-Bug**
- `mla_attention.py:766 get_and_maybe_dequant_weights(kv_b_proj)` AttributeError
- XFP-V2 hat `del layer.weight` gemacht und `xfp_packed` registriert, MLA findet weder weight/qweight/weight_packed
- Vermutlich gleicher Bug wie xfpglm-serve exit(1) am 02.05.
- **Skip für jetzt** — eigenständiger Engineering-Fix nötig (kv_b_proj von XFP ausnehmen oder xfp_packed als `weight_packed` mit dequant-callback registrieren)

**Q3.6-27B XFP-V2a**: ⏳ 3. Container-Versuch mit allen 5 Fixes läuft

## Stand am Ende des Tages

| Modell | BF16 (n=50) | XFP-V2a (n=50) | Δ | Status |
|---|---|---|---|---|
| 35B-A3B | 76.02% (full) | **76.0%** ±6.1 | within stderr | ✅ |
| 122B-A10B | n/a (>1 GPU) | **98.0%** ±2.0 | (V2 ref 94.62%) | ✅ |
| Q3.6-27B (dense) | **64.0%** ±6.86 | ⏳ | — | loading |
| GLM-4.7-Flash | **68.0%** ±6.66 | ❌ MLA bug | — | blocker |

## Offene Folgeaufgaben

1. Q3.6-27B XFP-V2a GSM8K-Probe abschließen
2. GLM MLA + XFP-V2 Bug fixen (`kv_b_proj` weight-attribute)
3. Volle GSM8K (1319 × 3 seeds) für 122B + Q3.6-27B XFP-V2a (Δ-Sicherheit)
4. xfp_moe_gemm_v17_lib.cu in DGX commiten + JIT-loader integrieren (Task #62)
5. Custom_op Performance-Regression untersuchen (122B 138 → 107 tok/s)

## Commits heute

- `0199a1fc7` fix(xfp-v2): eager-load V2 kernels before forward to fix aot_compile
- `03b50b40d` fix(xfp-v2): @torch.compiler.disable on _xfp_v2_log_dispatch (DEAD-END)
- `33dbb4394` fix(xfp-v2): remove logger calls from dispatch_v2_linear_gemm
- `f3d5cc2fb` fix(xfp-v2): register V2 dispatch as torch custom_op (xfp_v2_apply)
- `a0aa1c92d` fix(xfp-v2): online_moe.py — _load_xfp_v2_kernels now returns 3-tuple
- `629066cfa` fix(xfp-v2): remove torch.cuda.synchronize from MoE V2 hot path
