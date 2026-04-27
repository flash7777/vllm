# TASK: XFP TP=2 397B auf RTX zum Laufen bringen

**Ziel:** Qwen3.5-397B-A17B (multimodal Qwen3-VL Vision-Tower) als XFP
TP=2 auf RTX PRO 6000 (2× 96 GiB, SM120) servierfähig — kein
`--enforce-eager` als Dauerlösung, keine ad-hoc Per-Layer-Fixes.

**Status:** Loader-Refactor und Linear-Forward laufen. Crash jetzt im
**MoE-Forward (xfp_moe_gemm CUDA kernel)** bei TP=2. Letzter Stand
LOAD #18 (27.04. ~10:28).

## Hardware / Topology

- Host: Spiegel 2, `ssh -p 2020 root@10.249.0.99`, x86_64
- GPU: 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GiB je)
- Container: `localhost/vllm-multiquant:top` (image-tag im start.multiquant)
- Repo: `/root/vllm-riy`, branch `multiquant`

## Modell + Cache

- Source: `/data/tensordata/Qwen3.5-397B-A17B/` (HF safetensors, ~700 GB)
- XFP-Cache: `/data/tensordata/mq-cache/Qwen3.5-397B-A17B/2d523455112557a3/`
  (173 GB Layer-Shards, **keine** residuals.safetensors — alter v1/v2 Cache)
- RIY-Profil: `/data/tensordata/riy_profile_397b_36pct.json` (36% Aktivierung)
- Tatsächlicher Loader-Pfad: HF-Source-Iterator (Modus 2 im
  refaktorisierten Loader), weil Cache keine residuals hat. XFP MoE
  Tensoren kommen per Cache-Hit in `process_weights_after_loading`.

## Was bereits gefixt ist (LOAD #1–#17)

| # | Fix | Commit |
|---|---|---|
| 1 | PACK/LOAD-Trennung Schema v3 | `1d391b4cb` |
| 2 | MoE-Filter Substring-Match (HF-prefix `model.language_model.*`) | `06c0ec07c` |
| 3 | Cache-only loader nutzt `model.load_weights(iterator)` | `fe628518c` |
| 4 | Streaming-loader Robustness post-pwal `del weight` | `2ebf778b4` |
| 5 | XFP cache-load `torch.cuda.current_device()` statt `W.device` | `d98b0ff39` |
| 6 | Defensive lazy device-migration in `apply()` | `50afd91f7` |
| 7 | Outlier-TP-remap (replicated → filter+shift gegen `_xfp_N`/`_xfp_K`) | `0a32df87a` |
| 8 | start.multiquant: passthrough CUDA_LAUNCH_BLOCKING / TORCH_USE_CUDA_DSA | `28f3f1806` |

Was erreicht ist: Modell lädt vollständig (~64.9 GiB pro Rank), MoE-PWAL
durchgelaufen, dummy_run startet. **Linear-Forward (in_proj_qkvz,
in_proj_ba, out_proj) funktioniert sauber unter TP=2.**

## Aktueller Crash (LOAD #18, identisch #17)

```
File "...vllm/multiquant/xfp/online_moe.py", line 88
    down = torch.zeros(BT, N2, dtype=torch.bfloat16, device=x.device)
torch.AcceleratorError: CUDA error: invalid argument
```

Das ist Sticky-Error aus einem **vorherigen Kernel-Launch** in derselben
Forward-Function. Vorgänger:
- Line 74: `gate_up = torch.zeros(BT, N13, ...)` (vermutlich OK)
- **Line 75–79: `moe_kernel.xfp_moe_gemm(x_bf16, w13_packed, w13_codebook,
  gate_up, sorted_token_ids, sorted_expert_ids, no_w, bits, K13, N13,
  topk, fpe13, num_valid)`** ← **Verdächtiger #1**
- Line 82–83: `silu_and_mul` (Standard-CUDA, wahrscheinlich OK)
- Line 88: nächster `torch.zeros` → CUDA error invalid argument

`CUDA_LAUNCH_BLOCKING=1` wurde in #18 gesetzt aber zeigt den Crash NICHT
am xfp_moe_gemm — bedeutet entweder env-passthrough nicht angekommen
ODER der Kernel sync't intern bevor er die OOB-Adresse trifft.

## Verdacht / Hypothesen

### H1 (wahrscheinlichste): MoE-Kernel TP-unaware

**`xfp_moe_gemm_v12` (CUDA-Kernel aus `/opt/mq_kernels`) rechnet
mit globalen Expert-IDs 0..N_experts_full, aber per-Rank ist nur ein
RIY+TP-Slice davon physisch im Cache.** 

Beweis: `topk_ids` enthält global Expert-IDs (0..511 bei 397B). Der
Kernel macht intern `expert_offset = expert_id * (K * N // 8)` für
packed weights. Wenn expert_id ≥ E_local (lokale Expert-Zahl, z.B.
187), pointer-arithmetik landet ausserhalb der allokierten
`w13_xfp_packed`-Tensoren → invalid pointer / illegal memory access.

**Mitigation:** vor `xfp_moe_gemm` den `topk_ids`-Tensor gegen das
`_expert_map` (lokale Maske der Experts) filtern. Tokens mit
non-local expert müssen vom anderen Rank gerechnet und per
all-to-all aggregiert werden — Standard-MoE-EP-Pattern.

In vllm gibt's das eigentlich schon: `FusedMoE.expert_map` und der
`select_experts`-Pfad maskt non-local experts auf -1. Wenn unser XFP
MoE Method das nicht respektiert, ist das der Fix-Punkt.

Code-Schaft: `vllm/multiquant/xfp/online_moe.py` ~line 100+ wo
`apply()` aufgerufen wird, vor `_xfp_moe_op(...)` einen
`local_topk_ids`-View bauen.

### H2: BT oder N2 ist 0/negativ

Edge case: bei dummy_run werden synthetische topk_ids generiert. Wenn
RIY ein Layer hat mit 0 lokalen experts (kann bei niedrigem RIY und
ungünstigem TP-shard passieren), dann läuft `flat_topk[sort_indices]`
durch, aber kein Token ist "selected" → `num_valid = 0`. Der Kernel
bekommt num_valid=0 und schreibt nirgendwohin — sollte OK sein. Aber
wenn dann `BT * N2 * 2 = 0`, der nächste `torch.zeros(0, N2, ...)`
sollte auch OK sein.

Weniger wahrscheinlich. Test: BT/N2 vor line 88 loggen.

### H3: SM120-spezifischer Kernel-Bug

Memory-Note: "FP8 MoE auf SM120: Kein `grouped_mm_c3x_sm120.cu` in
vLLM!". Unser xfp_moe_gemm ist allerdings handgeschrieben und nicht
CUTLASS — sollte aber kompiliert sein für SM120 in `/opt/mq_kernels`.
Weniger wahrscheinlich, weil 35B XFP TP=1 auf SM121 (DGX) den
gleichen Kernel nutzt und läuft.

## Nächste Schritte (Reihenfolge)

### 1. Diagnostik: was genau ist `topk_ids.max()` vor dem Crash?

Patch in `online_moe.py:_xfp_moe_forward_impl` direkt nach line 70:

```python
import os as _dbg_os
if _dbg_os.environ.get("XFP_MOE_DEBUG", "0") == "1":
    import torch.distributed as _dist
    _r = _dist.get_rank() if _dist.is_initialized() else 0
    print(
        f"[xfp_moe rank={_r}] B={B} topk={topk} BT={BT} "
        f"K13={K13} N13={N13} K2={K2} N2={N2} "
        f"E_global={int(topk_ids.max().item())+1} "
        f"E_local_w13={w13_packed.shape[0]} "
        f"E_local_w2={w2_packed.shape[0]} "
        f"num_valid={num_valid}",
        flush=True,
    )
```

Run mit `XFP_MOE_DEBUG=1` (passthrough in start.multiquant ergänzen).
Wenn `E_global > E_local_w13` → bestätigt H1.

### 2. Wenn H1 bestätigt: TP-aware topk_ids in online_moe.py

Vor `xfp_moe_gemm` line 75:

```python
expert_map = getattr(layer, "expert_map", None)  # vllm-Standard
if expert_map is not None:
    # Map global → local id (or -1 for non-local)
    local_ids = expert_map[topk_ids.reshape(-1)]
    # Drop tokens for non-local experts
    valid_mask = local_ids >= 0
    sorted_token_ids = sort_indices[valid_mask].to(torch.int32)
    sorted_expert_ids = local_ids[sort_indices][valid_mask].to(torch.int32)
    num_valid = int(valid_mask.sum().item())
```

Plus: am Ende des Forward, all-reduce über das `down`-Output, weil jeder
Rank nur seine local-Expert-Tokens verarbeitet hat.

`vllm.distributed.tensor_model_parallel_all_reduce(output)` macht das.

### 3. Falls in vllm-Standard MoE der Pfad anders ist:

Prüfen: `default_moe_runner.py:forward_impl` macht **vor**
`_apply_quant_method` schon eine `select_experts` mit `expert_map`. Das
heißt unsere `online_moe.apply` müsste topk_ids bereits TP-lokal
bekommen. Crash deutet darauf hin, dass das nicht passiert — vermutlich
weil unsere XFP-Method das `expert_map`-Attribut nicht im
`apply()`-Signature ausliest. Code-Lookup nötig.

### 4. CUDA_LAUNCH_BLOCKING-Verifikation

Falls CLB nicht im Container ankam: in `start.multiquant` die env-Vars
nicht über `EXTRA_PASSTHROUGH_ENV[]`, sondern direkt mit `-e` hardcoden
für Debug-Sessions, z.B.:

```bash
podman run ... \
    -e CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-} \
    ...
```

Oder Gegen-Test: in den Container reingehen `podman exec mq-serve env |
grep CUDA` während er läuft.

## Acceptance-Test

Wenn der Fix sitzt, sollte folgender LOAD durchlaufen:

```bash
ssh -p 2020 root@10.249.0.99
cd /root/vllm-riy && git pull --rebase origin multiquant
podman rm -f mq-serve
./start.multiquant \
    --model Qwen3.5-397B-A17B \
    --load-format multiquant --tp 2 \
    --weight-dtype xfp --kv fp8 --weight-dtype-lm-head fp8 \
    --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --mm-encoder-tp-mode data \
    --riy /data/tensordata/riy_profile_397b_36pct.json \
    --eager
```

**Erfolgskriterien:**
- `determine_available_memory()` läuft sauber durch (kein invalid argument)
- `curl http://localhost:8011/v1/models` liefert 200
- Single GSM8K prompt durchläuft mit sinnvoller Antwort
- Bench `bench.lm-eval --label 397b-xfp-tp2 --tasks gsm8k --limit 50`
  liefert flexible-extract > 0.05

## Tagebuch / Verlauf der LOAD-Versuche

| LOAD # | Datum | Crash-Stelle | Fix | Commit |
|---|---|---|---|---|
| 1–10 | bis 25.04. | diverse cache-corruption / TP-slice issues | siehe commits | — |
| 11 | 26.04. | MergedColumnParallelLinear has no 'weight' (post-pwal del) | streaming-loader graceful | `2ebf778b4` |
| 12 | 26.04. | XFP outlier scatter device-mismatch cuda:0 vs cuda:1 | current_device() fix | `d98b0ff39` |
| 13–14 | 26.04. | RMSNormGated Triton sticky-error | (Symptom, nicht Ursache) | — |
| 15 | 27.04. 06:54 | GDN warmup torch.tensor() sticky-error | env passthrough | `28f3f1806` |
| 16 | 27.04. 07:14 | scatter_add device-side assert (XFP outlier OOB) | TP-remap outlier | `0a32df87a` |
| 17 | 27.04. 07:34 | xfp_moe_gemm crash, surfaced an `torch.zeros(BT, N2)` | offen — H1 | — |
| 18 | 27.04. 10:14 | gleiche Stelle mit CLB=1 | offen — H1 | — |
| 19 | 27.04. 12:14 | gleicher Crash, jetzt mit XFP_MOE_DEBUG: E_global_max=29 ⇒ H1 widerlegt | — | — |
| 20 | 27.04. 12:35 | 122B XFP TP=2 (Smoke): Crash in clifford.py, `torch.tensor()` in cuda-graph capture | clifford.reverse via torch.cat ohne signs-Tensor | `6073e64` |
| 21 | 27.04. 13:00 | 122B w2_weight=(0,) bei cache-cold: PWAL crash bei `K2 = w2.shape[2]` | Diagnostik-Patch | `97abf2f5` |
| 22 | 27.04. 14:38 | gleicher w2-Bug, Diagnostik bestätigt: stub size-0 + EP-routing asymmetrisch | Stub+Filter gelöscht (PR 2) | `0581133` |
| 23 | 27.04. 15:14 | streaming-quant `_sq_load_numel` zählt pre-TP+EP-non-local-experts; PWAL feuert vor w2 | post-TP-Numel-fix (return_success) | `8fe4ad3` |
| 24 | 27.04. 15:43 | Forward läuft jetzt; xfp_moe_gemm `cudaErrorInvalidValue` bei BT=65536 | H4: gridDim.y>65535. Workaround: `--max-num-batched-tokens 4096` ⇒ BT=32768 | (workaround) |
| 25 | 27.04. 16:54 | API ready! Aber `_scaled_mm` mat2 cuda:0 vs cuda:1 in compute_logits (fp8 lm_head) | TP-rank-affinity migrate in fp8_embedding.apply | `9482d1b` |
| 26 | 27.04. 17:18 | API ready, Generation = `!`-Spam (TP-fix moved data zur falschen Device-Slice?) | bf16 lm_head A/B | — |
| 27 | 27.04. 18:00 | bf16 lm_head: Crash in vocab_parallel_embedding.apply (Z. 69), TP-affinity gleich | apply() TP-guard symmetrisch zur embedding() | `f9c3019f` |

## Out-of-Scope (wenn TP=2 läuft)

- Performance-Tuning, Cuda-Graph-Capture, Speculative Decoding
- 397B XFP Re-pack mit residuals.safetensors v3 → für späteren cache-only
  fast-load
- Vision-Tower-Inferenz (multimodal Bildeingabe testen) — separate
  Smoke-Test-Stufe nach Text-only läuft
