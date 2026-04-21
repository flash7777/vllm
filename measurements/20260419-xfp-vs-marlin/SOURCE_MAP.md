# Source Map: NVTX Range → Source Location

Each NVTX range name used in the capture maps to exactly one source region.
Line numbers are against commit `234491260` (`git describe` at measurement
time; see `git blame` for later drift).

All instrumentation is gated by `VLLM_NVTX_PROFILE=1`
(`vllm/multiquant/_profiler.py:25`).

## Layer-level ranges

File: `vllm/model_executor/models/qwen3_next.py`
Class: `Qwen3NextDecoderLayer.forward`

| NVTX Name | Source line(s) | What it wraps |
|---|---:|---|
| `layer_{idx}` | `qwen3_next.py:1289` | `with _nvtx(_layer_tag):` — whole layer body |
| `layer_{idx}/input_norm` | `qwen3_next.py:1290` | input_layernorm call |
| `layer_{idx}/attn` | `qwen3_next.py:1299` | self_attn / linear_attn + optional attn_layer_scale |
| `layer_{idx}/post_norm` | `qwen3_next.py:1326` | post_attention_layernorm |
| `layer_{idx}/mlp` | `qwen3_next.py:1331` | self.mlp(hidden_states) |

Notes:
- `self.mlp` is `Qwen3NextSparseMoeBlock` for MoE layers (47 of 48 in Qwen3.5-122B-A10B),
  otherwise `Qwen3NextMLP` (dense, 1 layer).
- Post-mlp `ffn_layer_scale` branch (line 1334+) is NOT inside `layer_{idx}` —
  Qwen3.5-122B does not use layer_scale, so the delta is nil.

## Attention-internal ranges

File: `vllm/model_executor/models/qwen3_next.py`
Class: `Qwen3NextAttention.forward`

| NVTX Name | Source line(s) | What it wraps |
|---|---:|---|
| `attn/qkv_proj` | `qwen3_next.py:1166` | `self.qkv_proj(hidden_states)` |
| `attn/qk_norm` | `qwen3_next.py:1181` | q_norm + k_norm |
| `attn/rope` | `qwen3_next.py:1189` | `self.rotary_emb(...)` |
| `attn/core` | `qwen3_next.py:1192` | `self.attn(q, k, v)` — FlashAttention / FlashInfer kernel |
| `attn/o_proj` | `qwen3_next.py:1199` | `self.o_proj(attn_output)` |

Note: `attn_output_gate` pre/post-math (sigmoid gate, ~5 small kernels) is not
split — will show as between-range bookkeeping if present.

## MoE XFP-path ranges

File: `vllm/multiquant/xfp/online_moe.py`
Function: `_xfp_moe_forward_impl`

| NVTX Name | Source line(s) | What it wraps |
|---|---:|---|
| `moe/xfp/sort` | `online_moe.py:68` | argsort + reshape, computes sorted_token_ids + sorted_expert_ids |
| `moe/xfp/gate_up_gemm` | `online_moe.py:76` | alloc gate_up + `xfp_moe_gemm` gate_up |
| `moe/xfp/silu` | `online_moe.py:85` | F.silu(gate) * up |
| `moe/xfp/down_gemm` | `online_moe.py:91` | alloc down + `xfp_moe_gemm` down |
| `moe/xfp/scatter` | `online_moe.py:102` | arange + scatter_add_ weighted reduce |

Underlying kernel: `kernels/multiquant/xfp_moe_gemm_v12.cu` (loaded via
`xfp_moe_kernel.py::_load_xfp_moe_gemm`).

## MoE Marlin-path ranges

File: `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`

### Outer dispatch (function `fused_marlin_moe`, entry ~line 210)

| NVTX Name | Source line(s) | What it wraps |
|---|---:|---|
| `moe/marlin/align_block` | `fused_marlin_moe.py:325` | moe_align_block_size |
| `moe/marlin/fused_kernel` | `fused_marlin_moe.py:335` | `_fused_marlin_moe(...)` call |
| `moe/marlin/reduce` | `fused_marlin_moe.py:376` | torch.sum / moe_sum weighted reduce |

### Inner kernels (function `_fused_marlin_moe`, ~line 51)

| NVTX Name | Source line(s) | What it wraps |
|---|---:|---|
| `moe/marlin/gate_up_gemm` | `fused_marlin_moe.py:127` | `ops.moe_wna16_marlin_gemm` gate+up call |
| `moe/marlin/silu` | `fused_marlin_moe.py:156` | `activation_func(...)` |
| `moe/marlin/down_gemm` | `fused_marlin_moe.py:181` | `ops.moe_wna16_marlin_gemm` down call |

Underlying C++ kernel: `csrc/moe/marlin_moe_wna16/ops.cu` → `marlin_template.h`.

## Quick comparison pairs

The following NVTX name pairs are directly comparable between runs:

| XFP range | Marlin range | What's being compared |
|---|---|---|
| `layer_{N}/attn` | same | Attention block wall-clock (should be ~equal — same backend) |
| `moe/xfp/gate_up_gemm` | `moe/marlin/gate_up_gemm` | Gate+Up GEMM kernel time |
| `moe/xfp/silu` | `moe/marlin/silu` | SiLU activation |
| `moe/xfp/down_gemm` | `moe/marlin/down_gemm` | Down GEMM kernel time |
| `moe/xfp/sort` | `moe/marlin/align_block` | Token permutation overhead |
| `moe/xfp/scatter` | `moe/marlin/reduce` | Weighted reduce back to [B,N] |

No comparable match:
- `attn/qk_norm` / `attn/rope` / `attn/o_proj` are structurally identical
  between paths — used for sanity-check.
- `moe/marlin/fused_kernel` is a parent of the three inner Marlin ranges;
  its total should equal Σ(inner ranges) + tiny Python overhead.
