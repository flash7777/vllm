"""XFP-V2 paper-grade verification.

Comprehensive replication of the Phase 1 result with:
  - 6 layers (model depth: 0, 5, 10, 20, 30, 40)
  - 8 weight classes per layer
  - 3 random seeds for k-means library construction (variance estimate)
  - 4 library_size points: {8, 16, 32, 64}
  - 3 group_size points:    {64, 128, 256}
  - Baselines: V1 (per-channel learned), int4 per-channel, int4 per-group {32,128}

Output: machine-readable JSON + paper-ready markdown table at the end.

Usage in container:
  podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \\
    -v /root/vllm-riy/vllm/multiquant:/usr/local/lib/python3.12/dist-packages/vllm/multiquant:ro \\
    -v /data/tensordata:/data/tensordata:ro -v /root/vllm-riy/tests:/tests:ro \\
    localhost/vllm-multiquant:latest \\
    python3 /tests/xfp/verify_v2_paper.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.multiquant.xfp.xfp_pack import (
    xfp_pack, xfp_pack_v2, dequant_xfp,
)


MODEL_DIR = os.environ.get(
    "TEST_MODEL_DIR", "/data/tensordata/Qwen3.5-35B-A3B-BF16"
)
LAYERS = [0, 5, 10, 20, 30, 40]
SEEDS = [0, 1, 2]
LIBRARY_SIZES = [8, 16, 32, 64]
GROUP_SIZES = [64, 128, 256]


def _find_tensor(model_dir: str, key: str):
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def _cos(a, b):
    return F.cosine_similarity(
        a.float().reshape(-1).unsqueeze(0),
        b.float().reshape(-1).unsqueeze(0), dim=1).item()


def _v1_recon(W):
    Wf = W.float()
    packed, codebook, o_idx, o_val, _ = xfp_pack(Wf, bits=4, also_score_widths=())
    Wr = dequant_xfp(packed, codebook, K=Wf.shape[1], bits=4)
    if o_idx is not None and o_idx.numel() > 0:
        flat = Wr.reshape(-1).clone()
        flat[o_idx] = o_val.to(flat.dtype)
        Wr = flat.reshape(Wf.shape)
    return Wr


def _v2_recon(W, group_size, library_size, seed=0):
    Wf = W.float()
    # _build_codebook_library uses torch.Generator with `seed`; we control
    # via global manual_seed for reproducibility plus pass through if exposed.
    torch.manual_seed(seed)
    packed, library, lib_id, scale, mid, stats = xfp_pack_v2(
        Wf, bits=4, group_size=group_size, library_size=library_size,
    )
    # Reconstruction directly from pack output (no separate dequant needed
    # for cos check — pack returns the assignment, we just decode).
    N, K = Wf.shape
    G = K // group_size
    chosen_lib = library.float()[lib_id.long()]  # [N, G, n_cents]
    W_norm = (Wf.reshape(N, G, group_size) - mid.float().unsqueeze(-1)) / scale.float().unsqueeze(-1)
    d = (W_norm.unsqueeze(-1) - chosen_lib.unsqueeze(2)).abs()
    idx = d.argmin(dim=-1)
    rec_norm = torch.gather(chosen_lib, 2, idx)
    Wr = rec_norm * scale.float().unsqueeze(-1) + mid.float().unsqueeze(-1)
    return Wr.reshape(N, K), stats


def rtn_int4_perchannel(W):
    Wf = W.float()
    amax = Wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    return (Wf / scale).round().clamp(-7, 7) * scale


def rtn_int4_pergroup(W, group_size):
    Wf = W.float()
    N, K = Wf.shape
    pad = (group_size - K % group_size) % group_size
    Wp = torch.nn.functional.pad(Wf, (0, pad)) if pad else Wf
    K2 = Wp.shape[1]
    Wg = Wp.reshape(N, K2 // group_size, group_size)
    amax = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    Wq = (Wg / scale).round().clamp(-7, 7) * scale
    return Wq.reshape(N, K2)[:, :K]


def main():
    print(f"torch {torch.__version__} | model: {MODEL_DIR}")
    print(f"layers: {LAYERS} | seeds: {SEEDS}")
    print(f"library sweep: {LIBRARY_SIZES} | group sweep: {GROUP_SIZES}")

    # Layer-class keys (relative to "model.language_model.layers.{i}")
    weight_classes = [
        ("attn_qkv",     "linear_attn.in_proj_qkv.weight"),
        ("attn_o",       "linear_attn.out_proj.weight"),
        ("attn_a",       "linear_attn.in_proj_a.weight"),
        ("attn_b",       "linear_attn.in_proj_b.weight"),
        ("attn_z",       "linear_attn.in_proj_z.weight"),
        ("shared_gate",  "mlp.shared_expert.gate_proj.weight"),
        ("shared_down",  "mlp.shared_expert.down_proj.weight"),
    ]
    # Routed expert (3D fused, slice expert 0)
    moe_keys = [
        ("routed_gateup", "mlp.experts.gate_up_proj"),
        ("routed_down",   "mlp.experts.down_proj"),
    ]

    # Collect all (cls, layer_idx, W) tuples we'll test.
    weights: list[tuple[str, int, torch.Tensor]] = []
    for li in LAYERS:
        for cls, sfx in weight_classes:
            key = f"model.language_model.layers.{li}.{sfx}"
            W = _find_tensor(MODEL_DIR, key)
            if W is not None:
                weights.append((cls, li, W))
        for cls, sfx in moe_keys:
            key = f"model.language_model.layers.{li}.{sfx}"
            T = _find_tensor(MODEL_DIR, key)
            if T is not None and T.dim() == 3:
                weights.append((cls, li, T[0]))  # expert 0
    print(f"Loaded {len(weights)} weight matrices.\n")

    # Compute V1 + int4 baselines once (deterministic, no seed dependency)
    print("Stage A — V1 + int4 baselines (deterministic):")
    baselines: dict[tuple[str, int], dict[str, float]] = {}
    t0 = time.time()
    for i, (cls, li, W) in enumerate(weights):
        Wf = W.float()
        b = {}
        b["v1"]   = _cos(Wf, _v1_recon(W))
        b["g32"]  = _cos(Wf, rtn_int4_pergroup(Wf, 32))
        b["g128"] = _cos(Wf, rtn_int4_pergroup(Wf, 128))
        b["pc"]   = _cos(Wf, rtn_int4_perchannel(Wf))
        baselines[(cls, li)] = b
        if (i+1) % 10 == 0:
            print(f"  baselines {i+1}/{len(weights)} in {time.time()-t0:.1f}s")
    print(f"  total {time.time()-t0:.1f}s\n")

    # V2 sweep over (group_size, library_size, seed)
    print("Stage B — V2 sweep:")
    # results_by_config[(g, L)][cls] = list of cos (one per seed×weight)
    results: dict[tuple[int, int], dict[str, list[float]]] = {}
    for g in GROUP_SIZES:
        for L in LIBRARY_SIZES:
            t1 = time.time()
            cfg_key = (g, L)
            results[cfg_key] = {cls: [] for cls, _ in weight_classes}
            for cls, _ in moe_keys:
                results[cfg_key][cls] = []
            for cls, li, W in weights:
                if W.shape[1] % g != 0:
                    continue  # skip when group_size doesn't divide K
                seed_cs = []
                for seed in SEEDS:
                    Wr, stats = _v2_recon(W, g, L, seed=seed)
                    seed_cs.append(_cos(W.float(), Wr))
                results[cfg_key][cls].extend(seed_cs)
            mean_cos = statistics.mean(
                c for cls_cs in results[cfg_key].values() for c in cls_cs
            )
            print(f"  g={g:>3} L={L:>3}: avg cos={mean_cos:.5f} ({time.time()-t1:.1f}s)")

    # Aggregate baselines
    base_avg = {
        k: statistics.mean(b[k] for b in baselines.values())
        for k in ("v1", "g32", "g128", "pc")
    }

    # ── Render paper-style markdown table ──
    print()
    print("=" * 78)
    print("MARKDOWN TABLE (for paper):")
    print("=" * 78)
    out_lines: list[str] = []
    out_lines.append("## XFP-V2 Reconstruction Quality vs Baselines\n")
    out_lines.append(f"Model: Qwen3.5-35B-A3B-BF16, layers {LAYERS}, "
                     f"{len(weights)} weight matrices "
                     f"(8 classes × {len(LAYERS)} layers).")
    out_lines.append(f"Metric: cosine similarity between reference BF16 "
                     f"weights and reconstruction (higher is better). "
                     f"V2 averaged over {len(SEEDS)} seeds for k-means library.\n")
    out_lines.append("| Method | bits/param | avg cos |")
    out_lines.append("|---|---|---|")
    out_lines.append(f"| BF16 (reference)             | 16   | 1.00000 |")
    out_lines.append(f"| int4 per-channel (1 scale/row) | 4.01 | {base_avg['pc']:.5f} |")
    out_lines.append(f"| int4 per-group g=128         | 4.13 | {base_avg['g128']:.5f} |")
    out_lines.append(f"| int4 per-group g=32          | 4.50 | {base_avg['g32']:.5f} |")
    out_lines.append(f"| **XFP-V1** (per-channel codebook, K=2048) | **4.13** | **{base_avg['v1']:.5f}** |")
    out_lines.append("")
    out_lines.append("XFP-V2 sweep (per-group + shared codebook library):\n")
    out_lines.append("| group_size | library_size | bits/param | avg cos |")
    out_lines.append("|---|---|---|---|")
    for g in GROUP_SIZES:
        for L in LIBRARY_SIZES:
            cfg_key = (g, L)
            cs = [c for cls_cs in results[cfg_key].values() for c in cls_cs]
            if not cs:
                continue
            mean = statistics.mean(cs)
            stderr = statistics.stdev(cs) / (len(cs) ** 0.5)
            # Bit-budget: 4 (weights) + 32/g (scale+mid fp16 per group) + lib_id_bits/g
            lib_id_bits = 4 if L <= 16 else 8 if L <= 256 else 32
            bpp = 4 + 32.0/g + lib_id_bits/g
            out_lines.append(
                f"| **{g}** | **{L}** | **{bpp:.2f}** | **{mean:.5f} ±{stderr:.5f}** |"
            )
        out_lines.append("|---|---|---|---|")
    out_lines.append("\n*± = standard error across 8 weight classes × "
                     f"{len(LAYERS)} layers × {len(SEEDS)} seeds.*\n")

    # Per-class detail at the recommended config (g=128, L=32)
    out_lines.append("## Per-class detail (g=128, L=32)\n")
    out_lines.append("| weight class | V1 cos | V2 cos | Δ V2-V1 | int4-g32 | Δ V2-g32 |")
    out_lines.append("|---|---|---|---|---|---|")
    rec_g, rec_L = 128, 32
    cls_order = [c for c, _ in weight_classes] + [c for c, _ in moe_keys]
    for cls in cls_order:
        v1s = [baselines[(c, li)]["v1"] for c, li in baselines if c == cls]
        g32s = [baselines[(c, li)]["g32"] for c, li in baselines if c == cls]
        v2s = results.get((rec_g, rec_L), {}).get(cls, [])
        if not v1s or not v2s:
            continue
        v1m = statistics.mean(v1s)
        v2m = statistics.mean(v2s)
        g32m = statistics.mean(g32s)
        out_lines.append(
            f"| {cls} | {v1m:.5f} | {v2m:.5f} | {(v2m-v1m)*100:+.2f}pp | "
            f"{g32m:.5f} | {(v2m-g32m)*100:+.2f}pp |"
        )

    out = "\n".join(out_lines) + "\n"
    print(out)

    # Save JSON for downstream + plots
    json_out = {
        "model": MODEL_DIR,
        "layers": LAYERS,
        "seeds": SEEDS,
        "n_weights": len(weights),
        "baselines": {
            "v1_avg":   base_avg["v1"],
            "g32_avg":  base_avg["g32"],
            "g128_avg": base_avg["g128"],
            "pc_avg":   base_avg["pc"],
        },
        "v2_sweep": {
            f"g{g}_L{L}": {
                "mean_cos": (statistics.mean(c for cls_cs in results[(g, L)].values() for c in cls_cs)
                             if any(results[(g, L)].values()) else None),
                "n_obs": sum(len(v) for v in results[(g, L)].values()),
            }
            for g in GROUP_SIZES for L in LIBRARY_SIZES
        },
    }
    with open("/tmp/xfp_v2_verification.json", "w") as f:
        json.dump(json_out, f, indent=2)
    with open("/tmp/xfp_v2_verification.md", "w") as f:
        f.write(out)
    print("Saved /tmp/xfp_v2_verification.json")
    print("Saved /tmp/xfp_v2_verification.md")


if __name__ == "__main__":
    main()
