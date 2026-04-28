"""How many library codebooks are needed to cover the per-channel codebooks?

Drives the Phase-1 design choice for XFP-codebook-library:
  Each row currently has its own 16-centroid Lloyd-fitted codebook.
  We hypothesise that across all rows × experts × layers, only a few
  hundred *prototype* codebooks suffice — because weight distributions
  are roughly Gaussian-symmetric with similar shapes.

Method:
  1. Fit Lloyd codebook per row (16 centroids) on real 35B weights.
  2. Normalize each codebook to [-1, +1] (subtract midpoint, scale by
     range/2). This factors out per-row magnitude so the library
     captures *shape* not *scale*.
  3. K-means over the [M, 16] codebook collection at multiple
     library_size values.
  4. Report per-row reconstruction quality vs library_size:
       - cos(original_codebook, nearest_lib_entry × scale)
       - p50, p5, worst per row

Goal: find the smallest library_size where p5 ≥ 0.99.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.multiquant.xfp.xfp_pack import _lloyd_per_channel


MODEL_DIR = os.environ.get(
    "TEST_MODEL_DIR", "/data/tensordata/Qwen3.5-35B-A3B-BF16"
)
LIBRARY_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)


def _find_tensor(model_dir: str, key: str) -> torch.Tensor | None:
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def _fit_codebooks(W: torch.Tensor, lloyd_iters: int = 20) -> torch.Tensor:
    """Return [N_rows, 16] Lloyd-fitted per-row codebook."""
    return _lloyd_per_channel(W.float(), 16, lloyd_iters)


def _normalize_codebook(cb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (normalized [-1, +1], midpoint, scale).

    Reconstruction:  cb_orig = midpoint + scale * normalized
    """
    cb_min = cb.amin(dim=1, keepdim=True)
    cb_max = cb.amax(dim=1, keepdim=True)
    midpoint = (cb_min + cb_max) / 2
    scale = (cb_max - cb_min) / 2
    scale = scale.clamp(min=1e-12)
    return (cb - midpoint) / scale, midpoint.squeeze(-1), scale.squeeze(-1)


def _kmeans_lite(X: torch.Tensor, k: int, iters: int = 30,
                 seed: int = 0) -> torch.Tensor:
    """Pure-PyTorch k-means on [N, D] points → returns [k, D] centroids.

    K-means++-style init: pick first centroid randomly, then sample each
    next from ||x - nearest_c||² distribution.
    """
    N, D = X.shape
    g = torch.Generator(device=X.device).manual_seed(seed)
    # Init centroids via k-means++
    idx0 = int(torch.randint(0, N, (1,), generator=g).item())
    cents = X[idx0:idx0+1].clone()
    while cents.shape[0] < k:
        d2 = ((X.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(-1).amin(dim=1)
        prob = d2 / d2.sum().clamp(min=1e-12)
        nxt = int(torch.multinomial(prob, 1, generator=g).item())
        cents = torch.cat([cents, X[nxt:nxt+1]], dim=0)

    for _ in range(iters):
        d2 = ((X.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(-1)  # [N, k]
        assign = d2.argmin(dim=1)
        for c in range(k):
            mask = assign == c
            if mask.any():
                cents[c] = X[mask].mean(dim=0)
    return cents


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"model: {MODEL_DIR}")

    # Collect codebooks from a representative slice of layers/experts.
    # Cheap: layer-0 only, but try every weight class.
    LAYER0 = "model.language_model.layers.0"
    keys = [
        (f"{LAYER0}.linear_attn.in_proj_qkv.weight",     "attn_qkv"),
        (f"{LAYER0}.linear_attn.out_proj.weight",        "attn_o"),
        (f"{LAYER0}.linear_attn.in_proj_a.weight",       "attn_a"),
        (f"{LAYER0}.linear_attn.in_proj_b.weight",       "attn_b"),
        (f"{LAYER0}.linear_attn.in_proj_z.weight",       "attn_z"),
        (f"{LAYER0}.mlp.shared_expert.gate_proj.weight", "shared_g"),
        (f"{LAYER0}.mlp.shared_expert.up_proj.weight",   "shared_u"),
        (f"{LAYER0}.mlp.shared_expert.down_proj.weight", "shared_d"),
    ]
    all_cbs = []
    all_tags = []  # for reporting
    print("\nFitting per-row codebooks ...")
    for key, tag in keys:
        W = _find_tensor(MODEL_DIR, key)
        if W is None:
            print(f"  skip {tag} (not found)")
            continue
        cb = _fit_codebooks(W)  # [N_rows, 16]
        all_cbs.append(cb)
        all_tags.extend([tag] * cb.shape[0])
        print(f"  {tag:>10}: shape={tuple(W.shape)} → {cb.shape[0]} codebooks")

    # Add a few routed experts (slices from fused 3D tensor)
    fused = _find_tensor(MODEL_DIR, f"{LAYER0}.mlp.experts.gate_up_proj")
    if fused is not None:
        for e in range(8):  # 8 experts is enough to test variance
            cb = _fit_codebooks(fused[e])
            all_cbs.append(cb)
            all_tags.extend([f"routed_e{e}"] * cb.shape[0])
        print(f"  routed gate_up_proj × 8 experts: {fused.shape[1] * 8} codebooks")
    fused_d = _find_tensor(MODEL_DIR, f"{LAYER0}.mlp.experts.down_proj")
    if fused_d is not None:
        for e in range(8):
            cb = _fit_codebooks(fused_d[e])
            all_cbs.append(cb)
            all_tags.extend([f"routed_e{e}_d"] * cb.shape[0])
        print(f"  routed down_proj × 8 experts: {fused_d.shape[1] * 8} codebooks")

    cbs = torch.cat(all_cbs, dim=0)  # [M, 16]
    M = cbs.shape[0]
    print(f"\nTotal codebooks collected: {M}")

    # Normalize → factor out scale (just keep shape)
    cbs_norm, midpoint, scale = _normalize_codebook(cbs)
    print(f"Normalized codebook shape: {tuple(cbs_norm.shape)} dtype={cbs_norm.dtype}")

    # Try multiple library sizes
    print(f"\n{'lib_size':>8} {'p5_cos':>8} {'p50_cos':>8} {'p95_cos':>8} "
          f"{'min_cos':>8} {'M/lib':>6}")
    print("-" * 60)
    for libsz in LIBRARY_SIZES:
        if libsz >= M:
            continue
        # K-means over normalized codebooks
        torch.manual_seed(0)
        # Subsample if too many for kmeans speed (use first 4096 + random rest)
        if M > 4096:
            sample = cbs_norm[torch.randperm(M, generator=torch.Generator().manual_seed(0))[:4096]]
        else:
            sample = cbs_norm
        lib = _kmeans_lite(sample, libsz, iters=20)

        # Assign each codebook to nearest library entry, measure cos
        # cos in shape-space (after normalization)
        d2 = ((cbs_norm.unsqueeze(1) - lib.unsqueeze(0)) ** 2).sum(-1)
        nearest_idx = d2.argmin(dim=1)
        nearest = lib[nearest_idx]  # [M, 16]
        cos = F.cosine_similarity(cbs_norm, nearest, dim=1)
        # Sort and percentile
        cos_sorted, _ = cos.sort()
        p5 = cos_sorted[int(0.05 * M)].item()
        p50 = cos_sorted[int(0.50 * M)].item()
        p95 = cos_sorted[int(0.95 * M)].item()
        mn = cos_sorted[0].item()
        print(f"{libsz:>8} {p5:>8.4f} {p50:>8.4f} {p95:>8.4f} {mn:>8.4f} "
              f"{M/libsz:>6.1f}")

    # Per-class breakdown for a chosen library size (best in sweep)
    libsz = 256
    print(f"\nPer-class coverage at lib_size={libsz}:")
    torch.manual_seed(0)
    sample = cbs_norm[torch.randperm(M, generator=torch.Generator().manual_seed(0))[:min(M, 4096)]]
    lib = _kmeans_lite(sample, libsz, iters=20)
    d2 = ((cbs_norm.unsqueeze(1) - lib.unsqueeze(0)) ** 2).sum(-1)
    nearest = lib[d2.argmin(dim=1)]
    cos = F.cosine_similarity(cbs_norm, nearest, dim=1).tolist()
    by_tag: dict[str, list[float]] = {}
    for c, t in zip(cos, all_tags):
        by_tag.setdefault(t, []).append(c)
    for tag, cs in sorted(by_tag.items()):
        sc = sorted(cs)
        n = len(sc)
        p5 = sc[int(0.05*n)] if n >= 20 else min(sc)
        med = sc[n//2]
        mn = min(sc)
        print(f"  {tag:>14} ({n:>5} cbs): p5={p5:.4f}  med={med:.4f}  min={mn:.4f}")


if __name__ == "__main__":
    main()
