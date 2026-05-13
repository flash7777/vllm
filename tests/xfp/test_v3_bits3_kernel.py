#!/usr/bin/env python3
"""Unit test for V3 BITS=3 kernel — vs Python reference.

Runs both BITS=4 (baseline, should pass) and BITS=3 (V3, target) on the
same random Linear layer. Identifies whether V3 kernel math is correct.

Usage:
    XFP_V2=3 XFP_GROUP_SIZE=128 python3 test_v3_bits3_kernel.py
"""

import os
os.environ.setdefault("XFP_V2", "3")
os.environ.setdefault("XFP_GROUP_SIZE", "128")

import torch
import torch.nn.functional as F

from vllm.multiquant.xfp.xfp_pack import (
    xfp_pack_v2, xfp_repack, xfp_repack_v3,
)
from vllm.multiquant.xfp.xfp_kernel import (
    _load_xfp_v2_kernels, dispatch_v2_linear_gemm,
)


# ── Python reference dequant ─────────────────────────────────────────


def dequant_v2_bits4(packed_2d, library, lib_id, scale, mid,
                    K, group_size=128):
    """Python reference for V2 BITS=4 flat layout.

    packed_2d: [K_packed=K/8, N_out] int32 (flat, [K, N] layout)
    library: [library_size, n_centroids=16] fp16/fp32
    lib_id, scale, mid: [N_out, G] (G = K / group_size)
    """
    bits, vpw, mask = 4, 8, 0xF
    K_packed, N_out = packed_2d.shape
    G = K // group_size

    W = torch.zeros(N_out, K, dtype=torch.float32, device=packed_2d.device)
    pk = packed_2d.to(torch.int64)  # treat as uint32 bit pattern
    for n in range(N_out):
        for k_word in range(K_packed):
            word = int(pk[k_word, n].item()) & 0xFFFFFFFF
            for slot in range(vpw):
                k = k_word * vpw + slot
                if k >= K:
                    break
                g = k // group_size
                lib_idx = int(lib_id[n, g].item())
                cb = library[lib_idx]
                s = float(scale[n, g].item())
                m = float(mid[n, g].item())
                idx = (word >> (slot * bits)) & mask
                W[n, k] = float(cb[idx].item()) * s + m
    return W


def dequant_v2_bits3(packed_3d, library, lib_id, scale, mid,
                    group_size=128):
    """Python reference for V3 BITS=3 per-group layout.

    packed_3d: [N_out, G, K_PER_GROUP=13] int32
    """
    bits, vpw, mask = 3, 10, 0x7
    N_out, G, K_PER_GROUP = packed_3d.shape
    K = G * group_size

    W = torch.zeros(N_out, K, dtype=torch.float32, device=packed_3d.device)
    pk = packed_3d.to(torch.int64)
    for n in range(N_out):
        for g in range(G):
            lib_idx = int(lib_id[n, g].item())
            cb = library[lib_idx]
            s = float(scale[n, g].item())
            m = float(mid[n, g].item())
            for k_word in range(K_PER_GROUP):
                word = int(pk[n, g, k_word].item()) & 0xFFFFFFFF
                for slot in range(vpw):
                    abs_slot = k_word * vpw + slot
                    if abs_slot >= group_size:
                        break
                    idx = (word >> (slot * bits)) & mask
                    W[n, g * group_size + abs_slot] = float(cb[idx].item()) * s + m
    return W


# ── Common kernel-call wrapper ───────────────────────────────────────


def run_kernel(x, packed_repacked, library, lib_id, scale, mid,
              bits, K, N_out, group_size=128):
    M = int(x.shape[0])
    x_bf16 = x.to(torch.bfloat16).contiguous()
    library_fp16 = library.to(torch.float16).contiguous()
    lib_id_i32 = lib_id.to(torch.int32).contiguous()
    scale_fp16 = scale.to(torch.float16).contiguous()
    mid_fp16 = mid.to(torch.float16).contiguous()
    C = torch.empty(M, N_out, dtype=torch.bfloat16, device=x.device)
    dispatch_v2_linear_gemm(
        x_bf16, packed_repacked, library_fp16, lib_id_i32,
        scale_fp16, mid_fp16, C, bits, K, group_size,
    )
    return C.float()


# ── Tests ────────────────────────────────────────────────────────────


def make_inputs(N_out, K, M, seed=42, device="cuda:0"):
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(N_out, K, generator=g, device=device,
                   dtype=torch.float32) * 0.1
    x = torch.randn(M, K, generator=g, device=device,
                   dtype=torch.float32) * 0.1
    return W, x


def test_bits4_baseline(N_out=32, K=256, M=4):
    """Sanity check: BITS=4 should already work (verifies test framework)."""
    print(f"\n=== BITS=4 baseline (N_out={N_out}, K={K}, M={M}) ===")
    device = torch.device("cuda:0")
    W, x = make_inputs(N_out, K, M, device=device)
    packed_2d, library, lib_id, scale, mid, stats = xfp_pack_v2(
        W, bits=4, group_size=128, library_size=16,
        lloyd_iters=10, library_iters=5,
    )
    print(f"  pack stats: mse={stats.mse:.4e}, cos_sim={stats.cos_sim:.4f}")
    print(f"  packed shape: {tuple(packed_2d.shape)}")

    W_deq = dequant_v2_bits4(packed_2d, library.float(), lib_id, scale.float(),
                            mid.float(), K=K, group_size=128)
    C_ref = (x @ W_deq.T).float()

    packed_repacked = xfp_repack(packed_2d).contiguous()
    C_ker = run_kernel(x, packed_repacked, library, lib_id, scale, mid,
                      bits=4, K=K, N_out=N_out)

    cos = F.cosine_similarity(C_ker.flatten().unsqueeze(0),
                             C_ref.flatten().unsqueeze(0), dim=1).item()
    max_err = (C_ker - C_ref).abs().max().item()
    print(f"  kernel vs ref: cos={cos:.6f}, max_err={max_err:.4e}")
    print(f"  C_ref[0,:4]:   {C_ref[0, :4].tolist()}")
    print(f"  C_ker[0,:4]:   {C_ker[0, :4].tolist()}")
    return cos > 0.99


def test_bits3_v3(N_out=32, K=256, M=4):
    print(f"\n=== BITS=3 V3 (N_out={N_out}, K={K}, M={M}) ===")
    device = torch.device("cuda:0")
    W, x = make_inputs(N_out, K, M, device=device)
    packed_3d, library, lib_id, scale, mid, stats = xfp_pack_v2(
        W, bits=3, group_size=128, library_size=16,
        lloyd_iters=10, library_iters=5,
    )
    print(f"  pack stats: mse={stats.mse:.4e}, cos_sim={stats.cos_sim:.4f}")
    print(f"  packed shape: {tuple(packed_3d.shape)}")
    assert packed_3d.dim() == 3, f"expected 3D, got {packed_3d.dim()}D"

    W_deq = dequant_v2_bits3(packed_3d, library.float(), lib_id, scale.float(),
                            mid.float(), group_size=128)
    pack_cos = F.cosine_similarity(W.flatten().unsqueeze(0),
                                    W_deq.flatten().unsqueeze(0), dim=1).item()
    print(f"  W_deq vs W cos: {pack_cos:.4f}")

    C_ref = (x @ W_deq.T).float()

    packed_repacked = xfp_repack_v3(packed_3d).contiguous()
    print(f"  repacked shape: {tuple(packed_repacked.shape)}, "
          f"expected flat={N_out * (K // 128 // 2) * 26}")
    C_ker = run_kernel(x, packed_repacked, library, lib_id, scale, mid,
                      bits=3, K=K, N_out=N_out)

    cos = F.cosine_similarity(C_ker.flatten().unsqueeze(0),
                             C_ref.flatten().unsqueeze(0), dim=1).item()
    max_err = (C_ker - C_ref).abs().max().item()
    print(f"  kernel vs ref: cos={cos:.6f}, max_err={max_err:.4e}")
    print(f"  C_ref[0,:4]:   {C_ref[0, :4].tolist()}")
    print(f"  C_ker[0,:4]:   {C_ker[0, :4].tolist()}")

    if cos < 0.99:
        # Detailed diff to locate bug
        per_n_cos = F.cosine_similarity(C_ker, C_ref, dim=0)
        per_m_cos = F.cosine_similarity(C_ker, C_ref, dim=1)
        print(f"  per-N cos:  min={per_n_cos.min().item():.3f}  "
              f"max={per_n_cos.max().item():.3f}  "
              f"mean={per_n_cos.mean().item():.3f}")
        print(f"  per-M cos:  min={per_m_cos.min().item():.3f}  "
              f"max={per_m_cos.max().item():.3f}  "
              f"mean={per_m_cos.mean().item():.3f}")
    return cos > 0.99


def main():
    print("V3 BITS=3 kernel unit test")
    print("=" * 60)
    # Force kernel JIT compile up front
    v17_lib, v17_splitm, v17_splitk = _load_xfp_v2_kernels()
    print(f"Loaded kernels: v17_lib={v17_lib is not None}, "
          f"v17_splitm={v17_splitm is not None}, v17_splitk={v17_splitk is not None}")

    b4_ok = test_bits4_baseline()
    print(f"BITS=4 baseline: {'PASS' if b4_ok else 'FAIL'}")
    if not b4_ok:
        print("  (test framework or pack code broken — fix this first)")
        return 1

    b3_ok = test_bits3_v3()
    print(f"\nBITS=3 V3:       {'PASS' if b3_ok else 'FAIL'}")
    return 0 if (b4_ok and b3_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
