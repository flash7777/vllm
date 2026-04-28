"""Phase 3.0a — xfp_gemm_v17_lib kernel correctness vs Python reference.

JIT-compiles xfp_gemm_v17_lib.cu, runs against Phase-1's xfp_pack_v2 +
dequant_xfp_v2_packed Python reference. Pass criterion:
  cos(y_kernel, y_python) ≥ 0.999  AND  rel_max_err ≤ 5e-3.

Tested shapes (fit K_SMEM_MAX=8192, K % group_size=128 == 0):
  (32, 256x2048)   - small smoke
  (8, 4096x2048)   - typical attn out_proj
  (16, 1024x4096)  - typical FFN gate
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


KERNEL_SRC_DIR = "/opt/mq_kernels"


def _load_kernel():
    src = os.path.join(KERNEL_SRC_DIR, "xfp_gemm_v17_lib.cu")
    if not os.path.exists(src):
        # Fallback: source mounted via the multiquant volume
        alt = "/work/kernels/multiquant/xfp_gemm_v17_lib.cu"
        if os.path.exists(alt):
            src = alt
        else:
            raise FileNotFoundError(f"v17_lib.cu not found at {src} or {alt}")
    print(f"compiling {src} ...")
    return load(
        name="xfp_gemm_v17_lib",
        sources=[src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_120,code=sm_120",  # RTX PRO 6000 Blackwell
        ],
        verbose=False,
    )


def main() -> None:
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    from vllm.multiquant.xfp.xfp_pack import (
        xfp_pack_v2, dequant_xfp_v2_packed, xfp_repack,
    )

    mod = _load_kernel()
    print(f"kernel module: {mod}")

    cases = [
        # (M, N, K)
        (32,  256,  2048),
        (8,   4096, 2048),
        (16,  1024, 4096),
    ]

    overall_ok = True
    for M, N, K in cases:
        bits = 4
        group_size = 128
        library_size = 32
        torch.manual_seed(M * 100 + N + K)
        W = torch.randn(N, K, dtype=torch.float32) * 0.02
        x = torch.randn(M, K, dtype=torch.float32) * 0.1

        # Pack via xfp_pack_v2 (CPU)
        packed_cpu, library_cpu, lib_id_cpu, scale_cpu, mid_cpu, stats = xfp_pack_v2(
            W, bits=bits, group_size=group_size, library_size=library_size,
        )

        # Python reference forward
        W_rec_cpu = dequant_xfp_v2_packed(
            packed_cpu, library_cpu, lib_id_cpu, scale_cpu, mid_cpu,
            K=K, bits=bits, group_size=group_size,
        )
        y_ref = F.linear(x, W_rec_cpu)  # [M, N]

        # Move to GPU for kernel.
        # Kernel expects warp-interleaved repacked layout (same as v12).
        packed_repacked = xfp_repack(packed_cpu)  # [K_groups * N * 32]
        device = "cuda:0"
        x_g = x.to(torch.bfloat16).to(device).contiguous()
        packed_g = packed_repacked.to(device).contiguous()
        library_g = library_cpu.to(device).contiguous()
        lib_id_g = lib_id_cpu.to(torch.int32).to(device).contiguous()
        scale_g = scale_cpu.to(device).contiguous()
        mid_g = mid_cpu.to(device).contiguous()
        C_g = torch.empty(M, N, dtype=torch.bfloat16, device=device)

        mod.xfp_gemm_v17_lib(
            x_g, packed_g, library_g, lib_id_g, scale_g, mid_g, C_g,
            bits, K, group_size,
        )
        torch.cuda.synchronize()
        y_kernel = C_g.float().cpu()

        # Compare
        cos = F.cosine_similarity(
            y_ref.reshape(-1).unsqueeze(0),
            y_kernel.reshape(-1).unsqueeze(0), dim=1,
        ).item()
        max_err = (y_ref - y_kernel).abs().max().item()
        rel_err = max_err / max(y_ref.abs().max().item(), 1e-12)

        ok = (cos >= 0.999) and (rel_err <= 5e-3)
        marker = "✓" if ok else "✗"
        print(f"  {marker} M={M:>3} N={N:>5} K={K:>5}: "
              f"cos={cos:.5f} max_err={max_err:.4g} rel_err={rel_err:.3g} "
              f"(pack cos={stats.cos_sim:.4f})")
        if not ok:
            overall_ok = False

    print("\n" + ("✓ ALL PASS" if overall_ok else "✗ SOME FAILED"))


if __name__ == "__main__":
    main()
