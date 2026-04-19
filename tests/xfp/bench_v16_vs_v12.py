# SPDX-License-Identifier: Apache-2.0
"""v16 (Tensor-Core MMA m16n8k16 with XFP codebook B-decode) vs v12.

v16 is NOT bitwise identical to v11 (tensor-core fp32 accumulation rounds
differently than scalar fp32). Correctness gate: cos ≥ 0.9999, same as
Marlin vs reference.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


def _kernel_src(name):
    for base in (os.path.expanduser("~/vllm-riy/kernels/multiquant"),
                 "/opt/mq_kernels"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def _load(mod_name, cu_name):
    from torch.utils.cpp_extension import load
    return load(
        name=mod_name,
        sources=[_kernel_src(cu_name)],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_121,code=sm_121",
            "-diag-suppress=177,3288",
        ],
        verbose=False,
    )


def _time(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[iters // 2] * 1000.0  # µs


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    num = (a * b).sum().item()
    den = (a.norm().item() * b.norm().item() + 1e-12)
    return num / den


def check_linear(k11, k16):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    print("=" * 74)
    print("LINEAR correctness — v16 cos ≥ 0.9999 vs v11 reference")
    print("=" * 74)
    torch.manual_seed(42)
    shapes = [
        # Qwen 122B shapes
        (4,   8, 128,  128),    # small sanity
        (4,   1, 3072, 2048),   # Qwen attn.qkv slice, M=1 decode
        (4,  16, 3072, 2048),   # M=16 (e.g. speculative decode draft batch)
        (3,   1, 2048, 3072),   # bits=3 shape
        (4,   1, 9216, 2048),   # Qwen Linear wider
    ]
    all_ok = True
    for bits, M, N, K in shapes:
        if K % 16 != 0:
            print(f"  SKIP bits={bits} M={M} N={N} K={K} (K not multiple of 16)")
            continue
        W = torch.randn(N, K, device="cuda").float()
        x = torch.randn(M, K, device="cuda").bfloat16()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        cb = codebook.to(torch.float16).cuda()
        C11 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        C16 = torch.zeros_like(C11)
        k11.xfp_gemm(x, repacked, cb, C11, bits, K)
        k16.xfp_gemm(x, repacked, cb, C16, bits, K)
        torch.cuda.synchronize()
        cos = cos_sim(C11, C16)
        maxdiff = (C11.float() - C16.float()).abs().max().item()
        ok = cos >= 0.9999
        all_ok = all_ok and ok
        flag = "✓" if ok else "✗"
        print(f"  {flag} bits={bits} M={M:>3} N={N:>5} K={K:>5}: "
              f"cos={cos:.6f}  maxdiff={maxdiff:.3e}")
    return all_ok


def bench_linear(k11, k12, k16):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    print()
    print("=" * 74)
    print("LINEAR bench — M=1 decode (primary target)")
    print("=" * 74)
    print(f"{'bits':>4} {'M':>3} {'N':>6} {'K':>5}   "
          f"{'v11 µs':>9} {'v12 µs':>9} {'v16 µs':>9}   "
          f"{'16vs12':>7} {'16vs11':>7}")
    shapes = [
        (4, 1, 3072, 2048),
        (4, 1, 9216, 2048),
        (3, 1, 2048, 3072),
        (4, 1, 4608, 4608),
        (4, 1, 2048, 2048),
        (4, 1, 8192, 4096),
        # M>1 cases (speculative/prefill-like)
        (4, 16, 3072, 2048),
    ]
    torch.manual_seed(42)
    for bits, M, N, K in shapes:
        if K % 16 != 0:
            continue
        W = torch.randn(N, K, device="cuda").float()
        x = torch.randn(M, K, device="cuda").bfloat16()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        cb = codebook.to(torch.float16).cuda()
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        t11 = _time(lambda: k11.xfp_gemm(x, repacked, cb, C, bits, K))
        t12 = _time(lambda: k12.xfp_gemm(x, repacked, cb, C, bits, K))
        t16 = _time(lambda: k16.xfp_gemm(x, repacked, cb, C, bits, K))
        d_16_12 = (t16 - t12) / t12 * 100.0
        d_16_11 = (t16 - t11) / t11 * 100.0
        print(f"{bits:>4} {M:>3} {N:>6} {K:>5}   "
              f"{t11:>9.2f} {t12:>9.2f} {t16:>9.2f}   "
              f"{d_16_12:>+6.1f}% {d_16_11:>+6.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="correctness only, skip bench")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    k11 = _load("xfp_gemm_v11_b16", "xfp_gemm_v11.cu")
    k12 = _load("xfp_gemm_v12_b16", "xfp_gemm_v12.cu")
    k16 = _load("xfp_gemm_v16_b16", "xfp_gemm_v16.cu")

    ok = check_linear(k11, k16)
    if not ok:
        print()
        print("CORRECTNESS FAIL — v16 bench skipped")
        sys.exit(1)
    if args.check:
        sys.exit(0)
    bench_linear(k11, k12, k16)
