# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark + correctness: v13 (SMEM-A + cp.async) vs v12 vs v11.

v13 combines the two orthogonal core optimizations. It MUST be bitwise
identical to v11 (same template, only load paths differ). The delta vs
v12 tells us whether cp.async B prefetch overlaps the slot compute on
GB10/SM120+SM121.

Usage:
    python3 tests/xfp/bench_v13_vs_v12_vs_v11.py           # bench + check
    python3 tests/xfp/bench_v13_vs_v12_vs_v11.py --check   # correctness only
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


def _kernel_src(name: str) -> str:
    for base in (
        os.path.expanduser("~/vllm-riy/kernels/multiquant"),
        "/opt/mq_kernels",
    ):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def _load(mod_name: str, cu_name: str):
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


def _time_median(fn, warmup=50, iters=500):
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
    return times[iters // 2]  # median (ms)


def check_linear():
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    k11 = _load("xfp_gemm_v11_chk", "xfp_gemm_v11.cu")
    k12 = _load("xfp_gemm_v12_chk", "xfp_gemm_v12.cu")
    k13 = _load("xfp_gemm_v13_chk", "xfp_gemm_v13.cu")

    print("=" * 72)
    print("LINEAR correctness — v13 vs v11 bitwise, v12 vs v11 bitwise")
    print("=" * 72)
    torch.manual_seed(42)
    shapes = [
        (4, 128, 128),
        (4, 3072, 2048),
        (3, 2048, 3072),
        (2, 4096, 8192),   # K at K_SMEM_MAX boundary
    ]
    for bits, N, K in shapes:
        W = torch.randn(N, K, device="cuda").float()
        x = torch.randn(1, K, device="cuda").bfloat16()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        cb = codebook.to(torch.float16).cuda()

        C11 = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")
        C12 = torch.zeros_like(C11)
        C13 = torch.zeros_like(C11)
        k11.xfp_gemm(x, repacked, cb, C11, bits, K)
        k12.xfp_gemm(x, repacked, cb, C12, bits, K)
        k13.xfp_gemm(x, repacked, cb, C13, bits, K)
        torch.cuda.synchronize()

        same_12 = torch.equal(C11, C12)
        same_13 = torch.equal(C11, C13)
        if same_12 and same_13:
            print(f"  bits={bits} N={N:>5} K={K:>5}: v11=v12=v13 ✓")
        else:
            maxdiff_12 = (C11.float() - C12.float()).abs().max().item()
            maxdiff_13 = (C11.float() - C13.float()).abs().max().item()
            print(f"  bits={bits} N={N:>5} K={K:>5}: "
                  f"v12={'=' if same_12 else f'≠ max|d|={maxdiff_12:.2e}'}  "
                  f"v13={'=' if same_13 else f'≠ max|d|={maxdiff_13:.2e}'}")
            if not (same_12 and same_13):
                return False
    return True


def check_moe():
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    k11 = _load("xfp_moe_gemm_v11_chk", "xfp_moe_gemm_v11.cu")
    k12 = _load("xfp_moe_gemm_v12_chk", "xfp_moe_gemm_v12.cu")
    k13 = _load("xfp_moe_gemm_v13_chk", "xfp_moe_gemm_v13.cu")

    print("=" * 72)
    print("MoE correctness — v13 vs v11 bitwise, v12 vs v11 bitwise")
    print("=" * 72)
    torch.manual_seed(1234)

    # (bits, B, topk, E, N, K)
    shapes = [
        (4, 1, 2,  4,  256,  256),
        (4, 1, 8,  8, 3072, 2048),    # Qwen 35B gate+up
        (3, 1, 8, 16, 2048, 1536),    # GLM down
        (4, 2, 8, 16, 3072, 2048),    # small prefill batch
    ]
    for bits, B, topk, E, N, K in shapes:
        packed_list = []
        cb_list = []
        for _ in range(E):
            W = torch.randn(N, K, device="cuda").float()
            packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
            repacked = xfp_repack(packed).cuda().reshape(-1)
            packed_list.append(repacked)
            cb_list.append(codebook.to(torch.float16).cuda())
        fpe = packed_list[0].numel()
        B_packed = torch.cat(packed_list, dim=0)
        CB = torch.cat(cb_list, dim=0)
        x = torch.randn(B, K, device="cuda").bfloat16()
        topk_ids = torch.randint(0, E, (B, topk), dtype=torch.int32,
                                 device="cuda")
        flat = topk_ids.reshape(-1)
        sort_idx = flat.argsort(stable=True)
        sorted_tok = sort_idx.to(torch.int32)
        sorted_exp = flat[sort_idx].to(torch.int32)
        num_valid = sorted_tok.shape[0]
        BT = B * topk
        no_w = torch.empty(0, dtype=torch.float32, device="cuda")

        C11 = torch.zeros(BT, N, dtype=torch.bfloat16, device="cuda")
        C12 = torch.zeros_like(C11)
        C13 = torch.zeros_like(C11)

        def call(k, C):
            k.xfp_moe_gemm(x, B_packed, CB, C, sorted_tok, sorted_exp,
                           no_w, int(bits), int(K), int(N), int(topk),
                           int(fpe), int(num_valid))

        call(k11, C11); call(k12, C12); call(k13, C13)
        torch.cuda.synchronize()

        same_12 = torch.equal(C11, C12)
        same_13 = torch.equal(C11, C13)
        if same_12 and same_13:
            print(f"  bits={bits} E={E:>3} N={N:>5} K={K:>5}: v11=v12=v13 ✓")
        else:
            maxdiff_12 = (C11.float() - C12.float()).abs().max().item()
            maxdiff_13 = (C11.float() - C13.float()).abs().max().item()
            print(f"  bits={bits} E={E:>3} N={N:>5} K={K:>5}: "
                  f"v12={'=' if same_12 else f'≠ {maxdiff_12:.2e}'}  "
                  f"v13={'=' if same_13 else f'≠ {maxdiff_13:.2e}'}")
            return False
    return True


def bench_linear():
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    k11 = _load("xfp_gemm_v11", "xfp_gemm_v11.cu")
    k12 = _load("xfp_gemm_v12", "xfp_gemm_v12.cu")
    k13 = _load("xfp_gemm_v13", "xfp_gemm_v13.cu")

    print("=" * 72)
    print("LINEAR bench — M=1 decode")
    print("=" * 72)
    print(f"{'bits':>4} {'N':>6} {'K':>5}   "
          f"{'v11 µs':>9} {'v12 µs':>9} {'v13 µs':>9}   "
          f"{'12vs11':>7} {'13vs11':>7} {'13vs12':>7}")
    print("-" * 88)

    shapes = [
        (4, 3072, 2048),
        (4, 9216, 2048),
        (3, 2048, 3072),
        (4, 4608, 4608),
        (4, 2048, 2048),
        (4, 8192, 4096),
        (4, 4096, 8192),
    ]
    torch.manual_seed(42)
    for bits, N, K in shapes:
        W = torch.randn(N, K, device="cuda").float()
        x = torch.randn(1, K, device="cuda").bfloat16()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        cb = codebook.to(torch.float16).cuda()
        C = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")

        t11 = _time_median(lambda: k11.xfp_gemm(x, repacked, cb, C, bits, K)) * 1000.0
        t12 = _time_median(lambda: k12.xfp_gemm(x, repacked, cb, C, bits, K)) * 1000.0
        t13 = _time_median(lambda: k13.xfp_gemm(x, repacked, cb, C, bits, K)) * 1000.0

        d_12_11 = (t12 - t11) / t11 * 100.0
        d_13_11 = (t13 - t11) / t11 * 100.0
        d_13_12 = (t13 - t12) / t12 * 100.0
        print(f"{bits:>4} {N:>6} {K:>5}   "
              f"{t11:>9.2f} {t12:>9.2f} {t13:>9.2f}   "
              f"{d_12_11:>+6.1f}% {d_13_11:>+6.1f}% {d_13_12:>+6.1f}%")


def bench_moe():
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    k11 = _load("xfp_moe_gemm_v11", "xfp_moe_gemm_v11.cu")
    k12 = _load("xfp_moe_gemm_v12", "xfp_moe_gemm_v12.cu")
    k13 = _load("xfp_moe_gemm_v13", "xfp_moe_gemm_v13.cu")

    print("=" * 72)
    print("MoE bench — Qwen 122B decode (B=1, topk=8)")
    print("=" * 72)
    print(f"{'bits':>4} {'E':>3} {'N':>5} {'K':>5}   "
          f"{'v11 µs':>9} {'v12 µs':>9} {'v13 µs':>9}   "
          f"{'12vs11':>7} {'13vs11':>7} {'13vs12':>7}")
    print("-" * 90)

    shapes = [
        (4, 1, 8,   8, 3072, 2048),
        (4, 1, 8,  16, 3072, 2048),
        (4, 1, 8,   8, 1536, 2048),
        (4, 1, 8,   8, 2048, 1536),
        (4, 1, 8,  64, 3072, 2048),
        (4, 2, 8,  64, 3072, 2048),
    ]
    torch.manual_seed(1234)
    for bits, B, topk, E, N, K in shapes:
        packed_list = []
        cb_list = []
        for _ in range(E):
            W = torch.randn(N, K, device="cuda").float()
            packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
            repacked = xfp_repack(packed).cuda().reshape(-1)
            packed_list.append(repacked)
            cb_list.append(codebook.to(torch.float16).cuda())
        fpe = packed_list[0].numel()
        B_packed = torch.cat(packed_list, dim=0)
        CB = torch.cat(cb_list, dim=0)

        x = torch.randn(B, K, device="cuda").bfloat16()
        topk_ids = torch.randint(0, E, (B, topk), dtype=torch.int32,
                                 device="cuda")
        flat = topk_ids.reshape(-1)
        sort_idx = flat.argsort(stable=True)
        sorted_tok = sort_idx.to(torch.int32)
        sorted_exp = flat[sort_idx].to(torch.int32)
        num_valid = sorted_tok.shape[0]

        BT = B * topk
        no_w = torch.empty(0, dtype=torch.float32, device="cuda")
        C = torch.zeros(BT, N, dtype=torch.bfloat16, device="cuda")

        def mk(k):
            return lambda: k.xfp_moe_gemm(
                x, B_packed, CB, C, sorted_tok, sorted_exp,
                no_w, int(bits), int(K), int(N), int(topk),
                int(fpe), int(num_valid))

        t11 = _time_median(mk(k11)) * 1000.0
        t12 = _time_median(mk(k12)) * 1000.0
        t13 = _time_median(mk(k13)) * 1000.0

        d_12_11 = (t12 - t11) / t11 * 100.0
        d_13_11 = (t13 - t11) / t11 * 100.0
        d_13_12 = (t13 - t12) / t12 * 100.0
        print(f"{bits:>4} {E:>3} {N:>5} {K:>5}   "
              f"{t11:>9.2f} {t12:>9.2f} {t13:>9.2f}   "
              f"{d_12_11:>+6.1f}% {d_13_11:>+6.1f}% {d_13_12:>+6.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only run correctness check, skip bench")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    ok = check_linear()
    print()
    ok = check_moe() and ok
    print()
    if not ok:
        print("CORRECTNESS FAILED — v13 bench skipped")
        sys.exit(1)
    if args.check:
        sys.exit(0)
    bench_linear()
    print()
    bench_moe()
