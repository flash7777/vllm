# SPDX-License-Identifier: Apache-2.0
"""v14 (SMEM-A + bulk cp.async B preload) vs v12 (SMEM-A only).

Correctness: v14 must be bitwise identical to v11 on shapes where the
16-group B-SMEM cap holds (K ≤ 4096 for bits=4, ≤ 5120 for bits=3,
≤ 8192 for bits=2).
"""
from __future__ import annotations

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


def check_linear(k11, k14):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    print("LINEAR correctness — v14 must be bitwise = v11")
    torch.manual_seed(42)
    shapes = [(4, 3072, 2048), (3, 2048, 3072), (2, 4096, 4096)]
    for bits, N, K in shapes:
        W = torch.randn(N, K, device="cuda").float()
        x = torch.randn(1, K, device="cuda").bfloat16()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        cb = codebook.to(torch.float16).cuda()
        C11 = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")
        C14 = torch.zeros_like(C11)
        k11.xfp_gemm(x, repacked, cb, C11, bits, K)
        k14.xfp_gemm(x, repacked, cb, C14, bits, K)
        torch.cuda.synchronize()
        same = torch.equal(C11, C14)
        print(f"  bits={bits} N={N:>5} K={K:>5}: {'✓' if same else '✗'} "
              f"{'equal' if same else 'DIFF maxabs=' + str((C11.float()-C14.float()).abs().max().item())}")
        if not same:
            return False
    return True


def check_moe(k11, k14):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    print("MoE correctness — v14 must be bitwise = v11")
    torch.manual_seed(1234)
    shapes = [(4, 1, 8, 8, 3072, 2048), (3, 1, 8, 16, 2048, 1536)]
    for bits, B, topk, E, N, K in shapes:
        packed_list, cb_list = [], []
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
        C14 = torch.zeros_like(C11)

        def run(k, C):
            k.xfp_moe_gemm(x, B_packed, CB, C, sorted_tok, sorted_exp,
                           no_w, int(bits), int(K), int(N), int(topk),
                           int(fpe), int(num_valid))
        run(k11, C11); run(k14, C14)
        torch.cuda.synchronize()
        same = torch.equal(C11, C14)
        print(f"  bits={bits} E={E:>3} N={N:>5} K={K:>5}: {'✓' if same else '✗'}")
        if not same:
            return False
    return True


def bench_linear(k11, k12, k14):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    print()
    print("LINEAR bench — M=1 decode")
    print(f"{'bits':>4} {'N':>6} {'K':>5}   "
          f"{'v11 µs':>9} {'v12 µs':>9} {'v14 µs':>9}   "
          f"{'14vs12':>7} {'14vs11':>7}")
    shapes = [
        (4, 3072, 2048),
        (4, 9216, 2048),
        (3, 2048, 3072),
        (4, 4608, 4608),  # K=4608 bits=4 ⇒ n_groups=18 → v14 REJECT (runtime check); skip
        (4, 2048, 2048),
        (4, 4096, 4096),  # K=4096 bits=4 ⇒ n_groups=16 → edge
        (2, 4096, 8192),  # bits=2 K=8192 ⇒ K_packed=512 → n_groups=16 → edge
    ]
    torch.manual_seed(42)
    for bits, N, K in shapes:
        vpw = 16 if bits == 2 else 10 if bits == 3 else 8
        K_packed = (K + vpw - 1) // vpw
        n_groups = (K_packed + 31) // 32
        if n_groups > 16:
            print(f"{bits:>4} {N:>6} {K:>5}   SKIP (n_groups={n_groups}>16)")
            continue

        W = torch.randn(N, K, device="cuda").float()
        x = torch.randn(1, K, device="cuda").bfloat16()
        packed, codebook, _, _, _ = xfp_pack(W, bits=bits, lloyd_iters=5)
        repacked = xfp_repack(packed).cuda().reshape(-1)
        cb = codebook.to(torch.float16).cuda()
        C = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")

        t11 = _time(lambda: k11.xfp_gemm(x, repacked, cb, C, bits, K))
        t12 = _time(lambda: k12.xfp_gemm(x, repacked, cb, C, bits, K))
        t14 = _time(lambda: k14.xfp_gemm(x, repacked, cb, C, bits, K))
        d14_12 = (t14 - t12) / t12 * 100.0
        d14_11 = (t14 - t11) / t11 * 100.0
        print(f"{bits:>4} {N:>6} {K:>5}   {t11:>9.2f} {t12:>9.2f} {t14:>9.2f}   "
              f"{d14_12:>+6.1f}% {d14_11:>+6.1f}%")


def bench_moe(k11, k12, k14):
    from vllm.multiquant.xfp.xfp_pack import xfp_pack, xfp_repack
    print()
    print("MoE bench — Qwen 122B decode (B=1, topk=8)")
    print(f"{'bits':>4} {'E':>3} {'N':>5} {'K':>5}   "
          f"{'v11 µs':>9} {'v12 µs':>9} {'v14 µs':>9}   "
          f"{'14vs12':>7} {'14vs11':>7}")
    shapes = [
        (4, 1, 8,   8, 3072, 2048),
        (4, 1, 8,  16, 3072, 2048),
        (4, 1, 8,   8, 1536, 2048),
        (4, 1, 8,   8, 2048, 1536),
        (4, 1, 8,  64, 3072, 2048),
    ]
    torch.manual_seed(1234)
    for bits, B, topk, E, N, K in shapes:
        packed_list, cb_list = [], []
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
                x, B_packed, CB, C, sorted_tok, sorted_exp, no_w,
                int(bits), int(K), int(N), int(topk),
                int(fpe), int(num_valid))

        t11 = _time(mk(k11)); t12 = _time(mk(k12)); t14 = _time(mk(k14))
        d14_12 = (t14 - t12) / t12 * 100.0
        d14_11 = (t14 - t11) / t11 * 100.0
        print(f"{bits:>4} {E:>3} {N:>5} {K:>5}   "
              f"{t11:>9.2f} {t12:>9.2f} {t14:>9.2f}   "
              f"{d14_12:>+6.1f}% {d14_11:>+6.1f}%")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    k11_lin = _load("xfp_gemm_v11_b14", "xfp_gemm_v11.cu")
    k12_lin = _load("xfp_gemm_v12_b14", "xfp_gemm_v12.cu")
    k14_lin = _load("xfp_gemm_v14_b14", "xfp_gemm_v14.cu")
    k11_moe = _load("xfp_moe_gemm_v11_b14", "xfp_moe_gemm_v11.cu")
    k12_moe = _load("xfp_moe_gemm_v12_b14", "xfp_moe_gemm_v12.cu")
    k14_moe = _load("xfp_moe_gemm_v14_b14", "xfp_moe_gemm_v14.cu")

    ok = check_linear(k11_lin, k14_lin)
    ok = check_moe(k11_moe, k14_moe) and ok
    if not ok:
        sys.exit("CORRECTNESS FAIL")
    bench_linear(k11_lin, k12_lin, k14_lin)
    bench_moe(k11_moe, k12_moe, k14_moe)
