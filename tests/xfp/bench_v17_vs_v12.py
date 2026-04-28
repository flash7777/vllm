"""Phase 3.0b — Throughput bench: xfp_gemm_v17_lib vs xfp_gemm_v12.

Measures wall-clock latency on shapes typical of Qwen3.5-A3B Linear
layers. Each kernel runs the full forward (no dequant pre-step) so we
compare apples-to-apples.

Pass criterion: v17_lib latency ≤ 1.20× v12 (max +20% overhead is the
budget from the design doc; +5-12% expected).
"""
from __future__ import annotations

import os
import sys
import time
import statistics
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


KERNEL_SRC = "/opt/mq_kernels"


def _build_extra_cuda_flags():
    return [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_120,code=sm_120",
    ]


def _load_v12():
    return load(
        name="xfp_gemm_v12",
        sources=[os.path.join(KERNEL_SRC, "xfp_gemm_v12.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=_build_extra_cuda_flags(),
        verbose=False,
    )


def _load_v17_lib():
    return load(
        name="xfp_gemm_v17_lib",
        sources=[os.path.join(KERNEL_SRC, "xfp_gemm_v17_lib.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=_build_extra_cuda_flags(),
        verbose=False,
    )


def _bench_call(fn, n_warm=10, n_iter=100):
    # CUDA event timing
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]  # milliseconds
    return statistics.median(times), statistics.stdev(times)


def main() -> None:
    if not torch.cuda.is_available():
        print("no CUDA — skipping")
        return

    from vllm.multiquant.xfp.xfp_pack import (
        xfp_pack, xfp_pack_v2, xfp_repack,
    )

    print("Loading kernels (JIT compile if cold) ...")
    mod_v12 = _load_v12()
    mod_v17 = _load_v17_lib()
    device = "cuda:0"

    # (label, M, N, K) — typical Qwen3.5-A3B linear shapes (K ≤ 8192)
    cases = [
        ("attn_qkv 8192x2048",   1,    8192,  2048),
        ("attn_qkv 8192x2048",   8,    8192,  2048),
        ("attn_qkv 8192x2048",   64,   8192,  2048),
        ("attn_out 2048x4096",   1,    2048,  4096),
        ("attn_out 2048x4096",   8,    2048,  4096),
        ("attn_out 2048x4096",   64,   2048,  4096),
        ("shared_gate 512x2048", 8,    512,   2048),
        ("shared_down 2048x512", 8,    2048,  512),
    ]

    print(f"\n{'shape':>22} {'M':>4} {'v12 ms':>8} {'v17 ms':>8} "
          f"{'ratio':>6} {'overhead':>10}")
    print("-" * 72)

    overheads = []
    for label, M, N, K in cases:
        torch.manual_seed(M * 1000 + N + K)
        W = torch.randn(N, K, dtype=torch.float32) * 0.02
        x = torch.randn(M, K, dtype=torch.float32) * 0.1

        # V1 pack (CPU) → repack → GPU
        v1_packed_cpu, v1_codebook_cpu, _, _, _ = xfp_pack(
            W, bits=4, also_score_widths=()
        )
        v1_packed_g = xfp_repack(v1_packed_cpu).to(device).contiguous()
        v1_codebook_g = v1_codebook_cpu.to(device).contiguous()

        # V2 pack (CPU) → repack → GPU
        v2_packed_cpu, v2_lib_cpu, v2_lib_id_cpu, v2_scale_cpu, v2_mid_cpu, _ = xfp_pack_v2(
            W, bits=4, group_size=128, library_size=32,
        )
        v2_packed_g = xfp_repack(v2_packed_cpu).to(device).contiguous()
        v2_lib_g = v2_lib_cpu.to(device).contiguous()
        v2_lib_id_g = v2_lib_id_cpu.to(torch.int32).to(device).contiguous()
        v2_scale_g = v2_scale_cpu.to(device).contiguous()
        v2_mid_g = v2_mid_cpu.to(device).contiguous()

        x_g = x.to(torch.bfloat16).to(device).contiguous()
        C_v12 = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        C_v17 = torch.empty(M, N, dtype=torch.bfloat16, device=device)

        def call_v12():
            mod_v12.xfp_gemm(x_g, v1_packed_g, v1_codebook_g, C_v12, 4, K)

        def call_v17():
            mod_v17.xfp_gemm_v17_lib(
                x_g, v2_packed_g, v2_lib_g, v2_lib_id_g, v2_scale_g, v2_mid_g,
                C_v17, 4, K, 128
            )

        ms_v12, _ = _bench_call(call_v12)
        ms_v17, _ = _bench_call(call_v17)
        ratio = ms_v17 / ms_v12
        overhead_pct = (ratio - 1.0) * 100.0
        overheads.append(overhead_pct)
        print(f"{label:>22} {M:>4} {ms_v12:>8.4f} {ms_v17:>8.4f} "
              f"{ratio:>6.3f} {overhead_pct:>+9.2f}%")

    print("-" * 72)
    print(f"  median overhead across {len(overheads)} shapes: "
          f"{statistics.median(overheads):+.2f}%")
    print(f"  max overhead: {max(overheads):+.2f}%")


if __name__ == "__main__":
    main()
