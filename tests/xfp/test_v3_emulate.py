#!/usr/bin/env python3
"""V3 BITS=3 kernel emulation in Python — vs Python reference.

If emulation matches reference, the kernel HOT-LOOP LOGIC is correct
and the bug is CUDA-specific (SMEM/shuffle/launch).

If emulation also fails, the indexing logic (lane mapping, B_packed
stride, k_base) is wrong.
"""

import os
os.environ.setdefault("XFP_V2", "3")
os.environ.setdefault("XFP_GROUP_SIZE", "128")

import torch
import torch.nn.functional as F

from vllm.multiquant.xfp.xfp_pack import (
    xfp_pack_v2, xfp_repack_v3,
)


def dequant_v2_bits3(packed_3d, library, lib_id, scale, mid, group_size=128):
    """Reference dequant for V3 BITS=3 per-group layout."""
    bits, vpw, mask = 3, 10, 0x7
    N_out, G, K_PER_GROUP = packed_3d.shape
    K = G * group_size
    W = torch.zeros(N_out, K, dtype=torch.float32, device=packed_3d.device)
    pk = packed_3d.to(torch.int64)
    for n in range(N_out):
        for g in range(G):
            cb = library[int(lib_id[n, g].item())]
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


def emulate_v3_kernel(x_fp32, packed_repacked_flat, library, lib_id,
                     scale, mid, K, N_out, group_size=128):
    """Mirror V3 kernel hot-loop in Python — uses SAME indexing math
    as the CUDA kernel. If this matches the dequant-reference, the
    kernel logic is correct (bug is CUDA-specific).

    Layout assumptions (must match xfp_repack_v3 + kernel):
      packed_repacked_flat: 1D [n_warp_iters * N_out * ACTIVE_LANES]
      Index: gi * N * 26 + n * 26 + lane

    Per warp at (m, n):
      - Iterate gi=0..G/2-1
      - For each lane in [0, 26): decode 10 vals (with bounds-skip for
        last 2 padding slots in group end).
      - lane_grp = lane // 13, lane_in_grp = lane % 13
      - my_group_idx = gi * 2 + lane_grp
      - k_base = my_group_idx * 128 + lane_in_grp * 10
      - For slot in 0..9: if k_base + slot - my_group_idx*128 >= 128: break
        Wait — actually: abs_slot = lane_in_grp * 10 + slot, check < 128.
      - acc[lane] = sum of w[slot] * x[k_base + slot] for valid slots
    """
    bits = 3
    vpw = 10
    mask = 0x7
    LUT_SIZE = 1 << bits  # 8
    LANES_PER_GROUP = 13
    CB_PER_ITER = 2
    ACTIVE_LANES = CB_PER_ITER * LANES_PER_GROUP  # 26
    G = K // group_size
    n_warp_iters = G // CB_PER_ITER

    M = x_fp32.shape[0]
    C = torch.zeros(M, N_out, dtype=torch.float32, device=x_fp32.device)

    for m in range(M):
        a_row = x_fp32[m]  # [K]
        for ctx_n in range(N_out):
            # Per-row metadata
            row_lib_id = lib_id[ctx_n]   # [G]
            row_scale  = scale[ctx_n]    # [G]
            row_mid    = mid[ctx_n]      # [G]
            # Each warp's accumulator (32 lanes)
            lane_acc = torch.zeros(32, dtype=torch.float32, device=x_fp32.device)
            for gi in range(n_warp_iters):
                for lane in range(32):
                    if lane >= ACTIVE_LANES:
                        continue
                    lane_grp = lane // LANES_PER_GROUP
                    lane_in_grp = lane % LANES_PER_GROUP
                    my_group_idx = gi * CB_PER_ITER + lane_grp

                    lib_id_val = int(row_lib_id[my_group_idx].item())
                    cb = library[lib_id_val]  # [LUT_SIZE]
                    scale_f = float(row_scale[my_group_idx].item())
                    mid_f = float(row_mid[my_group_idx].item())

                    # B_packed read with V3 stride
                    flat_idx = gi * N_out * ACTIVE_LANES + ctx_n * ACTIVE_LANES + lane
                    word = int(packed_repacked_flat[flat_idx].item()) & 0xFFFFFFFF

                    k_base = my_group_idx * group_size + lane_in_grp * vpw

                    for slot in range(vpw):
                        abs_slot = lane_in_grp * vpw + slot
                        if abs_slot >= group_size:
                            break
                        idx = (word >> (slot * bits)) & mask
                        w_norm = float(cb[idx].item())
                        w = w_norm * scale_f + mid_f
                        a = float(a_row[k_base + slot].item())
                        lane_acc[lane] += w * a
            # Warp-reduction: sum all 32 lanes
            C[m, ctx_n] = lane_acc.sum().item()

    return C


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    N_out, K, M = 32, 256, 4
    g = torch.Generator(device=device).manual_seed(42)
    W = torch.randn(N_out, K, generator=g, device=device, dtype=torch.float32) * 0.1
    x = torch.randn(M, K, generator=g, device=device, dtype=torch.float32) * 0.1

    print("=== Pack ===")
    packed_3d, library, lib_id, scale, mid, stats = xfp_pack_v2(
        W, bits=3, group_size=128, library_size=16,
        lloyd_iters=10, library_iters=5,
    )
    print(f"  packed shape: {tuple(packed_3d.shape)}")
    library_fp32 = library.float()
    scale_fp32 = scale.float()
    mid_fp32 = mid.float()

    print("\n=== Python reference dequant ===")
    W_deq = dequant_v2_bits3(packed_3d, library_fp32, lib_id, scale_fp32, mid_fp32)
    C_ref = (x @ W_deq.T).float()
    print(f"  C_ref[0, :4]:  {C_ref[0, :4].tolist()}")

    print("\n=== V3 emulation (Python mirror of kernel logic) ===")
    packed_repacked = xfp_repack_v3(packed_3d).contiguous()
    print(f"  repacked shape: {tuple(packed_repacked.shape)}")
    # The kernel uses bf16 input but we pass fp32 here for emulation purity.
    C_emu = emulate_v3_kernel(x, packed_repacked, library_fp32, lib_id,
                              scale_fp32, mid_fp32, K=K, N_out=N_out)
    print(f"  C_emu[0, :4]:  {C_emu[0, :4].tolist()}")

    cos = F.cosine_similarity(C_emu.flatten().unsqueeze(0),
                             C_ref.flatten().unsqueeze(0), dim=1).item()
    max_err = (C_emu - C_ref).abs().max().item()
    print(f"\n  emulation vs ref: cos={cos:.6f}, max_err={max_err:.4e}")
    if cos > 0.999:
        print("  PASS: V3 indexing logic is CORRECT.")
        print("  → Bug must be CUDA-specific (SMEM/shuffle/launch).")
    else:
        print("  FAIL: V3 indexing logic is WRONG.")
        print("  → Bug is in the Python emulation OR my understanding of layout.")
    return 0 if cos > 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
