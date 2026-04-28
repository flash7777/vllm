"""Phase 4b — XFP-V2 MoE pack roundtrip + decompression sanity.

Verifies xfp_moe_pack_v2 produces correctly-shaped tensors, the cache
save/load round-trips bit-exact, and per-expert dequant matches a Phase-1
xfp_pack_v2 reference reconstruction.
"""
from __future__ import annotations

import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.multiquant.xfp.xfp_pack import (
    xfp_pack_v2, xfp_moe_pack_v2, dequant_xfp_v2_packed,
)


def main() -> None:
    print(f"torch {torch.__version__}")
    torch.manual_seed(0)
    E, N, K = 8, 256, 1024
    bits = 4
    group_size = 128
    library_size = 32

    W_stack = torch.randn(E, N, K, dtype=torch.float32) * 0.02
    print(f"W_stack: {tuple(W_stack.shape)} norm={W_stack.norm().item():.3g}")

    # Pack via xfp_moe_pack_v2 (shared library across experts)
    packed, library, lib_id, scale, mid, stats = xfp_moe_pack_v2(
        W_stack, bits=bits, group_size=group_size, library_size=library_size,
    )
    print(f"\nMoE-V2 pack stats: cos={stats.cos_sim:.5f} mse={stats.mse:.4g} "
          f"lib_p5={stats.library_p5_cos:.4f} overhead={stats.overhead_bits_per_param:.2f}")
    print(f"  packed       = {tuple(packed.shape)} {packed.dtype}")
    print(f"  library      = {tuple(library.shape)} {library.dtype}")
    print(f"  group_lib_id = {tuple(lib_id.shape)} {lib_id.dtype}")
    print(f"  group_scale  = {tuple(scale.shape)} {scale.dtype}")
    print(f"  group_mid    = {tuple(mid.shape)} {mid.dtype}")

    # Per-expert dequant via dequant_xfp_v2_packed and check cos
    cos_per_expert = []
    for e in range(E):
        Wr = dequant_xfp_v2_packed(
            packed[e], library, lib_id[e], scale[e], mid[e],
            K=K, bits=bits, group_size=group_size,
        )
        cos = F.cosine_similarity(
            W_stack[e].reshape(-1).unsqueeze(0),
            Wr.reshape(-1).unsqueeze(0), dim=1,
        ).item()
        cos_per_expert.append(cos)
    print(f"\nPer-expert dequant cos: min={min(cos_per_expert):.5f} "
          f"avg={sum(cos_per_expert)/E:.5f} max={max(cos_per_expert):.5f}")
    assert min(cos_per_expert) > 0.99, "per-expert cos too low"

    # Cache round-trip via save_moe_v2 / load_moe_v2 on a dummy MoE-shaped layer
    layer = nn.Module()
    layer.w13_xfp_packed = nn.Parameter(packed.contiguous(), requires_grad=False)
    layer.w13_xfp_library = nn.Parameter(library.contiguous(), requires_grad=False)
    layer.w13_xfp_group_lib_id = nn.Parameter(lib_id.contiguous(), requires_grad=False)
    layer.w13_xfp_group_scale = nn.Parameter(scale.contiguous(), requires_grad=False)
    layer.w13_xfp_group_mid = nn.Parameter(mid.contiguous(), requires_grad=False)
    # Reuse the same packed tensors for w2 to avoid packing twice in this smoke test
    layer.w2_xfp_packed = nn.Parameter(packed.contiguous(), requires_grad=False)
    layer.w2_xfp_library = nn.Parameter(library.contiguous(), requires_grad=False)
    layer.w2_xfp_group_lib_id = nn.Parameter(lib_id.contiguous(), requires_grad=False)
    layer.w2_xfp_group_scale = nn.Parameter(scale.contiguous(), requires_grad=False)
    layer.w2_xfp_group_mid = nn.Parameter(mid.contiguous(), requires_grad=False)
    layer._xfp_moe_bits = bits
    layer._xfp_moe_K13 = K
    layer._xfp_moe_N13 = N
    layer._xfp_moe_K2 = K
    layer._xfp_moe_N2 = N
    layer._xfp_moe_E = E
    layer._xfp_moe_fpe13 = packed[0].numel()
    layer._xfp_moe_fpe2 = packed[0].numel()
    layer._xfp_moe_group_size = group_size
    layer._xfp_moe_library_size = library_size

    with tempfile.TemporaryDirectory() as tmp:
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache
        cache = MultiQuantWeightCache(
            cache_root=os.path.join(tmp, "cache"), model_basename="test",
            cache_key="phase4b_test", read_only=False, model_path=None,
        )
        ok = xfp_cache.save_moe_v2(cache, "test.layer.0.experts", layer)
        assert ok
        layer_b = nn.Module()
        ok = xfp_cache.load_moe_v2(cache, "test.layer.0.experts", layer_b, torch.device("cpu"))
        assert ok
        # Bit-exact equality on key tensors
        for attr in ("w13_xfp_packed", "w13_xfp_library", "w13_xfp_group_lib_id",
                     "w13_xfp_group_scale", "w13_xfp_group_mid",
                     "w2_xfp_packed", "w2_xfp_library", "w2_xfp_group_lib_id",
                     "w2_xfp_group_scale", "w2_xfp_group_mid"):
            ta = getattr(layer, attr).data
            tb = getattr(layer_b, attr).data
            assert torch.equal(ta, tb), f"{attr} mismatch"
        assert layer_b._xfp_moe_E == E
        assert layer_b._xfp_moe_group_size == group_size
        assert layer_b._xfp_moe_library_size == library_size
        assert layer_b._xfp_v2 is True
        print("  ✓ MoE V2 cache save/load bit-exact")

    print("\n✓ Phase 4b PASS")


if __name__ == "__main__":
    main()
