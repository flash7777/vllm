"""Phase-2 round-trip test: pack_v2 → save_linear_v2 → load_linear_v2.

Verifies the V2 cache format is symmetric and lossless. Builds an XFPLinear-
shaped layer, packs a real weight, saves to a tmp cache dir, then loads
into a fresh layer and asserts:
  - Every V2 tensor field matches bit-exact (no precision loss).
  - All metadata round-trips (bits, K, N, group_size, library_size).
  - Reconstruction from cache matches reconstruction from original pack.

Doesn't need GPU; runs purely on CPU.
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn as nn

from vllm.multiquant.weight_cache import MultiQuantWeightCache
from vllm.multiquant.xfp.xfp_pack import xfp_pack_v2, dequant_xfp_v2
from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache


class _DummyLayer(nn.Module):
    """Minimal layer to attach packed tensors to."""
    def __init__(self):
        super().__init__()


def _attach_v2(layer: nn.Module, packed, library, lib_id, scale, mid,
               bits: int, K: int, N: int, group_size: int, library_size: int):
    layer.xfp_packed       = nn.Parameter(packed.contiguous(), requires_grad=False)
    layer.xfp_library      = nn.Parameter(library.contiguous(), requires_grad=False)
    layer.xfp_group_lib_id = nn.Parameter(lib_id.contiguous(), requires_grad=False)
    layer.xfp_group_scale  = nn.Parameter(scale.contiguous(), requires_grad=False)
    layer.xfp_group_mid    = nn.Parameter(mid.contiguous(), requires_grad=False)
    layer._xfp_bits = bits
    layer._xfp_K = K
    layer._xfp_N = N
    layer._xfp_group_size = group_size
    layer._xfp_library_size = library_size
    layer._xfp_has_outliers = False


def main() -> None:
    print(f"torch {torch.__version__}")
    torch.manual_seed(0)
    N, K = 512, 2048
    bits = 4
    group_size = 128
    library_size = 32
    W = torch.randn(N, K, dtype=torch.float32) * 0.02

    print(f"\nW shape={tuple(W.shape)} norm={W.norm().item():.3g}")

    # Pack with V2
    packed, library, lib_id, scale, mid, stats = xfp_pack_v2(
        W, bits=bits, group_size=group_size, library_size=library_size,
    )
    print(f"V2 pack: cos_sim={stats.cos_sim:.5f} mse={stats.mse:.4g}")

    # Build dummy layer + attach
    layer_a = _DummyLayer()
    _attach_v2(layer_a, packed, library, lib_id, scale, mid,
               bits, K, N, group_size, library_size)

    with tempfile.TemporaryDirectory() as tmp:
        # Build a cache instance pointed at tmp
        cache_root = os.path.join(tmp, "cache")
        cache = MultiQuantWeightCache(
            cache_root=cache_root,
            model_basename="test_model",
            cache_key="phase2_test",
            read_only=False,
            model_path=None,
        )
        layer_prefix = "test.layer.0.xfp_proj"

        # Save
        ok = xfp_cache.save_linear_v2(cache, layer_prefix, layer_a)
        print(f"\nsave_linear_v2: ok={ok}")
        assert ok, "save returned False"

        # Inspect on-disk artefacts
        layer_dir = cache.layer_path(layer_prefix)
        files = sorted(os.listdir(layer_dir)) if os.path.exists(layer_dir) else []
        print(f"on-disk files: {files}")
        for fname in (
            "xfp_packed.safetensors",
            "xfp_library.safetensors",
            "xfp_group_lib_id.safetensors",
            "xfp_group_scale.safetensors",
            "xfp_group_mid.safetensors",
            "_manifest.json",
        ):
            assert fname in files, f"missing on disk: {fname}"

        # Load into fresh layer
        layer_b = _DummyLayer()
        ok = xfp_cache.load_linear_v2(cache, layer_prefix, layer_b, torch.device("cpu"))
        print(f"load_linear_v2: ok={ok}")
        assert ok, "load returned False"

        # Bit-exact tensor equality
        for attr in ("xfp_packed", "xfp_library", "xfp_group_lib_id",
                     "xfp_group_scale", "xfp_group_mid"):
            ta = getattr(layer_a, attr).data
            tb = getattr(layer_b, attr).data
            assert ta.shape == tb.shape, f"{attr}: shape mismatch {ta.shape} vs {tb.shape}"
            assert ta.dtype == tb.dtype, f"{attr}: dtype mismatch {ta.dtype} vs {tb.dtype}"
            assert torch.equal(ta, tb), f"{attr}: value mismatch"
            print(f"  ✓ {attr}: shape={tuple(ta.shape)} dtype={ta.dtype}")

        # Metadata round-trip
        assert layer_b._xfp_bits == bits
        assert layer_b._xfp_K == K
        assert layer_b._xfp_N == N
        assert layer_b._xfp_group_size == group_size
        assert layer_b._xfp_library_size == library_size
        assert getattr(layer_b, "_xfp_v2", False) is True
        print(f"  ✓ metadata: bits={layer_b._xfp_bits} K={layer_b._xfp_K} "
              f"N={layer_b._xfp_N} group={layer_b._xfp_group_size} "
              f"L={layer_b._xfp_library_size} v2={layer_b._xfp_v2}")

        # Functional check: dequant from cache should equal pack reconstruction
        # Use the unpack-from-W path (same as Phase-1 verify) since the
        # python dequant_xfp_v2 takes unpacked idx — easier than rolling
        # a pack-aware unpacker for this test.
        N2, K2 = layer_b._xfp_N, layer_b._xfp_K
        G = K2 // layer_b._xfp_group_size
        Wf = W.float()
        chosen_lib = layer_b.xfp_library.float()[layer_b.xfp_group_lib_id.long()]
        W_norm = (Wf.reshape(N2, G, layer_b._xfp_group_size)
                  - layer_b.xfp_group_mid.float().unsqueeze(-1)
                 ) / layer_b.xfp_group_scale.float().unsqueeze(-1)
        d = (W_norm.unsqueeze(-1) - chosen_lib.unsqueeze(2)).abs()
        idx = d.argmin(dim=-1)
        rec_norm = torch.gather(chosen_lib, 2, idx)
        W_rec_b = (rec_norm * layer_b.xfp_group_scale.float().unsqueeze(-1)
                   + layer_b.xfp_group_mid.float().unsqueeze(-1)).reshape(N2, K2)
        cos = torch.nn.functional.cosine_similarity(
            Wf.reshape(-1).unsqueeze(0),
            W_rec_b.reshape(-1).unsqueeze(0), dim=1).item()
        print(f"  ✓ post-load reconstruction cos vs original W: {cos:.5f}")
        assert abs(cos - stats.cos_sim) < 1e-4, (
            f"reconstruction differs from pre-save: {cos} vs {stats.cos_sim}"
        )

    print("\n✓ Phase 2 round-trip PASS")


if __name__ == "__main__":
    main()
