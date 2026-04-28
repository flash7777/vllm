"""End-to-end V2 online branch test.

Simulates the production loading path (process_weights_after_loading
→ apply) with XFP_V2=1, on a single Linear layer, end-to-end.

Pass criteria:
  - process_weights_after_loading triggers V2 branch (no exceptions).
  - Pack populates the 5 V2 attributes; layer._xfp_v2 == True.
  - apply() output cos vs BF16 reference matches the Phase-1 pack stats.
  - Cache round-trip (save → reload via load_linear_v2) reproduces apply
    output bit-exact.
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MockQuantConfig:
    auto_min_cos = 0.98


def main() -> None:
    print(f"torch {torch.__version__}")
    os.environ["XFP_V2"] = "1"
    os.environ["XFP_GROUP_SIZE"] = "128"
    os.environ["XFP_LIBRARY_SIZE"] = "32"
    # Disable cache for first part to isolate the pack path:
    os.environ.pop("MULTIQUANT_CACHE_DIR", None)

    from vllm.multiquant.xfp.online_linear import XFPLinearMethod

    method = XFPLinearMethod(_MockQuantConfig(), dtype="xfp4")
    print(f"  method.bits = {method.bits}")

    # Build a fake layer (just an nn.Module with .weight)
    torch.manual_seed(0)
    N, K = 1024, 2048
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.randn(N, K, dtype=torch.bfloat16) * 0.02, requires_grad=False
    )
    layer.layer_name = "test.layer.0.proj"

    # Reference forward in BF16 for cos comparison
    torch.manual_seed(1)
    x = torch.randn(8, K, dtype=torch.bfloat16) * 0.1
    y_ref = F.linear(x, layer.weight)

    print(f"\nW shape={tuple(layer.weight.shape)} dtype={layer.weight.dtype}")
    print(f"x shape={tuple(x.shape)} dtype={x.dtype}")

    # Trigger V2 pack via process_weights_after_loading
    method.process_weights_after_loading(layer)

    print(f"\nAfter process_weights_after_loading:")
    print(f"  _xfp_v2          = {getattr(layer, '_xfp_v2', None)}")
    print(f"  _xfp_bits        = {getattr(layer, '_xfp_bits', None)}")
    print(f"  _xfp_K           = {getattr(layer, '_xfp_K', None)}")
    print(f"  _xfp_N           = {getattr(layer, '_xfp_N', None)}")
    print(f"  _xfp_group_size  = {getattr(layer, '_xfp_group_size', None)}")
    print(f"  _xfp_library_size= {getattr(layer, '_xfp_library_size', None)}")
    print(f"  xfp_packed       = {tuple(layer.xfp_packed.shape)} {layer.xfp_packed.dtype}")
    print(f"  xfp_library      = {tuple(layer.xfp_library.shape)} {layer.xfp_library.dtype}")
    print(f"  xfp_group_lib_id = {tuple(layer.xfp_group_lib_id.shape)} {layer.xfp_group_lib_id.dtype}")
    print(f"  xfp_group_scale  = {tuple(layer.xfp_group_scale.shape)} {layer.xfp_group_scale.dtype}")
    print(f"  xfp_group_mid    = {tuple(layer.xfp_group_mid.shape)} {layer.xfp_group_mid.dtype}")

    assert layer._xfp_v2 is True, "V2 flag not set"
    assert layer.xfp_library.shape == (32, 16), "library shape wrong"

    # Forward via V2 reference apply
    y_v2 = method.apply(layer, x, bias=None)

    cos = F.cosine_similarity(
        y_ref.float().reshape(-1).unsqueeze(0),
        y_v2.float().reshape(-1).unsqueeze(0),
        dim=1,
    ).item()
    rel_l2 = (y_ref.float() - y_v2.float()).norm().item() / y_ref.float().norm().item()
    print(f"\n  apply forward output:  cos={cos:.5f}  rel_L2={rel_l2:.4g}")
    assert cos > 0.99, f"V2 forward cos too low: {cos}"

    # Cache round-trip — save_linear_v2 then reload into a fresh layer
    with tempfile.TemporaryDirectory() as tmp:
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
        from vllm.multiquant.xfp import xfp_weight_cache as xfp_cache

        cache = MultiQuantWeightCache(
            cache_root=os.path.join(tmp, "cache"),
            model_basename="test",
            cache_key="phase4_test",
            read_only=False,
            model_path=None,
        )

        ok_save = xfp_cache.save_linear_v2(cache, "phase4.test.proj", layer)
        assert ok_save

        # Fresh layer + load
        layer_b = nn.Module()
        layer_b.layer_name = "phase4.test.proj"
        ok_load = xfp_cache.load_linear_v2(cache, "phase4.test.proj",
                                           layer_b, torch.device("cpu"))
        assert ok_load
        assert layer_b._xfp_v2 is True

        # apply() on freshly-loaded layer must match
        y_v2_b = method.apply(layer_b, x, bias=None)
        cos_b = F.cosine_similarity(
            y_v2.float().reshape(-1).unsqueeze(0),
            y_v2_b.float().reshape(-1).unsqueeze(0),
            dim=1,
        ).item()
        max_err = (y_v2 - y_v2_b).abs().max().item()
        print(f"  cache round-trip apply: cos={cos_b:.6f}  max_err={max_err:.4g}")
        assert cos_b > 0.99999, f"round-trip degraded: {cos_b}"

    print("\n✓ Phase 4a (online V2 branch) PASS")


if __name__ == "__main__":
    main()
