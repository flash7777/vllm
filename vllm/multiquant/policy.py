# SPDX-License-Identifier: Apache-2.0
"""MultiQuant Policy Registry — per-class quantization configuration.

Defines what quantization scheme is used for each component:
- K-Cache, V-Cache (separate or joint)
- Weights: shared experts, routed experts, attention
- MTP, DeltaNet

Built from CLI args + model quantization config at startup.
Logged as banner so the active policy is always visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class QuantPolicy:
    """Quantization policy for a single component class."""
    dtype: str = "auto"       # "bf16", "fp8", "fp16", "int2", "int3", "int4",
                               # "tq2w", "tq3w", "tq4w", "tq3r", etc.
    bits: int = 16             # effective bit width (16=bf16, 8=fp8, 4=int4, etc.)
    source: str = "default"    # "cli", "model", "default"
    group_size: int = 0        # 0 = use model default or N/A
    description: str = ""      # human-readable, for startup log

    @property
    def is_quantized(self) -> bool:
        return self.bits < 16

    @property
    def is_multiquant(self) -> bool:
        """True if this dtype is handled by MultiQuant (tqXw, rqX, etc.)."""
        from vllm.multiquant.registry import is_multiquant_dtype
        return is_multiquant_dtype(self.dtype)


# Component class names (keys for the registry)
K_CACHE = "k_cache"
V_CACHE = "v_cache"
WEIGHTS_SHARED = "weights_shared"
WEIGHTS_ROUTED = "weights_routed"
WEIGHTS_ATTN = "weights_attn"
WEIGHTS_ALL = "weights"  # shorthand for all weight classes
MTP = "mtp"
DELTANET = "deltanet"

# All component classes
ALL_CLASSES = [
    K_CACHE, V_CACHE,
    WEIGHTS_SHARED, WEIGHTS_ROUTED, WEIGHTS_ATTN,
    MTP, DELTANET,
]

# Map dtype strings to bit widths
DTYPE_BITS = {
    "bf16": 16, "fp16": 16, "fp32": 32,
    "fp8": 8, "fp8_e4m3": 8, "fp8_e5m2": 8,
    "int8": 8, "int4": 4, "int3": 3, "int2": 2,
    "tq2w": 2, "tq3w": 3, "tq4w": 4,
    "tq3r": 3, "tq4r": 4,
    "tq3": 3, "tq4": 4,
    "rq2": 2, "rq3": 3, "rq4": 4,
}


def _bits_for_dtype(dtype: str) -> int:
    """Get effective bit width for a dtype string."""
    return DTYPE_BITS.get(dtype, 16)


def _desc_for_dtype(dtype: str) -> str:
    """Human-readable description for startup log."""
    descs = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp8": "FP8 E4M3",
        "int4": "INT4 (AutoRound/GPTQ)",
        "int3": "INT3 (AutoRound/GPTQ, 3-in-4)",
        "int2": "INT2 (AutoRound/GPTQ)",
        "tq2w": "WHT 2-bit (Lloyd-Max, block-32)",
        "tq3w": "WHT 3-bit (Lloyd-Max, block-32)",
        "tq4w": "WHT 4-bit (Lloyd-Max, block-32)",
        "tq3r": "Block-rot 3-bit (random orthogonal)",
        "auto": "auto (from model)",
    }
    return descs.get(dtype, dtype)


class MultiQuantPolicyRegistry:
    """Registry of quantization policies per component class.

    Built at startup from CLI args + model config.
    Immutable after construction.
    """

    def __init__(self):
        # Default: everything bf16
        self._policies: dict[str, QuantPolicy] = {
            K_CACHE: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
            V_CACHE: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
            WEIGHTS_SHARED: QuantPolicy("auto", 16, "default", 0,
                                        "auto (from model)"),
            WEIGHTS_ROUTED: QuantPolicy("auto", 16, "default", 0,
                                        "auto (from model)"),
            WEIGHTS_ATTN: QuantPolicy("auto", 16, "default", 0,
                                      "auto (from model)"),
            MTP: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
            DELTANET: QuantPolicy("bf16", 16, "default", 0, "bfloat16"),
        }

    def set(self, cls: str, dtype: str, source: str = "cli",
            group_size: int = 0) -> None:
        """Set policy for a component class."""
        self._policies[cls] = QuantPolicy(
            dtype=dtype,
            bits=_bits_for_dtype(dtype),
            source=source,
            group_size=group_size,
            description=_desc_for_dtype(dtype),
        )

    def get(self, cls: str) -> QuantPolicy:
        """Get policy for a component class."""
        return self._policies.get(cls, QuantPolicy())

    @property
    def k_cache(self) -> QuantPolicy:
        return self._policies[K_CACHE]

    @property
    def v_cache(self) -> QuantPolicy:
        return self._policies[V_CACHE]

    @property
    def kv_cache_dtype(self) -> str:
        """Legacy: single KV dtype (K and V same). Returns K dtype."""
        return self._policies[K_CACHE].dtype

    def get_weight_policy(self, layer_type: str) -> QuantPolicy:
        """Get weight policy for a specific layer type.

        layer_type: 'shared_expert', 'routed_expert', 'attn', or generic
        """
        if layer_type == "shared_expert":
            return self._policies[WEIGHTS_SHARED]
        elif layer_type == "routed_expert":
            return self._policies[WEIGHTS_ROUTED]
        elif layer_type == "attn":
            return self._policies[WEIGHTS_ATTN]
        # Generic: return routed (most common)
        return self._policies[WEIGHTS_ROUTED]

    @classmethod
    def from_cli(
        cls,
        kv_cache_dtype: str = "auto",
        k_dtype: Optional[str] = None,
        v_dtype: Optional[str] = None,
        weight_dtype: Optional[str] = None,
        weight_dtype_shared: Optional[str] = None,
        weight_dtype_routed: Optional[str] = None,
        weight_dtype_attn: Optional[str] = None,
        model_quant_config: Optional[dict] = None,
    ) -> "MultiQuantPolicyRegistry":
        """Build registry from CLI arguments + model config."""
        reg = cls()

        # 1. KV-Cache: --kv-cache-dtype sets both K and V
        if kv_cache_dtype != "auto":
            reg.set(K_CACHE, kv_cache_dtype, "cli")
            reg.set(V_CACHE, kv_cache_dtype, "cli")

        # 2. --k-dtype / --v-dtype override individually
        if k_dtype is not None:
            reg.set(K_CACHE, k_dtype, "cli")
        if v_dtype is not None:
            reg.set(V_CACHE, v_dtype, "cli")

        # 3. Weights from model config (if available)
        if model_quant_config:
            model_method = model_quant_config.get("quant_method", "")
            model_bits = model_quant_config.get("bits", 16)
            model_gs = model_quant_config.get("group_size", 0)
            extra_config = model_quant_config.get("extra_config", {})

            if model_method in ("gptq", "awq", "auto-round"):
                w_dtype = f"int{model_bits}"
                # Default: all weights at model bits
                for w_cls in [WEIGHTS_ROUTED, WEIGHTS_ATTN]:
                    reg.set(w_cls, w_dtype, "model", model_gs)
                # Check extra_config for mixed-precision (shared experts BF16)
                has_bf16_shared = any(
                    "shared_expert" in k and v.get("bits", 0) >= 16
                    for k, v in extra_config.items()
                )
                if has_bf16_shared:
                    reg.set(WEIGHTS_SHARED, "bf16", "model")
                else:
                    reg.set(WEIGHTS_SHARED, w_dtype, "model", model_gs)
            elif model_method in ("fp8", "compressed-tensors"):
                for w_cls in [WEIGHTS_SHARED, WEIGHTS_ROUTED, WEIGHTS_ATTN]:
                    reg.set(w_cls, "fp8", "model")

        # 4. --weight-dtype overrides all weights
        if weight_dtype is not None:
            for w_cls in [WEIGHTS_SHARED, WEIGHTS_ROUTED, WEIGHTS_ATTN]:
                reg.set(w_cls, weight_dtype, "cli")

        # 5. Per-class weight overrides
        if weight_dtype_shared is not None:
            reg.set(WEIGHTS_SHARED, weight_dtype_shared, "cli")
        if weight_dtype_routed is not None:
            reg.set(WEIGHTS_ROUTED, weight_dtype_routed, "cli")
        if weight_dtype_attn is not None:
            reg.set(WEIGHTS_ATTN, weight_dtype_attn, "cli")

        return reg

    def log_policy(self) -> None:
        """Log the full policy as startup banner."""
        logger.info("MultiQuant Policy:")
        display = [
            ("K-Cache", K_CACHE),
            ("V-Cache", V_CACHE),
            ("Weights (shared)", WEIGHTS_SHARED),
            ("Weights (routed)", WEIGHTS_ROUTED),
            ("Weights (attn)", WEIGHTS_ATTN),
            ("MTP", MTP),
            ("DeltaNet", DELTANET),
        ]
        for label, cls in display:
            p = self._policies[cls]
            src = f" [{p.source}]" if p.source != "default" else ""
            gs = f", group={p.group_size}" if p.group_size > 0 else ""
            logger.info("  %s %s (%s%s)%s",
                        f"{label:20s}", f"{p.dtype:8s}",
                        p.description, gs, src)

    def needs_onthefly_quant(self, layer_type: str,
                             model_dtype: str) -> bool:
        """Check if on-the-fly quantization is needed for a layer.

        True when CLI requests a quantized dtype but the model layer
        is in a higher-precision format (BF16/FP16/FP8).
        """
        policy = self.get_weight_policy(layer_type)
        if policy.source != "cli":
            return False  # no CLI override → load 1:1
        if policy.dtype == "auto" or policy.dtype == model_dtype:
            return False  # same format → no conversion
        # CLI requests lower precision than model has
        return _bits_for_dtype(policy.dtype) < _bits_for_dtype(model_dtype)
