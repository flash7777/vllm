# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN config for MultiQuant.

Per-class quantization: uses MultiQuantPolicyRegistry to decide
which layers to quantize and at what bit width. Layers whose policy
is bf16/fp16 (no quantization) get None → loaded as-is.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)


class AutoRoundRTNConfig(QuantizationConfig):
    """On-the-fly RTN quantization at model load time.

    Packs BF16 weights to GPTQ int32 format via symmetric RTN.
    Inference uses the same kernels as pre-quantized GPTQ models:
    - INT2/INT3: MQSub4LinearMethod / MQSub4MoEMethod (fused GEMM)
    - INT4: Marlin kernel

    Per-class routing via MultiQuantPolicyRegistry:
    - Only layers whose policy requests quantization get packed.
    - Other layers stay BF16 (get_quant_method returns None).

    Usage: --quantization autoround_rtn
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        nsamples: int = 128,
        seqlen: int = 2048,
        dataset: str = "NeelNanda/pile-10k",
    ):
        self.bits = bits
        self.group_size = group_size
        self.nsamples = nsamples
        self.seqlen = seqlen
        self.dataset = dataset
        # Per-class policy snapshot (survives process serialization)
        # Stores dtype strings (e.g. "int3", "bf16", "tq3w") not just bits.
        # Future-proof: different algorithms with same bit width are
        # distinguishable (int4 vs nf4 vs mxfp4).
        self._class_dtype: dict[str, str] = {}
        self._class_gs: dict[str, int] = {}

    @classmethod
    def get_name(cls) -> str:
        return "autoround_rtn"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundRTNConfig":
        instance = cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
            nsamples=config.get("nsamples", 128),
        )
        # Read policy from config (injected by arg_utils before serialization)
        mq_dtype = config.get("_mq_class_dtype", {})
        mq_gs = config.get("_mq_class_gs", {})
        if mq_dtype:
            instance._class_dtype = mq_dtype
            instance._class_gs = mq_gs
            logger.info("RTN: policy loaded from config: %s",
                        {k: v for k, v in mq_dtype.items()
                         if v not in ("bf16", "fp16", "auto")})
        return instance

    def _snapshot_policy(self) -> None:
        """Capture policy into serializable dicts (call once from API server)."""
        if self._class_dtype:
            return  # already captured
        from vllm.multiquant.policy import (
            MultiQuantPolicyRegistry,
            ALL_CLASSES,
        )
        reg = MultiQuantPolicyRegistry.get_active()
        if reg is None:
            logger.warning("RTN: no active policy registry — "
                           "all layers will use default bits=%d", self.bits)
            return
        for cls in ALL_CLASSES:
            p = reg.get(cls)
            self._class_dtype[cls] = p.dtype
            self._class_gs[cls] = p.group_size
        logger.info("RTN: policy snapshot captured: %s",
                    {k: v for k, v in self._class_dtype.items()
                     if v not in ("bf16", "fp16", "auto")})

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """Route to RTN method based on per-class policy.

        Uses snapshot of MultiQuantPolicyRegistry that survives
        process serialization (Engine Core runs in separate process).
        """
        from vllm.multiquant.policy import (
            MultiQuantPolicyRegistry,
            classify_layer,
            WEIGHTS_SHARED,
            WEIGHTS_ROUTED,
            WEIGHTS_ATTN,
            WEIGHTS_DENSE,
            MTP,
            DELTANET,
        )

        # Snapshot policy on first call (API server process)
        self._snapshot_policy()

        # Map layer type → component class
        layer_type = classify_layer(prefix)
        type_to_cls = {
            "shared_expert": WEIGHTS_SHARED,
            "routed_expert": WEIGHTS_ROUTED,
            "attn": WEIGHTS_ATTN,
            "dense_mlp": WEIGHTS_DENSE,
            "mtp": MTP,
            "deltanet": DELTANET,
        }
        cls = type_to_cls.get(layer_type, WEIGHTS_ROUTED)

        # Check policy
        from vllm.multiquant.policy import DTYPE_BITS
        if self._class_dtype:
            dtype = self._class_dtype.get(cls, "bf16")
            group_size = self._class_gs.get(cls, 0) or self.group_size
            if dtype in ("bf16", "fp16", "fp32", "auto"):
                # 1:1, no quantization — return unquantized method
                from vllm.model_executor.layers.linear import (
                    LinearBase,
                    UnquantizedLinearMethod,
                )
                if isinstance(layer, LinearBase):
                    return UnquantizedLinearMethod()
                return None
            bits = DTYPE_BITS.get(dtype, self.bits)
        else:
            # Fallback: no policy data → use config defaults for all layers
            bits = self.bits
            group_size = self.group_size

        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            from vllm.multiquant.autoround.online_linear import (
                AutoRoundRTNLinearMethod,
            )
            return AutoRoundRTNLinearMethod(self, bits=bits,
                                            group_size=group_size)

        # FusedMoE support (INT2/INT3 only — INT4 MoE uses standard path)
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
            if isinstance(layer, FusedMoE):
                if bits in (2, 3):
                    from vllm.multiquant.autoround.online_moe import (
                        AutoRoundRTNMoEMethod,
                    )
                    moe_config = getattr(layer, "moe_config", None)
                    return AutoRoundRTNMoEMethod(
                        self, bits=bits, group_size=group_size,
                        moe_config=moe_config)
                # INT4 MoE: return None → UnquantizedFusedMoEMethod (BF16)
                # Marlin per-expert MoE is a future optimization
                return None
        except ImportError:
            pass

        return None
