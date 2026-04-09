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
        return cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
            nsamples=config.get("nsamples", 128),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """Route to RTN method based on per-class policy.

        Checks MultiQuantPolicyRegistry to determine if this layer
        should be quantized, and at what bit width.
        """
        from vllm.multiquant.policy import (
            MultiQuantPolicyRegistry,
            classify_layer,
        )

        # Determine layer class and check policy
        layer_type = classify_layer(prefix)
        policy_reg = MultiQuantPolicyRegistry.get_active()
        if policy_reg is not None:
            policy = policy_reg.get_weight_policy(layer_type)
            if not policy.is_quantized:
                return None  # 1:1, no quantization
            bits = policy.bits
            group_size = policy.group_size or self.group_size
        else:
            # No registry → use config defaults
            bits = self.bits
            group_size = self.group_size

        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            from vllm.multiquant.autoround.online_linear import (
                AutoRoundRTNLinearMethod,
            )
            return AutoRoundRTNLinearMethod(self, bits=bits,
                                            group_size=group_size)

        # FusedMoE support
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
            if isinstance(layer, FusedMoE):
                from vllm.multiquant.autoround.online_moe import (
                    AutoRoundRTNMoEMethod,
                )
                return AutoRoundRTNMoEMethod(self, bits=bits,
                                             group_size=group_size)
        except ImportError:
            pass

        return None
