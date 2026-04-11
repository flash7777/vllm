# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN config for MultiQuant.

Thin entry-point for the `autoround_rtn` CLI quant_method. The actual
per-layer method dispatch (RTN / XFP / Archer TQ-RQ / pass-through) is
delegated to `vllm.multiquant.policy.create_weight_method`, which reads
the active `MultiQuantPolicyRegistry` and routes by dtype string.

The Registry-based dispatch is the single source of truth across all
multiquant quant-on-load methods. `AutoRoundRTNConfig` only survives as
the CLI alias that activates the dispatcher pipeline — without a matching
`--quantization` string, vLLM would never instantiate our code path.
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
    """On-the-fly weight quantization at model load time.

    This config is the CLI entry point for `--quantization autoround_rtn`.
    All per-layer routing is delegated to `create_weight_method` in
    `vllm.multiquant.policy`, which reads the active MultiQuant registry.

    The `bits` / `group_size` / etc. fields remain for backward
    compatibility with `from_config` callers and as plausible defaults when
    the policy registry is missing an entry.
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
        """Re-construct the config in the Engine Core process.

        Also rebuilds `MultiQuantPolicyRegistry._active` from the
        `_mq_class_dtype` dict that `arg_utils.py` injected into
        `hf_config.quantization_config`. Without this rebuild, the
        Registry singleton in the Engine Core process is empty and the
        central dispatcher cannot route layers.
        """
        instance = cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
            nsamples=config.get("nsamples", 128),
        )

        mq_dtype: dict[str, str] = config.get("_mq_class_dtype", {})
        mq_gs: dict[str, int] = config.get("_mq_class_gs", {})

        if mq_dtype:
            from vllm.multiquant.policy import (
                MultiQuantPolicyRegistry,
                ALL_CLASSES,
            )
            if MultiQuantPolicyRegistry.get_active() is None:
                reg = MultiQuantPolicyRegistry()
                for cls_key in ALL_CLASSES:
                    if cls_key in mq_dtype:
                        reg.set(
                            cls_key,
                            mq_dtype[cls_key],
                            "cli",
                            group_size=mq_gs.get(cls_key, 0),
                        )
                MultiQuantPolicyRegistry._active = reg
                logger.info(
                    "autoround_rtn: rebuilt MultiQuant registry from "
                    "config dict: %s",
                    {k: v for k, v in mq_dtype.items()
                     if v not in ("bf16", "fp16")},
                )
        return instance

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """Delegate to the central Registry-based dispatcher."""
        from vllm.multiquant.policy import create_weight_method
        return create_weight_method(self, layer, prefix)
