# SPDX-License-Identifier: Apache-2.0
"""AutoRound RTN config for MultiQuant."""

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
    """AutoRound opt_rtn mode (--iters 0): fast INT4 at load time.

    Uses AutoRound's optimized Round-to-Nearest without iterative tuning.
    Requires auto_round package + minimal calibration data (auto-downloaded).
    Output is GPTQ-compatible → can use Marlin kernel for inference.

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
    def from_config(cls, config: dict[str, Any]) -> AutoRoundRTNConfig:
        return cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
            nsamples=config.get("nsamples", 128),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            from vllm.multiquant.autoround.online_linear import (
                AutoRoundRTNLinearMethod,
            )
            return AutoRoundRTNLinearMethod(self)
        return None
