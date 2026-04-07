# SPDX-License-Identifier: Apache-2.0
"""AutoRound INT2/INT3/INT4 Loader — reads GPTQ-packed safetensors,
dequantizes to FP16, yields standard (name, weight) pairs.

Usage:
    from vllm.multiquant.weight_quant.autoround_loader import load_autoround_as_fp16
    for name, tensor in load_autoround_as_fp16(model_path):
        # name: "model.layers.10.mlp.experts.0.gate_proj.weight"
        # tensor: [N, K] float16
        ...

Supports bits=2 (16 per int32), bits=3 (bit-stream), bits=4 (8 per int32).
Mixed-precision: layers with .weight (BF16) pass through unchanged.
Layers with .qweight/.scales/.qzeros get dequantized to FP16.
"""

import glob
import json
import os
from collections import defaultdict
from typing import Iterator

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _dequant_column(qweight: torch.Tensor, scales: torch.Tensor,
                    qzeros: torch.Tensor, bits: int,
                    group_size: int) -> torch.Tensor:
    """Dequantize one GPTQ-packed weight matrix to float16.

    Args:
        qweight: [K_packed, N] int32
        scales:  [n_groups, N] float16
        qzeros:  [n_groups, N_zp_packed] int32
        bits:    2, 3, or 4
        group_size: typically 128

    Returns: [N, K] float16 (transposed, ready for F.linear)
    """
    K_packed, N = qweight.shape
    mask = (1 << bits) - 1

    if bits == 2:
        K = K_packed * 16
        shifts = torch.arange(0, 32, 2, device=qweight.device,
                              dtype=torch.int32)
        expanded = qweight.unsqueeze(1).expand(-1, 16, -1)
        unpacked = ((expanded >> shifts.view(1, 16, 1)) & mask).reshape(K, N)
    elif bits == 3:
        K = (K_packed * 32) // 3
        # Bit-stream unpack on CPU (cross-boundary extraction)
        unpacked_cols = []
        qw_cpu = qweight.cpu()
        for col in range(N):
            words = qw_cpu[:, col].tolist()
            vals = []
            buf = 0
            buf_bits = 0
            for w in words:
                buf |= (w & 0xFFFFFFFF) << buf_bits
                buf_bits += 32
                while buf_bits >= 3 and len(vals) < K:
                    vals.append(buf & 0x7)
                    buf >>= 3
                    buf_bits -= 3
            vals.extend([0] * (K - len(vals)))
            unpacked_cols.append(vals[:K])
        unpacked = torch.tensor(unpacked_cols, device=qweight.device,
                                dtype=torch.int32).T
    elif bits == 4:
        K = K_packed * 8
        shifts = torch.arange(0, 32, 4, device=qweight.device,
                              dtype=torch.int32)
        expanded = qweight.unsqueeze(1).expand(-1, 8, -1)
        unpacked = ((expanded >> shifts.view(1, 8, 1)) & mask).reshape(K, N)
    else:
        raise ValueError(f"Unsupported bits={bits}")

    n_groups = K // group_size

    # Unpack zero points
    if bits in (2, 4):
        pf = 32 // bits
        zp_shifts = torch.arange(0, 32, bits, device=qzeros.device,
                                 dtype=torch.int32)[:pf]
        zp_exp = qzeros.unsqueeze(1).expand(-1, pf, -1)
        zp_all = ((zp_exp >> zp_shifts.view(1, -1, 1)) & mask)
        zp_all = zp_all.reshape(n_groups, -1)[:, :N]
    elif bits == 3:
        zp_list = []
        qz_cpu = qzeros.cpu()
        for g in range(n_groups):
            words = qz_cpu[g].tolist()
            vals = []
            buf = 0
            buf_bits = 0
            for w in words:
                buf |= (w & 0xFFFFFFFF) << buf_bits
                buf_bits += 32
                while buf_bits >= 3 and len(vals) < N:
                    vals.append(buf & 0x7)
                    buf >>= 3
                    buf_bits -= 3
            vals.extend([0] * (N - len(vals)))
            zp_list.append(vals[:N])
        zp_all = torch.tensor(zp_list, device=qzeros.device,
                              dtype=torch.int32)

    group_idx = torch.arange(K, device=qweight.device) // group_size
    w = scales[group_idx] * (unpacked.float() - zp_all[group_idx].float())
    return w.T.contiguous().to(torch.float16)


def load_autoround_as_fp16(
    model_path: str,
    device: str = "cpu",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load AutoRound INT2/INT3/INT4 model, yield (name, fp16_tensor) pairs.

    Dequantizes qweight+scales+qzeros → FP16 on the fly.
    BF16/FP16 .weight tensors pass through unchanged.
    Skips g_idx (not needed after dequant).
    """
    # Read config
    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path))
    qc = config.get("quantization_config", {})
    bits = qc.get("bits", 4)
    group_size = qc.get("group_size", 128)

    logger.info("AutoRound loader: bits=%d, group_size=%d, path=%s",
                bits, group_size, model_path)

    # Collect all safetensor shards
    shards = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    # Group tensors by layer prefix
    # e.g. "model.layers.10.mlp.experts.0.gate_proj" → {qweight, scales, qzeros}
    from safetensors import safe_open

    pending: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    yielded = set()

    for shard_path in shards:
        f = safe_open(shard_path, framework="pt", device=device)
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key)

            # Pass-through: non-quantized tensors (.weight, .bias, etc.)
            if not any(key.endswith(s) for s in
                       (".qweight", ".scales", ".qzeros", ".g_idx")):
                yield key, tensor
                yielded.add(key)
                continue

            # Skip g_idx (not needed for dequant)
            if key.endswith(".g_idx"):
                continue

            # Collect qweight/scales/qzeros by prefix
            prefix = key.rsplit(".", 1)[0]  # remove .qweight/.scales/.qzeros
            suffix = key.rsplit(".", 1)[1]
            pending[prefix][suffix] = tensor

            # When we have all three, dequantize
            if all(s in pending[prefix] for s in
                   ("qweight", "scales", "qzeros")):
                qw = pending[prefix]["qweight"]
                sc = pending[prefix]["scales"]
                qz = pending[prefix]["qzeros"]

                # Move to GPU for dequant, then back
                if device == "cpu":
                    qw_g = qw.cuda()
                    sc_g = sc.cuda()
                    qz_g = qz.cuda()
                    w = _dequant_column(qw_g, sc_g, qz_g, bits, group_size)
                    w = w.cpu()
                    del qw_g, sc_g, qz_g
                    torch.cuda.empty_cache()
                else:
                    w = _dequant_column(qw, sc, qz, bits, group_size)

                # Yield as .weight (standard Linear format)
                weight_name = prefix + ".weight"
                yield weight_name, w
                yielded.add(weight_name)

                # Cleanup
                del pending[prefix]

    # Warn about incomplete groups
    for prefix, bufs in pending.items():
        logger.warning("AutoRound loader: incomplete group %s: %s",
                       prefix, list(bufs.keys()))
