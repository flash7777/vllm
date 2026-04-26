#!/usr/bin/env python3
"""Verify MoE weight layout transformation for INT2/INT3.

Loads one expert's weights directly from safetensor (reference),
simulates the MoE weight loader path, applies process_weights_after_loading
transform, and verifies the result matches the reference.
"""
import torch
from safetensors import safe_open

MODEL_PATH = "/data/tensordata/GLM-4.7-Flash-int2-ar/GLM-4.7-Flash-w2g128"
SHARD = f"{MODEL_PATH}/model-00001-of-00009.safetensors"
EXPERT = 0
LAYER = 1


def load_reference(f, proj: str):
    """Load raw GPTQ tensors from safetensor (reference format for kernel)."""
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}.{proj}"
    qw = f.get_tensor(f"{prefix}.qweight")   # [K/16, N] int32
    sc = f.get_tensor(f"{prefix}.scales")     # [n_groups, N] float16
    qz = f.get_tensor(f"{prefix}.qzeros")     # [n_groups, N_zp] int32
    return qw, sc, qz


def simulate_moe_loader(qw, sc, qz):
    """Simulate what MoeWNA16Method weight_loader does for GPTQ."""
    # qweight: .T.contiguous().view(uint8)
    qw_moe = qw.T.contiguous().view(torch.uint8)  # [N, K_u8]

    # scales: .T
    sc_moe = sc.T.contiguous()  # [N, n_groups]

    # qzeros: .view(uint8) → convert_gptq_int4_qzeros → .T
    qz_u8 = qz.view(torch.uint8)  # [n_groups, N_zp*4] uint8
    # convert_gptq_int4_qzeros
    shifter = torch.tensor([0, 4], dtype=torch.uint8)
    qz_nib = (qz_u8[:, :, None] >> shifter) & 0xF
    qz_nib = qz_nib + 1
    qz_conv = qz_nib[:, :, 0] + qz_nib[:, :, 1] * 16  # [n_groups, N_zp*4]
    qz_moe = qz_conv.T.contiguous()  # [N_zp*4, n_groups] uint8

    return qw_moe, sc_moe, qz_moe


def simulate_process_weights(qw_moe, sc_moe, qz_moe):
    """Simulate process_weights_after_loading transform."""
    # qweight: [N, K_u8] uint8 → view(int32) → [N, K_i32] → transpose → [K_i32, N]
    qw_i32 = qw_moe.contiguous().view(torch.int32)
    qw_kern = qw_i32.T.contiguous()  # NOTE: 2D, so transpose is .T

    # scales: [N, n_groups] → transpose → [n_groups, N]
    sc_kern = sc_moe.T.contiguous()

    # qzeros: [N_zp_u8, n_groups] → transpose → [n_groups, N_zp_u8]
    qz_t = qz_moe.T.contiguous()
    # undo convert_gptq_int4_qzeros
    lo = (qz_t & 0xF).to(torch.int16) - 1
    hi = ((qz_t >> 4) & 0xF).to(torch.int16) - 1
    qz_orig = ((lo & 0xF) | ((hi & 0xF) << 4)).to(torch.uint8)
    qz_kern = qz_orig.contiguous().view(torch.int32)

    return qw_kern, sc_kern, qz_kern


def main():
    print("Loading reference tensors from safetensor...")
    f = safe_open(SHARD, framework="pt", device="cpu")

    for proj in ["gate_proj", "down_proj", "up_proj"]:
        qw_ref, sc_ref, qz_ref = load_reference(f, proj)
        print(f"\n=== {proj} ===")
        print(f"  Reference: qweight={qw_ref.shape} scales={sc_ref.shape} "
              f"qzeros={qz_ref.shape}")

        # Simulate MoE loader
        qw_moe, sc_moe, qz_moe = simulate_moe_loader(qw_ref, sc_ref, qz_ref)
        print(f"  MoE stored: qweight={qw_moe.shape}(u8) scales={sc_moe.shape} "
              f"qzeros={qz_moe.shape}(u8)")

        # Simulate process_weights_after_loading
        qw_kern, sc_kern, qz_kern = simulate_process_weights(
            qw_moe, sc_moe, qz_moe)
        print(f"  Kernel fmt: qweight={qw_kern.shape}(i32) scales={sc_kern.shape} "
              f"qzeros={qz_kern.shape}(i32)")

        # Verify: kernel format should match reference exactly
        qw_match = torch.equal(qw_kern, qw_ref)
        sc_match = torch.equal(sc_kern, sc_ref)
        qz_match = torch.equal(qz_kern, qz_ref)
        print(f"  qweight match: {qw_match}")
        print(f"  scales  match: {sc_match}")
        print(f"  qzeros  match: {qz_match}")

        if not qw_match:
            diff = (qw_kern != qw_ref).sum().item()
            print(f"  !! qweight differs at {diff}/{qw_ref.numel()} positions")
        if not sc_match:
            diff = (sc_kern != sc_ref).sum().item()
            print(f"  !! scales differs at {diff}/{sc_ref.numel()} positions")
        if not qz_match:
            diff = (qz_kern != qz_ref).sum().item()
            print(f"  !! qzeros differs at {diff}/{qz_ref.numel()} positions")
            print(f"  ref qzeros unique: {qz_ref.unique()[:5]}")
            print(f"  kern qzeros unique: {qz_kern.unique()[:5]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
