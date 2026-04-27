"""Quick scan: does the 122B HF safetensors source contain NaN tensors?

Bug under investigation: 122B XFP TP=2 PACK reports many layers with
mse=nan cos=nan, including [64x3072] (likely linear-attention dt/A/b
projections) and [10240x3072], [8704x3072] (likely combined in_proj_qkvz).

Step 1 of bisection — is the source data clean?
"""
import os
import sys
import torch
from safetensors import safe_open

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/tensordata/Qwen3.5-122B-A10B"

shards = sorted(p for p in os.listdir(MODEL_DIR) if p.endswith(".safetensors"))
print(f"Scanning {len(shards)} safetensors shards in {MODEL_DIR}")

n_total = 0
n_nan = 0
n_inf = 0
n_zero = 0
nan_keys = []
zero_keys = []

target_shapes = {(64, 3072), (10240, 3072), (8704, 3072)}
print("Highlighting tensors with shapes in:", target_shapes)
print()

for fn in shards:
    path = os.path.join(MODEL_DIR, fn)
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                continue
            n_total += 1
            tf = t.float()
            has_nan = torch.isnan(tf).any().item()
            has_inf = torch.isinf(tf).any().item()
            all_zero = (tf == 0).all().item()
            shape = tuple(t.shape)
            interesting = shape in target_shapes or (
                len(shape) >= 2 and (shape[-2:] == (64, 3072)
                                     or shape[-2:] == (10240, 3072)
                                     or shape[-2:] == (8704, 3072))
            )
            if has_nan:
                n_nan += 1
                nan_keys.append((key, shape))
                print(f"  NaN  {key:<60} shape={shape}")
            elif has_inf:
                n_inf += 1
                print(f"  INF  {key:<60} shape={shape}")
            elif all_zero:
                n_zero += 1
                zero_keys.append((key, shape))
            elif interesting:
                norm = tf.norm().item()
                amax = tf.abs().max().item()
                print(f"  ok   {key:<60} shape={shape} norm={norm:.3g} amax={amax:.3g}")

print(f"\n=== summary: {n_total} float tensors | NaN={n_nan} | Inf={n_inf} | all_zero={n_zero} ===")
if zero_keys:
    print("\nAll-zero tensors:")
    for k, s in zero_keys[:30]:
        print(f"  {k:<60} shape={s}")
