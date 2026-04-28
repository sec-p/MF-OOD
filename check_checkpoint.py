import torch
import os

ckpt_path = '/home/yeliu/huhuhu/data/ICML2026/GL_MCM_FA/logs/locoop-1-2.0_identity_1_20260129_151928/checkpoints/epoch_999.pt'
print(f"Checking {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"Type: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Keys: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: type={type(v)}")

print("\n" + "="*80)
ckpt_path2 = '/home/yeliu/huhuhu/data/ICML2026/GL_MCM_FA/logs/GL_MCM_FA_identity_1_20260127_041009/checkpoints/epoch_999.pt'
print(f"Checking {ckpt_path2}")
ckpt2 = torch.load(ckpt_path2, map_location='cpu')
print(f"Type: {type(ckpt2)}")
if isinstance(ckpt2, dict):
    print(f"Keys: {list(ckpt2.keys())}")
    for k, v in ckpt2.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: type={type(v)}")
