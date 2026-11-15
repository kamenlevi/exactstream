"""Bytes Qwen3.5-2B (mlx-community 4-bit) reads per forward pass, from its safetensors headers (read-only): the text trunk
(every layer), the final norm, and the tied embedding used as the lm_head (read in full by as_linear). The input-embedding
lookup reads only the rows it needs, so it is not counted. Vision weights are excluded. MoonShots, Sun 15 Nov 2026.
usage: python3 -I bytes_2b.py"""
import glob, json, os, struct

import sys
M = sys.argv[1] if len(sys.argv) > 1 else "mlx-community--Qwen3.5-2B-MLX-4bit"       # any same-family repo in the HF cache
D = sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{M}/snapshots/*")))[0]
tot = {"layers": 0, "embed_as_lm_head": 0, "lm_head": 0, "norm_other": 0, "vision_or_mtp": 0}
for f in sorted(glob.glob(os.path.join(D, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name, v in hdr.items():
        if name == "__metadata__":
            continue
        b = v["data_offsets"][1] - v["data_offsets"][0]
        if "vision" in name or "visual" in name or "mtp" in name:
            tot["vision_or_mtp"] += b
        elif ".layers." in name:
            tot["layers"] += b
        elif "embed_tokens" in name:
            tot["embed_as_lm_head"] += b
        elif "lm_head" in name:
            tot["lm_head"] += b
        else:
            tot["norm_other"] += b
per_pass = tot["layers"] + tot["norm_other"] + (tot["lm_head"] or tot["embed_as_lm_head"])
print(json.dumps({k: round(v / 1e9, 4) for k, v in tot.items()}), "GB")
print(f"per forward pass (trunk + lm_head): {per_pass / 1e9:.3f} GB; trunk only: {(tot['layers'] + tot['norm_other']) / 1e9:.3f} GB")
