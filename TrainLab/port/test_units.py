"""PORT unit tests (MTP Head, Wed 11-04 20:4x), CPU, small:
1. attention: "sdpa" (one F.scaled_dot_product_attention over [context; blocks] + block-diagonal mask, the CUDA path) vs
   "explicit" (the MLX trainer's layout) on the synthetic model; and explicit in anchor chunks of 3 vs all at once;
2. shards: L1-20's 109 npz windows -> port/pack_shards.py -> memory-mapped windows; every train batch (A 64, bs 32,
   pmask 0.5 with the trainer's RNG) must be bitwise identical to the npz batch.
Usage: [TrainLab .venv-torch] python -I test_units.py SCRATCH_DIR"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import dflash2_torch as M
import data_port as DP

tmp = os.path.expanduser(sys.argv[1]); os.makedirs(tmp, exist_ok=True)
torch.set_num_threads(4)
ok = True

# 1. attention paths
m = M.build_synthetic(layers=2)
wins = DP.synthetic_windows()
tr = [g for w_ in wins if w_["split"] == "train" for g in DP.groups_of(w_, 8)]
b = DP.to_torch(DP.make_batch(tr[:2], 8, 8, m.MASK), "cpu")
with torch.no_grad():
    ref = m.fwd(b["hc"], b["upos"], b["anc"], b["tok"])
    m.attn_chunk = 3
    chk = m.fwd(b["hc"], b["upos"], b["anc"], b["tok"])
    m.attn_chunk = 0; m.attn_impl = "sdpa"
    sdp = m.fwd(b["hc"], b["upos"], b["anc"], b["tok"])
d_chunk, d_sdpa = float((chk - ref).abs().max()), float((sdp - ref).abs().max())
ok &= d_chunk < 1e-5 and d_sdpa < 1e-4
print(f"1. attention: anchor-chunked vs whole max|diff| {d_chunk:.2e}; sdpa vs explicit max|diff| {d_sdpa:.2e} "
      f"(hidden max|x| {float(ref.abs().max()):.2f})")

# 2. shards
sh = os.path.join(tmp, "l120_shards")
subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pack_shards.py"), sh,
                os.path.expanduser("~/Leap/data/l120"), "--shard-gb", "0.05"], check=True)
wa, wb = DP.load_windows("~/Leap/data/l120"), DP.load_windows(sh)
ta = [g for w_ in wa if w_["split"] == "train" for g in DP.groups_of(w_, 64)]
tb = [g for w_ in wb if w_["split"] == "train" for g in DP.groups_of(w_, 64)]
ra, rb = np.random.default_rng(20), np.random.default_rng(20)
same = len(ta) == len(tb)
for ga, gb in zip(ta, tb):
    xa, xb = DP.make_batch([ga], 64, 32, 248077, 0.5, ra), DP.make_batch([gb], 64, 32, 248077, 0.5, rb)
    same &= all(np.array_equal(xa[k], xb[k]) for k in xa)
ok &= same
print(f"2. shards: {len(wb)} windows from {len(set(e['shard'] for e in __import__('json').load(open(os.path.join(sh, 'index.json')))['windows']))} "
      f"shards; {len(tb)} train batches bitwise identical to the npz path: {same}")
print("UNITS:", "PASS" if ok else "FAIL")
