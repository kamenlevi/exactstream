"""SDPA row-band probe (LONG1): every row i of a verify-shaped call (S query rows, keys [0, T), causal tail mask, MTPLX's
_verify_sdpa = head chunking for S*GQA > 32) vs the decode call for that token (one row, keys [0, T-S+i], no mask),
bitwise, at T keys; 125B attention shapes. Prints, per T and S, which rows differ.
usage: sdpa_band.py H HKV D [T,T,...]   (small GPU work)"""
import os, sys
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mtplx.models.qwen4_exp as Q
H, HKV, D = (int(a) for a in sys.argv[1:4])
mx.random.seed(0)
TS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [256, 512, 900, 1000, 1023, 1024, 1025, 1100, 1500, 2048, 4096]
for T in TS:
    k = mx.random.normal((1, HKV, T, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, T, D)).astype(mx.bfloat16)
    qa = mx.random.normal((1, H, 8, D)).astype(mx.bfloat16)
    cells = []
    for S in (2, 3, 4, 5, 8):
        q = qa[:, :, -S:]
        qpos = (T - S) + mx.arange(S)
        mask = (mx.arange(T)[None, :] <= qpos[:, None])[None, None]
        o = Q._verify_sdpa(q, k, v, scale=D ** -0.5, mask=mask)
        bad = []
        for i in range(S):
            n = T - S + i + 1
            ref = Q._verify_sdpa(q[:, :, i:i + 1], k[:, :, :n], v[:, :, :n], scale=D ** -0.5, mask=None)
            if not mx.array_equal(o[:, :, i:i + 1], ref).item():
                bad.append(i)
        cells.append(f"S={S}:{'ok' if not bad else 'rows' + ''.join(map(str, bad))}")
    print(f"T={T:5d}  " + "  ".join(cells), flush=True)
