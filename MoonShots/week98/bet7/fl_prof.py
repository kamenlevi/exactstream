"""WEEK98 BET 7: a per-step profile of the fused level (information; for the quiet slot beside fl_time.py).
Each step's kernels run N times back to back on the same inputs (async, one sync at the end): ms per call and its
weight bytes -> GB/s. Calls are independent, so latency-bound steps (route, tail, attention) may overlap and read low;
bandwidth-bound steps read close to their real cost. The sum is compared with the level's own ms.
usage: ~/BigMoE/.venv/bin/python -I -B fl_prof.py [--n 30] [--out <json>]   (MEM ~3 GB)"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx                                       # noqa: E402

import fl_attn                                              # noqa: E402
import fl_fused                                             # noqa: E402
import fl_hc                                                # noqa: E402
import fl_level                                             # noqa: E402
import fl_moe                                               # noqa: E402
import fl_stock as FS                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--T0", type=int, default=400)
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_prof.json"))
A = ap.parse_args()
st = FS.Stock(nh=32768, T0=A.T0)
fu = fl_fused.Fused(st)
mtp, L0 = st.mtp, st.mtp.layers[0]
attn, cache = L0.self_attn, st.cache[0]
T = int(cache.offset)
tok = mx.array([[1000]], dtype=mx.int32)
sc = mx.zeros((1,), mx.float32)
# one pass to get every step's real inputs
hpre = fl_level.s1_prep(mtp, st.emb, st.h0, tok)
mixed1, inj1, _, _ = fl_hc.hc_read(L0.attn_hyper_connection, hpre)
qf, kf, vf, iqk = fl_level.s4_proj(attn, mixed1)
kb, vb = cache.kv.keys, cache.kv.values
gated, krot = fl_attn.fused_attn(attn, qf, kf, vf, kb[0], vb[0], T)
bo1 = fl_level.s6_oproj(attn, gated)
mixed2, inj2, hw2, _ = fl_hc.hc_read(L0.mlp_hyper_connection, hpre, pending=(bo1, inj1))
bo2, it = fl_moe.moe(L0.mlp, mixed2)
x, _, hh, _ = fl_hc.hc_read(mtp.hyper_connection_mixer, hw2, pending=(bo2, inj2))
lg = fl_level.s14_head(st.sh, x)
hn0, en0 = fl_level.s0_prep(mtp, st.emb, st.h0, tok)
mx.eval(hpre, mixed1, inj1, qf, kf, vf, iqk, gated, krot, bo1, mixed2, inj2, hw2, bo2, it, x, hh, lg, hn0, en0)
MBY = {"S1 prep": 26.2, "S0 prep (fused2, first level)": 0.0, "S1G GEMVs (fused2)": 26.2,
       "S15P tail + next prep (fused2)": 0.0, "S2+S3 attn HC": 13.1, "S4 q/k/v/idx": 20.9, "S5 attention": 0.0, "S6 o_proj": 8.85,
       "S7+S8 MLP HC": 13.1, "S9-S11 MoE": 30.7 + 2.76 + 1.39, "S12+S13 mixer": 13.1, "S14 head": 47.2,
       "S15 tail": 0.0}
STEPS = {
    "S1 prep": lambda: fl_level.s1_prep(mtp, st.emb, st.h0, tok),
    "S0 prep (fused2, first level)": lambda: fl_level.s0_prep(mtp, st.emb, st.h0, tok)[0],
    "S1G GEMVs (fused2)": lambda: fl_level.s1g(mtp, hn0, en0),
    "S2+S3 attn HC": lambda: fl_hc.hc_read(L0.attn_hyper_connection, hpre)[0],
    "S4 q/k/v/idx": lambda: fl_level.s4_proj(attn, mixed1)[0],
    "S5 attention": lambda: fl_attn.fused_attn(attn, qf, kf, vf, kb[0], vb[0], T)[0],
    "S6 o_proj": lambda: fl_level.s6_oproj(attn, gated),
    "S7+S8 MLP HC": lambda: fl_hc.hc_read(L0.mlp_hyper_connection, hpre, pending=(bo1, inj1))[0],
    "S9-S11 MoE": lambda: fl_moe.moe(L0.mlp, mixed2)[0],
    "S12+S13 mixer": lambda: fl_hc.hc_read(mtp.hyper_connection_mixer, hw2, pending=(bo2, inj2))[0],
    "S14 head": lambda: fl_level.s14_head(st.sh, x),
    "S15 tail": lambda: fl_level.s15_tail(lg, sc, st.sh.ids, 8)[0],
    "S15P tail + next prep (fused2)": lambda: fl_level.s15p_tail(lg, sc, st.sh.ids, hh, mtp, st.emb, 8)[0],
}
res = {}
tot = 0.0
for name, fn in STEPS.items():
    for rep in range(2):
        outs = []
        t0 = time.perf_counter()
        for _ in range(A.n):
            o = fn()
            mx.async_eval(o)
            outs.append(o)
        mx.eval(outs)
        dt = (time.perf_counter() - t0) / A.n
    ms = 1e3 * dt
    gbs = MBY[name] / ms if MBY[name] else float("nan")
    res[name] = dict(ms=round(ms, 4), mb=MBY[name], gbs=round(gbs, 1))
    tot += ms
    print(f"  {name:16s} {ms:.4f} ms per call | {MBY[name]:6.2f} MB -> {gbs:6.1f} GB/s", flush=True)
f1 = sum(v["ms"] for k, v in res.items() if "fused2" not in k)
f2 = f1 - res["S1 prep"]["ms"] - res["S15 tail"]["ms"] + res["S1G GEMVs (fused2)"]["ms"] + res["S15P tail + next prep (fused2)"]["ms"]
print(f"  sum of steps: fused {f1:.4f} ms, fused2 {f2:.4f} ms (the level's bytes 177.6 MB at 292.7 GB/s = {177.6 / 292.7:.4f} ms)")
res["sum_ms_fused"] = round(f1, 4)
res["sum_ms_fused2"] = round(f2, 4)
json.dump(res, open(A.out, "w"), indent=1)
