"""WEEK98 BET 7, after deliverable (3): where the fused level's ~0.29 ms per level between steps goes (MoonShots, Mon
16 Nov 2026; a diagnostic for NewIdeas' choice between a persistent kernel with weight prefetch and other levers).
Arms (interleaved, order alternating per round, the first 2 rounds dropped; L = 21 levels per block, async_eval per
level, one sync per block, as fl_time.py), at T0 = 400 keys, full head (177.6 MB):
  stock   the stock level (anchor: fl_time_quiet 0.960 ms)
  fused2  the fused level as timed in fl_time (anchor: 0.876 ms), with the cache maintenance (stock ops)
  chain   the same kernels, each fed by its predecessor as in fused2, WITHOUT the cache maintenance (T stays fixed)
  cut     the same steps in the same order, each fed the reference pass's inputs: no data dependency between steps
          (within a step, e.g. HC down -> up, the kernels stay dependent), so the encoder may overlap them
Read-out: fused2 - chain = the maintenance's cost; chain - cut = what the dependencies between steps cost, an upper
bound on what cross-step prefetch could recover (overlap also hides the latency-bound steps, which prefetch can't).
Checks: chain's level 0 equals fused2's level 0 bitwise; cut's outputs equal the reference pass's.
usage: ~/BigMoE/.venv/bin/python -I -B fl_overlap.py [--rounds 30] [--out <json>]   (MEM ~3 GB)"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402

import fl_attn                                              # noqa: E402
import fl_fused                                             # noqa: E402
import fl_hc                                                # noqa: E402
import fl_level                                             # noqa: E402
import fl_moe                                               # noqa: E402
import fl_stock as FS                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--L", type=int, default=21)
ap.add_argument("--T0", type=int, default=400)
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_overlap.json"))
A = ap.parse_args()
BW, MB = 292.7, 177.6


def say(s=""):
    print(s, flush=True)


t00 = time.time()
st = FS.Stock(nh=32768, T0=A.T0)
fu2 = fl_fused.Fused2(st)
mtp, L0 = st.mtp, st.mtp.layers[0]
attn, cache = L0.self_attn, st.cache[0]
T = int(cache.offset)
kb, vb = cache.kv.keys, cache.kv.values
K = 8


def kernels(hn, en_in, sc):
    """fused2's kernels for one level, chained (Fused2.level without the cache maintenance)"""
    hpre = fl_level.s1g(mtp, hn, en_in)
    mixed1, inj1, _, _ = fl_hc.hc_read(L0.attn_hyper_connection, hpre)
    qf, kf, vf, iqk = fl_level.s4_proj(attn, mixed1)
    gated, krot = fl_attn.fused_attn(attn, qf, kf, vf, kb[0], vb[0], T)
    bo1 = fl_level.s6_oproj(attn, gated)
    mixed2, inj2, hw2, _ = fl_hc.hc_read(L0.mlp_hyper_connection, hpre, pending=(bo1, inj1))
    bo2, _ = fl_moe.moe(L0.mlp, mixed2)
    x, _, hh, _ = fl_hc.hc_read(mtp.hyper_connection_mixer, hw2, pending=(bo2, inj2))
    lg = fl_level.s14_head(st.sh, x)
    kidx, klp, cand, sel, nsc, ntok, hn2, en2 = fl_level.s15p_tail(lg, sc, st.sh.ids, hh, mtp, st.emb, K)
    return dict(hpre=hpre, mixed1=mixed1, inj1=inj1, qf=qf, kf=kf, vf=vf, iqk=iqk, gated=gated, krot=krot, bo1=bo1,
                mixed2=mixed2, inj2=inj2, hw2=hw2, bo2=bo2, x=x, hh=hh, lg=lg, kidx=kidx, klp=klp, nsc=nsc, ntok=ntok,
                hn2=hn2, en2=en2)


# the reference pass: level 0 from h0 and token 1000
sc0 = mx.zeros((1,), mx.float32)
tok_ref = mx.array([[1000]], dtype=mx.int32)
hn0, en0 = fl_level.s0_prep(mtp, st.emb, st.h0, tok_ref)
R = kernels(hn0, en0, sc0)
mx.eval(hn0, en0, *R.values())


def cut_level():
    """the same steps in the same order, each fed the reference inputs (no dependency between steps)"""
    o = [fl_level.s1g(mtp, hn0, en0),
         fl_hc.hc_read(L0.attn_hyper_connection, R["hpre"])[0],
         fl_level.s4_proj(attn, R["mixed1"])[0],
         fl_attn.fused_attn(attn, R["qf"], R["kf"], R["vf"], kb[0], vb[0], T)[0],
         fl_level.s6_oproj(attn, R["gated"]),
         fl_hc.hc_read(L0.mlp_hyper_connection, R["hpre"], pending=(R["bo1"], R["inj1"]))[0],
         fl_moe.moe(L0.mlp, R["mixed2"])[0],
         fl_hc.hc_read(mtp.hyper_connection_mixer, R["hw2"], pending=(R["bo2"], R["inj2"]))[0],
         fl_level.s14_head(st.sh, R["x"])]
    t = fl_level.s15p_tail(R["lg"], sc0, st.sh.ids, R["hh"], mtp, st.emb, K)
    return o + [t[0], t[5]]


# checks
o0 = fu2.level(st.h0, tok_ref, sc0, 1, K, prep=None)
mx.eval(o0["tok"], o0["kidx"], o0["hh"])
st.cache[0].trim(1)
chk_chain = bool(mx.array_equal(o0["kidx"].reshape(-1), R["kidx"].reshape(-1)).item()) and \
    bool(mx.array_equal(o0["hh"].reshape(-1), R["hh"].reshape(-1)).item()) and \
    int(o0["tok"].reshape(-1)[0].item()) == int(R["ntok"].reshape(-1)[0].item())
c = cut_level()
mx.eval(c)
chk_cut = bool(mx.array_equal(c[0], R["hpre"]).item()) and bool(mx.array_equal(c[8], R["lg"]).item()) and \
    bool(mx.array_equal(c[9].reshape(-1), R["kidx"].reshape(-1)).item())
say(f"built at {time.time() - t00:.0f} s; T {T}; chain level 0 == fused2 level 0: {chk_chain}; cut outputs == "
    f"reference: {chk_cut}")


def run(arm, tok0):
    h, tok, sc = st.h0, mx.array([[tok0]], dtype=mx.int32), mx.zeros((1,), mx.float32)
    prep = None
    keep = []
    t0 = time.perf_counter()
    for _ in range(A.L):
        if arm == "stock":
            o = st.level(h, tok, sc, 1, K)
            h, tok, sc = o["h"], o["tok"], o["sc"]
            mx.async_eval(h, tok, sc, o["kidx"], o["klp"])
            keep.append(tok)
        elif arm == "fused2":
            o = fu2.level(h, tok, sc, 1, K, prep=prep)
            h, tok, sc, prep = o["h"], o["tok"], o["sc"], o["prep"]
            mx.async_eval(h, tok, sc, o["kidx"], o["klp"], prep[0], prep[1])
            keep.append(tok)
        elif arm == "chain":
            if prep is None:
                prep = fl_level.s0_prep(mtp, st.emb, h, tok)
            o = kernels(prep[0], prep[1], sc)
            prep, sc, tok = (o["hn2"], o["en2"]), o["nsc"], o["ntok"]
            mx.async_eval(o["hh"], tok, sc, o["kidx"], o["klp"], prep[0], prep[1])
            keep.append(tok)
        else:
            o = cut_level()
            mx.async_eval(o)
            keep.append(o[-1])
    mx.eval(keep)
    t1 = time.perf_counter()
    if arm in ("stock", "fused2"):
        st.cache[0].trim(A.L)
    return (t1 - t0) / A.L


ARMS = ["stock", "fused2", "chain", "cut"]
W = {a: [] for a in ARMS}
for r in range(A.rounds + 2):
    for a in (ARMS if r % 2 == 0 else ARMS[::-1]):
        w = run(a, 1000 + r)
        if r >= 2:
            W[a].append(1e3 * w)
res = {"rounds": A.rounds, "L": A.L, "T": T, "chk_chain": chk_chain, "chk_cut": chk_cut, "arms": {}}
say("ms per level, median [p10-p90] | f = ms - 177.6 MB / 292.7")
for a in ARMS:
    x = np.array(W[a])
    res["arms"][a] = dict(ms=round(float(np.median(x)), 4), p10=round(float(np.percentile(x, 10)), 4),
                          p90=round(float(np.percentile(x, 90)), 4), f=round(float(np.median(x)) - MB / BW, 4))
    v = res["arms"][a]
    say(f"  {a:7s} {v['ms']:.4f} [{v['p10']:.4f}-{v['p90']:.4f}] | f {v['f']:.4f}")
m = {a: res["arms"][a]["ms"] for a in ARMS}
say(f"maintenance (fused2 - chain) {m['fused2'] - m['chain']:.4f} ms; dependencies between steps (chain - cut) "
    f"{m['chain'] - m['cut']:.4f} ms per level")
json.dump(res, open(A.out, "w"), indent=1)
say(f"wrote {A.out} ({time.time() - t00:.0f} s)")
