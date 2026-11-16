"""WEEK98 MODULE 2 (MoonShots): bit test + per-kernel GPU time of candidate forms (m2_cand.py) against ooo1cq on exp32's
real blocks (m2_common.py's setup). Per block: the reference call (ref2._ooo_sel -> ooo1cq, captured), then the same
stages with the candidate's kernel A and B: h, y and the combined output must equal ooo1cq's bit for bit. Timing (GPU us
per kernel, libmtlprof, median of --reps; ref and candidate interleaved) is indicative unless run in a quiet slot.
usage: ~/BigMoE/.venv/bin/python -I -B m2_bits.py [--forms C1] [--blocks 24] [--reps 5]   (MEM ~3 GB)"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--forms", default="C1")
ap.add_argument("--blocks", type=int, default=24)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/m2/m2_bits"))
A_ = ap.parse_args()
os.environ["M2_BLOCKS"] = str(A_.blocks)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_common import *                                  # noqa: E402,F403
from m2_common import L_, say                            # noqa: E402
import m2_cand as MC                                     # noqa: E402

FORMS = A_.forms.split(",")


def gy(form, nseg):
    """grid y: P (= nseg here, the pair count) or, for the capped forms, min(P, M2_GY)"""
    return min(nseg, MC.GY_CAP) if form in MC.CAPPED else nseg


def gyt(form, nseg):
    return [("GY_C", gy(form, nseg))] if form in MC.CAPPED else []


def run_a(st, form, grp, xs):
    pidx, seg, nsg_, nseg = grp
    g, u = st.msb_n["gate_proj"], st.msb_n["up_proj"]
    L = st.lsb_n
    c3 = form in MC.C3SET
    return MC.ka(form)(
        inputs=[st.x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], g["b3"], u["b3"],
                L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], st.ms, st.ls,
                pidx, seg, nsg_, st.region, st.offs, st.status],
        template=st.tmpl + [("K_C", st.Kd), ("N_C", st.N), ("KS_C", st.ks)] + gyt(form, nseg)
        + ([("TPT_C", MC.TPT)] if c3 else []),
        grid=(st.N // (32 * (MC.TPT if c3 else 1)) * 64 * st.ks, gy(form, nseg), 1), threadgroup=(64 * st.ks, 1, 1),
        output_shapes=[(st.M * st.R, st.N), (st.M * st.R,), (max(st.M * st.r4, 1),)]
        + ([(st.M * st.R, st.N // 64)] if c3 else []),
        output_dtypes=[st.x.dtype, mx.int32, mx.int32] + ([mx.float32] if c3 else []))


def run_b(st, form, grp, h, hs, rs_m, rs_l):
    """kernel B's y; for the C4 forms (y, out): B also writes the combined rows"""
    pidx, seg, nsg_, nseg = grp
    d = st.msb_n["down_proj"]
    L = st.lsb_n
    c4 = form in MC.C4SET
    extra = [mx.zeros((st.M * (st.D // 32),), mx.int32), st.comb_a[3].astype(st.x.dtype).reshape(-1)] if c4 else []
    outs = MC.kb(form)(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], d["b3"], L["down_proj"]["l1"], L["down_proj"]["b"], rs_m, rs_l,
                pidx, seg, nsg_] + extra,
        template=st.tmpl + [("N_C", st.N), ("D_C", st.D), ("NSG_C", st.nsg_b), ("KS_C", st.ks_b)] + gyt(form, nseg)
        + ([("TPB_C", MC.TPB)] if form in MC.C3BSET else []) + ([("LY_C", 8)] if c4 else []),
        grid=(st.D // (32 * st.nsg_b * (MC.TPB if form in MC.C3BSET else 1)) * 32 * st.nsg_b * st.ks_b, gy(form, nseg), 1),
        threadgroup=(32 * st.nsg_b * st.ks_b, 1, 1),
        output_shapes=[(st.M * st.R, st.D)] + ([(st.M, st.D)] if c4 else []),
        output_dtypes=[st.x.dtype] + ([st.x.dtype] if c4 else []))
    return (outs[0], outs[1]) if c4 else outs[0]


def whole(st, f):
    """the whole one-call + combine with the candidate's kernels, in one graph (one command buffer)"""
    grp = st.grp()
    xs = st.xs()
    oa = run_a(st, f, grp, xs)
    h, rs_m, rs_l = oa[:3]
    hs = oa[3] if f in MC.C3SET else st.hs(h)
    ob = run_b(st, f, grp, h, hs, rs_m, rs_l)
    return ob[1] if f in MC.C4SET else st.comb(ob)


def exp32_way(form, rounds=7):
    """Leap's exp32_bench timing (NB calls in one graph through ref2, wall per call; median of rounds 2..) with ref2's
    f1c swapped for the candidate's (form None: ooo1cq); also the GPU busy per call"""
    F0 = OQ.tier_glu_nax6n_1cq
    if form is not None:
        OQ.tier_glu_nax6n_1cq = MC.tier_glu_1cq(form)
    try:
        ts, bz = [], []
        for _ in range(rounds):
            mx.synchronize()
            c0, t0 = MP.counts(), MP.now()
            ys = [call(b_) for b_ in B32[:NB]]
            mx.eval(ys)
            mx.synchronize()
            t1, c1 = MP.now(), MP.counts()
            ts.append(t1 - t0)
            bz.append(MP.busy(MP.cbs(c0["ncb"], c1["ncb"]), t0, t1))
        return 1e6 * float(np.median(ts[1:])) / NB, 1e6 * float(np.median(bz[1:])) / NB
    finally:
        OQ.tier_glu_nax6n_1cq = F0


STS = []


def graph_way(form, rounds=7):
    """NB whole calls (grp .. combine) on the captured inputs in one graph; wall per call (median of rounds 2..) and GPU
    busy per call; form None: ooo1cq's kernels + combine_phase"""
    ts, bz = [], []
    for _ in range(rounds):
        mx.synchronize()
        c0, t0 = MP.counts(), MP.now()
        ys = [(s_.whole() if form is None else whole(s_, form)) for s_ in STS]
        mx.eval(ys)
        mx.synchronize()
        t1, c1 = MP.now(), MP.counts()
        ts.append(t1 - t0)
        bz.append(MP.busy(MP.cbs(c0["ncb"], c1["ncb"]), t0, t1))
    return 1e6 * float(np.median(ts[1:])) / len(STS), 1e6 * float(np.median(bz[1:])) / len(STS)


for blk in B32[:2]:
    mx.eval(call(blk)); mx.synchronize()
ROWS = []
t_run = time.time()
eqall = {f: True for f in FORMS}
for bi in range(NB):
    CAP.clear()
    yref = call(B32[bi])
    mx.eval(yref)
    st = Staged(*CAP["oq"], CAP["comb"])
    STS.append(st)
    grp0, _, _, _ = gpu(st.grp)
    xs0, _, _, _ = gpu(st.xs)
    (h0, rsm0, rsl0), _, _, _ = gpu(lambda: st.ka(grp0, xs0))
    hs0, _, _, _ = gpu(lambda: st.hs(h0))
    y0, _, _, _ = gpu(lambda: st.kb(grp0, h0, hs0, rsm0, rsl0))
    row = dict(block=bi)
    for f in FORMS:
        oa = run_a(st, f, grp0, xs0)
        hc, rmc, rlc = oa[:3]
        hsc = oa[3] if f in MC.C3SET else st.hs(hc)
        ob = run_b(st, f, grp0, hc, hsc, rmc, rlc)
        yc, oc = ob if f in MC.C4SET else (ob, st.comb(ob))
        mx.eval(hc, rmc, rlc, yc, oc)
        e = dict(h=bool(mx.array_equal(hc, h0).item()), hs=bool(mx.array_equal(hsc, hs0).item()),
                 rs_m=bool(mx.array_equal(rmc, rsm0).item()),
                 rs_l=bool(mx.array_equal(rlc, rsl0).item()), y=bool(mx.array_equal(yc, y0).item()),
                 out=bool(mx.array_equal(oc.reshape(-1), yref.reshape(-1)).item()))
        row[f"eq_{f}"] = e
        eqall[f] = eqall[f] and all(e.values())
    T_ = {k: [] for k in ["A_ref", "B_ref", "W_ref"] + [f"{p}_{f}" for f in FORMS for p in "ABW"]}
    for rep in range(A_.reps):
        T_["A_ref"].append(gpu(lambda: st.ka(grp0, xs0))[1])
        T_["B_ref"].append(gpu(lambda: st.kb(grp0, h0, hs0, rsm0, rsl0))[1])
        T_["W_ref"].append(gpu(st.whole)[1])
        for f in FORMS:
            T_[f"A_{f}"].append(gpu(lambda: run_a(st, f, grp0, xs0))[1])
            T_[f"B_{f}"].append(gpu(lambda: run_b(st, f, grp0, h0, hs0, rsm0, rsl0))[1])
            T_[f"W_{f}"].append(gpu(lambda: whole(st, f))[1])
    med = {k: round(float(np.median(v)), 2) for k, v in T_.items()}
    row.update({f"us_{k}": v for k, v in med.items()})
    ROWS.append(row)
    say(f"  block {bi:2d}: " + "; ".join(f"{f} bits {'ALL ==' if all(row[f'eq_{f}'].values()) else row[f'eq_{f}']}"
                                         for f in FORMS)
        + " | GPU us A " + ", ".join(f"{k[2:]} {med[k]:.1f}" for k in med if k.startswith("A_"))
        + " | B " + ", ".join(f"{k[2:]} {med[k]:.1f}" for k in med if k.startswith("B_"))
        + " | whole call " + ", ".join(f"{k[2:]} {med[k]:.1f}" for k in med if k.startswith("W_")))
col = lambda k: np.array([r_[k] for r_ in ROWS], float)
say(f"\nM2 BITS {time.strftime('%a %d %b %H:%M:%S')}, {NB} blocks: " + "; ".join(
    f"{f} == ooo1cq bit for bit (h, rs_m, rs_l, y, combined): {eqall[f]}" for f in FORMS))
say("  mean GPU us: A ref {:.1f}, B ref {:.1f}; ".format(np.mean(col("us_A_ref")), np.mean(col("us_B_ref"))) + "; ".join(
    f"{f}: A {np.mean(col(f'us_A_{f}')):.1f}, B {np.mean(col(f'us_B_{f}')):.1f}" for f in FORMS)
    + f"  (indicative unless quiet; {time.time() - t_run:.0f} s)")
say("  whole call in one command buffer, mean GPU us: ref {:.1f}; ".format(np.mean(col("us_W_ref"))) + "; ".join(
    f"{f} {np.mean(col(f'us_W_{f}')):.1f}" for f in FORMS))
E32 = {}
for rnd in range(2):                                     # exp32's way, interleaved: ref, forms, ref, forms
    for f in [None] + [f_ for f_ in FORMS if f_ not in MC.C4SET]:
        E32.setdefault(f or "ref", []).append(exp32_way(f))
say(f"  exp32's way ({NB} calls per graph, us per call, wall / GPU busy; two interleaved passes): " + "; ".join(
    f"{k} " + " & ".join(f"{w:.1f} / {g:.1f}" for w, g in v) for k, v in E32.items()))
GW = {}
for rnd in range(2):                                     # graph way: NB blocks' whole calls (captured inputs) in one graph
    for f in [None] + FORMS:
        GW.setdefault(f or "ref", []).append(graph_way(f))
say(f"  graph way ({NB} whole calls per graph on the captured inputs, no ref2 prep ops; us per call, wall / GPU busy; "
    f"two interleaved passes): " + "; ".join(f"{k} " + " & ".join(f"{w:.1f} / {g:.1f}" for w, g in v) for k, v in GW.items()))
open(A_.out + ".txt", "a").write("\n".join(L_) + "\n")
json.dump(dict(rows=ROWS, eq=eqall, exp32=E32, graph=GW), open(A_.out + ".json", "w"), indent=1)
