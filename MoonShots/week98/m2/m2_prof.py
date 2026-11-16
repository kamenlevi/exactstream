"""WEEK98 MODULE 2 (Leap's split 11-16, ~/Leap/data/week98/check25/SPLIT_1116.md "Module 2"): where the lossy expert
one-call's time goes, stage by stage (MoonShots, Mon 16 Nov 2026). Leap's first question: fixed cost or rate?
Setup = a copy of Leap's exp32_bench.py (layer 20's converted pools via BigMoE's loader, all 512 resident; the decision
map's real 1c-tiered code windows in 32-row blocks; km path; 2-D nolsb), BigMoE / Leap modules imported read-only (-I -B).
Per block:
  - the reference call (ref2._ooo_sel as exp32 runs it, COH off) with ooo1cq.tier_glu_nax6n_1cq and tierglu.combine_phase
    captured, so every stage below replays on the call's own inputs;
  - each stage in its own command buffer (libmtlprof: GPU start / end): grp, rowsum64(x), kernel A, rowsum64(h),
    kernel B, combine; the whole call (grp .. combine) in one command buffer; and exp32's way (NB calls in one graph);
  - kernel A / B with the grid's y cut to the block's real segment count (read on the host; diagnosis only) and with
    nseg forced to 0 (every threadgroup returns at once): what the P = 320-row grid's empty threadgroups cost;
  - bytes: per segment its expert's pieces (A: gate + up = 2/3 of the MSB piece, + 2/3 of the LSB piece at tier 4;
    B: 1/3 each), summed over segments (what the kernels stream) and over distinct experts (the DRAM floor).
Fit over the blocks, per stage and for the call: GPU us = fixed + MB / rate (least squares, with an intercept).
Outputs ~/MoonShots/week98/m2/m2_prof.{txt,json}. Indicative unless run in a quiet slot (MEM ~3 GB, share lock).
usage: ~/BigMoE/.venv/bin/python -I -B m2_prof.py [--blocks 24] [--reps 9]"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", type=int, default=24)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/m2/m2_prof"))
A_ = ap.parse_args()
os.environ["M2_BLOCKS"] = str(A_.blocks)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_common import *                                  # noqa: E402,F403  (libmtlprof before mlx; the setup runs here)
from m2_common import L_, say                            # noqa: E402

# ---- warm: build every pipeline once ----
for blk in B32[:2]:
    mx.eval(call(blk)); mx.synchronize()
STAGES = ["grp", "xs", "A", "hs", "B", "comb"]
ROWS = []
t_run = time.time()
for bi in range(NB):
    CAP.clear()
    yref = call(B32[bi])
    mx.eval(yref)
    st = Staged(*CAP["oq"], CAP["comb"])
    grp0, _, _, _ = gpu(st.grp)
    xs0, _, _, _ = gpu(st.xs)
    (h0, rsm0, rsl0), _, _, _ = gpu(lambda: st.ka(grp0, xs0))
    hs0, _, _, _ = gpu(lambda: st.hs(h0))
    y0, _, _, _ = gpu(lambda: st.kb(grp0, h0, hs0, rsm0, rsl0))
    out0, _, _, _ = gpu(lambda: st.comb(y0))
    eq = bool(mx.array_equal(out0.reshape(-1), yref.reshape(-1)).item())   # the staged replay == the reference call
    sb = seg_bytes(st, grp0)
    ny = max(1, sb["nseg"])
    z0 = mx.zeros((1,), mx.int32)
    R_ = {k: [] for k in STAGES + ["whole", "A_ycut", "B_ycut", "A_empty", "B_empty"]}
    for rep in range(A_.reps):
        R_["grp"].append(gpu(st.grp)[1])
        R_["xs"].append(gpu(st.xs)[1])
        R_["A"].append(gpu(lambda: st.ka(grp0, xs0))[1])
        R_["hs"].append(gpu(lambda: st.hs(h0))[1])
        R_["B"].append(gpu(lambda: st.kb(grp0, h0, hs0, rsm0, rsl0))[1])
        R_["comb"].append(gpu(lambda: st.comb(y0))[1])
        R_["whole"].append(gpu(st.whole)[1])
        R_["A_ycut"].append(gpu(lambda: st.ka(grp0, xs0, ny=ny))[1])
        R_["B_ycut"].append(gpu(lambda: st.kb(grp0, h0, hs0, rsm0, rsl0, ny=ny))[1])
        R_["A_empty"].append(gpu(lambda: st.ka(grp0, xs0, nseg_over=z0))[1])
        R_["B_empty"].append(gpu(lambda: st.kb(grp0, h0, hs0, rsm0, rsl0, nseg_over=z0))[1])
    med = {k: float(np.median(v)) for k, v in R_.items()}
    ROWS.append(dict(block=bi, eq_ref=eq, **sb, **{f"us_{k}": round(v, 2) for k, v in med.items()}))
    say(f"  block {bi:2d}: staged == ref {eq}; {sb['nseg']} segments ({sb['live']} live), {sb['n_exp']} experts, "
        f"{sb['n_lsb']} LSB; MB grp {sb['grp_mb']:.1f} / uniq {sb['uniq_mb']:.1f} | GPU us: "
        + ", ".join(f"{k} {med[k]:.1f}" for k in STAGES) + f" | sum {sum(med[k] for k in STAGES):.1f}, whole "
        f"{med['whole']:.1f} | A ycut {med['A_ycut']:.1f} empty {med['A_empty']:.1f}; B ycut {med['B_ycut']:.1f} "
        f"empty {med['B_empty']:.1f}")

# exp32's way on the same blocks (NB calls in one graph, wall per call) and its GPU busy
DISP = {}


def exp32_way(rounds=7):
    ts, bz = [], []
    for _ in range(rounds):
        mx.synchronize()
        c0, t0, p0 = MP.counts(), MP.now(), MP.psos()
        ys = [call(b) for b in B32[:NB]]
        mx.eval(ys)
        mx.synchronize()
        t1, c1 = MP.now(), MP.counts()
        ts.append(t1 - t0)
        bz.append(MP.busy(MP.cbs(c0["ncb"], c1["ncb"]), t0, t1))
        DISP.update(disp=(c1["disp"] - c0["disp"]) / NB, bar=(c1["bar"] - c0["bar"]) / NB, ncb=(c1["ncb"] - c0["ncb"]) / NB,
                    kernels={k[:70]: round(v / NB, 2) for k, v in MP.pso_delta(p0, MP.psos()).items()})
    return 1e6 * float(np.median(ts[1:])) / NB, 1e6 * float(np.median(bz[1:])) / NB


e32_wall, e32_busy = exp32_way()


def fit(xmb, yus):
    X = np.stack([np.ones_like(xmb), xmb], 1)
    (c0, c1), res, _, _ = np.linalg.lstsq(X, yus, rcond=None)
    pred = X @ np.array([c0, c1])
    ss = float(np.sum((yus - pred) ** 2))
    r2 = 1 - ss / float(np.sum((yus - yus.mean()) ** 2)) if len(yus) > 2 else float("nan")
    return dict(fixed_us=round(float(c0), 1), us_per_mb=round(float(c1), 3),
                rate_gbs=round(1e3 / float(c1), 1) if c1 > 0 else None, r2=round(r2, 3))


col = lambda k: np.array([r_[k] for r_ in ROWS], float)
FITS = {"A vs A MB": fit(col("a_mb"), col("us_A")), "B vs B MB": fit(col("b_mb"), col("us_B")),
        "whole vs grp MB": fit(col("grp_mb"), col("us_whole")), "whole vs uniq MB": fit(col("uniq_mb"), col("us_whole"))}
S = {k: float(np.mean(col(f"us_{k}"))) for k in STAGES + ["whole", "A_ycut", "B_ycut", "A_empty", "B_empty"]}
say(f"\nmeans over {NB} blocks (GPU us; each stage alone in its own command buffer): "
    + ", ".join(f"{k} {S[k]:.1f}" for k in STAGES) + f"; sum {sum(S[k] for k in STAGES):.1f}; whole call in one "
    f"command buffer {S['whole']:.1f}")
say(f"  MB per call: grp {np.mean(col('grp_mb')):.1f} (A {np.mean(col('a_mb')):.1f}, B {np.mean(col('b_mb')):.1f}), "
    f"distinct {np.mean(col('uniq_mb')):.1f}; segments {np.mean(col('nseg')):.1f} of P = 320 grid rows")
say(f"  rates: A {np.mean(col('a_mb')) / S['A'] * 1e3:.1f} GB/s, B {np.mean(col('b_mb')) / S['B'] * 1e3:.1f}, A + B "
    f"{np.mean(col('grp_mb')) / (S['A'] + S['B']) * 1e3:.1f} (segment bytes); whole call "
    f"{np.mean(col('uniq_mb')) / S['whole'] * 1e3:.1f} GB/s on distinct bytes")
say(f"  empty threadgroups: A with y cut to the segments {S['A_ycut']:.1f} (vs {S['A']:.1f}), all returning at once "
    f"{S['A_empty']:.1f}; B {S['B_ycut']:.1f} (vs {S['B']:.1f}), empty {S['B_empty']:.1f}")
say(f"  exp32's way on these blocks: {e32_wall:.1f} us per call wall, GPU busy {e32_busy:.1f} us per call; per call "
    f"{DISP.get('disp', 0):.1f} dispatches, {DISP.get('bar', 0):.1f} barriers, {DISP.get('ncb', 0):.2f} command buffers; "
    f"kernels per call: " + "; ".join(f"{k} {v}" for k, v in sorted(DISP.get("kernels", {}).items(), key=lambda kv: -kv[1])))
for k, v in FITS.items():
    say(f"  fit {k}: fixed {v['fixed_us']} us + {v['us_per_mb']} us/MB (rate {v['rate_gbs']} GB/s), R2 {v['r2']}")
say(f"all staged == ref: {all(r_['eq_ref'] for r_ in ROWS)}; peak {mx.get_peak_memory() / 1e9:.2f} GB; "
    f"{time.time() - t_run:.0f} s")
os.makedirs(os.path.dirname(A_.out), exist_ok=True)
open(A_.out + ".txt", "a").write("\n".join(L_) + "\n")
json.dump(dict(rows=ROWS, means=S, fits=FITS, exp32_wall_us=e32_wall, exp32_busy_us=e32_busy, exp32_disp=DISP), open(A_.out + ".json", "w"),
          indent=1)
