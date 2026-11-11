"""LOSSY350-FAR, MISS-DEFER replay (Fable FAR, Thu 2026-11-05 16:1x; CPU, numpy, one core, < 1 GB, ~1 min).
The rule broken: "a check waits for the right expert". Never wait: a missed expert's read is issued when the router names it
and the check moves on; its term w_r E_r(x_L) is computed on the SAVED layer-L input and added to the residual stream at the
first layer boundary after the read lands (d >= 1 layers late; KTransformers' expert deferral, keyed on cache misses).
Frame: far_frame.py (M-38's chained cap45 replay: six prompts, 16-row EXACT-loop checks at C 160, per-layer LRU of K
slots, counted on prompts 2-6). Policies:
  EXACT     every row's top-10 is loaded on demand (the check stalls on each layer with a miss)
  QOM       today's FAST preset, simplified (qom_keep.py: THR 0, KS 2, bonus row gated): ranks 0-1 of every row are demand
            loads; a rank 2-9 term is computed only if its expert is resident or among the check's loads, else DROPPED
  rows: all 16 as captured, or "live" = the first min(16, kept + 2) rows (a stand-in for the engine's running-cf cut).
Timeline of a zero-wait check: layer l's reads are issued at l * TAU; one SSD stream serves batches FIFO, a batch of m
experts takes A0 + B * m ms (FAR.md 7b's law per stalled layer: A0 = 12 ms floor / ~45 stalled layers, B = 0.155 ms per
miss); the terms land d = ceil((finish - issue) / TAU) layers late; a term with l + d > 47 cannot be added (tail).
Mass = the router's renormalised weight by rank (Fable C / cloud3 P2 sec 4, domain-independent), 1.0 per row-layer.
Effective mass of a deferred term [G model]: rho = min(1, d / (47 - l)): the share of the remaining blocks that ran without
it (all of its effect taken as indirect and spread evenly over the later blocks: conservative, the direct path is kept).
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_defer.py -> far_defer.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, math
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("far_frame", os.path.expanduser("~/FableGod/far/far_frame.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
P, NL, E = F.P, F.NL, F.E
OUT = os.path.expanduser("~/FableGod/far/far_defer.txt")
W = np.array([0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010])
W = W / W.sum()
A0, B = 0.27, 0.155
t0 = time.time()
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def rank_check():
    """are cap45's ids rank-sorted? recompute one layer's router logits from the captured x"""
    try:
        z = np.load(os.path.join(F.CAP, F.ORDER[0] + "_gate_0.npz"))
        r = np.load(os.path.join(F.CAP, "routers.npz"))
        keys = sorted(r.keys())
        seq, L, rows, roff, xoff = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"]
        ok = tot = 0
        for n in range(len(seq)):
            l = int(L[n])
            if int(rows[n]) != 16 or l not in (5, 20, 40):
                continue
            k = [q for q in keys if q == f"w{l}"]
            if not k:
                continue
            R = r[k[0]].astype(np.float32)
            x = (z["x"][xoff[n]:xoff[n] + 16].astype(np.uint32) << 16).view(np.float32)
            lg = x @ (R.T if R.shape[1] == x.shape[1] else R)
            ids = z["ids"][roff[n]:roff[n] + 16].astype(np.int64)
            g = np.take_along_axis(lg, ids, 1)
            ok += int((np.diff(g, axis=1) <= 1e-3).all(1).sum()); tot += 16
            if tot >= 16 * 30:
                break
        return f"router logits of the captured ids are non-increasing in {ok} / {tot} rows (keys e.g. {keys[:2]})"
    except Exception as e:
        return f"rank check failed: {e!r}"


def run(K, pol, live, tau):
    res = np.zeros((NL, E), bool); last = np.zeros((NL, E))
    A = {"checks": 0, "kept": 0, "rows": 0, "miss": 0, "stl": 0, "block": 0.0, "drop": 0.0, "dmass": 0.0, "eff": 0.0, "eff2": 0.0,
         "tail": 0.0, "dh": 0.0, "d1": 0.0, "d2": 0.0, "d3": 0.0, "terms": 0, "dterms": 0, "ssd_busy": 0.0, "m01": 0.0, "span": 0.0}
    i = 0
    for p, (_, cks, res0) in enumerate(P):
        for t in range(-1, len(cks)):
            i += 1
            cnt = p >= 1 and t >= 0
            if t < 0:
                for l in range(NL):
                    need = res0[l]
                    res[l, need] = True; last[l, need] = i
                    over = int(res[l].sum()) - K
                    if over > 0:
                        cand = np.nonzero(res[l])[0]; cand = cand[~np.isin(cand, need)]
                        res[l, cand[np.argsort(last[l, cand], kind="stable")[:over]]] = False
                continue
            c = cks[t]
            nr = min(16, c["kept"] + 2) if live else 16
            ssd_free = 0.0
            if cnt:
                A["checks"] += 1; A["kept"] += c["kept"]; A["rows"] += nr
            for l in range(NL):
                rw = c["rows"][l][:nr]
                if pol == "EXACT":
                    need = np.unique(rw)
                    miss = need[~res[l, need]]
                    used = need
                    dropm = 0.0
                else:
                    d01 = np.unique(rw[:, :2])
                    miss = d01[~res[l, d01]]
                    avail = res[l].copy(); avail[miss] = True
                    ok = avail[rw]
                    dropm = float((W[None, :] * ~ok).sum())
                    used = np.unique(rw[ok])
                mm = np.zeros(E, bool); mm[miss] = True
                late = mm[rw]                                        # the terms whose expert is a demand miss
                dm = float((W[None, :] * late).sum())
                m = len(miss)
                if cnt:
                    A["miss"] += m; A["stl"] += m > 0; A["drop"] += dropm; A["terms"] += rw.size; A["dterms"] += int(late.sum())
                    A["m01"] += float((W[None, :2] * late[:, :2]).sum())
                    if m:
                        A["block"] += A0 + B * m
                        ti = l * tau
                        st = max(ti, ssd_free); fin = st + A0 + B * m; ssd_free = fin
                        A["ssd_busy"] += A0 + B * m
                        d = max(1, math.ceil((fin - ti) / tau))
                        if fin - ti <= tau / 2 and l < NL - 1:
                            A["dh"] += dm                            # lands within half a layer (B12LL's lateness)
                        A["dmass"] += dm
                        if l + d > NL - 1:
                            A["tail"] += dm; A["eff"] += dm; A["eff2"] += dm
                        else:
                            rho = min(1.0, d / (NL - 1 - l))
                            A["eff"] += dm * rho; A["eff2"] += dm * min(1.0, 2 * rho)
                            A["d1" if d == 1 else "d2" if d == 2 else "d3"] += dm
                last[l, used] = i
                res[l, miss] = True
                over = int(res[l].sum()) - K
                if over > 0:
                    nm = np.zeros(E, bool); nm[used] = True
                    cand = np.nonzero(res[l] & ~nm)[0]
                    if cand.size < over:
                        cand = np.nonzero(res[l])[0]
                    res[l, cand[np.argsort(last[l, cand], kind="stable")[:over]]] = False
            if cnt:
                A["span"] += max(0.0, ssd_free - NL * tau)
    return A


say(f"FAR DEFER ({time.strftime('%a %Y-%m-%d %H:%M')}): the never-wait check on M-38's cap45 frame (prompts 2-6); A0 {A0} ms per read batch, "
    f"B {B} ms per miss; mass = renormalised router weight by rank, 1.0 per row-layer")
say("  " + rank_check())
say("per K, policy, rows, layer time TAU: misses per check (per kept token) | stalled layers | blocking wait today's way (model, ms per check) | "
    "QOM's DROPPED mass per row-layer | late terms: share of terms, mass per row-layer (ranks 0-1 part) | lands 1 / 2 / 3+ layers late, tail "
    "(cannot land); half = the share landing within half a layer, B12LL's lateness (it is inside d1) | effective mass per row-layer rho / 2 x rho | SSD busy ms per check")
for K in F.KS:
    for pol in ("QOM", "EXACT"):
        for live in (True, False):
            for tau in ((1.56, 1.0, 0.63) if (pol == "QOM" and live) or (pol == "EXACT" and live) else (1.56,)):
                A = run(K, pol, live, tau)
                n = A["checks"]; rl = A["rows"] * NL
                dm = max(A["dmass"], 1e-9)
                say(f"  K {K} {pol:5s} {'live' if live else '16r '} TAU {tau:4.2f}: miss {A['miss'] / n:6.1f} ({A['miss'] / A['kept']:5.1f}) | stl {A['stl'] / n:4.1f} | "
                    f"block {A['block'] / n:5.1f} ms | drop {A['drop'] / rl:.4f} | late {100 * A['dterms'] / A['terms']:4.1f} % of terms, mass {A['dmass'] / rl:.4f} "
                    f"(r0-1 {A['m01'] / rl:.4f}) | half {100 * A['dh'] / dm:4.1f} %, d1 {100 * A['d1'] / dm:4.1f} %, d2 {100 * A['d2'] / dm:4.1f} %, d3+ {100 * A['d3'] / dm:4.1f} %, tail "
                    f"{100 * A['tail'] / dm:4.1f} % | eff {A['eff'] / rl:.4f} / {A['eff2'] / rl:.4f} | ssd {A['ssd_busy'] / n:5.1f} ms, overrun "
                    f"{A['span'] / n:4.1f} ms ({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
