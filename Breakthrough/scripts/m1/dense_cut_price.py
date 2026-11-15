#!/usr/bin/env python3
"""Lossy cuts to the DENSE read of a 32-row check at 45 GB (Breakthrough, Sun 15 Nov 2026 01:1x; WEEK98 goal B addendum,
the orchestrator 01:0x: "challenge 'dense stays'") [K: weight bytes from the safetensors headers, measured map flips;
P: combinations]. CPU only.
- Dense bytes by module: ~/BigMoE/models/bare/*.safetensors headers (no weights loaded).
- Measured map flips (all-row / non-tie vs baseline.json; tie rows from ~/FableGod/floor_rows.json): hc8w (HC mixers +
  PLE at 8-bit), bm_d4 vs bm_res (the bf16 / 8-bit trunk modules at 4-bit, both with resident-only experts).
- lm_head shortlist: the 125B's top-1 token frequency on BigMoE's bank (~/datasets/deepdraft/bank, 524 files, disjoint
  from the map windows) -> misses = the map's baseline top-1 outside the K most frequent (an upper bound: a draft that
  equals the true argmax is always scored).
- Combined code plan: the 1c-family expert tiers (lossy_check_price.py's calibration, re-derived here) + dense cuts;
  dense flips added to the expert prediction (conservative on all rows: tie flips counted twice).
Output: ~/Breakthrough/data/m1/dense_cut_price.out. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B dense_cut_price.py
"""
import glob
import json
import os
import re
import struct
import collections

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
BW, EFF, EXP_MB, NL, ROWS = 280.0, 0.84, 2.135, 48, 32
LEVEL = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}
NT_LEVEL = {"code": 0.87, "ifeval": 2.22, "multi": 1.63}
DOMS = ["code", "ifeval", "multi"]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


# ---- 1. dense bytes by module ----
DT = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1, "I8": 1, "I64": 8, "F64": 8}
grp = collections.Counter()
for f in sorted(glob.glob(os.path.expanduser("~/BigMoE/models/bare/*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__":
            continue
        b = int(np.prod(v["shape"])) * DT[v["dtype"]]
        if "switch_mlp" in k or "embed_tokens" in k or "mtp" in k or k.startswith("ngram") or ".ngram" in k:
            continue
        cat = ("lm_head" if "lm_head" in k else "GDN projections" if "linear_attn" in k and "conv" not in k else
               "attention projections" if "self_attn" in k and "indexer" not in k else "indexer (8-bit)" if "indexer" in k else
               "shared expert" if "shared_expert" in k else "router (8-bit)" if "mlp.gate" in k else
               "HC mixers (bf16)" if "hyper_connection" in k else "PLE (bf16)" if "ple" in k else "norms / conv / other")
        grp[cat] += b
dense = sum(grp.values()) / 1e9
say(f"DENSE read per pass from the weight headers: {dense:.3f} GB (Leap's D32 figure: 3.44 GB)")
for c, b in grp.most_common():
    say(f"  {c:24s} {b / 1e9:6.3f} GB ({100 * b / 1e9 / dense:4.1f} %)")

# ---- 2. measured map flips for the dense levers ----
base = json.load(open(os.path.join(M, "baseline.json")))
TIE = {(r["nm"], int(r["i"])): r["cls"] == "tie" for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))}


def flips(arm):
    a = json.load(open(os.path.join(M, arm + ".json"))); r = {}
    for nm, v in a.items():
        if nm not in base:
            continue
        x = np.array([t[0] for t in v["tk"]]); y = np.array([t[0] for t in base[nm]["tk"]]); k = min(len(x), len(y))
        tie = np.array([TIE.get((nm, base[nm]["off"] + base[nm]["a"] + j), False) for j in range(k)])
        f = x[:k] != y[:k]; s = r.setdefault(v["dom"], [0, 0, 0, 0])
        s[0] += int(f.sum()); s[1] += k; s[2] += int((f & ~tie).sum()); s[3] += int((~tie).sum())
    return {d: (100 * s[0] / s[1], 100 * s[2] / s[3]) for d, s in r.items()}


F = {arm: flips(arm) for arm in ("hc8w", "t4_all", "bm_res", "bm_d4", "bm_d3")}
say("")
say("MEASURED map flips % (all-row / non-tie):")
for arm, lab in (("hc8w", "HC mixers + PLE at 8-bit (exact experts)"), ("t4_all", "bit-plane path at 4 bits (numeric-noise reference)"),
                 ("bm_res", "resident-only experts (BigMoE D2 control)"), ("bm_d4", "+ bf16 / 8-bit trunk modules at 4-bit"),
                 ("bm_d3", "+ the whole trunk at 3-bit RTN")):
    say(f"  {arm:8s} {lab:52s} " + "  ".join(f"{d} {F[arm][d][0]:.2f} / {F[arm][d][1]:.2f}" for d in DOMS))
D4 = {d: (F["bm_d4"][d][0] - F["bm_res"][d][0], F["bm_d4"][d][1] - F["bm_res"][d][1]) for d in DOMS}
say("  mixers + inject + PLE + indexer at 4-bit, increment over resident-only (confounded): "
    + "  ".join(f"{d} +{D4[d][0]:.2f} / +{D4[d][1]:.2f}" for d in DOMS))

# ---- 3. lm_head shortlist misses ----
cnt = collections.Counter()
for f in sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/*.npz"))):
    z = np.load(f)
    cnt.update(z["tk"][:, 0].tolist())
order = [t for t, _ in cnt.most_common()]
say("")
say(f"LM_HEAD SHORTLIST: the bank's top-1 tokens ({sum(cnt.values())} rows, {len(cnt)} distinct) -> map misses (baseline top-1 "
    f"outside the K most frequent; an upper bound)")
SL = {}
for K in (2048, 4096, 8192, 16384, 32768):
    S = set(order[:K]); r = {}
    for nm, v in base.items():
        tk = np.array([t[0] for t in v["tk"]]); s = r.setdefault(v["dom"], [0, 0])
        s[0] += int(sum(1 for t in tk if t not in S)); s[1] += len(tk)
    gb = 0.358 * (1 - K / 248320)
    SL[K] = ({d: 100 * r[d][0] / r[d][1] for d in DOMS}, gb)
    say(f"  K {K:6d}: saves {gb:.3f} GB; misses % " + "  ".join(f"{d} {SL[K][0][d]:.3f}" for d in DOMS))

# ---- 4. expert tiers (re-derived from lossy_check_price.py's calibration) ----
TIERS3 = [("4-bit", 0.0, 1.0), ("MSB-3", 0.08, 0.778), ("drop", 1.0, 0.0)]
W = {}
for f in sorted(os.listdir(os.path.join(M, "xcap"))):
    z = np.load(os.path.join(M, "xcap", f)); W[f] = (base[f]["dom"], z["ids"].astype(np.int32), z["w"].astype(np.float32), int(z["a"]))


def e_rows(kind, nm):
    dom, ids, w, a = W[nm]; w2 = w ** 2
    if kind[0] == "topk":
        return w2[:, :, kind[1]:].sum((1, 2)) / NL
    keep = np.load(os.path.join(M, kind[1], nm))["keep"].astype(bool)[a:a + w.shape[0]]
    return (w2 * ~keep).sum((1, 2)) / NL


CAL = [("topk2", ("topk", 2)), ("topk5", ("topk", 5)), ("top_k8", ("topk", 8)), ("q1u_cf09", ("mask", "q1u_cf09")),
       ("q1u_cf099", ("mask", "q1u_cf099")), ("qom_0_live_c51", ("mask", "bt/qom_0_live_c51")),
       ("qom_0_live_c147", ("mask", "ni/qom_0_live_c147"))]
PTS = {}
for nt in (False, True):
    for d in DOMS:
        P = []
        for arm, kind in CAL:
            a = json.load(open(os.path.join(M, arm + ".json"))); es, fs = [], []
            for nm, (dom, ids, w, aa) in W.items():
                if dom != d or nm not in a:
                    continue
                x = np.array([t[0] for t in a[nm]["tk"]]); y = np.array([t[0] for t in base[nm]["tk"]]); k = min(len(x), len(y), w.shape[0])
                tie = np.array([TIE.get((nm, aa + j), False) for j in range(k)])
                sel = ~tie if nt else np.ones(k, bool)
                es.append(e_rows(kind, nm)[:k][sel]); fs.append((x[:k] != y[:k])[sel])
            P.append((float(np.concatenate(es).mean()), 100 * float(np.concatenate(fs).mean())))
        PTS[(nt, d)] = np.array(P)


def predict(nt, d, E):
    P = PTS[(nt, d)]
    b, la = np.polyfit(np.log(P[:, 0]), np.log(P[:, 1]), 1)
    loo = []
    for i in range(len(P)):
        Q = np.delete(P, i, 0); bb, aa = np.polyfit(np.log(Q[:, 0]), np.log(Q[:, 1]), 1); loo.append(np.exp(aa) * E ** bb)
    return np.exp(la) * E ** b, min(loo), max(loo)


GIDX, W2, RD, RDOM, G_of = [], [], [], [], []
g0 = nbl = 0
for nm, (dom, ids, w, a) in W.items():
    for bi in range(ids.shape[0] // ROWS):
        I, w2 = ids[bi * ROWS:(bi + 1) * ROWS], w[bi * ROWS:(bi + 1) * ROWS] ** 2
        gi = np.empty((ROWS, NL, 10), np.int64)
        for l in range(NL):
            u, inv = np.unique(I[:, l].ravel(), return_inverse=True)
            gi[:, l, :] = (inv + g0).reshape(ROWS, 10); g0 += len(u); nbl += 1
        GIDX.append(gi); W2.append(w2); RD.append(np.arange(ROWS)); RDOM.append(np.full(ROWS, DOMS.index(dom)))
GIDX = np.concatenate(GIDX); W2 = np.concatenate(W2); RD = np.concatenate(RD); RDOM = np.concatenate(RDOM)


def expert_option(s, lam):
    rw = s ** RD
    S = np.bincount(GIDX.ravel(), weights=(W2 * rw[:, None, None]).ravel(), minlength=g0)
    err = np.array([t[1] for t in TIERS3]); size = np.array([t[2] for t in TIERS3])
    t = (err[None, :] * S[:, None] + lam * size[None, :]).argmin(1)
    keq = size[t].sum() / nbl
    Erow = (err[t][GIDX] * W2).sum((1, 2)) / NL
    sel = RDOM == 0
    Ec = float((Erow[sel] * rw[sel]).sum() / rw[sel].sum())
    return keq, Ec


say("")
say("COMBINED CODE PLAN: expert tiers (3 tiers, acceptance-weighted) + dense cuts; code flips = the expert prediction "
    "[leave-one-arm-out range] + the dense levers' measured standalone flips (all-row / non-tie)")
MIX, PLE = grp["HC mixers (bf16)"] / 1e9, grp["PLE (bf16)"] / 1e9      # bf16 -> b bits + group-64 scales (b + 0.5 bits)
CUT8, CUT4 = (MIX + PLE) * (1 - 8.5 / 16), (MIX + PLE) * (1 - 4.5 / 16)
say(f"  dense cuts (GB saved per check): HC mixers + PLE to 8-bit {CUT8:.3f}, to 4-bit {CUT4:.3f}; lm_head K16K {SL[16384][1]:.3f}; "
    f"GDN + attention 4 -> 3-bit {(grp['GDN projections'] + grp['attention projections']) / 1e9 / 4.5:.3f} (unmeasured); "
    f"shared expert MSB-3 {grp['shared expert'] / 1e9 * 0.222:.3f}")
DENSE_OPTS = [
    ("dense as is", 0.0, (0.0, 0.0)),
    ("mixers 8-bit + lm_head K16K", CUT8 + SL[16384][1], (F["hc8w"]["code"][0] + SL[16384][0]["code"], F["hc8w"]["code"][1] + SL[16384][0]["code"])),
    ("mixers 4-bit + lm_head K16K", CUT4 + SL[16384][1], (D4["code"][0] + SL[16384][0]["code"], D4["code"][1] + SL[16384][0]["code"])),
]
say("| experts (s, lambda) | dense option | GB | ms floor / 84 % | tok/s HE+ at 84 % (today / 98 %) | code all-row % (level 1.34) | code non-tie % (level 0.87) | verdict (code) |")
say("|---|---|---|---|---|---|---|---|")
for s, lam in ((0.98, 8e-3), (0.98, 1.2e-2), (0.98, 1.6e-2), (0.98, 1.8e-2)):
    keq, Ec = expert_option(s, lam)
    pa, pa_lo, pa_hi = predict(False, "code", Ec); pn, pn_lo, pn_hi = predict(True, "code", Ec)
    for name, cut, (fa, fn) in DENSE_OPTS:
        gb = dense - cut + EXP_MB * NL * keq / 1000
        a, b = 1000 * gb / BW, 1000 * gb / (BW * EFF)
        ta, tn = pa + fa, pn + fn
        v = "within" if (pa_hi + fa) <= LEVEL["code"] and (pn_hi + fn) <= NT_LEVEL["code"] else (
            "within (point)" if ta <= LEVEL["code"] and tn <= NT_LEVEL["code"] else "over")
        say(f"| s {s} lam {lam:.1e} ({keq:.1f} eq) | {name} | {gb:.2f} | {a:.1f} / {b:.1f} | {13.8 / ((b + 7) / 1000):.0f} / "
            f"{23.3 / ((b + 7) / 1000):.0f} | {ta:.2f} [{pa_lo + fa:.2f}, {pa_hi + fa:.2f}] | {tn:.2f} [{pn_lo + fn:.2f}, {pn_hi + fn:.2f}] | {v} |")
open(os.path.expanduser("~/Breakthrough/data/m1/dense_cut_price.out"), "w").write("\n".join(lines) + "\n")
