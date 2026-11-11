#!/usr/bin/env python3
"""D2's scoring (docs/PREREG_D2_2026-11-11.md; Breakthrough's rules, 18:4x). Reads the map's out/bm_*.json and the windows.
- Agreement per step: pooled over a set's answer rows, each row weighted the same. agree[i] = the arm's top-1 at answer
  row i equals the recorded next token (the 125B's greedy answer). Also reported against out/baseline.json's top-1,
  where that file holds the window.
- Kept per check: Breakthrough's accounting. 16-row checks from the first answer row; accepted = the leading True flags
  in agree[anchor .. anchor + 14]; kept = accepted + 1 (at most 16); the next anchor is anchor + kept.
- 95 % intervals: a window bootstrap (2,000 resamples, seed 1111), resampling the windows of a set with replacement.
- chat = IFEval + multi pooled. Lines: PASS = IFEval >= 0.85 AND multi >= 0.85; KILL = either < 0.80.
- Determinism: bm_d2_rep's top-8 ids and log-probs equal bm_d2's on that window, bit for bit.
usage: ~/BigMoE/.venv/bin/python -I scripts/d2_score.py [out.json]"""
import json, os, sys

import numpy as np

DM = os.path.expanduser("~/FableGod/decision_map")
TD = os.path.expanduser("~/TrainLab/data")
SETS = {"code": "d02b_walk_code.npz", "ifeval": "d02b_walk_d02a_ifeval.npz", "multi": "d02b_walk_d02a_multi.npz",
        "agent": "d02b_walk_d02a_agent.npz"}
TOK = {}
for d, f in SETS.items():
    z = np.load(os.path.join(TD, f))
    for s, nm in enumerate(z["names"]):
        TOK[str(nm)] = (d, z[f"{s}_ids"].astype(np.int64))
base = json.load(open(os.path.join(DM, "out", "baseline.json")))


def stored_top1(nm, dom):
    """D-02a's stored top-1 for a window (code: ~/Leap/data/l118_tgt; others: TrainLab data/d02a), with its offset."""
    f = os.path.join(os.path.expanduser("~/Leap/data/l118_tgt") if dom == "code" else os.path.join(TD, "d02a"), nm)
    if not os.path.exists(f):
        return None
    z = np.load(f)
    ref = z["tk"][:, 0] if "tk" in z else z["top1"]
    return np.asarray(ref, np.int64), int(z["off"]) if "off" in z else 0


def flags(res):
    """{window: (set, agree vs recorded, agree vs baseline or None)}"""
    out = {}
    for nm, r in res.items():
        d, ids = TOK[nm]
        a = int(r["a"])
        top1 = np.array([t[0] for t in r["tk"]], np.int64)
        nxt = ids[a + 1:a + 1 + len(top1)]
        n = min(len(top1), len(nxt))
        ab = None
        if nm in base:
            b1 = np.array([t[0] for t in base[nm]["tk"]], np.int64)
            m = min(n, len(b1))
            ab = top1[:m] == b1[:m]
        st = stored_top1(nm, d)
        sv = None
        if st is not None:
            ref, roff = st
            pos = int(r["off"]) + a + np.arange(n) - roff
            ok = (pos >= 0) & (pos < len(ref))
            sv = top1[:n][ok] == ref[pos[ok]]
        out[nm] = (d, top1[:n] == nxt[:n], ab, sv)
    return out


def kept(ag):
    k, c, i = 0, 0, 0
    while i < len(ag):
        acc = 0
        while acc < 15 and i + acc < len(ag) and ag[i + acc]:
            acc += 1
        kk = min(acc + 1, 16)
        k += kk; c += 1; i += kk
    return k, c


def stats(wins, rng):
    A = [w[1] for w in wins]
    agr = sum(int(a.sum()) for a in A) / max(sum(len(a) for a in A), 1)
    KC = [kept(a) for a in A]
    kpc = sum(k for k, _ in KC) / max(sum(c for _, c in KC), 1)
    bs_a, bs_k = [], []
    n = len(A)
    for _ in range(2000):
        ix = rng.integers(0, n, n)
        bs_a.append(sum(int(A[j].sum()) for j in ix) / max(sum(len(A[j]) for j in ix), 1))
        bs_k.append(sum(KC[j][0] for j in ix) / max(sum(KC[j][1] for j in ix), 1))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    B = [w[2] for w in wins if w[2] is not None]
    agb = (sum(int(b.sum()) for b in B) / max(sum(len(b) for b in B), 1)) if B else None
    V = [w[3] for w in wins if w[3] is not None]
    ags = (sum(int(v.sum()) for v in V) / max(sum(len(v) for v in V), 1)) if V else None
    return dict(windows=n, rows=int(sum(len(a) for a in A)), agree=agr, agree_ci=q(bs_a), kept=kpc, kept_ci=q(bs_k),
                checks=int(sum(c for _, c in KC)), agree_vs_baseline=agb, baseline_windows=len(B),
                agree_vs_stored=ags, stored_rows=int(sum(len(v) for v in V)))


def main():
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/BigMoE/data/d2/d2_score.json")
    R = {}
    for arm in ("bm_d2", "bm_res", "bm_ops", "bm_d2t", "bm_d3", "bm_d4", "bm_base_agent"):
        p = os.path.join(DM, "out", arm + ".json")
        if not os.path.exists(p):
            continue
        F = flags(json.load(open(p)))
        R[arm] = {}
        for d in ("code", "ifeval", "multi", "agent", "chat"):
            ws = [v for v in F.values() if (v[0] in ("ifeval", "multi") if d == "chat" else v[0] == d)]
            if ws:
                R[arm][d] = stats(ws, np.random.default_rng(1111))
    if "bm_d2" in R:
        i, m = R["bm_d2"].get("ifeval", {}).get("agree"), R["bm_d2"].get("multi", {}).get("agree")
        v = "PASS" if (i is not None and m is not None and i >= 0.85 and m >= 0.85) else \
            "KILL" if (i is not None and m is not None and (i < 0.80 or m < 0.80)) else "INCONCLUSIVE"
        R["verdict"] = dict(rule="PASS: IFEval >= 0.85 and multi >= 0.85; KILL: either < 0.80", ifeval=i, multi=m, verdict=v)
    g = R.get("bm_base_agent", {}).get("agent")
    if g is not None:                                    # the agent set's gate (Breakthrough 18:5x)
        R["agent_gate"] = dict(rule="the stock exact pass vs D-02a's stored top-1 on the agent answer rows >= 99.9 %",
                               agree=g["agree_vs_stored"], rows=g["stored_rows"],
                               ok=g["agree_vs_stored"] is not None and g["agree_vs_stored"] >= 0.999)
    p1, p2 = os.path.join(DM, "out", "bm_d2.json"), os.path.join(DM, "out", "bm_d2_rep.json")
    if os.path.exists(p1) and os.path.exists(p2):
        a, b = json.load(open(p1)), json.load(open(p2))
        same = {nm: (a[nm]["tk"] == b[nm]["tk"] and a[nm]["lp"] == b[nm]["lp"]) for nm in b if nm in a}
        R["determinism"] = dict(windows=list(same), identical=all(same.values()) and bool(same))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(R, open(dst, "w"), indent=1)
    for arm, v in R.items():
        if arm in ("verdict", "determinism", "agent_gate"):
            print(arm, v)
            continue
        for d, s in v.items():
            print(f"{arm:9s} {d:6s} windows {s['windows']:>2} rows {s['rows']:>5}: agree {s['agree']:.4f} "
                  f"[{s['agree_ci'][0]:.4f}, {s['agree_ci'][1]:.4f}]  kept/check {s['kept']:.2f} [{s['kept_ci'][0]:.2f}, "
                  f"{s['kept_ci'][1]:.2f}] ({s['checks']} checks)  vs baseline "
                  f"{'-' if s['agree_vs_baseline'] is None else format(s['agree_vs_baseline'], '.4f')} ({s['baseline_windows']} w)  vs stored "
                  f"{'-' if s['agree_vs_stored'] is None else format(s['agree_vs_stored'], '.4f')} ({s['stored_rows']} rows)")


if __name__ == "__main__":
    main()
