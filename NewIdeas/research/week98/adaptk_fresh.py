#!/usr/bin/env python3
"""ADAPTIVE K: the FRESH-CELL re-read, frozen before the data (NewIdeas, Tue 17 Nov; written ~01:1x by date).

The gate (the orchestrator, accepted Tue ~01:1x; ADAPTK.md): on fresh cells (Wednesday night's capture + BigMoE's
127-node tree dumps, Thu by day), re-read the policy at the FROZEN settings:
  tau 0.9 (MBPP's in-sample tau; out of set for HE+), KERN2 (16 / 32 rows), L 0.66 ms per draft level, E14 costs
  (costs_e14.json: A16 62.49 ms, R32 1.4372, c 1.025 / 1.04, overhead 4 ms).
KILL if the HE+-like set's pooled gain over fixed 16 rows is < +3 %. PASS if >= +3 %. Everything else is information:
the MBPP-like set, the other L, the window bootstrap, the ONEPASS variant.

usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_fresh.py --he TAG --mbpp TAG [--dump-fmt FMT] [--out NAME]
  TAG = the cells / dump tag in ~/BigMoE/data/deepdraft/ceiling (cells_t30b_{TAG}.npz + the dump FMT.format(tag=TAG)).
  Dry check on the dev tags: --he heplus24 --mbpp mbppbank must give HE+ +8.6 % (adaptk_windows.txt).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_audit as TA  # noqa: E402
import adaptk_draft as AD  # noqa: E402

TAU, L_FROZEN, KILL = 0.9, 0.66, 0.03
A16, OV, R32 = 62.49, 4.0, 1.4372
C16, C32 = 1.025, 1.04
T = {16: 1.0 * C16 * A16 + OV, 32: R32 * C32 * A16 + OV}


def read(tag, fmt, L, rng):
    W = TA.load_windows(tag, float("nan"))
    z = np.load(os.path.join(TA.CE, fmt.format(tag=tag)), allow_pickle=True)
    idx, PP, A, Hh, mono = AD.dense({k: z[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p")})
    S = AD.steps(W, idx)
    cost = (lambda r, i: T[r] + (L * float(Hh[r][i]) if i >= 0 else 0.0))
    npop = np.maximum(15, (PP >= TAU).sum(1))
    ro = np.where(npop <= 15, 16, 32)

    def per_window(choose):
        out = []
        for ids in S:
            kept = tm = 0.0; n = len(ids); p = 1
            while p < n:
                i = ids[p]; r = choose(i) if i >= 0 else 16; a = int(A[r][i]) if i >= 0 else 0
                kept += a + 1; tm += cost(r, i); p += a + 1
            out.append((kept, tm))
        return np.array(out)
    base = per_window(lambda i: 16)
    fix32 = per_window(lambda i: 32)
    pol = per_window(lambda i: int(ro[i]))
    tps = lambda X: 1000 * X[:, 0].sum() / X[:, 1].sum()
    g = tps(pol) / tps(base) - 1
    bs = []
    for _ in range(2000):
        j = rng.integers(0, len(S), len(S))
        bs.append((1000 * pol[j, 0].sum() / pol[j, 1].sum()) / (1000 * base[j, 0].sum() / base[j, 1].sum()) - 1)
    gw = (pol[:, 0] / pol[:, 1]) / (base[:, 0] / base[:, 1]) - 1
    calib = (float(np.mean(1 + PP[:, :15].sum(1))), float(np.mean(A[16] + 1)))
    return {"windows": len(S), "anchors": len(idx), "fixed16": tps(base), "fixed32": tps(fix32), "policy": tps(pol),
            "gain": g, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "windows_up": int((gw > 0.005).sum()), "windows_down": int((gw < -0.005).sum()),
            "share32": float(np.mean(ro[[i for ids in S for i in ids if i >= 0]] == 32)),
            "calib16_E_vs_real": calib, "monotone": mono}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", required=True)
    ap.add_argument("--mbpp", required=True)
    ap.add_argument("--dump-fmt", default="tree_t30b_{tag}_n127k8.npz")
    ap.add_argument("--out", default="adaptk_fresh")
    a = ap.parse_args()
    rng = np.random.default_rng(17)
    lines = [f"ADAPTIVE K fresh-cell re-read, run {time.strftime('%a %d %b %H:%M')}; frozen: tau {TAU}, KERN2, "
             f"L {L_FROZEN} ms per level, E14 costs; KILL if the HE+-like gain < +{100 * KILL:.0f} %"]
    res = {}
    for name, tag in (("he", a.he), ("mbpp", a.mbpp)):
        for L in (L_FROZEN, 0.40, 0.92):
            r = read(tag, a.dump_fmt, L, rng)
            res.setdefault(name, {})[str(L)] = r
            lines.append(f"{name} ({tag}) L {L:.2f}: fixed16 {r['fixed16']:.1f} fixed32 {r['fixed32']:.1f} policy "
                         f"{r['policy']:.1f} -> {100 * r['gain']:+.1f} % [95 % window bootstrap {100 * r['ci95'][0]:+.1f}, "
                         f"{100 * r['ci95'][1]:+.1f}]; windows up {r['windows_up']} / down {r['windows_down']} of "
                         f"{r['windows']}; 32 rows at {100 * r["share32"]:.1f} % of positions; E[kept16] {r['calib16_E_vs_real'][0]:.2f}"
                         f" vs real {r['calib16_E_vs_real'][1]:.2f}")
    g = res["he"][str(L_FROZEN)]["gain"]
    verdict = "PASS" if g >= KILL else "KILL"
    lines.append(f"VERDICT (HE+-like, frozen settings): {verdict} ({100 * g:+.1f} % vs the +{100 * KILL:.0f} % line)")
    res["verdict"] = verdict
    print("\n".join(lines))
    json.dump(res, open(os.path.join(HERE, a.out + ".json"), "w"), indent=1)
    open(os.path.join(HERE, a.out + ".txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
