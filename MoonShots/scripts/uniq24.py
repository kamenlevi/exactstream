"""R-29 UNIQ (MoonShots Tue 20:3x; NON-exact, bytes side; published kin: Lynx batch-aware expert selection): in a 16-row
check, drop a row's rank >= R0 expert (0-indexed, of its top-10) only if NO other row of the check routes to it and its
top-10-renormalized weight < tau: single-use, low-weight experts cost a whole read for little. wall24 LFU replay (16-row
checks accepting 10, tokens 400+, exact stack C_m 159 / C_l 44 > union, W24-ec + 11-bit sizes) vs DYNK / fixed top-8 at
equal dropped gate mass. PASS (written first): some setting cuts >= 15 % of bytes / token while dropping <= 3 % of gate
mass per token-layer (DYNK tau 0.06: -13.2 % at 4.7 %); KILL: nothing beats DYNK's bytes-per-mass by >= 1.5x.
usage: [BigMoE venv] taskpolicy -b python uniq24.py -> stdout + data/uniq24.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

R, A = int(os.environ.get("UNIQ_R", "16")), int(os.environ.get("UNIQ_A", "10"))
CMU, CLU = int(os.environ.get("UNIQ_CM", "159")), int(os.environ.get("UNIQ_CL", "44"))
W.MSB, W.LSB = 1900544, 720896
WT = {}
for k, d in W.SETS.items():
    g = np.stack([d[f"G{l}"] for l in range(W.NL)], 1)[:, :, :10].astype(np.float64)
    WT[k] = g / g.sum(-1, keepdims=True)
STATE = {"r0": 10, "tau": 0.0, "mass": 0.0, "uses": 0, "mode": os.environ.get("UNIQ_MODE", "single")}


def needs(ids, R_, A_, wt=None):
    n, out, pos = ids.shape[0], [], 0
    while pos + R_ <= n:
        lay = []
        meas = W.MEAS[0] <= pos < W.MEAS[1]
        for l in range(W.NL):
            blk = ids[pos:pos + R_, l]                     # [R, 10]
            cnt = {}
            for r in range(R_):
                for e in blk[r]:
                    cnt[int(e)] = cnt.get(int(e), 0) + 1
            best = {}
            tot = {}
            if STATE["mode"] == "mass":                        # check-level: summed weight of e over the rows, ranks >= r0 only
                for r in range(R_):
                    for k, e in enumerate(blk[r]):
                        tot[int(e)] = tot.get(int(e), 0.0) + (wt[pos + r, l, k] if k >= STATE["r0"] else 9.0)
            for r in range(R_):
                for k, e in enumerate(blk[r]):
                    e = int(e)
                    if k >= STATE["r0"] and ((STATE["mode"] == "mass" and tot[e] < STATE["tau"]) or
                                             (STATE["mode"] != "mass" and cnt[e] == 1 and wt[pos + r, l, k] < STATE["tau"])):
                        if meas:
                            STATE["mass"] += wt[pos + r, l, k]
                        continue
                    if e not in best or k < best[e]:
                        best[e] = k
            if meas:
                STATE["uses"] += R_
            lay.append((frozenset(best), frozenset(e for e, k in best.items() if k < 2)))
        out.append((meas, lay))
        pos += A_
    return out


lines = []
base = None
GRID = ((10, 0.0), (8, 1.0), (8, 0.06), (6, 0.06), (6, 0.08), (4, 0.08), (6, 1.0), (4, 1.0)) if STATE["mode"] != "mass" else \
    ((10, 0.0), (4, 0.06), (4, 0.08), (4, 0.10), (4, 0.12), (2, 0.08), (2, 0.12), (2, 0.16))
for r0, tau in GRID:
    STATE.update(r0=r0, tau=tau, mass=0.0, uses=0)
    W._NEEDS.clear()
    tot = 0.0
    for k, ids in W.IDS.items():
        W._NEEDS[(k, R, A)] = needs(ids, R, A, WT[k])
    b = W.bytes_per_token("lfu", Cm=CMU, Cl=CLU, R=R, A=A)[0]
    base = base or b
    m = STATE["mass"] / max(STATE["uses"], 1)
    s = (f"drop {'experts (ranks >= ' + str(r0) + ' uses) whose summed check weight < ' + format(tau, '.2f') if STATE['mode'] == 'mass' else 'single-use ranks >= ' + str(r0) + ' with weight < ' + format(tau, '.2f')}: {b / 1e6:.1f} MB / token ({100 * (b / base - 1):+.1f} %), gate mass "
         f"dropped {100 * m:.2f} % per row-layer" + (f", {100 * (1 - b / base) / (100 * m):.2f} % bytes per % mass" if m else ""))
    print(s, flush=True); lines.append(s)
lines.append("reference (R-25 DYNK, same replay): fixed top-8 -25.7 % at 12.1 % mass (2.12 % / %), tau 0.06 -13.2 % at 4.7 % (2.81 % / %)")
print(lines[-1])
open(os.path.expanduser(("~/MoonShots/data/uniq24.txt" if STATE["mode"] != "mass" else "~/MoonShots/data/uniq24_mass.txt") if (R, A, CMU) == (16, 10, 159) else f"~/MoonShots/data/uniq_{STATE['mode']}_{CMU}_{CLU}_r{R}a{A}.txt"), "w").write("\n".join(lines) + "\n")
