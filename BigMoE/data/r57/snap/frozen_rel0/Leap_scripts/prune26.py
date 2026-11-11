"""Orchestrator 22:3x (Leap, Sat 2026-10-24): would a 256-expert pruned 125B (ISTA-DASLab GSQ-RCO-Coder: 256 of 512 routed
experts kept per layer by RCO, retained weights ~3.5 bpw) reach <= ~16 SSD misses per generated token at 24 GB? CPU only,
no download: Breakthrough's 125B routing traces (data/expA/traces125/{code,chat,math,bulgarian}_800_g.npz: router top-20
ids L<l> and their softmax probabilities G<l>, 800 decode tokens per set, 48 layers).
- RCO stand-in: per layer keep the 256 experts with the most top-10 gate mass on calibration tokens (first half of each
  set: code only, or all four sets); evaluate on the second halves. A token then routes to the first 10 (or 8) kept
  experts among its router top-20 (fewer if the top-20 holds fewer kept experts: counted).
- Kept mass: share of the original top-10 gate mass that stays on kept experts (a quality proxy; RCO optimises KL).
- Checks: R rows, T accepted tokens per check (16 rows / 9.5 tokens for DFlash2; 4 / 3.4 for MTP); the rows of a check
  are approximated by the next R true tokens. Per layer an LRU pool of C experts (warm-started with the C most used);
  each check loads its union's missing experts and touches the rest. Misses per generated token = misses / tokens.
- Calibration of the simulator: the unpruned 512 at C 192 with MTP checks vs Fit12's RES1 (155-283 reads per 4-row
  check at C_m 192, i.e. ~46-83 per token).
Usage: python3 prune26.py -> ~/Leap/data/prune26.txt"""
import glob, os
import numpy as np
from collections import OrderedDict

NL, E = 48, 512
T = "/Users/kamen/Breakthrough/data/expA/traces125/"
sets = {os.path.basename(f).split("_")[0]: np.load(f) for f in sorted(glob.glob(T + "*_800_g.npz"))}
IDS = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}   # [tok, L, 20]
GP = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def usage(names, half):
    """per layer, top-10 gate mass per expert over the chosen half of the chosen sets"""
    u = np.zeros((NL, E))
    for k in names:
        n = IDS[k].shape[0]; sl = slice(0, n // 2) if half == 0 else slice(n // 2, n)
        ids, gp = IDS[k][sl, :, :10], GP[k][sl, :, :10]
        for l in range(NL):
            np.add.at(u[l], ids[:, l].ravel(), gp[:, l].ravel())
    return u

def route(ids20, gp20, keep, k):
    """per token / layer: the first k kept experts of the router's top-20 (list per layer), kept-mass share"""
    tok = ids20.shape[0]
    out = [[None] * NL for _ in range(tok)]
    short, kept_mass = 0, []
    for l in range(NL):
        kl = keep[l]
        for t in range(tok):
            sel = [e for e in ids20[t, l] if kl[e]][:k]
            if len(sel) < k:
                short += 1
            out[t][l] = sel
        m = gp20[:, l, :10]
        kept_mass.append((m * kl[ids20[:, l, :10]]).sum(1) / m.sum(1))
    return out, short / (tok * NL), float(np.mean(kept_mass))

def simulate(routes, warm, C, R, Tacc):
    """LRU pools of C per layer; checks of R rows advancing Tacc tokens; returns misses per generated token"""
    pools = []
    for l in range(NL):
        od = OrderedDict()
        for e in np.argsort(-warm[l])[:C]:
            od[int(e)] = True
        pools.append(od)
    n = len(routes); pos, acc, miss, gen = 0.0, 0, 0, 0
    while int(pos) + R <= n:
        s = int(pos)
        for l in range(NL):
            un = set()
            for t in range(s, s + R):
                un.update(routes[t][l])
            od = pools[l]
            for e in un:
                if e in od:
                    od.move_to_end(e)
                else:
                    miss += 1
                    od[e] = True
                    if len(od) > C:
                        od.popitem(last=False)
        gen_now = int(pos + Tacc) - s
        gen += gen_now
        pos += Tacc
    return miss / max(gen, 1)

say("Pruned-125B estimate (Breakthrough's routing traces, 4 sets x 800 decode tokens, 48 layers); misses per generated token")
full_keep = np.ones((NL, E), bool)
for calib_name, calib in (("code", ["code"]), ("all four sets", list(IDS))):
    u = usage(calib, 0)
    keep = np.zeros((NL, E), bool)
    for l in range(NL):
        keep[l, np.argsort(-u[l])[:256]] = True
    say(f"RCO stand-in calibrated on {calib_name} (first halves); evaluated on second halves")
    for test in ("code", "chat", "math", "bulgarian"):
        n = IDS[test].shape[0]
        ids20, gp20 = IDS[test][n // 2:], GP[test][n // 2:]
        r10_full, _, _ = route(ids20, gp20, full_keep, 10)
        r10, short10, km = route(ids20, gp20, keep, 10)
        r8, short8, _ = route(ids20, gp20, keep, 8)
        warm_full = usage(calib, 0)                                    # warm start: most used (calibration)
        cells = []
        for lab, routes, warm, C in (("512 @ C192", r10_full, warm_full, 192), ("256 @ C192", r10, warm_full * keep, 192),
                                     ("256 top-8 @ C192", r8, warm_full * keep, 192), ("256 @ C224", r10, warm_full * keep, 224),
                                     ("256 @ C240", r10, warm_full * keep, 240)):
            m16 = simulate(routes, warm, C, 16, 9.5)
            m4 = simulate(routes, warm, C, 4, 3.4)
            cells.append(f"{lab}: {m16:5.1f} / {m4:5.1f}")
        def union16(routes):
            us = []
            for s0 in range(0, len(routes) - 16 + 1, 8):
                for l in range(NL):
                    us.append(len(set(e for t in range(s0, s0 + 16) for e in routes[t][l])))
            return float(np.mean(us))
        say(f"  {test:9s} distinct experts per layer in a 16-row check (all resident, bytes): 512 top-10 {union16(r10_full):.1f}, "
            f"256 top-10 {union16(r10):.1f}, 256 top-8 {union16(r8):.1f}")
        say(f"  {test:9s} kept top-10 mass {km:.3f} (rows with < 10 kept in the top-20: {short10:.1%}); misses per token "
            f"at 16-row (9.5) / 4-row (3.4) checks: " + "; ".join(cells))
say("Need <= ~16 misses per token at 24 GB for 350 (with full read / compute overlap; ~30 MB per token at 10.6 GB/s).")
open(os.path.expanduser("~/Leap/data/prune26.txt"), "w").write("\n".join(lines) + "\n")
