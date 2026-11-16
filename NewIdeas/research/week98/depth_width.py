"""PLAN900 bet #9: DRAFT DEATH + WIDE-SHALLOW (ZERO THINK_1116), NewIdeas, Mon 16 Nov 12:3x. CPU only, read-only.
On BigMoE's real 127-node dumps (k 8): choose K nodes (15 / 31 / 63 / 127) in pop order with
  - WIDE-SHALLOW: a depth cap D (6 / 8 / 10 / 12 / 16 / 20 / 24 / none), or
  - DRAFT DEATH: a confidence cut, path_p >= tau (0.5 / 0.3 / 0.1 / 0.03 / 0.01); the first K such pops.
Every selection is closed (pop order puts parents first; both filters are monotone along a path).
Report kept (each tree under its own walk, K + 1 rows) and head LEVELS per block (deepest selected node + 1).
tok/s with drafting NOT hidden = kept / (check(K) + levels x per-level + 4 ms):
  - check models: (a) the 900 target, 25 ms at 32 rows scaled by Leap's floor R(K) / R(32) and c(K); (b) Leap's byte
    floor R(K) x 62 ms x c(K);
  - per level: 0.917 ms (today's sync-free, full head) and 0.30 ms (QAT 26 MB head + a fused level kernel, FAR's
    target).
LINE (ZERO's framing, written before the run): PASS if some (K, cut) beats tree-32 (no cut) on tok/s by >= +10 % under
model (a) at today's per-level cost, on HE+ and MBPP both; else the depth / width trade is already near-optimal.
usage: ~/BigMoE/.venv/bin/python -I -B depth_width.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

OUT = os.path.expanduser("~/NewIdeas/research/week98/depth_width")
RF = {16: 38.8 / 62, 32: 0.85, 48: 1.00, 64: 1.12, 96: 1.30, 128: 1.44}     # Leap's floor R(rows), 05eea11
CK = {16: 1.025, 32: 1.04, 48: 1.055, 64: 1.07, 96: 1.09, 128: 1.11}
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def main():
    TA.say = lambda s="": None
    res = {}
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        tz = np.load(f"{TA.CE}/tree_t30b_{tag}_n127k8.npz", allow_pickle=True)
        fi, an, nd, dp, ot, pp = (tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p"))
        order = np.lexsort((nd, an, fi)); fi, an, nd, dp, ot, pp = (x[order] for x in (fi, an, nd, dp, ot, pp))
        b = np.flatnonzero(np.append(True, (np.diff(fi) != 0) | (np.diff(an) != 0))); b = np.append(b, len(fi))
        T = {(int(fi[b0]), int(an[b0])): (b0, b1) for b0, b1 in zip(b[:-1], b[1:])}
        say(f"\n=== {s} ({tag}) ===")
        rows = []
        def run(K, D, tau):
            ks, lv = [], []
            for X in W:
                p = 1
                while p < X["n"]:
                    if p < len(X["an"]):
                        b0, b1 = T[(X["fi"], int(X["an"][p]))]
                        m = (dp[b0:b1] < D) & (pp[b0:b1] >= tau)
                        idx = np.flatnonzero(m)[:K]
                        sel = np.zeros(b1 - b0, bool); sel[idx] = True
                        hit = ot[b0:b1].astype(bool) & sel
                        a = int(dp[b0:b1][hit].max()) + 1 if hit.any() else 0
                        lv.append(int(dp[b0:b1][sel].max()) + 1 if sel.any() else 0)
                    else:
                        a = 0
                    ks.append(a + 1); p += a + 1
            return float(np.mean(ks)), float(np.mean(lv))
        for K in (15, 31, 63, 127):
            rk = K + 1
            for D, tau in [(99, 0.0)] + [(d, 0.0) for d in (6, 8, 10, 12, 16, 20, 24)] + [(99, t) for t in (0.5, 0.3, 0.1, 0.03, 0.01)]:
                kept, lev = run(K, D, tau)
                ca = 25.0 * RF[rk] / RF[32] * CK[rk]; cb = RF[rk] * 62 * CK[rk]
                t = {f"{m}_{pl}": 1000 * kept / (c + lev * pl + 4) for m, c in (("a25", ca), ("bfloor", cb)) for pl in (0.917, 0.30)}
                nm = f"K{K:3d} " + (f"depth<{D}" if D < 99 else ("" if tau == 0 else f"p>={tau}"))
                rows.append({"name": nm.strip(), "K": K, "D": D, "tau": tau, "kept": kept, "levels": lev, **t})
        base = next(r for r in rows if r["K"] == 31 and r["D"] == 99 and r["tau"] == 0.0)
        say(f"  tree-32 (no cut): kept {base['kept']:.3f}  levels {base['levels']:.2f}  tok/s a25 {base['a25_0.917']:.0f} / {base['a25_0.3']:.0f}, floor {base['bfloor_0.917']:.0f} / {base['bfloor_0.3']:.0f}")
        for key, lab in (("a25_0.917", "25 ms model, today's level cost"), ("a25_0.3", "25 ms model, QAT + fused level"),
                         ("bfloor_0.917", "byte floor, today's level cost"), ("bfloor_0.3", "byte floor, QAT + fused level")):
            top = sorted(rows, key=lambda r: -r[key])[:4]
            say(f"  best under {lab}: " + "; ".join(f"{r['name']} {r[key]:.0f} (kept {r['kept']:.2f}, lv {r['levels']:.1f})" for r in top)
                + f"  | tree-32 {base[key]:.0f}")
        res[s] = {"rows": rows, "base": base}
    ok = all(max(r["a25_0.917"] for r in res[s]["rows"]) >= 1.10 * res[s]["base"]["a25_0.917"] for s in res)
    for s in res:
        bst = max(res[s]["rows"], key=lambda r: r["a25_0.917"])
        say(f"  {s}: best a25 today {bst['name']} {bst['a25_0.917']:.0f} vs tree-32 {res[s]['base']['a25_0.917']:.0f} (x{bst['a25_0.917'] / res[s]['base']['a25_0.917']:.3f})")
    say(f"\nLINE (>= +10 % tok/s over tree-32 under the 25 ms model at today's level cost, both sets): {'PASS' if ok else 'KILL'}")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
