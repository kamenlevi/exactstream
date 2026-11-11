"""QUEUE B9 + B7 (NewIdeas Sun ~06:00): SSD bytes per token at a 24 GB pool budget, all 512 experts, exact top-10 routing.
125B traces (Breakthrough's expA traces125: bulgarian / chat / code / math, 800 tokens, rank-ordered top-20 per layer).
Per layer, LRU pools under the byte budget of today's 24 GB split (C_m 147 MSB-3 + C_l 46 LSB-1 = 353.3 MB per layer):
  A0  today's split, plan 4433333333 (ranks 0-1 need MSB + LSB, ranks 2-9 MSB)
  A   the same plan, split rebalanced (C_l swept, C_m from the rest of the budget)
  B   + a 2-bit WARM tier (HOBBIT-style multi-precision residency): ranks 4-9 use the MSB-3 copy if resident, else the
      2-bit copy if resident, else read the 2-bit copy (Q2 bytes); ranks 0-3 as in A (2-bit never serves them);
      C_l and C_2 swept, C_m from the rest of the budget
  C   reference plans: 3333333333 (one MSB pool) and 4444444444 (one full-record pool)
Each set: tokens 0-399 warm the pools (not counted), tokens 400-799 measured. Per-token replay (misses are drafting-
invariant; NewIdeas) plus 16-row checks accepting 10 tokens (union per check) for the best configs.
usage: budget24.py [Q2_bytes]   (CPU only)"""
import glob, os, sys, time
from collections import OrderedDict
import numpy as np

NL, E = 48, 512
MSB, LSB = 2162688, 770048                      # records3 parts (index.json): msb_sz, lsb_sz
Q2 = int(sys.argv[1]) if len(sys.argv) > 1 else 1540096   # planar MSB-2 + bf16 g64 scales / biases, 16 KB aligned
BUD = 147 * MSB + 46 * LSB                      # today's 24 GB split per layer (353.3 MB)
T = "/Users/kamen/Breakthrough/data/expA/traces125/"
SETS = {os.path.basename(f).split("_")[0]: np.load(f) for f in sorted(glob.glob(T + "*_800_g.npz"))}
IDS = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1)[:, :, :10].astype(np.int32) for k, d in SETS.items()}


class LRU:
    __slots__ = ("C", "od")

    def __init__(self, C):
        self.C, self.od = C, OrderedDict()

    def has(self, e):
        return e in self.od

    def touch(self, e):                         # hit: refresh; miss: insert (evict LRU); returns 1 on a miss
        od = self.od
        if e in od:
            od.move_to_end(e)
            return 0
        if self.C <= 0:
            return 1
        od[e] = True
        if len(od) > self.C:
            od.popitem(last=False)
        return 1


def replay(ids, cfg, R=1, A=1):
    """cfg: dict(plan, Cm, Cl, C2, Cf). R rows per check, A tokens accepted per check. Returns (bytes per token, stats)."""
    n = ids.shape[0]
    plan = cfg["plan"]
    pm = [LRU(cfg.get("Cm", 0)) for _ in range(NL)]
    pl = [LRU(cfg.get("Cl", 0)) for _ in range(NL)]
    p2 = [LRU(cfg.get("C2", 0)) for _ in range(NL)]
    pf = [LRU(cfg.get("Cf", 0)) for _ in range(NL)]
    st = dict(bytes=0, msb=0, lsb=0, q2=0, full=0, low_uses=0, low_2bit=0, tok=0)
    pos = 0
    while pos + R <= n:
        meas = pos >= n // 2
        for l in range(NL):
            need = {}                                # expert -> best (lowest) rank among the check's rows
            for r in range(R):
                for k, e in enumerate(ids[pos + r, l]):
                    e = int(e)
                    if e not in need or k < need[e]:
                        need[e] = k
            for e, k in need.items():
                b = 0
                if plan == "4444444444":
                    b += pf[l].touch(e) * (MSB + LSB); st["full"] += b > 0
                elif plan == "3333333333":
                    m = pm[l].touch(e); b += m * MSB; st["msb"] += m
                elif plan == "3B" and k < 4:         # 3B: plan 3333333333, ranks 4-9 cold -> 2-bit
                    m = pm[l].touch(e); b += m * MSB; st["msb"] += m
                elif k < 2 and plan != "3B":         # 4-bit ranks: MSB + LSB
                    m = pm[l].touch(e); s = pl[l].touch(e); b += m * MSB + s * LSB; st["msb"] += m; st["lsb"] += s
                elif k < 4 or plan == "4433333333":  # 3-bit ranks (plans 4433333333, B, Bnp)
                    m = pm[l].touch(e); b += m * MSB; st["msb"] += m
                else:                                # plan "B" / "Bnp" / "3B": ranks 4-9 take MSB-3 if resident, else 2-bit
                    st["low_uses"] += meas
                    if pm[l].has(e):
                        if plan != "Bnp":                # Bnp: a low-rank hit does not refresh the MSB LRU
                            pm[l].touch(e)
                    else:
                        q = p2[l].touch(e); b += q * Q2; st["q2"] += q; st["low_2bit"] += meas
                if meas:
                    st["bytes"] += b
        if meas:
            st["tok"] += A
        pos += A
    return st["bytes"] / max(1, st["tok"]), st


def mean_bpt(cfg, R=1, A=1):
    out = {k: replay(v, cfg, R, A) for k, v in IDS.items()}
    return float(np.mean([b for b, _ in out.values()])), out


def fit(Cl, C2=0, q2=Q2):
    return int((BUD - Cl * LSB - C2 * q2) // MSB)


if __name__ == "__main__":
    t0 = time.time()
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say(f"budget per layer {BUD / 1e6:.1f} MB (C_m 147 + C_l 46); parts MSB {MSB}, LSB {LSB}, Q2 {Q2}")
    a0, o0 = mean_bpt(dict(plan="4433333333", Cm=147, Cl=46))
    say(f"A0 today 147 / 46: {a0 / 1e6:.1f} MB/token  (" + ", ".join(f"{k} {b / 1e6:.1f}" for k, (b, _) in o0.items()) + ")")
    best_a = (a0, 46)
    for Cl in range(0, 161, 8):
        b, _ = mean_bpt(dict(plan="4433333333", Cm=fit(Cl), Cl=Cl))
        say(f"A  C_m {fit(Cl)} C_l {Cl}: {b / 1e6:.1f} MB/token ({100 * (b / a0 - 1):+.1f} %)")
        best_a = min(best_a, (b, Cl))
    say(f"A best: C_l {best_a[1]} (C_m {fit(best_a[1])}): {best_a[0] / 1e6:.1f} MB/token ({100 * (best_a[0] / a0 - 1):+.1f} %)")
    best_b = None
    for Cl in (16, 32, 48, 64, 96):
        for C2 in (0, 32, 64, 96, 128, 160, 192, 224):
            Cm = fit(Cl, C2)
            if Cm < 16:
                continue
            b, o = mean_bpt(dict(plan="B", Cm=Cm, Cl=Cl, C2=C2))
            lu = sum(s["low_uses"] for _, s in o.values()); l2 = sum(s["low_2bit"] for _, s in o.values())
            say(f"B  C_m {Cm} C_l {Cl} C_2 {C2}: {b / 1e6:.1f} MB/token ({100 * (b / a0 - 1):+.1f} %), rank 4-9 uses at 2-bit "
                f"{100 * l2 / max(1, lu):.0f} %")
            if best_b is None or b < best_b[0]:
                best_b = (b, Cm, Cl, C2, l2 / max(1, lu))
    say(f"B best: C_m {best_b[1]} C_l {best_b[2]} C_2 {best_b[3]}: {best_b[0] / 1e6:.1f} MB/token "
        f"({100 * (best_b[0] / a0 - 1):+.1f} %), rank 4-9 uses at 2-bit {100 * best_b[4]:.0f} %")
    c3, _ = mean_bpt(dict(plan="3333333333", Cm=int(BUD // MSB)))
    c4, _ = mean_bpt(dict(plan="4444444444", Cf=int(BUD // (MSB + LSB))))
    say(f"C  3333333333 C {int(BUD // MSB)}: {c3 / 1e6:.1f} MB/token ({100 * (c3 / a0 - 1):+.1f} %); 4444444444 C "
        f"{int(BUD // (MSB + LSB))}: {c4 / 1e6:.1f} MB/token ({100 * (c4 / a0 - 1):+.1f} %)")
    for lab, cfg in (("A0", dict(plan="4433333333", Cm=147, Cl=46)), ("A best", dict(plan="4433333333", Cm=fit(best_a[1]), Cl=best_a[1])),
                     ("B best", dict(plan="B", Cm=best_b[1], Cl=best_b[2], C2=best_b[3]))):
        b16, _ = mean_bpt(cfg, R=16, A=10)
        say(f"16-row checks (10 accepted): {lab}: {b16 / 1e6:.1f} MB/token")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser(f"~/MoonShots/data/budget24_q{Q2}.txt"), "w").write("\n".join(lines) + "\n")
