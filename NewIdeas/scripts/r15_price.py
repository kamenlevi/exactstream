"""R15 / R15b BYTES, an independent re-derivation of Cloud3's claim (NewIdeas, Thu 5 Nov 2026 07:0x; CPU only, one process).
Cloud3 (REPORT_P3 sections 21 / 25, its sim_lru4 / sim_lru5, MSB records 2.16 MB, its own pool convention): the chat
route at 24 GB 46.5 / 70.5 / 75.4 MB per kept token -> + u8 43.2 / 57.2 / 61.7 (x0.93 / 0.81 / 0.82) -> + u8 + reach 0.5
40.2 / 45.4 / 48.4 (x0.86 / 0.64 / 0.64). This is my replay (chat_route_price.py's: 24 GB, 11-bit parts, 4-bit sure rows,
best RAM split per domain), with a per-row read depth:
  sure rows (drafted cf >= 0.5; the bonus row when its next draft's cf >= 0.5) read top-2;
  unsure rows read top-10 (the route), top-8 (+ u8), or top-2 when their reach (the product of the check's draft cf
  before them) is < 0.5 (+ reach); union members and resident experts cost no read.
Re-derived if my ratios are within ~10 % of Cloud3's. Writes ~/NewIdeas/data/ledger/r15_price.txt."""
import os, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
import chat_route_price as CR  # noqa: E402
DOMS = ("code", "ifeval", "multi")
THR = 0.5
CLOUD3 = {"route": (46.5, 70.5, 75.4), "u8": (43.2, 57.2, 61.7), "u8 + reach 0.5": (40.2, 45.4, 48.4)}


def blocks(w, ku, tau):
    """per live segment: rows, per-row read depth k, kept count (LL.blocks_conv's rows; the bonus gate at THR)."""
    g = w["g"]; N = len(w["ids"]); bl = []
    for (k0_, m0_, A0_), dc0_ in zip(w["checks"], w["dcf"]):
        for (k, m, A, dc, nxt) in LL.segments_nxt(k0_, m0_, A0_, dc0_):
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            reach = np.r_[1.0, np.cumprod(np.array(dc, float))]
            rows, ks = [], []
            for j, p in enumerate(range(A, last + 1)):
                is_last = p - A == m
                if p <= fm: rows.append(p); c = -1.0 if (k == m and is_last) else g[p]
                else: rows.append((p + 97) % N); c = dc[p - A] if p - A < m else -1.0
                if is_last: c = 1.0 if nxt >= THR else -1.0
                kk = 2 if c >= THR else ku
                if kk == ku and tau is not None and reach[min(j, len(reach) - 1)] < tau: kk = 2
                ks.append(kk)
            if rows: bl.append((rows, ks, min(k, fm - A) + 1))
    return bl


def replay(dom, cl, ku, tau):
    cm = (CR.B24 - cl * CR.LR11) // CR.MR11; MR, LR = CR.MR11, CR.LR11
    bp = [OrderedDict() for _ in range(48)]; lp = [OrderedDict() for _ in range(48)]
    mb = ml = kept = 0
    for w in T.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, cm)
                for e in ids[p, l, :2].tolist(): T.touch(lp[l], e, cl)
        for rows, ks, kp in blocks(w, ku, tau):
            for l in range(48):
                Ub, Ul = set(), set()
                for r, kk in zip(rows, ks):
                    Ub.update(ids[r, l, :kk].tolist()); Ul.update(ids[r, l, :2].tolist())
                mb += sum(1 for e in Ub if e not in bp[l]); ml += sum(1 for e in Ul if e not in lp[l])
                for e in Ub: T.touch(bp[l], e, cm)
                for e in Ul: T.touch(lp[l], e, cl)
            kept += kp
    return (mb * MR + ml * LR) / max(kept, 1) / 1e6


def main():
    t0 = time.time(); T.W = T.windows()
    R = {}
    for name, ku, tau in (("route", 10, None), ("u8", 8, None), ("u8 + reach 0.5", 8, 0.5)):
        R[name] = {d: min((replay(d, cl, ku, tau), cl) for cl in CR.CLS) for d in DOMS}
    L = ["R15 / R15b BYTES, re-derived (24 GB, 11-bit parts, 4-bit sure rows, best split; SSD MB per kept token in decode [P])",
         "| arm | code | IFEval | multi | x the route (mine) | x the route (Cloud3) |", "|---|---|---|---|---|---|"]
    for name in R:
        mine = [R[name][d][0] / R["route"][d][0] for d in DOMS]; c3 = [CLOUD3[name][j] / CLOUD3["route"][j] for j in range(3)]
        L.append(f"| {name} | " + " | ".join(f"{R[name][d][0]:.1f} (Cl {R[name][d][1]})" for d in DOMS) + " | " +
                 " / ".join(f"{x:.2f}" for x in mine) + " | " + " / ".join(f"{x:.2f}" for x in c3) + " |")
    L.append(f"check: the route here = chat_route_price.txt's 4-bit best split (61.8 / 87.9 / 87.2) ({time.time() - t0:.0f} s)")
    open(os.path.join(T.DST, "r15_price.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
