"""CHAT ROUTE PRICE (NewIdeas, Wed 4 Nov 2026 20:0x; CPU only). THE SWEEP v1.2 says no store reaches chat PASS at 24 GB
under QOM cf 0; the lever left is the rule. Fable C's 1(c) (qom_0.5_live_bonus5g, measured 19:38) is QOM at cf 0.5 with
the bonus row gated by its own draft's cf at 0.5: 0.48 / 2.88 / 1.91 % on the map (45 GB pool, 4-bit sure rows).
This prices that rule's decode bytes at 24 GB on today's 11-bit parts, with the RAM split searched:
  rule (0.5, False) = sure rows read their top-2's LSB parts too (4-bit sure rows, the measured form);
  rule (0.5, True)  = "3b": sure rows read no LSB part (sure rows' top-2 at MSB-3, the t3 form);
  unsure rows read top-10 MSB + top-2 LSB (REF2-T's own 4433333333 reads); conv 'engine' = the bonus row is sure iff
  its next draft's cf >= 0.5 (bonus5g's rule).
RAM per layer = the 24 GB line's budget: Cm = (B - Cl x LR11) // MR11. ten_x_ledger.replay conventions (prefill warms
both pools; plain LRU per layer). Quality [G], verdict frame (vs REF2-T; additive KL from measured map arms):
  bonus5g 0.0010 / 0.0057 / 0.0039; + t3 term (qom_0.5_live_t3 - qom_0.5_live) for 3b; + the pool term at 164 slots
  (qom_0.5_live_c164 - qom_0.5_live), the nearest measured pool. Writes ~/NewIdeas/data/ledger/chat_route_price.txt."""
import os, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
import map_kl as M  # noqa: E402
MR11, LR11 = 2064384, 720896
B24 = 147 * 2162688 + 46 * 770048
CLS = [0, 23, 46, 69, 92, 138]
FIT = {"code": (-0.078, 15.68), "ifeval": (0.420, 39.04), "multi": (-0.173, 36.17)}
LINE = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}
DOMS = ("code", "ifeval", "multi")


def job(task):
    dom, cl, t3, conv = task[:4]; thr = task[4] if len(task) > 4 else 0.5
    cm = (B24 - cl * LR11) // MR11
    T.MR, T.LR = MR11, LR11
    T.POOLS["x"] = ([cm] * 48, [cl] * 48)
    T.live_blocks = lambda w: LL.blocks_conv(w, conv)
    _, mbk, _, _, _ = T.replay((dom, "x", "live", (thr, t3)))
    return task, (cm, mbk)


def fl(d, kl):
    a, b = FIT[d]; return a + b * max(kl, 0) ** 0.5


def vd(d, f):
    return "PASS" if f <= LINE[d] else ("BETWEEN" if f <= 1.5 * LINE[d] else "KILL")


def main():
    t0 = time.time(); T.W = T.windows()
    tasks = [(d, cl, t3, "engine") for d in DOMS for cl in CLS for t3 in (False, True)]
    tasks += [(d, 0, True, "all", 0.0) for d in DOMS]                  # today's cf 0 stack (thr 0), the scale check
    tasks += [(d, cl, False, "engine", None) for d in DOMS for cl in CLS]   # EXACT (REF2-T reads) at the same splits
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "4")), maxtasksperchild=1) as P:
        R = dict(P.map(job, tasks, chunksize=1))
    K = {n: M.arm(n) for n in ("qom_0.5_live", "qom_0.5_live_bonus5g", "qom_0.5_live_t3", "qom_0.5_live_c164")}
    kl = {n: {d: K[n][d][1] for d in DOMS} for n in K}
    L = ["CHAT ROUTE: QOM cf 0.5 + the bonus row gated at 0.5 (1(c)'s rule), 24 GB, 11-bit parts; SSD MB per kept token in decode [P]",
         "| Cl (LSB slots) | Cm (MSB slots) | 4-bit sure rows: code / IFEval / multi | 3b sure rows: code / IFEval / multi | EXACT (REF2-T reads) |",
         "|---|---|---|---|---|"]
    for cl in CLS:
        a = [R[(d, cl, False, "engine")] for d in DOMS]; b = [R[(d, cl, True, "engine")] for d in DOMS]
        L.append(f"| {cl} | {a[0][0]} | " + " / ".join(f"{x[1]:.1f}" for x in a) + " | " +
                 " / ".join(f"{x[1]:.1f}" for x in b) + " | " + " / ".join(f"{R[(d, cl, False, 'engine', None)][1]:.1f}" for d in DOMS) + " |")
    best = {t3: {d: min(CLS, key=lambda c: R[(d, c, t3, "engine")][1]) for d in DOMS} for t3 in (False, True)}
    for t3 in (False, True):
        L.append(f"best split, {'3b' if t3 else '4-bit'} sure rows: " + "; ".join(
            f"{d} {R[(d, best[t3][d], t3, 'engine')][1]:.1f} MB (Cl {best[t3][d]})" for d in DOMS))
    L.append("this replay's cf 0 FAST stack (QOM cf 0 + 3b + every last row gated, Cl 0 / Cm 171), for scale: " +
             " / ".join(f"{R[(d, 0, True, 'all', 0.0)][1]:.1f}" for d in DOMS) + " (trellis_draft17.txt: 19.5 / 14.3 / 10.6)")
    L += ["", "QUALITY [G], verdict frame (vs REF2-T), additive KL from measured map arms; the pool term is the 164-slot arm's",
          "| form | KL code / IFEval / multi | flips % | verdicts |", "|---|---|---|---|"]
    pool = {d: kl["qom_0.5_live_c164"][d] - kl["qom_0.5_live"][d] for d in DOMS}
    t3t = {d: kl["qom_0.5_live_t3"][d] - kl["qom_0.5_live"][d] for d in DOMS}
    for nm, extra in (("4-bit sure rows", {d: 0.0 for d in DOMS}), ("3b sure rows", t3t)):
        k = {d: kl["qom_0.5_live_bonus5g"][d] + extra[d] + pool[d] for d in DOMS}
        f = {d: fl(d, k[d]) for d in DOMS}
        L.append(f"| {nm} at ~164 slots | " + " / ".join(f"{k[d]:.4f}" for d in DOMS) + " | " +
                 " / ".join(f"{f[d]:.2f}" for d in DOMS) + " | " + " / ".join(vd(d, f[d]) for d in DOMS) + " |")
    L.append(f"(terms: pool at 164 slots " + " / ".join(f"{pool[d]:+.4f}" for d in DOMS) + "; t3 " +
             " / ".join(f"{t3t[d]:+.4f}" for d in DOMS) + f"; {time.time() - t0:.0f} s)")
    open(os.path.join(T.DST, "chat_route_price.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
