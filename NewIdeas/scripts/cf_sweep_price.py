"""CF SWEEP, BYTES SIDE (NewIdeas, Wed 4 Nov 2026 22:0x; CPU only, one process). The chat PASS route (QUEUE 20:0x / 20:1x)
uses QOM at cf 0.5 + the bonus row gated at 0.5 and costs ~88 MB per kept token on chat at 24 GB [P], while cf 0 (the FAST
stack) is ~10-14 MB but BETWEEN / KILL on chat. Idea: an intermediate threshold may pass chat for far fewer bytes.
Cheapest kill test first, bytes only (quality needs map arms): the same replay as chat_route_price.py (24 GB, 11-bit
parts, 4-bit sure rows, conv 'engine' = the bonus row gated by its own next draft's cf at the same threshold), at
cf 0.1 / 0.2 / 0.3 / 0.4 / 0.5, best RAM split per domain.
Pre-registered (22:0x, before the run): KILL if cf 0.3's best chat bytes are >= 0.8x cf 0.5's on both IFEval and multi
(a lever too small to chase); ALIVE (worth a quality estimate or a map arm) if <= 0.6x on both; between = weak.
Also prints the map's measured quality at cf 0 / 0.5 / 0.9 (45 GB pool, 4-bit sure rows) for the curve's shape.
Writes ~/NewIdeas/data/ledger/cf_sweep_price.txt."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import chat_route_price as CR  # noqa: E402
import map_kl as M  # noqa: E402
DOMS = ("code", "ifeval", "multi")
THRS = (0.1, 0.2, 0.3, 0.4, 0.5)


def main():
    t0 = time.time(); T.W = T.windows()
    R = {}
    for thr in THRS:
        for d in DOMS:
            for cl in CR.CLS:
                task, v = CR.job((d, cl, False, "engine", thr)); R[task] = v
    best = {(thr, d): min(CR.CLS, key=lambda c: R[(d, c, False, "engine", thr)][1]) for thr in THRS for d in DOMS}
    mb = {(thr, d): R[(d, best[(thr, d)], False, "engine", thr)][1] for thr in THRS for d in DOMS}
    L = ["CF SWEEP, BYTES: QOM at cf t + the bonus row gated at t, 24 GB, 11-bit parts, 4-bit sure rows; SSD MB per kept token "
         "in decode [P], best split per domain (Cl)", "| cf t | code | IFEval | multi | x cf 0.5 (code / IFEval / multi) |",
         "|---|---|---|---|---|"]
    for thr in THRS:
        L.append(f"| {thr} | " + " | ".join(f"{mb[(thr, d)]:.1f} (Cl {best[(thr, d)]})" for d in DOMS) + " | " +
                 " / ".join(f"{mb[(thr, d)] / mb[(0.5, d)]:.2f}" for d in DOMS) + " |")
    r3 = {d: mb[(0.3, d)] / mb[(0.5, d)] for d in ("ifeval", "multi")}
    v = ("KILL" if all(x >= 0.8 for x in r3.values()) else "ALIVE" if all(x <= 0.6 for x in r3.values()) else "WEAK")
    L.append(f"pre-registered read (cf 0.3 vs cf 0.5, chat): IFEval x{r3['ifeval']:.2f}, multi x{r3['multi']:.2f} -> {v}")
    L.append(f"check: cf 0.5 here = chat_route_price.txt's 4-bit best split (61.8 / 87.9 / 87.2)")
    L += ["", "map quality, measured (45 GB pool, 4-bit sure rows, bonus row not gated), flips % / KL:"]
    for n in ("qom_0_live", "qom_0.5_live", "qom_0.9_live"):
        a = M.arm(n); L.append(f"  {n}: " + "; ".join(f"{d} {a[d][0]:.2f} / {a[d][1]:.4f}" for d in DOMS))
    L.append(f"({time.time() - t0:.0f} s)")
    open(os.path.join(T.DST, "cf_sweep_price.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
