"""TRELLIS RESCUE PRICE (NewIdeas, Wed 4 Nov 2026 22:3x; CPU only, read-only on the map outputs). The orchestrator's
22:3x ask after screen 7 (the 2.40 b mix = BETWEEN / KILL / KILL in-frame): price a SELECTIVE trellis store (the mix on
part of the model, MSB-3 elsewhere) and TRL-3, at 24 GB and at the 45 GB line all-resident. All quality numbers are [G]:
the SWEEP's model (sweep_engine.py) with screen 7's MEASURED damage rate for the mix instead of its MSE ratio:
  r_eff = (q (4 T01 + 5 E29) + T01) / (T01 + E29), q = (s7_trl - s7_ref) / (s7_cal - s7_ref) at the same 4 layers.
Sizes from bits per expert parameter (4,718,592 params per expert-layer, 24,576 expert-layers; MSB-3 part 2,064,384 B =
3.5 b): MSB-3 3.5 b, the mix 2.40 b, TRL-2 2.06 b, TRL-3 3.07 b. Expert budget at a RAM line = the line minus ~7.0 GB
(the 24 GB line's split leaves 16.96 GB for experts).
  24 GB, per layer: a fraction phi of layers holds the mix (its 239-slot pool and bytes), the rest today's FAST stack
  (MSB-3 parts, 171 slots); per-layer LRU, so bytes and KL mix linearly in phi:
    KL(phi) = (1 - phi) mapkl(171) + phi [mapkl(239) + (r_eff - 1)(T01 + E29(239))];
    bytes(phi) = (1 - phi) x today's + phi x the mix's (SWEEP rows 0 and 2, [P]).
  45 GB all-resident (zero SSD reads), vs REF2-T (ranks 0-1 at 4 bits, 2-9 at MSB-3): an MSB-3 expert costs T01 (its
    ranks 0-1 lose the LSB), a mix expert r_eff T01 + (r_eff - 1) E29 / keep(239):
    KL(phi) = T01 + phi (r_eff - 1)(T01 + E29 / keep(239)); GB(phi) = 50.7 - phi x (50.7 - 34.8).
  TRL-3 at 24 GB: SWEEP row 3 (185 slots, bytes [P]) with r between its MSE ratio 0.52 and 0.52 x the mix's miss factor
    (r_eff / r_MSE, r_MSE 1.62-1.87).
Writes ~/NewIdeas/data/ledger/rescue_price.txt."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_kl as M  # noqa: E402
import s7_read as S  # noqa: E402
from sweep_engine import FIT, LINE, KEEP, G  # noqa: E402
DOMS = ("code", "ifeval", "multi")
NPAR, NEL = 4718592, 512 * 48
GB = {b: NPAR * NEL * b / 8 / 1e9 for b in (3.5, 2.40, 2.06, 3.07)}
BYTES = {"today": (19.5, 14.3, 10.6), "mix": (7.2, 5.0, 4.1), "trl3": (15.8, 11.4, 8.5)}   # SWEEP rows 0 / 2 / 3, [P]
REST = 24.0 - 16.96


def fl(d, kl):
    a, b = FIT[d]; return a + b * max(kl, 0) ** 0.5


def vd(d, f):
    return "PASS" if f <= LINE[d] else ("BETWEEN" if f <= 1.5 * LINE[d] else "KILL")


def main():
    i = {d: n for n, d in enumerate(DOMS)}
    K = {n: M.arm(n) for n in ("qom_0_live_ball_c147_t3", "qom_0_live_bq_c147_t3", "s7_ref", "qom_0_live_bq_t3", "qom_0_live_bq")}
    kl = {n: {d: K[n][d][1] for d in DOMS} for n in K}
    A = {n: S.per_window(n) for n in ("s7_ref", "s7_ref3", "s7_cal", "s7_trl")}
    P = {n: S.paired(A[n], A["s7_ref"]) for n in ("s7_ref3", "s7_cal", "s7_trl")}
    T01 = {d: kl["qom_0_live_bq_t3"][d] - kl["qom_0_live_bq"][d] for d in DOMS}
    E29 = {d: P["s7_ref3"][d][0] for d in DOMS}
    D17 = {d: (kl["qom_0_live_ball_c147_t3"][d] - kl["qom_0_live_bq_c147_t3"][d]) if d == "code" else 0.0 for d in DOMS}
    A0 = {d: kl["qom_0_live_ball_c147_t3"][d] for d in DOMS}; A1 = {d: kl["s7_ref"][d] for d in DOMS}
    A2 = {d: kl["qom_0_live_bq_t3"][d] + D17[d] for d in DOMS}

    def mapkl(d, c):
        g, g1 = G[c][i[d]], G[239][i[d]]
        return A0[d] + (A1[d] - A0[d]) * g / g1 if g <= g1 else A1[d] + (A2[d] - A1[d]) * (g - g1) / (1 - g1)

    def e29(d, c): return E29[d] * KEEP[c][i[d]] / KEEP[239][i[d]]
    q = {d: P["s7_trl"][d][0] / P["s7_cal"][d][0] for d in DOMS}
    r = {d: (q[d] * (4 * T01[d] + 5 * E29[d]) + T01[d]) / (T01[d] + E29[d]) for d in DOMS}
    L = ["TRELLIS RESCUE PRICE (orchestrator 22:3x); quality [G] from screen 7's measured damage rate; bytes [P] from SWEEP rows",
         "r_eff (the mix, measured at screen 7's 4 layers): " + " / ".join(f"{r[d]:.2f}" for d in DOMS) + " vs MSE 1.62-1.87",
         f"expert store sizes (GB): MSB-3 {GB[3.5]:.1f}, the mix {GB[2.40]:.1f}, TRL-2 {GB[2.06]:.1f}, TRL-3 {GB[3.07]:.1f}; "
         f"rest of the footprint ~{REST:.1f} GB (24 GB split), so experts get ~{45 - REST:.0f} GB at the 45 GB line", "",
         "A. 24 GB, the mix on a fraction phi of layers, MSB-3 (today's FAST stack) elsewhere:",
         "| phi | code / IFEval / multi flips % [G] | verdicts | MB per kept token [P-mix] | expert store GB (SSD) |", "|---|---|---|---|---|"]
    best = None
    for phi in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 1.0):
        k = {d: (1 - phi) * mapkl(d, 171) + phi * (mapkl(d, 239) + (r[d] - 1) * (T01[d] + e29(d, 239))) for d in DOMS}
        f = {d: fl(d, k[d]) for d in DOMS}; b = [(1 - phi) * BYTES["today"][j] + phi * BYTES["mix"][j] for j in range(3)]
        if f["code"] <= LINE["code"]: best = (phi, b[0])
        L.append(f"| {phi} | " + " / ".join(f"{f[d]:.2f}" for d in DOMS) + " | " + " / ".join(vd(d, f[d]) for d in DOMS) +
                 " | " + " / ".join(f"{x:.1f}" for x in b) + f" | {(1 - phi) * GB[3.5] + phi * GB[2.40]:.1f} |")
    L.append(f"largest phi on this grid keeping code's line: {best[0]} ({best[1]:.1f} MB on code); chat is over its lines at every phi "
             "(today's stack is already BETWEEN / KILL on chat; the chat PASS route is the cf 0.5 rule, which needs 4-bit top-2 = MSB-3 + LSB parts)")
    L += ["", "B. 45 GB line all-resident (zero SSD reads), the mix on a fraction phi of expert-layers, MSB-3 elsewhere:",
          "| phi | expert GB | fits ~38 GB? | code / IFEval / multi flips % [G] | verdicts |", "|---|---|---|---|---|"]
    for phi in (0.0, 0.5, 0.8, 1.0):
        k = {d: T01[d] + phi * (r[d] - 1) * (T01[d] + E29[d] / (KEEP[239][i[d]] / 100)) for d in DOMS}
        f = {d: fl(d, k[d]) for d in DOMS}; gb = (1 - phi) * GB[3.5] + phi * GB[2.40]
        L.append(f"| {phi} | {gb:.1f} | {'yes' if gb <= 45 - REST else 'no (' + format(100 * (45 - REST) / gb, '.0f') + ' % resident)'} | " +
                 " / ".join(f"{f[d]:.2f}" for d in DOMS) + " | " + " / ".join(vd(d, f[d]) for d in DOMS) + " |")
    L.append("phi 0 (all MSB-3, no trellis) already misses multi: REF2-T keeps ranks 0-1 at 4 bits, so any store without "
             "4-bit ranks 0-1 pays T01 (" + " / ".join(f"{T01[d]:.4f}" for d in DOMS) + " KL)")
    L += ["", "C. TRL-3 (3.07 b) at 24 GB, SWEEP row 3 (185 slots), r between its MSE ratio and the mix's miss factor:",
          "| r | code / IFEval / multi flips % [G] | verdicts | MB per kept token [P] |", "|---|---|---|---|"]
    miss = (min(r[d] for d in DOMS) / 1.87, max(r[d] for d in DOMS) / 1.62)
    for rr in (0.52, round(0.52 * miss[0], 2), round(0.52 * miss[1], 2)):
        k = {d: mapkl(d, 185) + (rr - 1) * (T01[d] + e29(d, 185)) for d in DOMS}; f = {d: fl(d, k[d]) for d in DOMS}
        L.append(f"| {rr} | " + " / ".join(f"{f[d]:.2f}" for d in DOMS) + " | " + " / ".join(vd(d, f[d]) for d in DOMS) +
                 " | " + " / ".join(f"{x}" for x in BYTES["trl3"]) + " |")
    k = {d: mapkl(d, 171) for d in DOMS}
    L.append("today's FAST stack for comparison (r 1, 171 slots): " + " / ".join(f"{fl(d, k[d]):.2f}" for d in DOMS) +
             " % at " + " / ".join(str(x) for x in BYTES["today"]) + f" MB; TRL-3 at 45 GB: {GB[3.07]:.1f} GB of experts "
             f"({100 * (45 - REST) / GB[3.07]:.0f} % resident at ~38 GB)")
    open(os.path.expanduser("~/NewIdeas/data/ledger/rescue_price.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
