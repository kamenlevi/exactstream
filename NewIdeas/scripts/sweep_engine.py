"""SWEEP IN THE ENGINE FRAME (NewIdeas, Wed 4 Nov 2026 19:5x; CPU only, read-only on the map outputs).
THE SWEEP v1 (research/SWEEP.md) priced quality in the map's frame: the map runs D-02a's 4-bit model, so in C's form
the kept ranks 2-9 sit at 4 bits, while the engine's FAST pool serves them at MSB-3. Screen 7's arms measure what v1
guessed. This moves every store row to the engine frame from measured anchors:
  map(c): C's form + DRAFT17 at c MSB slots, piecewise linear in g = the keep gap closed (147 -> the 45 GB pool, PQ LRU
          MASS shares from data/ledger/trellis_draft17.txt) between three measured anchors:
          g = 0: screen 6 (qom_0_live_ball_c147_t3); g(239): s7_ref; g = 1: qom_0_live_bq_t3 + DRAFT17's measured code
          term (screen 6 - C at 147; chat 0).
  E29(c) = (s7_ref3 - s7_ref) x keep(c) / keep(239): the ranks 2-9 MSB-3 term the engine pays.
  A store with MSE ratio r vs MSB-3 (Fit12's E4 units: errors vs the 4-bit weights, MSB-3 = 3 E4):
          KL = map(c) + (r - 1) x T01 + r x E29(c), T01 = bq_t3 - bq (sure rows' top-2 at MSB-3 vs 4 bits).
  Check: MSB-2 (r = 5 in these units: 15 E4 vs 3 E4) at 239 = s7_ref + 4 T01 + 5 E29 vs the measured s7_cal_all.
flips = the pre-registered fit a + b sqrt(KL); lines PASS <= 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x, KILL above.
FRAME FIX (19:5x): the FAST stack certifies against REF2-T (EvalSuite PREREG_f1), and REF2-T serves ranks 2-9 at 3 bits
(4433333333). So MSB-3 kept ranks 2-9 match the reference: the verdict frame is amendment 2's in-frame one,
  KL_in = map(c) + (r - 1) x (T01 + E29(c))   (= KL_ref + (est_abs - KL_ref3) for a store over all 48 layers),
and the formula above (KL_abs, vs the 4-bit model) is the conservative column, as amendment 2 reports est_abs beside it.
MoonShots (cd7e1f0): the credit view is the expected-score view; disagreement with REF2-T for an INDEPENDENT quantizer
(TRL / GPTQ / the mix, errors not nested in MSB-3) is (r + 1) x E29 at ranks 2-9: KL_strict = map(c) + (r - 1) x T01 +
(r + 1) x E29(c). MSB-3 (row 0) is the reference's own bits: 0 under every view. Both views print for rows 1-4.
Writes ~/NewIdeas/data/ledger/sweep_engine.txt."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_kl as M  # noqa: E402
DOMS = ("code", "ifeval", "multi")
FIT = {"code": (-0.078, 15.68), "ifeval": (0.420, 39.04), "multi": (-0.173, 36.17)}
LINE = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}
# trellis_draft17.txt (24 GB, DRAFT17 'all'): keep % and g (keep gap closed 147 -> 45 GB pool) per slot count
KEEP = {171: (80.2, 82.6, 83.6), 185: (82.0, 83.9, 84.7), 239: (87.9, 88.2, 88.2), 276: (91.1, 90.3, 89.8)}
G = {147: (0.0, 0.0, 0.0), 171: (0.21, 0.23, 0.25), 185: (0.32, 0.36, 0.38), 239: (0.67, 0.75, 0.81),
     276: (0.87, 0.94, 1.00)}
ROWS = [("0 today's FAST stack (MSB-3 11-bit, POOLSPLIT, DRAFT17)", 1.0, 171, "1.12 / 7.67 / 6.01"),
        ("1 B10 TRL-2 (2.06 b)", 1.854, 276, "1.27 / 7.96 / 7.09"),
        ("2a the 2.40 b mix, r 1.62", 1.62, 239, "1.23 / 7.88 / 6.81"),
        ("2b the 2.40 b mix, r 1.87", 1.87, 239, "1.32 / 8.34 / 7.27"),
        ("3 TRL-3 (3INST 3.07 b)", 0.52, 185, "0.87 / 6.33 / 4.66"),
        ("4 GPTQ-3 g64", 0.61, 171, "0.96 / 6.87 / 5.09")]


def fl(d, kl):
    a, b = FIT[d]; return a + b * max(kl, 0) ** 0.5


def vd(d, f):
    return "PASS" if f <= LINE[d] else ("BETWEEN" if f <= 1.5 * LINE[d] else "KILL")


def main():
    K = {n: M.arm(n) for n in ("qom_0_live_ball_c147_t3", "qom_0_live_bq_c147_t3", "s7_ref", "s7_ref3", "s7_cal_all",
                               "qom_0_live_bq_t3", "qom_0_live_bq")}
    kl = {n: {d: K[n][d][1] for d in DOMS} for n in K}
    i = {d: n for n, d in enumerate(DOMS)}
    T01 = {d: kl["qom_0_live_bq_t3"][d] - kl["qom_0_live_bq"][d] for d in DOMS}
    D17 = {d: kl["qom_0_live_ball_c147_t3"][d] - kl["qom_0_live_bq_c147_t3"][d] for d in DOMS}
    D17["ifeval"] = D17["multi"] = 0.0 if abs(D17["ifeval"]) < 1e-4 else D17["ifeval"]   # chat: no measured term
    E29 = {d: kl["s7_ref3"][d] - kl["s7_ref"][d] for d in DOMS}
    A0 = {d: kl["qom_0_live_ball_c147_t3"][d] for d in DOMS}
    A1 = {d: kl["s7_ref"][d] for d in DOMS}
    A2 = {d: kl["qom_0_live_bq_t3"][d] + D17[d] for d in DOMS}
    g239 = {d: G[239][i[d]] for d in DOMS}

    def mapkl(d, c):
        g = G[c][i[d]]
        if g <= g239[d]: return A0[d] + (A1[d] - A0[d]) * g / g239[d]
        return A1[d] + (A2[d] - A1[d]) * (g - g239[d]) / (1 - g239[d])

    def e29(d, c): return E29[d] * KEEP[c][i[d]] / KEEP[239][i[d]]

    L = ["SWEEP IN THE ENGINE FRAME (24 GB, Cl 0, every store's pool at its slot count; KL(REF2-T || arm), [G] from "
         "measured anchors)",
         "anchors (map frame, C's form + DRAFT17): " + "; ".join(
             f"{d} 147 {A0[d]:.4f} / 239 {A1[d]:.4f} / 45 GB {A2[d]:.4f}" for d in DOMS),
         "terms: T01 " + " / ".join(f"{T01[d]:.4f}" for d in DOMS) + "; E29 at 239 " + " / ".join(
             f"{E29[d]:.4f}" for d in DOMS) + "; DRAFT17 " + " / ".join(f"{D17[d]:+.4f}" for d in DOMS),
         "pool term check, measured KL gap closed at 239 vs the keep proxy g: " + "; ".join(
             f"{d} {(A0[d] - A1[d]) / (A0[d] - A2[d]):.2f} vs {g239[d]:.2f}" for d in DOMS)]
    chk = {d: A1[d] + 4 * T01[d] + 5 * E29[d] for d in DOMS}
    L.append("check, MSB-2 at 239 (r 5): model " + " / ".join(f"{chk[d]:.4f}" for d in DOMS) + " vs s7_cal_all " +
             " / ".join(f"{kl['s7_cal_all'][d]:.4f}" for d in DOMS) + " (implied r: " + " / ".join(
                 f"{(kl['s7_cal_all'][d] - A1[d] + T01[d]) / (T01[d] + E29[d]):.2f}" for d in DOMS) + ")")
    L += ["", "| row | r (MSE vs MSB-3) | slots | in-frame KL code / IFEval / multi | in-frame flips % (VERDICT) | "
          "verdicts | strict flips % (independent quantizer vs REF2-T) | strict verdicts | absolute flips % (vs the 4-bit model) | v1 (map frame, guessed terms) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r, c, v1 in ROWS:
        k = {d: mapkl(d, c) + (r - 1) * (T01[d] + e29(d, c)) for d in DOMS}
        ka = {d: mapkl(d, c) + (r - 1) * T01[d] + r * e29(d, c) for d in DOMS}
        ks = {d: mapkl(d, c) + (r - 1) * T01[d] + ((r + 1) * e29(d, c) if r != 1.0 else 0.0) for d in DOMS}
        fs = {d: fl(d, ks[d]) for d in DOMS}
        f = {d: fl(d, k[d]) for d in DOMS}
        L.append(f"| {name} | {r} | {c} | " + " / ".join(f"{k[d]:.4f}" for d in DOMS) + " | " +
                 " / ".join(f"{f[d]:.2f}" for d in DOMS) + " | " + " / ".join(vd(d, f[d]) for d in DOMS) +
                 " | " + " / ".join(f"{fs[d]:.2f}" for d in DOMS) + " | " + " / ".join(vd(d, fs[d]) for d in DOMS) +
                 " | " + " / ".join(f"{fl(d, ka[d]):.2f}" for d in DOMS) + f" | {v1} |")
    L.append("(E29's window-cluster SE on code is 0.00065 (z 1.8; s7_read.txt): code rows carry about +-0.1 flip points "
             "from it; chat terms are z 8-10)")
    open(os.path.expanduser("~/NewIdeas/data/ledger/sweep_engine.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
