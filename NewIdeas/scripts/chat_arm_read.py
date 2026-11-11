"""CHAT ARM READER (NewIdeas, Wed 4 Nov 2026 21:4x; CPU only, read-only on the map outputs). The kill test pre-registered
in QUEUE.md 20:1x: arm qom_0.5_live_c147_bonus5g (QOM cf 0.5 + the bonus row gated by its own draft's cf at 0.5, the
24 GB line's 147 MSB slots, 4-bit sure rows). Verdict on measured flips (top-1 changed vs REF2-T's baseline rows):
  PASS <= 1.34 / 5.42 / 3.69 % on every domain = the chat route holds at 24 GB (bytes 61.8 / 87.9 / 87.2 MB per kept token
  [P], chat_route_price.txt); BETWEEN <= 1.5x the line; KILL above. Prediction (20:1x, [G] additive): ~0.7 / ~4.6 / ~3.0 %.
For information: the same rule at the 45 GB pool (qom_0.5_live_bonus5g, Fable C's 1(c)), cf 0.5 at 164 slots
(qom_0.5_live_c164) and cf 0.5 at the 45 GB pool (qom_0.5_live); paired flips vs 1(c) (McNemar, exact two-sided).
Usage: python chat_arm_read.py [arm]   (default the cf 0.5 arm -> data/ledger/chat_arm_read.txt; another arm ->
data/ledger/chat_arm_read_<arm>.txt). 22:0x: also reads the cf 0.3 arm (QUEUE 22:0x, [G] range in PRED)."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_kl as M  # noqa: E402
DOMS = ("code", "ifeval", "multi")
LINE = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}
ARM = sys.argv[1] if len(sys.argv) > 1 else "qom_0.5_live_c147_bonus5g"
PREDS = {"qom_0.5_live_c147_bonus5g": {"code": "~0.7", "ifeval": "~4.6", "multi": "~3.0"},
         "qom_0.3_live_c147_bonus3g": {"code": "0.66-0.88", "ifeval": "4.5-6.9", "multi": "2.9-4.8"},
         "qom_0.3_live_c147_bonus4g": {"code": "0.7-0.8", "ifeval": "4.9-5.5", "multi": "3.2-3.8"},
         "qom_0.5_u8_live_c147_bonus5g": {"code": "C3 <= 0.81; FC +0.2-0.4", "ifeval": "C3 <= 4.40; FC 4.5-4.7",
                                          "multi": "C3 <= 2.83; FC 2.9-3.1"},
         "qom_0.5_u8_reach0.5_live_c147_bonus5g": {"code": "C3 ~0.74", "ifeval": "C3 ~4.65; FC ~4.95",
                                                   "multi": "C3 ~3.27; FC ~3.6"},
         "qom_0_split0.9_live_bq": {"code": "FC 1.2-1.5 (decisive)", "ifeval": "control", "multi": "control"},
         "qom_0_split0.99_live_bq": {"code": "FC 0.9-1.2 (decisive)", "ifeval": "control", "multi": "control"},
         "qom_0_split0.9_live_ball": {"code": "FC 1.2-1.5 (decisive)", "ifeval": "control", "multi": "control"},
         "qom_0_split0.99_live_ball": {"code": "FC 0.9-1.2 (decisive)", "ifeval": "control", "multi": "control"}}
PRED = PREDS.get(ARM, {d: "-" for d in ("code", "ifeval", "multi")})
INFOS = {"qom_0.3_live_c147_bonus3g": ("qom_0.5_live_c147_bonus5g", "qom_0_live_bq_c147_t3", "qom_0_live"),
         "qom_0.3_live_c147_bonus4g": ("qom_0.3_live_c147_bonus3g", "qom_0.5_live_c147_bonus5g"),
         "qom_0.5_u8_live_c147_bonus5g": ("qom_0.5_live_c147_bonus5g",),
         "qom_0.5_u8_reach0.5_live_c147_bonus5g": ("qom_0.5_live_c147_bonus5g", "qom_0.5_u8_live_c147_bonus5g"),
         "qom_0_split0.9_live_bq": ("qom_0_live_bq", "qom_0_split0.99_live_bq"),
         "qom_0_split0.99_live_bq": ("qom_0_live_bq", "qom_0_split0.9_live_bq"),
         "qom_0_split0.9_live_ball": ("qom_0_live_ball", "qom_0_live_bq", "qom_0_split0.99_live_ball"),
         "qom_0_split0.99_live_ball": ("qom_0_live_ball", "qom_0_live_bq", "qom_0_split0.9_live_ball"),
         "qom_0_split0.9_live_ball+code": ("qom_0_live_ball+code", "qom_0_live_bq", "qom_0_split0.99_live_ball+code"),
         "qom_0_split0.99_live_ball+code": ("qom_0_live_ball+code", "qom_0_live_bq", "qom_0_split0.9_live_ball+code")}
INFO = INFOS.get(ARM, ("qom_0.5_live_bonus5g", "qom_0.5_live_c164", "qom_0.5_live"))
DST = os.path.expanduser("~/NewIdeas/data/ledger/chat_arm_read" + ("" if len(sys.argv) < 2 else "_" + ARM) + ".txt")


def rows(name):
    """per window: (dom, array of (kl, dlp, flip)) over the baseline's rows; None if the arm has not landed."""
    p = os.path.join(M.OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p))
    return {nm: (b["dom"], np.array(M.rows_kl(b, R[nm]), float)) for nm, b in M.BASE.items() if nm in R}


def summary(A):
    out = {}
    for d in DOMS:
        xs = [v[1] for v in A.values() if v[0] == d]
        if not xs: out[d] = (float('nan'), float('nan'), 0, 0); continue        # Thu 18:5x: a domain not run (e.g. +code)
        x = np.concatenate(xs)
        out[d] = (100 * np.nanmean(x[:, 2]), float(np.nanmean(x[:, 0])), len(x), len(xs))
    return out


def verdict(d, f):
    return "PASS" if f <= LINE[d] else ("BETWEEN" if f <= 1.5 * LINE[d] else "KILL")


def mcnemar(A, B, d):
    """discordant rows: b = A flips and B does not, c = the reverse; exact two-sided binomial p."""
    b = c = 0
    for nm, (dom, x) in A.items():
        if dom != d or nm not in B: continue
        y = B[nm][1]; n = min(len(x), len(y))
        fa, fb = x[:n, 2] > 0.5, y[:n, 2] > 0.5
        b += int((fa & ~fb).sum()); c += int((fb & ~fa).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p


def main():
    L = [f"CHAT ARM READ (QUEUE 20:1x / 22:0x): {ARM} vs REF2-T's rows; flips % (measured), mean KL, rows / windows"]
    A = rows(ARM)
    if A is None:
        L.append(f"{ARM}: not landed yet")
    else:
        S = summary(A)
        L += ["| domain | flips % | line | verdict | prediction (pre-registered) | mean KL | rows / windows |", "|---|---|---|---|---|---|---|"]
        for d in DOMS:
            f, k, n, w = S[d]
            if n == 0: L.append(f"| {d} | not run | {LINE[d]} | - | {PRED[d]} | - | 0 |"); continue
            L.append(f"| {d} | {f:.2f} | {LINE[d]} | {verdict(d, f)} | {PRED[d]} | {k:.4f} | {n} / {w} |")
        allpass = all(verdict(d, S[d][0]) == "PASS" for d in DOMS if S[d][2])
        if ARM in ("qom_0.5_u8_live_c147_bonus5g", "qom_0.5_u8_reach0.5_live_c147_bonus5g"):   # amendment Thu 07:1x
            R = summary(rows("qom_0.5_live_c147_bonus5g"))
            g5 = all(S[d][0] <= R[d][0] + 0.5 for d in DOMS); g13 = all(S[d][0] <= 1.3 * R[d][0] for d in DOMS)
            kill = any(verdict(d, S[d][0]) != "PASS" for d in ("ifeval", "multi"))
            L.append(f"amended verdict (Fable C's lines): {'KILL' if kill else ('PASS' if allpass and g5 else 'BETWEEN')} "
                     f"(within +0.5 of the route: {g5}; Cloud3's 1.3x guard, information: {g13})")
        L.append(f"pre-registered verdict: {'PASS on every domain -> the chat route holds at 24 GB' if allpass else 'not PASS on every domain (see rows)'}")
    L += ["", "| arm (information) | code flips % / KL | ifeval | multi | paired vs " + ARM + " (b arm-only flips / c / p) |",
          "|---|---|---|---|---|"]
    for nm in INFO:
        B = rows(nm)
        if B is None: L.append(f"| {nm} | missing | | | |"); continue
        S2 = summary(B)
        pc = "; ".join(f"{d} {x[0]} / {x[1]} / p {x[2]:.2g}" for d in DOMS if S2[d][2] for x in [mcnemar(A, B, d)]) if A else "-"
        L.append(f"| {nm} | " + " | ".join((f"{S2[d][0]:.2f} / {S2[d][1]:.4f}" if S2[d][2] else "-") for d in DOMS) + f" | {pc} |")
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    open(DST, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
