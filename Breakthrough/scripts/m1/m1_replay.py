#!/usr/bin/env python3
"""M1 of the CHAT150 strike (NewIdeas Sun 2026-11-08 00:4x; Breakthrough with BigMoE): the 45 GB chat miss levers priced
on the decision map's chat windows (12 IFEval + 12 multi), CPU only, one core.

Core: bytes10_core.py = NewIdeas' ~/NewIdeas/scripts/bytes10_replay_fix.py (Fable C's BYTES10 replay + NewIdeas' rank-axis
fix; NewIdeas commit 1d53325; sha256 49892423dc06b9d1...), copied unchanged. Its 45 GB EXACT chat row (38.1 MB per kept
token on IFEval) sits within ~5 % of the engine's N45 exact run (~40 MB per kept token: 101 MB per check / 2.53 kept,
~/BigMoE/data/r57/nie1_256_checks.json), so the ratios here are the engine's ratios to first order.

Added rules (tuple = thr, ks, ku, t3, cold, t3all, bonus; sure rows = cf >= thr load ranks 0-1 only, the others ku ranks;
QOM's reads equal Q1's by construction, so 'QOM' rows here price the map's qom_* arms):
  QOM0          every drafted row skips its non-resident, non-union ranks 2-9 (cf >= 0), the bonus row exact
  QOM0+ball     QOM0 and every check's last row too (the map's '_ball' = DRAFT17's bound): the 45 GB FAST code rule
  QOM0+next     the last row gated by the next draft's cf (realizable proxy)
  QOM0.5(+ball) the same at cf >= 0.5
  TOP8          top-8 on every row (CH5); TOP8+QOM0.5
  U3 all / U3 sure0.5: a cold LSB sidecar is not read (ranks 0-1 at 3 bits when cold). Bytes == t3 on those rows (only
                misses cost bytes); quality differs (U3 keeps resident sidecars).
Pool splits at the 45 GB line (laysz45_256): S base slots per layer moved to the sidecar pool at equal bytes
(S x MR / LR sidecar slots), for EXACT and QOM0+ball.
Output: ~/Breakthrough/data/m1/m1_replay.out and .json. Usage: nice -n 15 <BigMoE venv python> -I -B m1_replay.py
"""
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bytes10_core as B  # noqa: E402

DST = os.path.expanduser("~/Breakthrough/data/m1")
B.RULES.update({
    "QOM0": (0.0, 2, 10, False, None, False, None),
    "QOM0+ball": (0.0, 2, 10, False, None, False, "all"),
    "QOM0+next": (0.0, 2, 10, False, None, False, "next"),
    "QOM0.5+ball": (0.5, 2, 10, False, None, False, "all"),
    "QOM0.5+next": (0.5, 2, 10, False, None, False, "next"),
    "TOP8": (None, 10, 8, False, None, False, None),
    "TOP8+QOM0.5": (0.5, 2, 8, False, None, False, None),
    "U3 all": (0.0, 10, 10, True, None, False, "all"),
    "U3 sure0.5": (0.5, 10, 10, True, None, False, None),
})
ORDER = ["EXACT", "Q1 0.5", "QOM0", "QOM0+next", "QOM0+ball", "QOM0.5+next", "QOM0.5+ball", "TOP8", "TOP8+QOM0.5",
         "U3 sure0.5", "U3 all", "Q1 0.5 t3", "Q1 0 t3 +bonus all"]
SPLIT_RULES = ("EXACT", "QOM0", "QOM0.5+next", "QOM0+next", "QOM0+ball")
POOL = os.environ.get("M1_POOL", "45")          # Sun 04:1x: the 24 / 12 GB pool lines (the ledger's C_m 147 + C_l 46 / 51 + 32)
SPLITS = {"45": (0, 16, 32, 48, 64), "24": (0, 8, 16, 32, 48), "12": (0, 4, 8, 16, 24)}[POOL]
SFX = "" if POOL == "45" else f"_{POOL}"
OPT_RULES = ("EXACT", "QOM0+ball")


def main():
    t0 = time.time(); lines = []; res = {}
    def say(s=""): print(s, flush=True); lines.append(s)
    W = B.windows(); W = {d: W[d] for d in ("ifeval", "multi")}
    Cm0, Cl0 = B.POOLS[POOL]
    say(f"M1 replay ({POOL} GB line: C_m mean {np.mean(Cm0):.0f}, C_l mean {np.mean(Cl0):.0f} per layer; MSB record {B.MR} B, "
        f"LSB sidecar {B.LR} B); windows {dict((d, len(v)) for d, v in W.items())}; live checks (t 0.3); LRU unless named")
    say("| domain | rule | MB per kept (MSB) | vs EXACT | misses per check (MSB + LSB) | MB per check | OPT MB per kept | kept | rows |")
    say("|---|---|---|---|---|---|---|---|---|")
    for dom in W:
        base = None
        for rule in ORDER:
            bm = B.RULES[rule][6]
            seqs = [(B.warm_sets(w), B.block_sets(w, B.live_blocks(w, bm), rule, None)) for w in W[dom]]
            r = B.simulate(seqs, Cm0, Cl0, "LRU")
            mbk = B.mbk(r); msb = r["mb"] * B.MR / r["kept"] / 1e6
            mpc_b, mpc_l = r["mb"] / r["nchk"], r["ml"] / r["nchk"]
            mbc = (r["mb"] * B.MR + r["ml"] * B.LR) / r["nchk"] / 1e6
            opt = None
            if rule in OPT_RULES:
                ro = B.simulate(seqs, Cm0, Cl0, "OPT"); opt = B.mbk(ro)
            if rule == "EXACT": base = mbk
            res[f"{dom}|{rule}"] = dict(mbk=mbk, msb=msb, mpc_b=mpc_b, mpc_l=mpc_l, mbc=mbc, opt=opt,
                                        kept=r["kept"] / r["nchk"], rows=r["nrow"] / r["nchk"])
            say(f"| {dom} | {rule} | {mbk:.1f} ({msb:.1f}) | {mbk / base - 1:+.0%} | {mpc_b + mpc_l:.1f} ({mpc_b:.1f} + {mpc_l:.1f}) | "
                f"{mbc:.1f} | {'-' if opt is None else f'{opt:.1f}'} | {r['kept'] / r['nchk']:.2f} | {r['nrow'] / r['nchk']:.2f} |")
            if rule in SPLIT_RULES:
                cells = []
                for S in SPLITS[1:]:
                    Cm = [max(c - S, 8) for c in Cm0]; Cl = [c + int(round(S * B.MR / B.LR)) for c in Cl0]
                    rs = B.simulate(seqs, Cm, Cl, "LRU"); v = B.mbk(rs)
                    res[f"{dom}|{rule}|split{S}"] = v
                    cells.append(f"S {S}: {v:.1f} ({v / mbk - 1:+.0%}; misses {rs['mb'] / rs['nchk']:.1f} + {rs['ml'] / rs['nchk']:.1f})")
                say(f"|   | {rule}: pool split, S base slots -> {B.MR / B.LR:.2f} S sidecar slots per layer | " + "; ".join(cells) + " | | | | | | |")
            print(f"  ({dom} {rule}: {time.time() - t0:.0f} s)", flush=True)
    say(f"done in {time.time() - t0:.0f} s")
    os.makedirs(DST, exist_ok=True)
    open(os.path.join(DST, f"m1_replay{SFX}.out"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(DST, f"m1_replay{SFX}.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
