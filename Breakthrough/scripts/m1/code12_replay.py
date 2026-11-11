#!/usr/bin/env python3
"""12 GB -> 90 tok/s on code (Kamen, Mon 9 Nov 2026 18:5x): SSD bytes per kept token on the decision map's CODE windows
(12 HumanEval), CPU only, one core. Breakthrough, Mon 9 Nov 2026 19:2x.

Core: bytes10_core.py (Fable C's BYTES10 replay + NewIdeas' rank-axis fix; see m1_replay.py), used as a library only:
its own main() writes into NewIdeas' folder and is NOT called here. Rules: bytes10_core's RULES plus m1_replay.py's added
rules (QOM0 / +ball / +next, QOM0.5, TOP8, U3), same tuples.
Pools: the 12 GB line's C_m 51 base + C_l 32 sidecar slots per layer (Fit12's 12 GB route, the ledger's "12" pool), then
a base-pool sweep at the same sidecar share (C_l = round(C_m x 32 / 51)) to see how bytes scale with the pool a ~12 GB
engine can hold.
Check: the 12 GB EXACT row should land near Fit12's measured 441 MB per kept token on code (A' chw14ha, 16 HumanEval);
a replay more than ~1.5x off the engine is a bug until shown otherwise.
Output: ~/Breakthrough/data/m1/code12_replay.out and .json. Usage: nice -n 15 <python with numpy> -I -B code12_replay.py
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
    "TOP8": (None, 10, 8, False, None, False, None),
    "TOP8+QOM0.5": (0.5, 2, 8, False, None, False, None),
    "U3 all": (0.0, 10, 10, True, None, False, "all"),
})
ORDER = ["EXACT", "Q1 0.5", "QOM0", "QOM0+next", "QOM0+ball", "QOM0.5+ball", "TOP8", "TOP8+QOM0.5", "U3 all",
         "Q1 0.5 t3", "Q1 0 t3 +bonus all"]
SWEEP_RULES = ("EXACT", "QOM0+ball", "Q1 0 t3 +bonus all")
SWEEP_CM = (24, 32, 40, 64, 80, 112)


def main():
    t0 = time.time(); lines = []; res = {}
    def say(s=""): print(s, flush=True); lines.append(s)
    W = B.windows(); W = {"code": W["code"]}
    Cm0, Cl0 = B.POOLS["12"]
    say(f"code12 replay (12 GB line: C_m {Cm0[0]}, C_l {Cl0[0]} per layer; MSB record {B.MR} B, LSB sidecar {B.LR} B); "
        f"code windows {len(W['code'])}; live checks (t 0.3); LRU unless named")
    say("| rule | MB per kept (MSB) | vs EXACT | misses per check (MSB + LSB) | MB per check | OPT MB per kept | kept | rows |")
    say("|---|---|---|---|---|---|---|---|")
    base = None
    seqs_of = {}
    for rule in ORDER:
        bm = B.RULES[rule][6]
        seqs = [(B.warm_sets(w), B.block_sets(w, B.live_blocks(w, bm), rule, None)) for w in W["code"]]
        seqs_of[rule] = seqs
        r = B.simulate(seqs, Cm0, Cl0, "LRU")
        mbk = B.mbk(r); msb = r["mb"] * B.MR / r["kept"] / 1e6
        mpc_b, mpc_l = r["mb"] / r["nchk"], r["ml"] / r["nchk"]
        mbc = (r["mb"] * B.MR + r["ml"] * B.LR) / r["nchk"] / 1e6
        opt = B.mbk(B.simulate(seqs, Cm0, Cl0, "OPT")) if rule in ("EXACT", "QOM0+ball") else None
        if rule == "EXACT": base = mbk
        res[rule] = dict(mbk=mbk, msb=msb, mpc_b=mpc_b, mpc_l=mpc_l, mbc=mbc, opt=opt, kept=r["kept"] / r["nchk"],
                         rows=r["nrow"] / r["nchk"])
        say(f"| {rule} | {mbk:.1f} ({msb:.1f}) | {mbk / base - 1:+.0%} | {mpc_b + mpc_l:.1f} ({mpc_b:.1f} + {mpc_l:.1f}) | "
            f"{mbc:.1f} | {'-' if opt is None else f'{opt:.1f}'} | {r['kept'] / r['nchk']:.2f} | {r['nrow'] / r['nchk']:.2f} |")
        print(f"  ({rule}: {time.time() - t0:.0f} s)", flush=True)
    say("")
    say("Base-pool sweep (C_l = round(C_m x 32 / 51)): MB per kept token, LRU")
    say("| rule | " + " | ".join(f"C_m {c}" for c in SWEEP_CM) + " |")
    say("|---|" + "---|" * len(SWEEP_CM))
    for rule in SWEEP_RULES:
        cells = []
        for c in SWEEP_CM:
            r = B.simulate(seqs_of[rule], [c] * 48, [int(round(c * 32 / 51))] * 48, "LRU")
            v = B.mbk(r); res[f"{rule}|Cm{c}"] = v; cells.append(f"{v:.1f}")
        say(f"| {rule} | " + " | ".join(cells) + " |")
        print(f"  (sweep {rule}: {time.time() - t0:.0f} s)", flush=True)
    say(f"done in {time.time() - t0:.0f} s")
    os.makedirs(DST, exist_ok=True)
    open(os.path.join(DST, "code12_replay.out"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(DST, "code12_replay.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
