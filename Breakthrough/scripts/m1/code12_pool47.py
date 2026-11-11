#!/usr/bin/env python3
"""QOM0's SSD reads at Fit12's call pool C_m 47 vs the map's passing pool C_m 51 (C_l 32 at both: Fit12's F12_CMCL=47,32),
on the decision map's code windows, live checks, LRU (bytes10_core.simulate, exactly as code12_replay.py). Breakthrough,
Tue 10 Nov 2026 09:1x, after qom_0_live_c47 failed the code map line by one flip while qom_0_live_c51 passes: what the four
extra slots per layer buy in reads [P, CPU replay]. Self-check: the C_m 51 rows must reproduce code12_replay.out's 12 GB
rows (QOM0 162.5, QOM0+ball 101.8, EXACT 636.8 MB per kept token), else this script is wrong.
Output: ~/Breakthrough/data/m1/code12_pool47.out. Usage: nice -n 15 <python with numpy> -I -B code12_pool47.py
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code12_replay as R  # noqa: E402  (registers QOM0 / QOM0+ball in bytes10_core's RULES; its main() is not called)

B = R.B
REF51 = {"QOM0": 162.5, "QOM0+ball": 101.8, "EXACT": 636.8}   # code12_replay.out, C_m 51 / C_l 32, LRU
t0 = time.time(); lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


W = B.windows()["code"]
Cm0, Cl0 = B.POOLS["12"]
say(f"QOM0 reads at C_m 47 vs 51 (C_l 32 at both; code windows {len(W)}; live checks; LRU; MSB record {B.MR} B, LSB "
    f"sidecar {B.LR} B; the 12 GB pool in bytes10_core: C_m {sorted(set(Cm0))}, C_l {sorted(set(Cl0))})")
say("| rule | C_m | MB per kept | MB per check | misses per check (MSB + LSB) | kept per check | C_m 51 self-check |")
say("|---|---|---|---|---|---|---|")
ok = True
res = {}
for rule in ("QOM0", "QOM0+ball", "EXACT"):
    bm = B.RULES[rule][6]
    seqs = [(B.warm_sets(w), B.block_sets(w, B.live_blocks(w, bm), rule, None)) for w in W]
    for c in (47, 51):
        r = B.simulate(seqs, [c] * 48, [32] * 48, "LRU")
        v = B.mbk(r); res[(rule, c)] = v
        mbc = (r["mb"] * B.MR + r["ml"] * B.LR) / r["nchk"] / 1e6
        chk = ""
        if c == 51:
            good = abs(v - REF51[rule]) < 0.15
            ok = ok and good
            chk = f"{'PASS' if good else 'FAIL'} ({REF51[rule]})"
        say(f"| {rule} | {c} | {v:.1f} | {mbc:.1f} | {(r['mb'] + r['ml']) / r['nchk']:.1f} ({r['mb'] / r['nchk']:.1f} + "
            f"{r['ml'] / r['nchk']:.1f}) | {r['kept'] / r['nchk']:.2f} | {chk} |")
say("")
for rule in ("QOM0", "QOM0+ball", "EXACT"):
    say(f"{rule}: C_m 47 -> 51 changes MB per kept token by {res[(rule, 51)] / res[(rule, 47)] - 1:+.1%}")
say(f"SELF-CHECK {'PASS' if ok else 'FAIL'}; done in {time.time() - t0:.0f} s")
open(os.path.expanduser("~/Breakthrough/data/m1/code12_pool47.out"), "w").write("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
