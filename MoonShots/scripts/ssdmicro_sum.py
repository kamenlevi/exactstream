"""M-28 SSDMICRO summary (MoonShots, Tue 2026-11-03; pre-registered in ~/NewIdeas/QUEUE.md M-28 before any timing): prices
scripts/ssdmicro.c's raw calls (medians per point over the rounds) against the lines.
R0 = the median C 256 loader call: 2 base records (2 x 2,162,688 B) as 2 whole F_NOCACHE reads by one GCD dispatch_apply
after a 0.8 ms idle gap. Fixed cost of a call = its time - its bytes at 12.7 GB/s (SSD_WALL2's fit: 0.283 ms per call
+ MB / 12.7 GB/s at C 256).
Lines:
  L1 (Fable A's D1 screen, SSD_WALL2 section 5): pieces = 1 base record whole - in 8 pieces; joint = base call then sidecar
     call (whole) - one call of 8 + 3 pieces; gap 0.8, no keepalive. PASS pieces >= 0.10 ms AND joint >= 0.20 ms;
     KILL pieces < 0.04 AND joint < 0.10; BETWEEN otherwise (each part also judged alone).
  L2 (the orchestrator's question): R0's fixed cost F0. >= 0.20 ms: the drive + dispatch path makes most of the engine's
     0.28 (levers 1b-1c act there); <= 0.10 ms: most of it is outside the I/O path (release pickup, rebuild, call
     overhead), so BigMoE's per-burst trace must find it; between: split.
  L3 (lever 1's target, <= 5 ms per check at ~45 calls): the best measured 2-record call at gap 0.8 (any dispatch /
     pieces / keepalive / cache): fixed cost PASS <= 0.11 ms, KILL > 0.20 ms.
usage: python ssdmicro_sum.py raw.tsv [out.txt] -> data/ssdmicro.txt"""
import os, sys, time
import numpy as np

GBS, F_ENGINE, BASE, SIDE = 12.7, 283.0, 2162688, 770048
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/MoonShots/data/ssdmicro.txt")
rows, kar, meta, cols = {}, {}, [], None
for ln in open(sys.argv[1]):
    f = ln.rstrip("\n").split("\t")
    if ln.startswith("#ka"):
        kar.setdefault(int(f[1]), np.zeros(2))[:] += (int(f[2]), float(f[3]))
    elif ln.startswith("#"):
        meta.append(ln[2:].strip())
    elif f[0] == "round":
        cols = f[11:]
    else:
        rows.setdefault(" ".join(f[3:11]), []).append([float(x) for x in f[11:]])
R = {k: np.array(v) for k, v in rows.items()}
C = {c: i for i, c in enumerate(cols)}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def key(disp="gcd", gap=800, ka=0, pc=False, b=(0, 1), s=(0, 1), m=(0, 0), seq=False):
    return f"{disp} g{gap} ka{ka} {'pc' if pc else 'nc'} b{b[0]}x{b[1]} s{s[0]}x{s[1]} m{m[0]}x{m[1]} {'seq' if seq else 'one'}"


def med(k, c="T_us"):
    return float(np.median(R[k][:, C[c]]))


def fixed(k):
    return med(k) - R[k][0, C["bytes"]] / (GBS * 1e3)


R0 = key(b=(2, 1))
n = len(next(iter(R.values())))
say(f"== M-28 SSDMICRO ({time.strftime('%a %H:%M')}): {len(R)} points x {n} rounds; {'; '.join(meta)}")
say(f"  R0 = 2 base records as 2 whole F_NOCACHE reads, one GCD dispatch_apply, after 0.8 ms idle; fixed = time - bytes at "
    f"{GBS} GB/s; us medians")
fast = sum(int(v[:, C["fast"]].sum()) for v in R.values())
say(f"  sanity: {fast} reads >= 1 MB faster than 16 GB/s (page-cache hits, not the drive)")

say("  S1 lone read, in-thread (QD 1): size -> us at gap 0 / 0.8 ms")
sz = [4096, 65536, 262144, 770048, 1 << 20, BASE, 4 << 20]
k1 = lambda s, g: key("serial", g, b=(1, 1)) if s == BASE else key("serial", g, m=(1, s))
for s in sz:
    say(f"    {s / 1024:7.0f} KB  {med(k1(s, 0)):7.1f} / {med(k1(s, 800)):7.1f}")
fit = {}
for g in (0, 800):
    sl, a = np.polyfit(np.array(sz[1:], float), [med(k1(s, g)) for s in sz[1:]], 1)
    fit[g] = (a, 1e-3 / sl)
    say(f"    fit 64 KB-4 MB, gap {g / 1e3:.1f} ms: first byte {a:.1f} us + bytes at {1e-3 / sl:.2f} GB/s")

say("  S2 queue depth: k parallel 256 KB reads (GCD): k -> call us, GB/s, median read us (gap 0.8 | back to back)")
for k in (1, 2, 4, 8, 16, 32, 64):
    a = key(m=(k, 262144))
    b = key(gap=0, m=(k, 262144))
    tail = f" | {med(b):7.1f} {k * 262144 / med(b) / 1e3:5.2f} {med(b, 'mean_read_us'):6.1f}" if b in R else ""
    say(f"    k {k:2d}: {med(a):7.1f} {k * 262144 / med(a) / 1e3:5.2f} {med(a, 'mean_read_us'):6.1f}{tail}")

say("  S3 pieces (GCD, gap 0.8): 1 base record in k pieces -> us (fixed)")
say("    " + "  ".join(f"k {k}: {med(key(b=(1, k))):.1f} ({fixed(key(b=(1, k))):.1f})" for k in (1, 2, 4, 8, 16, 32)))
say("    2 records: " + "  ".join(f"k {k}: {med(key(b=(2, k))):.1f} ({fixed(key(b=(2, k))):.1f})" for k in (1, 4, 8)))

say("  S4 idle gap -> R0 us (+ vs back to back) | R0 in 8 pieces")
for g in (0, 100, 250, 500, 800, 2000, 5000, 10000):
    a, b = key(gap=g, b=(2, 1)), key(gap=g, b=(2, 8))
    say(f"    {g / 1e3:5.2f} ms: {med(a):7.1f} ({med(a) - med(key(gap=0, b=(2, 1))):+6.1f}) | {med(b):7.1f} "
        f"({med(b) - med(key(gap=0, b=(2, 8))):+6.1f})")

say("  S5 keepalive spacing (4 KB pread + usleep(n)): R0 us back to back / after 0.8 ms (penalty), reads/s")
for ka in (0, 25, 50, 100, 200, 500):
    a, b = key(gap=0, ka=ka, b=(2, 1)), key(ka=ka, b=(2, 1))
    rate = f"{kar[ka][0] / kar[ka][1]:7.0f}" if ka in kar else "      -"
    say(f"    {'off' if not ka else f'{ka} us':6s}: {med(a):7.1f} / {med(b):7.1f} ({med(b) - med(a):+6.1f}) {rate} /s")

say("  S6 page cache vs F_NOCACHE (GCD, gap 0.8): us")
for nm, b in (("R0", (2, 1)), ("R0 8 pieces", (2, 8)), ("1 record", (1, 1))):
    say(f"    {nm:12s}: F_NOCACHE {med(key(b=b)):7.1f} | page cache {med(key(pc=True, b=b)):7.1f}")

say("  S7 dispatch (gap 0.8): us; critical read's start after the call")
for nm, b in (("1 record", (1, 1)), ("R0", (2, 1)), ("1 rec 8 pieces", (1, 8)), ("2 rec 4 pieces", (2, 4))):
    g, s = key(b=b), key("spin", b=b)
    say(f"    {nm:15s}: GCD {med(g):7.1f} (start {med(g, 'crit_start_us'):5.1f}) | pre-spun {med(s):7.1f} "
        f"(start {med(s, 'crit_start_us'):5.1f})")
say(f"    R0 serial (2 reads one after the other): {med(key('serial', b=(2, 1))):.1f}; pre-spun + keepalive 50: R0 "
    f"{med(key('spin', ka=50, b=(2, 1))):.1f}, 2 rec 4 pieces {med(key('spin', ka=50, b=(2, 4))):.1f}")

say("  S8 JOINT (gap 0.8): us for base + sidecar records; two calls vs one")
for ka in (0, 50):
    sq, j2 = key(ka=ka, b=(1, 1), s=(1, 1), seq=True), key(ka=ka, b=(1, 1), s=(1, 1))
    sq8, j11 = key(ka=ka, b=(1, 8), s=(1, 3), seq=True), key(ka=ka, b=(1, 8), s=(1, 3))
    say(f"    1+1, keepalive {'off' if not ka else '50'}: two calls {med(sq):7.1f}, one call {med(j2):7.1f}; in 8 + 3 pieces: "
        f"two calls {med(sq8):7.1f}, one call {med(j11):7.1f}")
say(f"    2+2: two calls {med(key(b=(2, 1), s=(2, 1), seq=True)):.1f}, one call {med(key(b=(2, 1), s=(2, 1))):.1f}, "
    f"one call in 8 + 3 pieces {med(key(b=(2, 8), s=(2, 3))):.1f}")

# L1: Fable A's D1 screen
dp = med(key(b=(1, 1))) - med(key(b=(1, 8)))
dj = med(key(b=(1, 1), s=(1, 1), seq=True)) - med(key(b=(1, 8), s=(1, 3)))
pv = "PASS" if dp >= 100 else "KILL" if dp < 40 else "BETWEEN"
jv = "PASS" if dj >= 200 else "KILL" if dj < 100 else "BETWEEN"
say(f"  L1 (D1 screen): pieces cut a lone record by {dp:.1f} us ({pv}: line 100 / 40); joint in 8 + 3 pieces cuts base + "
    f"sidecar by {dj:.1f} us ({jv}: line 200 / 100) -> " + ("PASS" if pv == jv == "PASS" else "KILL" if pv == jv == "KILL" else "BETWEEN"))
# L2: what R0's fixed cost is made of
T, f0 = med(R0), fixed(R0)
wake = med(R0) - med(key(gap=0, b=(2, 1)))
say(f"  L2 (what the 0.28 ms is): R0 {T:.1f} us = GCD start {med(R0, 'crit_start_us'):.1f} + critical read "
    f"{med(R0, 'crit_read_us'):.1f} + return {med(R0, 'tail_us'):.1f}; fixed {f0:.1f} us of the engine's {F_ENGINE:.0f} "
    f"({100 * f0 / F_ENGINE:.0f} %); its parts: drive first byte ~{fit[0][0]:.0f} (S1 intercept), idle wake {wake:+.1f} "
    f"(0.8 ms vs back to back) -> " + ("the I/O path makes most of it" if f0 >= 200 else
                                       "mostly outside the I/O path (pickup / rebuild / call overhead)" if f0 <= 100 else "split"))
# L3: the best measured 2-record call
cand = [k for k in R if k.split()[1] == "g800" and k.split()[4].startswith("b2x") and k.split()[5].startswith("s0")
        and k.split()[6] == "m0x0"]
best = min(cand, key=fixed)
fb = fixed(best)
say(f"  L3 (lever 1 target): best 2-record call '{best}': {med(best):.1f} us, fixed {fb:.1f} us (R0 {f0:.1f}) -> "
    + ("PASS" if fb <= 110 else "KILL" if fb > 200 else "BETWEEN") + " (line 110 / 200)")
open(OUT, "a").write("\n".join(lines) + "\n")
