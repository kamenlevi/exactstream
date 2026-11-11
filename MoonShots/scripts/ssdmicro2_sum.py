"""M-29 SSDMICRO2 summary (MoonShots, Tue 2026-11-03; pre-registered in ~/NewIdeas/QUEUE.md M-29 before any timing):
prices scripts/ssdmicro2.m's raw calls (medians per point over the rounds) against the lines.
Lines:
  Z (zero-copy): the best zero-copy arm in Metal slot memory (Z1m contiguous, Z3m 4 KB-padded pieces into 16 KB-aligned
     slots, Z3bm the same into 4 KB-aligned slots) against today's Z0m (bounce + copy into Metal pool slots), 2 base
     records per call: PASS = saves >= 50 us per call AND its call is within 15 us of Z0n (the same reads with no copy:
     zero-copy costs no DMA penalty); KILL = saves < 20 us; else BETWEEN. Reported with it: whether 4 KB-aligned slots
     match 16 KB-aligned ones (Z3b vs Z3 within 15 us), REC1 (Z2) and the 16 KB-padded layout (Z4).
  Copy tail (descriptive): the copy's median per base / sidecar record on the call's critical job, against M-28's
     ~96 us per call outside the I/O path.
  K (descriptive): the widest real keepalive spacing whose 0.8 ms penalty is <= 10 us, and the read-size effect.
  P (wake-ahead ping): removal = 1 - (ping call - back to back) / (no-ping call after 0.8 ms - back to back);
     PASS = some T <= 200 us removes >= 70 %; KILL = no T removes >= 30 %; else BETWEEN.
usage: python ssdmicro2_sum.py raw.tsv [out.txt] -> data/ssdmicro2.txt"""
import os, sys, time
import numpy as np

OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/MoonShots/data/ssdmicro2.txt")
rows, kar, meta, cols = {}, {}, [], None
for ln in open(sys.argv[1]):
    f = ln.rstrip("\n").split("\t")
    if ln.startswith("#ka"):
        kar.setdefault((int(f[1]), int(f[2])), np.zeros(2))[:] += (int(f[3]), float(f[4]))
    elif ln.startswith("#"):
        meta.append(ln[2:].strip())
    elif f[0] == "round":
        cols = f[8:]
    else:
        rows.setdefault(tuple(f[2:8]), []).append([float(x) for x in f[8:]])
R = {k: np.array(v) for k, v in rows.items()}
C = {c: i for i, c in enumerate(cols)}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def key(s, a, n=2, g=800, ka=0, sz=0, p=0):
    return (s, a, str(n), f"g{g}", f"ka{ka}x{sz}", f"p{p}")


def med(k, c="T_us"):
    return float(np.median(R[k][:, C[c]]))


n = len(next(iter(R.values())))
say(f"== M-29 SSDMICRO2 ({time.strftime('%a %H:%M')}): {len(R)} points x {n} rounds; {'; '.join(meta)}")
fast = sum(int(v[:, C["fast"]].sum()) for v in R.values())
say(f"  sanity: {fast} reads faster than 16 GB/s (page-cache hits, not the drive)")
say("  Z zero-copy (gap 0.8 ms, GCD, F_NOCACHE): call us (read + copy on the critical job) | vs Z0n (same reads, no copy)")
arms = ["Z0", "Z0m", "Z0p", "Z0n", "Z1", "Z1m", "Z2", "Z3", "Z3m", "Z3b", "Z3bm", "Z4"]
for nr in (1, 2):
    base = med(key("Z", "Z0n", nr))
    for a in arms:
        k = key("Z", a, nr)
        if k in R:
            say(f"    {nr} rec {a:5s}: {med(k):7.1f} ({med(k, 'read_us'):6.1f} + {med(k, 'copy_us'):5.1f}) | {med(k) - base:+6.1f}")
cb, cbm, cs = med(key("Z", "Z0", 1), "copy_us"), med(key("Z", "Z0m", 1), "copy_us"), med(key("S", "S0", 1), "copy_us")
say(f"  copy tail per record on the critical job: base {cb:.1f} us (malloc slots), {cbm:.1f} (Metal slots), split over 3 "
    f"projections {med(key('Z', 'Z0p', 1), 'copy_us'):.1f}; sidecar {cs:.1f}; against M-28's ~96 us per call outside the I/O path")
say("  S sidecar part, 1 record: " + "; ".join(f"{a} {med(key('S', a, 1)):.1f}" for a in ("S0", "S0n", "S1", "S3", "S3b")))
z0m = med(key("Z", "Z0m", 2))
best = min(("Z1m", "Z3m", "Z3bm"), key=lambda a: med(key("Z", a, 2)))
sv, pen = z0m - med(key("Z", best, 2)), med(key("Z", best, 2)) - med(key("Z", "Z0n", 2))
al = {m: med(key("Z", "Z3b" + m, 2)) - med(key("Z", "Z3" + m, 2)) for m in ("", "m")}
say(f"  Z line: best zero-copy {best} saves {sv:.1f} us per 2-record call vs today's Z0m; {pen:+.1f} us vs the same reads "
    f"without any copy -> " + ("PASS" if sv >= 50 and pen <= 15 else "KILL" if sv < 20 else "BETWEEN")
    + f" (line 50 / 20; DMA penalty <= 15). 4 KB- vs 16 KB-aligned slots: {al['']:+.1f} us (malloc), {al['m']:+.1f} (Metal)"
    + (" -> 4 KB alignment suffices" if max(abs(al[""]), abs(al["m"])) <= 15 else " -> 16 KB alignment matters"))
say("  K keepalive (BigMoE's pread + usleep(n)): R0 back to back / after 0.8 ms (penalty) | real spacing")
b0, off8 = med(key("K", "R0", g=0)), med(key("Z", "Z0n", 2))
say(f"    off      : {b0:7.1f} / {off8:7.1f} ({off8 - b0:+6.1f})")
for ka, sz in ((50, 4096), (100, 4096), (125, 4096), (150, 4096), (175, 4096), (200, 4096), (200, 16384), (200, 65536)):
    a, b = med(key("K", "R0", g=0, ka=ka, sz=sz)), med(key("K", "R0", ka=ka, sz=sz))
    sp = kar[(ka, sz)][1] / kar[(ka, sz)][0] * 1e6 if (ka, sz) in kar and kar[(ka, sz)][0] else float("nan")
    say(f"    n {ka:3d} {sz // 1024:2d} KB: {a:7.1f} / {b:7.1f} ({b - a:+6.1f}) | {sp:5.0f} us")
say("  P wake-ahead ping (one 4 KB read T us before the R0 call, no trickle): call us, removal, ping read us, done before the call by")
rm = {}
for T in (25, 50, 100, 200, 400):
    k = key("P", "R0", p=T)
    rm[T] = 1 - (med(k) - b0) / (off8 - b0) if off8 > b0 else float("nan")
    say(f"    T {T:3d}: {med(k):7.1f} us, removes {100 * rm[T]:5.1f} %, ping {med(k, 'ping_read_us'):5.1f} us, "
        f"lead {med(k, 'ping_lead_us'):+6.1f} us")
ok = [T for T in rm if T <= 200 and rm[T] >= 0.70]
say(f"  P line: " + ("PASS" if ok else "KILL" if max(rm.values()) < 0.30 else "BETWEEN")
    + f" (best T {max(rm, key=rm.get)}: {100 * max(rm.values()):.0f} %; line 70 % at T <= 200 / 30 %)")
say(f"  C keepalive 50: today's copy {med(key('C', 'Z0', ka=50, sz=4096)):.1f} us, zero-copy Z3b "
    f"{med(key('C', 'Z3b', ka=50, sz=4096)):.1f} us (2 records; no keepalive: Z0 {med(key('Z', 'Z0', 2)):.1f})")
open(OUT, "a").write("\n".join(lines) + "\n")
