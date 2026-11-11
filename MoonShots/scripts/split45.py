"""M-08 SPLIT45 (MoonShots, Fri 2026-10-30; pricing for R-58's S45-T): REF2-T's MSB-3 / LSB pool split at the 45 GB
route's RAM points, on 16-row checks. REF2 (4444444444) needs every used expert's whole 4-bit record (MSB-3 2,048,000 B
+ LSB 720,896 B = 2,768,896 B; one pool of C slots per layer). REF2-T (4433333333) needs the MSB-3 part for every use and
the LSB part only when the expert's best rank among the check's rows is < 2 (two pools, C_m and C_l per layer). Equal
RAM: C_m x MSB + C_l x LSB = C x FULL per layer. RAM points: REF2 C 192 (S45: 36.3 GB peak), 224 (S45b: 40.5 GB), 256
(~45 GB), 288 (~49 GB).
wall24's checks: expA traces (bulgarian / chat / code / math, 800 tokens, top-10 of the rank-ordered top-20), tokens
0-399 warm the pools, checks starting at 400+ are measured; R 16 rows advancing A (8 ~ S45's mean kept 8.27; 10 code-like).
Pools are trimmed strictly to their size after each call (memory: wall24-free-staging-flaw), so a pool smaller than a
check's union never acts as free cache. Policies: lru (BigMoE's ext ensure default) and lfu (BIGMOE_EVICT_LFU 0.99).
Reads count an expert missing both parts in one check once (BIGMOE_JOINT reads them together).
Reports per RAM point: REF2 reads and MB per kept token; REF2-T per C_l: MSB / LSB / joint reads, MB per kept token; the
best split by reads; SSD ms per kept token at 10.6 GB/s (S45's rate); and RAM-side expert bytes per check (REF2: FULL
per distinct expert; REF2-T: MSB per distinct expert + LSB per distinct rank-0-1 expert).
Calibration: S45 measured REF2 C 192 at 51-86 misses per kept token (mean 69); true-token rows give fewer (Leap's
resid27: 49 on code), so read the ratios, not the absolutes.
Lines (M-08 row, written before the run): PASS = REF2-T reads per kept token -30 % or better vs REF2 at equal RAM with a
C_l that fits, at >= 2 of the 4 RAM points; KILL = under -10 % at every RAM point; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python split45.py -> stdout + data/split45.txt (CPU only, one core)"""
import heapq, os, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

MSB, LSB = 2048000, 720896
FULL = MSB + LSB
RATE = 10.6e9
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def lru(seq, C):
    """strict LRU: per call the set of misses; after the call the pool keeps at most C (oldest first out)"""
    od, out = OrderedDict(), []
    for N in seq:
        out.append(frozenset(e for e in N if e not in od))
        for e in N:
            if e in od:
                od.move_to_end(e)
            else:
                od[e] = True
        while len(od) > C:
            od.popitem(last=False)
    return out


def lfu(seq, C, d=0.99):
    """strict decayed LFU: experts not needed now go first, then the lowest score"""
    R, score, w, out = set(), {}, 1.0, []
    for N in seq:
        w /= d
        for e in N:
            score[e] = score.get(e, 0.0) + w
        out.append(frozenset(e for e in N if e not in R))
        R |= N
        if len(R) > C:
            for v in heapq.nsmallest(len(R) - C, R, key=lambda v: (v in N, score[v])):
                R.discard(v)
    return out


POL = {"lru": lru, "lfu": lfu}
CH = {}


def checks(R, A):
    for k, ids in W.IDS.items():
        if (k, R, A) not in CH:
            CH[(k, R, A)] = W.needs(ids, R, A)
    return {k: CH[(k, R, A)] for k in W.IDS}


def price(R, A, pol, C, Cm=None, Cl=None):
    """REF2 if Cm is None, else REF2-T. Mean over the 4 sets of per-kept-token reads / bytes and per-check RAM bytes."""
    f = POL[pol]
    acc = []
    for k, ch in checks(R, A).items():
        meas = [m for m, _ in ch]
        nm = sum(meas)
        rd = by = ram = rm = rl = jt = 0
        for l in range(W.NL):
            sm = [lay[l][0] for _, lay in ch]
            sl = [lay[l][1] for _, lay in ch]
            if Cm is None:
                mm = f(sm, C)
                for t, m in enumerate(meas):
                    if m:
                        rd += len(mm[t]); by += len(mm[t]) * FULL; ram += len(sm[t]) * FULL
            else:
                mm, ml = f(sm, Cm), f(sl, Cl)
                for t, m in enumerate(meas):
                    if m:
                        j = len(mm[t] & ml[t])
                        rm += len(mm[t]); rl += len(ml[t]); jt += j
                        rd += len(mm[t]) + len(ml[t]) - j
                        by += len(mm[t]) * MSB + len(ml[t]) * LSB
                        ram += len(sm[t]) * MSB + len(sl[t]) * LSB
        acc.append((rd / (A * nm), by / (A * nm), ram / nm, rm / (A * nm), rl / (A * nm), jt / (A * nm)))
    return np.mean(np.array(acc), 0)


if __name__ == "__main__":
    if os.path.exists(os.path.expanduser("~/.local/share/quiet/PAUSE")):
        sys.exit("PAUSE set: not running")
    t0 = time.time()
    say("M-08 SPLIT45: REF2 (one 4-bit pool) vs REF2-T (MSB-3 pool + LSB pool) at equal RAM, 16-row checks, expA traces "
        "(mean of bulgarian / chat / code / math), per kept token")
    verdict = {}
    for R, A in ((16, 8), (16, 10)):
        for pol in ("lru", "lfu"):
            say(f"== R {R} A {A} {pol}")
            for C in (192, 224, 256, 288):
                b = price(R, A, pol, C)
                say(f"  RAM of REF2 C {C} ({C * W.NL * FULL / 1e9:.1f} GB of pool): REF2 reads {b[0]:5.1f} / token, "
                    f"{b[1] / 1e6:6.1f} MB / token, SSD {1e3 * b[1] / RATE:5.2f} ms / token; RAM-side expert bytes "
                    f"{b[2] / 1e9:5.2f} GB / check")
                best = None
                for Cl in (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320):
                    Cm = int((C * FULL - Cl * LSB) // MSB)
                    if Cl > C or Cm > 512 or Cm < 0:
                        continue
                    t = price(R, A, pol, C, Cm, Cl)
                    say(f"    REF2-T C_m {Cm:3d} C_l {Cl:3d}: reads {t[0]:5.1f} (MSB {t[3]:5.1f}, LSB {t[4]:5.1f}, joint "
                        f"{t[5]:4.1f}) {100 * (t[0] / b[0] - 1):+6.1f} %, {t[1] / 1e6:6.1f} MB {100 * (t[1] / b[1] - 1):+6.1f} %, "
                        f"SSD {1e3 * t[1] / RATE:5.2f} ms / token; RAM-side {t[2] / 1e9:5.2f} GB / check "
                        f"({100 * (t[2] / b[2] - 1):+.1f} %)")
                    if best is None or t[0] < best[1][0]:
                        best = ((Cm, Cl), t)
                (Cm, Cl), t = best
                d = 100 * (t[0] / b[0] - 1)
                verdict.setdefault((R, A, pol), []).append(d)
                say(f"    best by reads: C_m {Cm} C_l {Cl}: reads {d:+.1f} %, MB {100 * (t[1] / b[1] - 1):+.1f} % vs REF2")
    say("Lines: PASS = reads -30 % or better at >= 2 of 4 RAM points; KILL = under -10 % at every point")
    for key, ds in verdict.items():
        n30 = sum(d <= -30 for d in ds)
        call = "PASS" if n30 >= 2 else ("KILL" if all(d > -10 for d in ds) else "BETWEEN")
        say(f"  R {key[0]} A {key[1]} {key[2]}: best reads vs REF2 at C 192 / 224 / 256 / 288: "
            + " / ".join(f"{d:+.1f} %" for d in ds) + f" -> {call}")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/split45.txt"), "w").write("\n".join(lines) + "\n")
