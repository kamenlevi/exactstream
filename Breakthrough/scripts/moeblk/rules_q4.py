"""rules_q4.py (Breakthrough, Tue 17 Nov 2026 19:4x; MOE BLOCK DIAG): applies the pre-registered rules (docs/MOEBLOCK.md
DIAG + its amendments) to the quiet run's JSONs. Paired per block: the median over blocks of per-block differences, each
per-block value being bench.py's median over rounds. GPU-busy = libmtlprof's union of command-buffer intervals, us/call.
  front fold  F2 - F1, Run B (bench_q4_r1, serialized), 32 rows, both layers: <= -10 us -> Leap's slot #4 knob
  megakernel  knob - c_exp - c_t1 (the glue without the 1c tiers), Run B, 32 rows, both layers:
              >= 150 us -> propose (BIG); <= 60 -> close; between -> reported, no proposal
  DIAG        Run A (bench_q4_r6), 32 rows, per arm: CPU (build + encode) >= 0.8 x busy -> CPU-BOUND;
              CPU <= 0.5 x busy and idle (wall - busy) <= 15 % of wall -> GPU-BOUND; else MIXED
Also the round-0 cache check: the warm-up covered blocks 1-2 only, so c_exp / c_t1 computed their cached inputs inside
the timed region once (round 0) on blocks 3-12. If the 9-round median absorbed that, the component / knob ratios match
between blocks 1-2 and 3-12.
Usage: python3 -I -B scripts/moeblk/rules_q4.py   -> data/moeblk/rules_q4.txt (stdlib only)"""
import json
import os
import statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "moeblk")
A = json.load(open(os.path.join(D, "bench_q4_r6.json")))
B = json.load(open(os.path.join(D, "bench_q4_r1.json")))
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def v(R, L, r, arm, k="busy"):
    return R[f"L{L}_{r}_{arm}_split"][k]


def q(x):
    x = sorted(x)
    return f"median {S.median(x):7.1f} [min {x[0]:7.1f}, max {x[-1]:7.1f}]"


say("MOEBLK RULES on the quiet run of Tue 17 Nov 19:39 (Run A = bench_q4_r6, Run B = bench_q4_r1; GPU-busy, us per block)")
ff, mk, mk2 = {}, {}, {}
for L in (20, 44):
    for r in (32, 16):
        f2, f1, kn, ce, ct = (v(B, L, r, a) for a in ("f2", "f1", "knob", "c_exp", "c_t1"))
        d = [a - b for a, b in zip(f2, f1)]
        g = [k - e - t for k, e, t in zip(kn, ce, ct)]
        g2 = [k - e for k, e in zip(kn, ce)]
        dA = [a - b for a, b in zip(v(A, L, r, "f2"), v(A, L, r, "f1"))]
        say(f"  L{L} {r} rows: Run B F2 - F1 {q(d)}; blocks <= -10: {sum(x <= -10 for x in d)}/{len(d)}")
        say(f"  L{L} {r} rows: Run B glue w/o t1cf (knob - c_exp - c_t1) {q(g)}")
        say(f"  L{L} {r} rows: Run B glue with t1cf (knob - c_exp)        {q(g2)}")
        say(f"  L{L} {r} rows: Run A F2 - F1 {q(dA)} (info: independent calls)")
        if r == 32:
            ff[L], mk[L], mk2[L] = S.median(d), S.median(g), S.median(g2)
fv = "PASS -> Leap's slot #4 knob" if all(ff[L] <= -10 for L in ff) else "FAIL -> the line closes with tgl16eu as its result"
if all(mk[L] >= 150 for L in mk):
    mv = "PROPOSE (BIG)"
elif all(mk[L] <= 60 for L in mk):
    mv = "CLOSE"
else:
    mv = "BETWEEN -> reported, no proposal"
say(f"FRONT FOLD (Run B, 32 rows): F2 - F1 = {ff[20]:.1f} / {ff[44]:.1f} us per block (L20 / L44); rule <= -10 on both -> {fv}")
say(f"MEGAKERNEL (Run B, 32 rows): glue without t1cf = {mk[20]:.1f} / {mk[44]:.1f} us per block (L20 / L44; with t1cf "
    f"{mk2[20]:.1f} / {mk2[44]:.1f}); rule >= 150 propose, <= 60 close -> {mv}; x 48 layers = "
    f"{48 * mk[20] / 1000:.1f} / {48 * mk[44] / 1000:.1f} ms per 32-row check [P]")
say("DIAG reading (Run A, 32 rows):")
for L in (20, 44):
    for arm in ("today", "knob", "f1", "f2", "f2c"):
        bu, bd, en, wa = (v(A, L, 32, arm, k) for k in ("busy", "build", "encode", "wall"))
        cpu = S.median([a + b for a, b in zip(bd, en)])
        gb, w = S.median(bu), S.median(wa)
        idle = S.median([x - y for x, y in zip(wa, bu)])
        cr, ir = cpu / gb, idle / w
        rd = "CPU-BOUND" if cr >= 0.8 else ("GPU-BOUND" if cr <= 0.5 and ir <= 0.15 else "MIXED")
        say(f"  L{L} {arm:5s}: CPU {cpu:6.1f} = {cr:.2f} x busy {gb:6.1f}; idle {idle:6.1f} = {100 * ir:4.1f} % of wall {w:6.1f} -> {rd}")
say("Round-0 cache check (Run B, 32 rows): component / knob GPU-busy, median over blocks 1-2 (warmed) vs 3-12:")
for L in (20, 44):
    kn = v(B, L, 32, "knob")
    for arm in ("c_exp", "c_t1", "c_comb"):
        rt = [c / k for c, k in zip(v(B, L, 32, arm), kn)]
        say(f"  L{L} {arm:6s}: blocks 1-2 {S.median(rt[:2]):.3f}, blocks 3-12 {S.median(rt[2:]):.3f}")
open(os.path.join(D, "rules_q4.txt"), "w").write("\n".join(OUT) + "\n")
