"""crossover32.py (Fable B, mission B3): when does a 32-row check beat a 16-row one, on the EXACT and FAST tracks, at the
floor engine? A first-order cycle model on measured terms. CPU only, numpy.
Terms per check (ms) [K unless marked]: draft 12.0 at 16 rows / 15.5 at 32 (CA; the ANE-2 batched call 12.4 at 16);
dense GPU (the dense constant) D16 = 34 today (C0 eval), 23 EXACT expected / 19 EXACT floor, 17 FAST expected (HC8) /
15 FAST floor (BOUNDARIES); dense at 32 rows = D16 + 4.2 (D32: the GEMMs 17.96 -> 22.19) + 16 x 0.35 (GDN rows +
attention rows, C0 / B3: 0.3-0.4 per row) = D16 + 9.8 [P], or + 8.0 with G3 (the GDN row term halved) [P];
experts E16 = 50.9 (3c) / 43.2 (NAX6) / 25.7-28.6 floor at 258-287 GB/s (C2); the row slope 16 -> 32: 3c 3.03, NAX6 1.41
(B3), floor 0.9 = C2's bytes per added row 16 -> 32 (11.33 - 7.38 GB) / 16 at 258-287 GB/s (so experts(32) at the floor =
41 ms, as C2's 39.5-43.9) [K/P]; RELAX_350's "1.2 ms per row at 32" is the all-32-rows average (32 x 1.2 = 38), the same thing; commit 4.7 at 16 (R-12) + 0.09 per
kept row above 10 (R-21: keep 1 -> 15 adds ~0.85 ms over 35 layers) [P]; rest ~5; SSD exposed: 65 today at C 256 on
code (RELAX_350), x1.6 at 32 rows (the union x1.5-1.6), the floor engine hides ~70 % (the runahead) and night C 320 is
x0.65 [G, RELAX_350's model]. Kept: code 10.54 (16) / 10.5 (32) today [K D-23d]; the copy override x1.041 / x1.211
[K COPYGATE G2]; DFlash3-XL 13 / 16 [G NEXT_DRAFTER]; MBPP x1.000 / x1.001, agent x1.032 / x1.036 [K COPYGATE].
Run: nice -n 15 ~/BigMoE/.venv/bin/python crossover32.py > crossover32.out"""
import numpy as np


def cycle(rows, D16, E16, slope, kept, ssd16, hide, draft16, draft32, g3=False, dense32_extra=None, hide32=None):
    draft = draft16 if rows == 16 else draft32
    dense = D16 if rows == 16 else D16 + (8.0 if g3 else 9.8) if dense32_extra is None else D16 + dense32_extra
    experts = E16 if rows == 16 else E16 + 16 * slope
    ssd = ssd16 * (1 - hide) if rows == 16 else ssd16 * 1.6 * (1 - (hide if hide32 is None else hide32))
    commit = 4.7 + 0.09 * max(0, kept - 10)
    rest = 5.0
    return draft + dense + experts + ssd + commit + rest


def toks(rows, kept, **kw):
    return 1e3 * kept / cycle(rows, kept=kept, **kw)


print("== tok/s on code, 16 vs 32 rows, by engine and kept pair [P on K; see the docstring for the terms] ==")
engines = [
    ("today (3c, C 256 day, SSD exposed 65)", dict(D16=34.0, E16=50.9, slope=3.03, ssd16=65.0, hide=0.0, draft16=12.0, draft32=15.5)),
    ("this week EXACT, NAX6, SSD exposed", dict(D16=30.0, E16=43.2, slope=1.41, ssd16=65.0, hide=0.0, draft16=12.0, draft32=15.5)),
    ("floor engine EXACT (dense 23 exp.), experts at the floor, night C 320 + runahead (hide 70 %, x0.65)", dict(D16=23.0, E16=27.0, slope=0.9, ssd16=65.0 * 0.65, hide=0.7, draft16=5.0, draft32=6.5)),
    ("floor engine EXACT at its floor (dense 19)", dict(D16=19.0, E16=27.0, slope=0.9, ssd16=65.0 * 0.65, hide=0.7, draft16=5.0, draft32=6.5, g3=True)),
    ("floor engine FAST (dense 17 exp. with HC8)", dict(D16=17.0, E16=27.0, slope=0.9, ssd16=65.0 * 0.65, hide=0.7, draft16=5.0, draft32=6.5, g3=True)),
    ("floor engine FAST at its floor (dense 15)", dict(D16=15.0, E16=27.0, slope=0.9, ssd16=65.0 * 0.65, hide=0.7, draft16=5.0, draft32=6.5, g3=True)),
    ("floor engine FAST, SSD fully hidden (dense 15)", dict(D16=15.0, E16=27.0, slope=0.9, ssd16=65.0 * 0.65, hide=1.0, draft16=5.0, draft32=6.5, g3=True)),
    ("floor engine EXACT (dense 23), the 32-row runahead hides 90 % vs 70 % at 16 rows [G]", dict(D16=23.0, E16=27.0, slope=0.9, ssd16=65.0 * 0.65, hide=0.7, hide32=0.9, draft16=5.0, draft32=6.5, g3=True)),
    ("floor engine FAST (dense 17), the 32-row runahead hides 100 % vs 70 % at 16 rows [G]", dict(D16=17.0, E16=27.0, slope=0.9, ssd16=65.0 * 0.65, hide=0.7, hide32=1.0, draft16=5.0, draft32=6.5, g3=True)),
]
kept_pairs = [("today's drafter 10.54 / 10.5", 10.54, 10.5), ("+ copy override 10.97 / 12.72 (HumanEval+)", 10.54 * 1.041, 10.5 * 1.211),
              ("DFlash3-XL 13 / 16", 13.0, 16.0), ("MBPP-like 10.54 / 10.51", 10.54, 10.51)]
for name, kw in engines:
    print("\n%s" % name)
    c16 = cycle(16, kept=10.54, **kw); c32 = cycle(32, kept=10.5, **kw)
    print("  cycle 16 rows %.0f ms (dense %.0f, experts %.0f, SSD %.0f); 32 rows %.0f ms (dense %.0f, experts %.0f, SSD %.0f): the 32-row check costs x%.2f, so it pays iff kept32 / kept16 > %.2f"
          % (c16, kw["D16"], kw["E16"], kw["ssd16"] * (1 - kw["hide"]), c32, kw["D16"] + (8.0 if kw.get("g3") else 9.8), kw["E16"] + 16 * kw["slope"], kw["ssd16"] * 1.6 * (1 - (kw["hide"] if kw.get("hide32") is None else kw["hide32"])), c32 / c16, c32 / c16))
    for kname, k16, k32 in kept_pairs:
        t16, t32 = toks(16, k16, **kw), toks(32, k32, **kw)
        print("    %-44s 16 rows %6.1f tok/s | 32 rows %6.1f | ratio %.2f %s" % (kname, t16, t32, t32 / t16, "<- 32 wins" if t32 > t16 else ""))

print("\n== the crossover surface: the kept ratio kept32 / kept16 that a 32-row check needs, by dense ms and expert slope (floor engine, SSD 70 % hidden, night) [P] ==")
print("%10s | " % "dense D16" + " | ".join("slope %.2f (%s)" % (s, n) for s, n in ((3.03, "3c"), (1.41, "NAX6"), (0.9, "floor"), (0.6, "below floor"))))
for D16 in (34, 26, 23, 19, 17, 15, 13):
    row = []
    for s in (3.03, 1.41, 0.9, 0.6):
        kw = dict(D16=D16, E16=27.0, slope=s, ssd16=65.0 * 0.65, hide=0.7, draft16=5.0, draft32=6.5, g3=True)
        row.append(cycle(32, kept=12.0, **kw) / cycle(16, kept=10.5, **kw))
    print("%10d | " % D16 + " | ".join("%17.3f" % r for r in row))
print("read: the copy override gives 1.163 on HumanEval+ (1.211 / 1.041), 1.00 on MBPP, 1.004 on agent; DFlash3-XL would give 1.23.")
