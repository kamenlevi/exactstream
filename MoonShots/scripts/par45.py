"""Parallel-answer fast mode at 45 GB (NewIdeas Mon 03:35, design only): price k independent streams of one answer x
16-row checks per pass on the all-resident non-exact 45 GB stack, from L1-32's measured expert unions (no runs).
Bytes per check = routed experts (NewIdeas' '! 350 at 45 GB' 1x16 figure 4.9 GB, scaled by L1-32's union ratio)
+ dense diet 2.1 GB read once per pass (shared by all rows) + each stream's GDN state read and written once (36 layers x
48 heads x 128 x 128 fp32 = 113 MB). Time = bytes / 280 GB/s + non-byte time: 'lean' 15 ms flat (the '! 350' row's
assumption) or 'rows' 15 -> 37.5 ms at 64 rows (GDN mixers, attention, drafter and glue grow with rows; L1-13: the 35B's
GDN mixers already cost 12.2 ms vs a 2.04 ms floor at 16 rows). Each stream accepts 9.5 tokens per check (L1-32's
code-like acceptance). Answer model: router 5 + skeleton 40 tokens serial, then k = 4 parts with lengths mean x
[0.75, 0.9, 1.1, 1.25]; streams retire as parts finish. usage: python par45.py  -> stdout"""
U8 = {1: 54.6, 2: 87.9, 3: 109.7, 4: 131.5}       # top-8 union per layer, L1-32 mean of 4 sets (3 streams interpolated)
U10 = {1: 66.1, 2: 105.4, 3: 130.5, 4: 155.5}     # top-10
E1, DENSE, GDN, BW, A = 4.9, 2.1, 0.113, 280.0, 9.5


def t_check(k, nb, U=U8):
    gb = E1 * U[k] / U[1] + DENSE + 2 * GDN * k
    return gb, gb / BW * 1e3 + nb(k)


NB = {"lean": lambda k: 15.0, "rows": lambda k: 15.0 + 22.5 * (k - 1) / 3}


def answer(parts, nb, serial=45):
    t = serial / A * t_check(1, nb)[1]
    left, done = sorted(parts), 0.0
    while left:
        k = len(left)
        step = left[0] - done                          # tokens until the shortest active part finishes
        t += step / A * t_check(k, nb)[1]
        done = left[0]
        left = [p for p in left if p > done]
    return t


for name, nb in NB.items():
    print(f"[{name} non-byte time]")
    for k in (1, 2, 3, 4):
        gb, ms = t_check(k, nb)
        gb10, ms10 = t_check(k, nb, U10)
        print(f"  {k} stream(s) x 16 rows: {gb:5.2f} GB, {ms:5.1f} ms per check, {k * A:4.1f} tokens -> {1e3 * k * A / ms:4.0f} tok/s"
              f"   (top-10: {gb10:5.2f} GB, {1e3 * k * A / ms10:4.0f} tok/s)")
    one = 1e3 * A / t_check(1, nb)[1]
    for normal, longer in ((600, 1.0), (600, 1.3)):
        body = normal * longer - 45
        parts = [body / 4 * f for f in (0.75, 0.9, 1.1, 1.25)]
        t = answer(parts, nb)
        print(f"  SoT answer {normal * longer:.0f} tokens (normal {normal}, x{longer}): {t:5.0f} ms -> {normal * longer / t * 1e3:4.0f} "
              f"tok/s visible ({normal * longer / t * 1e3 / one:.2f}x); time to answer {normal / one * 1e3:5.0f} -> {t:5.0f} ms "
              f"({normal / one * 1e3 / t:.2f}x)")
    g = (600 * 1.3 / answer([(600 * 1.3 - 45) / 4 * f for f in (0.75, 0.9, 1.1, 1.25)], nb) * 1e3) / one
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        print(f"  mix: {100 * f:3.0f} % of output tokens in parallel answers -> {one / ((1 - f) + f / g):4.0f} tok/s")
    print(f"  350 needs {100 * (1 - one / 350) / (1 - 1 / g):.0f} % of output tokens in parallel answers (gain {g:.2f}x on them)")
