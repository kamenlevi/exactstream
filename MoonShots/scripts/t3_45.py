"""R-17 T3-45 (NewIdeas Tue 19:3x; pricing only, no runs): TRL-3 (3INST, ~3.07 b, 0.50x MSB-3 error) as a NON-exact
45 GB fast mode, in par45's bytes model: one 16-row check = experts' time + the rest; TRL-3 scales the experts' bytes by
x0.72 (NewIdeas, vs today's exact 4-bit top-10 resident experts) or x0.68 (T3R's measured base record 1,884,160 B vs
MLX 4-bit's 2,764,800 B per expert). Rest = Leap's CA anatomy (exact 16-row check 143 ms, experts 58.7 ms, drafting
12.5 ms) for today, and CA's 'every lever together ~55-60 ms' cycle for all levers (experts at 90 % / 100 % of their
byte floor). Cross-check with MEASURED kernel time instead of bytes: Leap's B10(b) (data/b10_bench.txt, layer 24, real
routing, S = 16): 3c 4-bit 1.240 ms / layer, 3c 3-bit 0.914, 3INST trellis base (k = 2, ~2.1 b) 1.257.
PASS/KILL (NewIdeas): lever if tok/s rises >= 20 % at the all-levers time; KILL < 10 %.
usage: python3 t3_45.py -> stdout"""
U10, LAYERS, BW = 66.1, 48, 280.0                  # top-10 union per layer per 16-row check (L1-32), GB/s
E4 = U10 * LAYERS * 2764800 / 1e9                  # exact 4-bit experts read per 16-row check, GB
CA_CHECK, CA_EXP, DRAFT = 143.0, 58.7, 12.5        # Leap's CA (ms)


def row(name, exp, rest, f):
    t0, t1 = exp + rest, exp * f + rest
    return f"  {name:58s} cycle {t0:6.1f} -> {t1:6.1f} ms: {100 * (t0 / t1 - 1):+5.1f} % tok/s"


print(f"experts per 16-row check, exact 4-bit top-10: {E4:.2f} GB (byte floor {E4 / BW * 1e3:.1f} ms at {BW:.0f} GB/s)")
for f in (0.72, 0.68):
    print(f"TRL-3 at x{f:.2f} experts' bytes, experts' time proportional to bytes (decode free):")
    print(row("today (CA: check 143 + drafting 12.5; experts 58.7)", CA_EXP, CA_CHECK - CA_EXP + DRAFT, f))
    for cyc in (55.0, 60.0):
        for eff in (0.9, 1.0):
            exp = E4 / BW * 1e3 / eff
            print(row(f"all levers (cycle {cyc:.0f}), experts at {100 * eff:.0f} % of floor ({exp:.1f} ms)", exp, cyc - exp, f))
k4, k3, ktr = 1.240, 0.914, 1.257                  # ms / layer at S = 16 (B10(b))
print(f"measured kernels at S = 16 (B10(b)): 3c 4-bit {k4} ms / layer, 3c 3-bit {k3} ({k3 / k4:.2f}x for {3.52 / 4.77:.2f}x "
      f"bytes: 3c is read-bound), 3INST trellis k = 2 base {ktr} ({ktr / k4:.2f}x of 4-bit at 0.45x the bytes: decode-bound)")
for nm, rest in (("today", CA_CHECK - CA_EXP + DRAFT), ("all levers, cycle 55", 55.0 - E4 / BW * 1e3 / 0.9)):
    exp = CA_EXP if nm == "today" else E4 / BW * 1e3 / 0.9
    print(row(f"{nm}, TRL-3 kernel = B10's trellis time ({ktr / k4:.2f}x 4-bit)", exp, rest, ktr / k4))
