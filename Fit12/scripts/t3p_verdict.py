"""Fit12 T3P + P1-K1 read-out (PLAN.md 'T3P amendment', written Tue 3 Nov 10:4x before any run). CPU only.
Usage: t3p_verdict.py [DIR (/Users/kamen/Fit12/data/scr/t3p)]
Per process (c47 = the verdict, c51 = the replicate): the TOTAL lines, t80ck1u / t80ck2 tok/s and the line (>= 1.10x PASS,
<= 1.03x KILL, else BETWEEN); checks: t80ck2 tokens + lsha = T3t's, t80ck1u's K1U counters engaged, lifetime <= 11.95 GB,
the process finished."""
import os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/t3p"
T3T = {"156": ("ba7f9b3682", "674de3d0b4"), "42": ("33bbf6856a", "eae0a8bcd9"), "68": ("c96b655e9d", "685b9746a8"),
       "91": ("40348025ac", "d6162e5e65")}   # T3t's t80ck2 (tokens sha, lsha), /Users/kamen/Fit12/data/scr/t3t/q1.txt
for tag in ("c47", "c51"):
    f = os.path.join(D, tag + ".txt")
    if not os.path.exists(f):
        print(f"{tag}: no output"); continue
    txt = open(f).read()
    tot = {m.group(1): float(m.group(2)) for m in re.finditer(r"TOTAL (\w+): .*? = ([0-9.]+) tok/s", txt)}
    for l in txt.splitlines():
        if " TOTAL " in l:
            print(l)
    chk = []
    for m in re.finditer(r"HumanEval/(\d+) t80ck2 theta .*?\| tok \d+ sha (\w+) \| lsha (\w+)", txt):
        want = T3T.get(m.group(1))
        chk.append((m.group(1), want is not None and (m.group(2), m.group(3)) == want))
    k1u = [int(m.group(1)) for m in re.finditer(r"of row-layers over KU \((\d+) row-layers\)", txt)]
    k1u_txt = re.findall(r"HumanEval/(\d+) t80ck1u .*?K1U: ([0-9.]+) experts per row-layer, ([0-9.]+) of row-layers over KU", txt)
    life = [float(v) for v in re.findall(r"lifetime ([0-9.]+) GB", txt)]
    done = "] done" in txt
    errs = [l for l in txt.splitlines() if "Error" in l or "Traceback" in l]
    print(f"{tag}: finished {done}; t80ck2 = T3t (tokens + lsha) {chk}; K1U counters engaged "
          f"{bool(k1u) and all(v > 0 for v in k1u)} {k1u_txt}; lifetime max {max(life) if life else float('nan'):.3f} GB "
          f"(<= 11.95: {bool(life) and max(life) <= 11.95}){'; ERRORS: ' + errs[-1][:160] if errs else ''}")
    for a in ("t80ck2m3", "t80ck1um3"):                    # INFO arms (no line)
        if a in tot and "t80ck2" in tot:
            print(f"{tag}: INFO {a} {tot[a]:.2f} tok/s = x{tot[a] / tot['t80ck2']:.3f} of t80ck2"
                  + (f", x{tot[a] / tot['t80ck1u']:.3f} of t80ck1u" if "t80ck1u" in tot else ""))
    if "t80ck1u" in tot and "t80ck2" in tot:
        r = tot["t80ck1u"] / tot["t80ck2"]
        v = "PASS" if r >= 1.10 else "KILL" if r <= 1.03 else "BETWEEN"
        print(f"{tag}: t80ck1u {tot['t80ck1u']:.2f} / t80ck2 {tot['t80ck2']:.2f} tok/s = x{r:.3f} -> {v}"
              f"{' (THE VERDICT: C 47)' if tag == 'c47' else ' (replicate)'}")
