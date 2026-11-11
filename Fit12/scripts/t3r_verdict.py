"""Fit12 T3R / T3R-PF read-out (PLAN.md 'T3R' AGREED amendment, 'T3R-PF', 'T3R NOW'; every line was written before the
runs). CPU only. Usage: t3r_verdict.py [DIR (/Users/kamen/Fit12/data/scr/t3rn)]
T3R (r47): the shared-text metric (t3_shared.py) of t80ck1u, t80ck2m3 and t80ck1um3 vs t80ck2; coverage = shared tokens /
the arm's total tokens (< 50 % for either arm = no verdict); line: t80ck1u >= 1.10x PASS, <= 1.03x KILL, else BETWEEN
(t80ck2m3 / t80ck1um3 INFO). Checks: t80ck2 tokens + lsha = T3t's on HumanEval 156 / 42 / 68 / 91, K1U counters engaged,
lifetime <= 11.95 GB, '] done'.
T3R-PF (p47): every GATE PF2 line PASS (else no timing verdict: the add-on is not exact); summed tokens / summed decode s:
t80ck1upf / t80ck1u >= 1.05x PASS, <= 1.01x KILL, else BETWEEN; t80ck2pf / t80ck2 the second verdict on the same line;
PF2 parts per prompt (late ids posted, loader late reads, late ids forced to wait: must be 0)."""
import os, re, subprocess, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/t3rn"
T3T = {"156": ("ba7f9b3682", "674de3d0b4"), "42": ("33bbf6856a", "eae0a8bcd9"), "68": ("c96b655e9d", "685b9746a8"),
       "91": ("40348025ac", "d6162e5e65")}   # T3t's t80ck2 (tokens sha, lsha), /Users/kamen/Fit12/data/scr/t3t/q1.txt


def totals(txt):
    return {m.group(1): (int(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"TOTAL (\w+): \d+ prompts, (\d+) tokens in [0-9.]+ s = ([0-9.]+) tok/s", txt)}


def common(tag, txt):
    life = [float(v) for v in re.findall(r"lifetime ([0-9.]+) GB", txt)]
    errs = [l for l in txt.splitlines() if "Error" in l or "Traceback" in l]
    return (f"{tag}: finished {'] done' in txt}; lifetime max {max(life) if life else float('nan'):.3f} GB "
            f"(<= 11.95: {bool(life) and max(life) <= 11.95})" + (f"; ERRORS: {errs[-1][:160]}" if errs else ""))


f = os.path.join(D, "r47.txt")
if os.path.exists(f):
    txt = open(f).read()
    tot = totals(txt)
    print("\n".join(l for l in txt.splitlines() if " TOTAL " in l))
    chk = [(m.group(1), (m.group(2), m.group(3)) == T3T[m.group(1)])
           for m in re.finditer(r"HumanEval/(\d+) t80ck2 theta .*?\| tok \d+ sha (\w+) \| lsha (\w+)", txt) if m.group(1) in T3T]
    k1u = re.findall(r"HumanEval/(\d+) t80ck1u theta .*?K1U: ([0-9.]+) experts per row-layer, ([0-9.]+) of row-layers over KU", txt)
    print(common("r47", txt) + f"; t80ck2 = T3t (tokens + lsha) {chk} -> {bool(chk) and all(v for _, v in chk)}; "
          f"K1U engaged {bool(k1u) and all(float(v[1]) > 0 for v in k1u)} {k1u}")
    out = subprocess.run([sys.executable, "-B", "/Users/kamen/Fit12/scripts/t3_shared.py", D, "r47", "t80ck2", "t80ck1u",
                          "t80ck2m3", "t80ck1um3"], capture_output=True, text=True)
    print(out.stdout.rstrip())
    if out.returncode:
        print("t3_shared.py FAILED:", out.stderr[-600:])
    for m in re.finditer(r"shared text (\w+) vs (\w+): \w+ (\d+) tok / [0-9.]+ s = [0-9.]+ tok/s \(kept [0-9.]+\) \| \w+ (\d+) "
                         r"tok / [0-9.]+ s = [0-9.]+ tok/s \(kept [0-9.]+\) \| x([0-9.]+)", out.stdout):
        b, a, ta, tb, x = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), float(m.group(5))
        ca = ta / tot[a][0] if a in tot else float("nan")
        cb = tb / tot[b][0] if b in tot else float("nan")
        cov = ca >= 0.5 and cb >= 0.5
        if b == "t80ck1u":
            v = ("PASS" if x >= 1.10 else "KILL" if x <= 1.03 else "BETWEEN") if cov else "NO VERDICT (coverage < 50 %)"
            print(f"T3R VERDICT: t80ck1u / t80ck2 on shared text x{x:.3f}, coverage {ca:.0%} / {cb:.0%} -> {v}")
        else:
            print(f"T3R INFO: {b} / {a} on shared text x{x:.3f}, coverage {ca:.0%} / {cb:.0%}")
else:
    print(f"r47: no output in {D}")

f = os.path.join(D, "p47.txt")
if os.path.exists(f):
    txt = open(f).read()
    tot = totals(txt)
    print("\n".join(l for l in txt.splitlines() if " TOTAL " in l or "GATE PF2" in l))
    gates = re.findall(r"GATE PF2 \((\w+) = (\w+), tokens \+ logits\): (PASS|FAIL)", txt)
    pf2 = re.findall(r"HumanEval/(\d+) (t80ck\w*pf) theta .*?PF2: (\d+) late ids posted in (\d+) wide calls, loader late reads "
                     r"(\d+), forced to wait (\d+)", txt)
    gok = bool(gates) and all(g[2] == "PASS" for g in gates)
    print(common("p47", txt) + f"; GATE PF2 {len(gates)} lines, all PASS {gok}; PF2 (prompt, arm, posted, calls, late reads, "
          f"forced) {pf2}; forced to wait total {sum(int(p[5]) for p in pf2)}")
    if gok:
        for a, b, name in (("t80ck1upf", "t80ck1u", "VERDICT"), ("t80ck2pf", "t80ck2", "second verdict")):
            if a in tot and b in tot:
                x = tot[a][1] / tot[b][1]
                print(f"T3R-PF {name}: {a} {tot[a][1]:.2f} / {b} {tot[b][1]:.2f} tok/s = x{x:.3f} -> "
                      f"{'PASS' if x >= 1.05 else 'KILL' if x <= 1.01 else 'BETWEEN'}")
    else:
        print("T3R-PF: GATE PF2 not all PASS -> no timing verdict (the add-on is not exact)")
else:
    print(f"p47: no output in {D} (skipped or not run)")
