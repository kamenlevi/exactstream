"""b37 = R-37 SB-24 A/B (NewIdeas Tue ~21:00; Leap's R-11c 'SB' on the 24 GB v3 decode): 8 runs A1 B1 B2 A2 A3 B3 B4 A4
of scripts/s10_anatomy.py 0.25 256 (code / prose / edit, no trace). B = A + BIGMOE_VERIFY_MIDLOOP=4 (MTPLX's mid-loop
async_eval in verify forwards) + SPEC125_COMMIT_ASYNC=1 (the commit replay goes out on its own).
PASS: tok/s +2 % or more (B vs A, mean over the 3 prompts of each arm's mean) AND every sha equal. KILL: < +0.5 % or any
sha differs. Else BETWEEN. usage: r37_ab.py DIR"""
import json, os, sys
import numpy as np

D = sys.argv[1]
ARMS = {"A": ("A1", "A2", "A3", "A4"), "B": ("B1", "B2", "B3", "B4")}
R = {r: json.load(open(os.path.join(D, r + ".json")))["prompts"] for rs in ARMS.values() for r in rs}
P = list(R["A1"])
exact = all(R[r][p]["sha"] == R["A1"][p]["sha"] for r in R for p in P)
clean = all(R[r][p].get("clean", True) for r in R for p in P)
print("\n".join(__doc__.split("\n")[3:5]))
print(f"exact: {'yes, every sha equal across the 8 runs' if exact else 'NO'}; clean (no PAUSE / gaps): {clean}")
d_all = []
for p in P:
    t = {a: [R[r][p]["tok_s"] for r in rs] for a, rs in ARMS.items()}
    v = {a: [R[r][p]["verify_ms"] for r in rs] for a, rs in ARMS.items()}
    d = 100 * (np.mean(t["B"]) / np.mean(t["A"]) - 1)
    d_all.append(d)
    sd = 100 * np.std(t["A"], ddof=1) / np.mean(t["A"])
    print(f"  {p:6s} tok/s A {np.mean(t['A']):6.2f} ({' '.join(f'{x:.1f}' for x in t['A'])})  B {np.mean(t['B']):6.2f} "
          f"({' '.join(f'{x:.1f}' for x in t['B'])})  {d:+.1f} % (A run sd {sd:.1f} %); verify ms/pass A "
          f"{np.mean(v['A']):.1f} B {np.mean(v['B']):.1f}")
D_ = float(np.mean(d_all))
verdict = ("INVALID (not exact)" if not exact else "PASS" if D_ >= 2 else "KILL" if D_ < 0.5 else "BETWEEN")
if os.environ.get("AB_LINES") == "r16":                  # b42 (R-16 overlap): B = A + a paced GPU rANS decode process
    import glob, re
    bg = [open(f).read() for f in sorted(glob.glob(os.path.join(D, "bg_B*.log")))]
    rates = [float(m) for b in bg for m in re.findall(r"= ([\d.]+) GB/s of codes decoded", b)]
    bad = [int(m) for b in bg for m in re.findall(r"mismatching bytes (\d+)", b)]
    lo = min(rates) if rates else 0.0
    print(f"background decode: {' / '.join(f'{r:.2f}' for r in rates)} GB/s of codes over the B runs; mismatching bytes {bad}")
    print("lines (b42): PASS engine tok/s >= -2 % AND every B run's decode >= 5.5 GB/s AND 0 mismatches; KILL < -5 %, "
          "decode < 3 GB/s, or any mismatch")
    verdict = ("INVALID (not exact)" if not exact else
               "KILL" if D_ < -5 or lo < 3 or any(bad) or len(rates) < 4 else "PASS" if D_ >= -2 and lo >= 5.5 else "BETWEEN")
print(f"B vs A: tok/s {D_:+.1f} % (mean of the 3 prompts)  -> {verdict}")
json.dump(dict(d=d_all, mean=D_, exact=exact, clean=clean, verdict=verdict), open(os.path.join(D, "ab.json"), "w"), indent=1)
