"""R-41F REFILL verdict (NewIdeas Fri 17:5x; lines in SCOREBOARD): from refill_arms.sh's outputs in DIR (rf_hash.txt,
rf_A.txt, rf_B.txt, rf_A2.txt if it ran). Row time = prefill + decode as v3o_creep.py times it; decode = after the
prefill (the refill counts in the prefill). Reference = A (mean of A and A2 when A2 ran).
PASS: every hash = R-48's AND write14 and edit10 in B within +3 % of A's row time (A: no PFR on these short rows) AND
text23 (8,582 tokens) decode tok/s in B >= +5 % over A's (A: today's PFR, no refill). KILL: neither time line holds.
Usage: [BigMoE venv] refill_sum.py DIR"""
import os, re, sys

D = sys.argv[1]
RX = re.compile(r"(agent/\w+) RESULT row \d+ .*?\| tok (\d+) \w+ .*?suite tokens equal (\w+) \| prefill ([\d.]+) s, PFR (\w+), "
                r"refill (\[[^\]]*\]|no), decode ([\d.]+) s = ([\d.]+) tok/s, prefill max [\d.]+ \(rise ([-+\d.]+)\) \| footprint start "
                r"[\d.]+ max [\d.]+ \(rise ([-+\d.]+)\).*?\| ([\d.]+) s decode = ")


def rows(arm):
    p = os.path.join(D, f"rf_{arm}.txt")
    if not os.path.exists(p):
        return None
    out = {}
    for ln in open(p):
        m = RX.search(ln)
        if m:
            k, tok, eq, pf, pfr, rf, dec, tps, prise, rise, secs = m.groups()
            out[k.split("/")[1]] = dict(tok=int(tok), eq=eq == "True", pf=float(pf), pfr=pfr, rf=rf, dec=float(dec),
                                        tps=float(tps), prise=float(prise), rise=float(rise), secs=float(secs))
    return out


H = open(os.path.join(D, "rf_hash.txt")).read()
hok = "HASH VERDICT PASS" in H
A, B, A2 = rows("A"), rows("B"), rows("A2")
print(f"R-41F REFILL: hashes {'PASS' if hok else 'FAIL'}; arms A{' A2' if A2 else ''} (B12_PFR_MIN 4096, no refill) vs B "
      f"(PFR_MIN 512 + refill)")
for arm, R in (("A", A), ("B", B), ("A2", A2)):
    if R:
        for k, r in R.items():
            print(f"  {arm:2s} {k:8s} tok {r['tok']:3d} equal {r['eq']} | prefill {r['pf']:6.2f} s PFR {r['pfr']:3s} refill "
                  f"{r['rf']:14s} rise {r['prise']:+.3f} | decode {r['dec']:6.2f} s = {r['tps']:5.2f} tok/s | row {r['secs']:6.2f} s, "
                  f"row max rise {r['rise']:+.3f} GB")
ref = lambda k, f: (A[k][f] + A2[k][f]) / 2 if A2 and k in A2 else A[k][f]
ok_eq = all(r["eq"] for R in (A, B, A2) if R for r in R.values())
short = {k: B[k]["secs"] / ref(k, "secs") - 1 for k in ("write14", "edit10")}
long_ = B["text23"]["tps"] / ref("text23", "tps") - 1
s_ok, l_ok = all(v <= 0.03 for v in short.values()), long_ >= 0.05
if A2:
    print(f"  noise A vs A2: " + ", ".join(f"{k} row {100 * (A2[k]['secs'] / A[k]['secs'] - 1):+.1f} %, decode "
                                            f"{100 * (A2[k]['tps'] / A[k]['tps'] - 1):+.1f} %" for k in A))
v = ("PASS" if hok and s_ok and l_ok and ok_eq else "KILL" if not (s_ok or l_ok) else
     "FAIL (hash / tokens)" if not (hok and ok_eq) else "BETWEEN")
print(f"VERDICT {v}: short rows B vs A " + ", ".join(f"{k} {100 * x:+.1f} %" for k, x in short.items())
      + f" (line <= +3 %); text23 decode {100 * long_:+.1f} % (line >= +5 %); tokens equal everywhere {ok_eq}")
