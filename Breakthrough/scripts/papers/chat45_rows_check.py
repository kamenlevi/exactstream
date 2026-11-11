#!/usr/bin/env python3
"""SMALL_RAM section 5.4 number check (Breakthrough, Mon 2026-11-09 07:5x). Read-only on BigMoE's run records.
Re-derives, from ~/BigMoE/data/r57/<arm>_256_runs.json (one record per prompt; STYLE.md section 3 definitions):
  - whole decode pooled = sum(n) / sum(n / tps); geo and median shown for comparison (the reports print geo / median);
  - whole request pooled = sum(n) / sum(pfs + n / tps)  (pfs = prefill call -> first token readback, as BM chat_ttft.txt);
  - first token = median of pfs (min-max);
  - lifetime peak = the largest mem[6] (r57_screen.py: lifetime peak after decode, footprint_peak_gb, 10^9 bytes).
Arms: CHS2 x2c0 / x2c0b / x2qm1 / x2qm2 / x2qm2s / x2e (the 5.4 table), CHS5 z0 + z0b (the chat configuration, two passes),
PF-D pdz4a (chat as shipped) and its control pdz0. Writes ~/Breakthrough/data/papers/chat45_rows_check.txt.
Usage: python3 -B chat45_rows_check.py"""
import json, math, os, statistics as st

D = os.path.expanduser("~/BigMoE/data/r57")
OUT = os.path.expanduser("~/Breakthrough/data/papers/chat45_rows_check.txt")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def load(a):
    return json.load(open(os.path.join(D, f"{a}_256_runs.json")))


def row(rs):
    n = sum(r["n"] for r in rs); dec = sum(r["n"] / r["tps"] for r in rs); req = sum(r["pfs"] + r["n"] / r["tps"] for r in rs)
    t = sorted(r["pfs"] for r in rs)
    return dict(rows=len(rs), n=n, pooled=n / dec, geo=math.exp(sum(math.log(r["tps"]) for r in rs) / len(rs)),
                med=st.median(r["tps"] for r in rs), req=n / req, ttft=st.median(t), tmin=t[0], tmax=t[-1],
                tsum=sum(t), wall=req, peak=max(r["mem"][6] for r in rs))


def show(label, x):
    say(f"  {label:7s} rows {x['rows']:2d} tokens {x['n']:5d} | whole decode pooled {x['pooled']:.2f} (geo {x['geo']:.2f}, "
        f"median {x['med']:.2f}) | whole request pooled {x['req']:.2f} | first token median {x['ttft']:.4f} s "
        f"({x['tmin']:.2f}-{x['tmax']:.2f}; total {x['tsum']:.2f} of {x['wall']:.2f} s) | lifetime peak {x['peak']:.4f} GB")


say("SMALL_RAM 5.4 number check from BigMoE's run records (data/r57/<arm>_256_runs.json), read-only")
say("CHS2 (BM:chs2; the 5.4 table):")
for a in ("x2c0", "x2c0b", "x2qm1", "x2qm2", "x2qm2s", "x2e"):
    show(a, row(load(a)))
say("CHS5 chat configuration M1-B + COH (BM:chs5), per pass and two passes pooled:")
z = {a: load(a) for a in ("z0", "z0b")}
for a, rs in z.items():
    show(a, row(rs))
show("z0+z0b", row(z["z0"] + z["z0b"]))
say("PF-D (BM:pfd): chat as shipped (pdz4a = zs + PF-D v4 + read-ahead) and its control (pdz0 = zs):")
for a in ("pdz4a", "pdz0"):
    show(a, row(load(a)))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
