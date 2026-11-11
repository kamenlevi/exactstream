"""CT1 judge (NewIdeas Sun 11-08; lines pre-registered 08:19 in QUEUE c5a7e2c, before any run).
Question: does the 45 GB chat-turn prefill re-read experts its pools already hold (PF-D for chat)?
Data: BigMoE PFB's ct1 arm (zs + R57_PFSTAT=1, 12 IFEval = N45 keys, CAP 16): data/r57/ct1_256_runs.json, one record
per prompt with pfs (TTFT s) and pfst = res_base / res_side (pool expert-layers resident at prefill start), gb (the
prefill's SSD GB), routed (expert-layers the prefill routes to), routed_res_base / routed_res_side (of those, resident).
Reference: Fit12's 12 GB prefill of the same prompts (data/scr/chat12/chat12.txt PREFILL lines, SSD MB), which has
almost no pools, so it reads every routed expert-layer.
Lines (verbatim from the board):
  KILL if the 45 GB chat prefill already reads <= 60 % of 12 GB's bytes per prompt (pooled), or mean base residency at
  start < 25 %. PASS-to-build if >= 80 % of 12 GB's bytes AND mean base residency >= 50 %. Else BETWEEN.
'Base residency at start' is PFX's measure (res_base / (48 x 512): PFX's 36-57 %); the routed-weighted share
(routed_res_base / routed: what a resident-serving prefill would not read) is printed beside it as information.
usage: /usr/bin/python3 -I ct1_judge.py [runs.json]"""
import json, os, re, sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/BigMoE/data/r57/ct1_256_runs.json")
REF = os.path.expanduser("~/Fit12/data/scr/chat12/chat12.txt")
NLAY, NEXP = 48, 512

ref = {}
for line in open(REF):
    m = re.search(r"\] (\S+) ex PREFILL n (\d+) .*\| ([\d.]+) s, [\d.]+ tok/s, SSD (\d+) MB", line)
    if m:
        ref[m.group(1)] = dict(n=int(m.group(2)), s=float(m.group(3)), gb=int(m.group(4)) / 1e3)
runs = json.load(open(RUNS))
runs = runs if isinstance(runs, list) else [runs]
rows, miss = [], []
for r in runs:
    k, p = r.get("key"), r.get("pfst")
    if not p or k not in ref:
        miss.append(k); continue
    rows.append((k, r.get("P"), r.get("pfs"), p, ref[k]))
print(f"CT1 judge: {len(rows)} prompts with PFSTAT and a 12 GB reference" + (f"; skipped {miss}" if miss else ""))
print(f"{'key':14s} {'P':>4s} {'TTFT45':>7s} {'TTFT12':>7s} {'GB45':>7s} {'GB12':>7s} {'45/12':>6s} {'res_b%':>7s} "
      f"{'res_s%':>7s} {'routed':>7s} {'rt_res_b%':>9s} {'rt_res_s%':>9s}")
g45 = g12 = rb = rr = rrb = 0.0
for k, P, pfs, p, f in rows:
    resb = p["res_base"] / (NLAY * NEXP); ress = p["res_side"] / (NLAY * NEXP)
    rt = p.get("routed"); rtb = p.get("routed_res_base"); rts = p.get("routed_res_side")
    g45 += p["gb"]; g12 += f["gb"]; rb += resb
    if rt:
        rr += rt; rrb += rtb
    print(f"{k:14s} {P or 0:4d} {pfs or 0:7.3f} {f['s']:7.3f} {p['gb']:7.3f} {f['gb']:7.3f} {p['gb'] / f['gb']:6.3f} "
          f"{100 * resb:7.1f} {100 * ress:7.1f} {rt or 0:7d} "
          + (f"{100 * rtb / rt:9.1f} {100 * rts / rt:9.1f}" if rt else f"{'-':>9s} {'-':>9s}"))
n = len(rows)
if not n:
    sys.exit("no rows: nothing to judge")
ratio, mresb = g45 / g12, rb / n
print(f"POOLED: 45 GB reads {g45:.2f} GB vs 12 GB {g12:.2f} GB = {100 * ratio:.1f} %; mean base residency at start "
      f"{100 * mresb:.1f} %" + (f"; routed-weighted resident base share {100 * rrb / rr:.1f} % (information)" if rr else ""))
if ratio <= 0.60 or mresb < 0.25:
    v = "KILL" + (" (45 GB already reads <= 60 % of 12 GB's bytes)" if ratio <= 0.60 else "") + \
        (" (mean base residency < 25 %)" if mresb < 0.25 else "")
elif ratio >= 0.80 and mresb >= 0.50:
    v = "PASS-to-build (PF-D for chat: route to BigMoE)"
else:
    v = "BETWEEN"
print(f"VERDICT: {v}")
