"""Read-only check of the 24 GB gate's per-prompt peaks and prompt lengths (orchestrator's v1.1.0 number check, Wed 04:4x,
items 1, 4, 5). Source: EvalSuite/results/card/es_cdw24_100136_{fwd,rev}{1..20}_runs.json (EV:w24runs); per prompt, the
lifetime peak is the larger of mem[1] and mem[-1] (SMALL_RAM's key table). Also W24e's six prompts
(BigMoE/data/r57/pc{1024,512,256}g_100_runs.json) and W24g's math order (bg1a_100 / bg2a_100).
usage: python3 -I rd_w24_gate_peaks.py"""
import collections
import json
import pathlib

H = pathlib.Path.home()
E = H / "EvalSuite/results/card"
rows = []
for d in ("fwd", "rev"):
    for i in range(1, 21):
        p = E / f"es_cdw24_100136_{d}{i}_runs.json"
        rs = json.loads(p.read_text())
        proc_peak = max(max(r["mem"][1], r["mem"][-1]) for r in rs)
        for j, r in enumerate(rs, 1):
            pk = max(r["mem"][1], r["mem"][-1])
            rows.append(dict(proc=f"{d}{i}", row=j, key=r["key"], P=r["P"], peak=pk, sets=abs(pk - proc_peak) < 1e-9,
                             first=(j == 1)))
print(f"rows {len(rows)}, processes {len({r['proc'] for r in rows})}")

# item 1: the gate's longest distinct prompts
byP = collections.OrderedDict()
for r in sorted(rows, key=lambda r: -r["P"]):
    byP.setdefault(r["key"], r["P"])
top = list(byP.items())[:14]
print("longest distinct gate prompts:", [(k.split("/")[-1], P) for k, P in top])
w24e = []
for f in ("pc1024g_100", "pc512g_100", "pc256g_100"):
    rs = json.loads((H / f"BigMoE/data/r57/{f}_runs.json").read_text())
    w24e.append(sorted((r["P"] for r in rs), reverse=True))
print("W24e prompts (P):", w24e[0], "| same in all three arms:", w24e[0] == w24e[1] == w24e[2])

# items 4 and 5: around the line
under = [r for r in rows if r["peak"] < 24.0]
over = [r for r in rows if r["peak"] >= 24.0]
print(f"rows under 24.0: {len(under)}, at or over: {len(over)}")
print("shortest prompt at a crossing:", min(r["P"] for r in over))
print("longest prompt that stayed under:", max(r["P"] for r in under))
print("highest peak under the line, top 6:")
for r in sorted(under, key=lambda r: -r["peak"])[:6]:
    print(f"   {r['peak']:.3f} GB  P {r['P']}  {r['key']}  {r['proc']} row {r['row']}  sets proc peak {r['sets']}")
print("rows of 440-600 tokens:")
for r in sorted((r for r in rows if 440 <= r["P"] <= 600), key=lambda r: (r["P"], r["proc"])):
    print(f"   P {r['P']}  {r['peak']:.3f} GB  {r['key']}  {r['proc']} row {r['row']}  first {r['first']}  sets {r['sets']}")
same = collections.defaultdict(list)
for r in rows:
    same[r["key"]].append(r["peak"])
spreads = sorted(((max(v) - min(v), k, len(v)) for k, v in same.items() if len(v) > 1), reverse=True)
print("largest run-to-run spread of one prompt's peak:", [(round(s, 3), k.split('/')[-1], n) for s, k, n in spreads[:5]])
near = [(round(max(v) - min(v), 3), k.split('/')[-1], [round(x, 3) for x in v]) for k, v in same.items()
        if len(v) > 1 and min(v) < 24.0 <= max(v)]
print("prompts on both sides of the line:", near)

# item 8: W24g math order
for f in ("bg1a_100", "bg2a_100"):
    rs = json.loads((H / f"BigMoE/data/r57/{f}_runs.json").read_text())
    print(f, "rows", len(rs), "P by row:", [r["P"] for r in rs], "| 520 at row",
          [i for i, r in enumerate(rs, 1) if r["P"] == 520])
