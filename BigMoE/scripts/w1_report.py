"""W1B's scoreboard row (docs/WALL1_BENCH.md) at 45 GB from a traced arm's R57_STALL dump + an untraced arm's runs JSON,
with the 24 GB harness's definitions (s10b_trace.py drain_trace), so both lines' rows compare:
- SSD busy per check: the union of all read intervals (demand, prefetch, joint) inside each check's window;
- exposed per check: the union of the loader's [seen, release] intervals of calls with misses (kind 4), inside the window
  (an upper bound on the GPU's wait: under OOO the resident experts' work runs beside part of it);
- calls per check: calls with misses (kind 4);
- MB per kept token and steady tok/s (checks >= 2): the untraced arm's runs (mean over prompts);
- exact: both arms' tokens == the reference where stored (r57_screen's REFTOK line).
usage: .venv/bin/python scripts/w1_report.py TRACED_TAG UNTRACED_TAG [DIR=data/r57]   e.g. c45t_160 ka0_160"""
import json, os, sys
import numpy as np

TT, UT = sys.argv[1], sys.argv[2]
D = os.path.expanduser(sys.argv[3] if len(sys.argv) > 3 else "~/BigMoE/data/r57")


def union_len(iv):
    iv = sorted(iv); tot = 0.0; s = e = None
    for a, b in iv:
        if s is None or a > e:
            if s is not None:
                tot += e - s
            s, e = a, b
        else:
            e = max(e, b)
    return tot + (e - s if s is not None else 0.0)


def clip(iv, a, b):
    return [(max(x, a), min(y, b)) for x, y in iv if y > a and x < b]


d = json.load(open(os.path.join(D, f"{TT}_stall.json")))
busy, expo, calls = [], [], []
for run in d["runs"]:
    if run.get("paused"):
        continue
    rd = [(r[3], r[4]) for r in run["rd"]]
    slow = [(t[3], t[4]) for t in run["tr"] if int(t[2]) == 4]
    for c in run["chk"]:
        a, b = c[1], c[4]                                # check start -> commit submitted
        busy.append(union_len(clip(rd, a, b)))
        expo.append(union_len(clip(slow, a, b)))
        calls.append(sum(1 for x, y in slow if a <= x < b))
row = {"tag": f"{TT} / {UT}", "checks": len(busy),
       "ssd_busy_ms": round(1e3 * float(np.mean(busy)), 1) if busy else None,
       "exposed_ms": round(1e3 * float(np.mean(expo)), 1) if expo else None,
       "calls_per_check": round(float(np.mean(calls)), 1) if calls else None}
try:
    rs = [r for r in json.load(open(os.path.join(D, f"{UT}_runs.json"))) if not r.get("bad")]
    row["mb_per_kept"] = round(float(np.mean([r["mb"] for r in rs])), 1)
    row["tok_s_steady"] = round(float(np.mean([r["tpsw"] for r in rs])), 1)
except Exception as e:
    row["untraced"] = f"missing ({e})"
print(json.dumps(row))
print(f"| {UT.rsplit('_', 1)[-1]} | {row['ssd_busy_ms']} | {row['exposed_ms']} | {row['calls_per_check']} | "
      f"{row.get('mb_per_kept', '-')} | {row.get('tok_s_steady', '-')} | (see REFTOK lines) |")
