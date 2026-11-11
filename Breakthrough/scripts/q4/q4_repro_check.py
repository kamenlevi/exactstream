#!/usr/bin/env python3
"""Qwen 4 day-0: "today's capture reproduced" (pre-registered SCOREBOARD.md Sun 04:4x). Breakthrough, Sun 2026-11-08.
Compares the patched decision map's capture (DM_OUT=<scratch>/cap, 36 windows) and xcapture (<scratch>/xcap, --max 1 =
3 windows) with the stored ones in ~/FableGod/decision_map/out, read-only:
  1. capture.json tk / lp == out/baseline.json and out/capture.json, every window (also byte-for-byte as files);
  2. routing ids / tokens / gen == out/routing, every window (dtype and shape too);
  3. xcap x / w / ids / a == out/xcap for the 3 windows; the xcapture run's routing and tk / lp too.
PASS = every comparison exact. Writes ~/Breakthrough/data/q4/q4_repro_check.out.
Usage: python -B q4_repro_check.py <scratch dir>"""
import filecmp, json, os, sys
import numpy as np
OUT = os.path.expanduser("~/FableGod/decision_map/out"); DST = os.path.expanduser("~/Breakthrough/data/q4")
SP = sys.argv[1]
lines, bad = [], []
def say(s): print(s, flush=True); lines.append(s)
def same(a, b): return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
def fn(nm): return nm if nm.endswith(".npz") else nm + ".npz"   # window names already end in .npz (Sun 11:4x fix)


def cmp_json(new_p, ref_p, label, only=None):
    N, R = json.load(open(new_p)), json.load(open(ref_p))
    names = [n for n in R if only is None or n in only]
    miss = [n for n in names if n not in N]
    diff = [n for n in names if n in N and (N[n]["tk"] != R[n]["tk"] or N[n]["lp"] != R[n]["lp"]
                                            or N[n]["a"] != R[n]["a"] or N[n]["dom"] != R[n]["dom"])]
    rows = sum(len(R[n]["tk"]) for n in names)
    ok = not miss and not diff and len(names) > 0 and (len(N) == len(R) if only is None else len(names) == len(only))   # no vacuous pass
    say(f"  {label}: {len(names)} windows, {rows} answer rows; missing {len(miss)}, tk/lp/a/dom differ {len(diff)}"
        f"{'' if only else f'; file byte-equal {filecmp.cmp(new_p, ref_p, shallow=False)}'} -> {'OK' if ok else 'FAIL'}")
    if not ok: bad.append(label)
    return names


def cmp_npz(new_d, ref_d, names, keys, label):
    nd = 0; det = []
    for nm in names:
        p = os.path.join(new_d, fn(nm))
        if not os.path.exists(p): nd += 1; det.append(f"{nm}: missing"); continue
        A, B = np.load(p), np.load(os.path.join(ref_d, fn(nm)))
        for k in keys:
            if not same(A[k], B[k]):
                nd += 1; det.append(f"{nm}.{k}: {A[k].dtype}{A[k].shape} vs {B[k].dtype}{B[k].shape}"); break
    say(f"  {label}: {len(names)} windows, keys {keys}; differ {nd} -> {'OK' if nd == 0 else 'FAIL'}"
        + (f" ({'; '.join(det[:4])})" if det else ""))
    if nd: bad.append(label)


say(f"Qwen 4 day-0 capture reproduction: patched dm_common (Breakthrough a03955b) vs the stored map outputs; scratch {SP}")
names = cmp_json(os.path.join(SP, "cap", "capture.json"), os.path.join(OUT, "baseline.json"), "capture.json vs baseline.json")
cmp_json(os.path.join(SP, "cap", "capture.json"), os.path.join(OUT, "capture.json"), "capture.json vs capture.json (11-02)")
cmp_npz(os.path.join(SP, "cap", "routing"), os.path.join(OUT, "routing"), names, ("ids", "tokens", "gen"), "routing")
x3 = sorted(f for f in os.listdir(os.path.join(SP, "xcap", "xcap")) if f.endswith(".npz"))
say(f"  xcapture windows: {x3}")
if len(x3) != 3: bad.append("xcap window count"); say("  xcap window count != 3 -> FAIL")
cmp_json(os.path.join(SP, "xcap", "xcapture.json"), os.path.join(OUT, "xcapture.json"), "xcapture.json vs xcapture.json (11-03)", set(x3))
cmp_npz(os.path.join(SP, "xcap", "xcap"), os.path.join(OUT, "xcap"), x3, ("x", "w", "ids", "a"), "xcap")
cmp_npz(os.path.join(SP, "xcap", "routing"), os.path.join(OUT, "routing"), x3, ("ids", "tokens", "gen"), "xcapture's routing")
say(f"VERDICT: {'PASS' if not bad else 'FAIL: ' + ', '.join(bad)}")
os.makedirs(DST, exist_ok=True)
open(os.path.join(DST, "q4_repro_check.out"), "w").write("\n".join(lines) + "\n")
