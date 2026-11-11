#!/usr/bin/env python3
"""Bit-check of the q1u_0_live masks (NewIdeas' builder dm_oea_masks.py, built into ~/Breakthrough/data/dm_masks/q1u_0_live)
against NewIdeas' q1u_0.9_live masks (~/NewIdeas/data/dm_masks/q1u_0.9_live), per NewIdeas' list (Mon 9 Nov 2026 21:16):
(1) the same 36 window files and shapes; (2) every keep is a subset of 0.9's (theta 0 makes more rows sure, so the unions
are smaller); (3) prompt rows keep all 10 in both; (4) the kept share is below 0.9's. Also counts rows that keep all 10 in
the answer part (the bonus rows / undrafted rows) for both. Exit 0 only if (1)-(4) hold. Breakthrough, Mon 9 Nov 21:4x.
Usage: ~/BigMoE/.venv/bin/python -I -B q1u0_mask_check.py
"""
import json, os, sys
import numpy as np

A = os.path.expanduser("~/Breakthrough/data/dm_masks/q1u_0_live")
R = os.path.expanduser("~/NewIdeas/data/dm_masks/q1u_0.9_live")
base = json.load(open(os.path.expanduser("~/FableGod/decision_map/out/baseline.json")))
fa = sorted(f for f in os.listdir(A) if f.endswith(".npz"))
fr = sorted(f for f in os.listdir(R) if f.endswith(".npz"))
ok = True


def say(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL ") + m, flush=True)


say(fa == fr and len(fa) == 36, f"(1) window files: {len(fa)} vs {len(fr)}; same names: {fa == fr}")
shapes = sub = prompt = True
ka = kr = tot = full_a = full_r = 0
per_dom = {}
for f in fa:
    if f not in fr: continue
    a = np.load(os.path.join(A, f))["keep"]; r = np.load(os.path.join(R, f))["keep"]
    if a.shape != r.shape:
        shapes = False; print(f"  shape {f}: {a.shape} vs {r.shape}"); continue
    if (a & ~r).any():
        sub = False; print(f"  not a subset: {f}: {int((a & ~r).sum())} keeps")
    # rows before the walk: row a - 1 (the last prompt row) is the first check's anchor A, whose logits verify the first
    # answer token (dm_q1u_masks: checks are rows A .. A + m), so it is masked like any check row. FIX Mon 23:5x: p0 was
    # a, which failed every window, NewIdeas' scored q1u_0.9_live masks included.
    p0 = int(base[f]["a"]) - 1 if f in base else None
    if p0 is None:
        prompt = False; print(f"  no baseline row for {f}"); continue
    if not (a[:p0].all() and r[:p0].all()):
        prompt = False; print(f"  prompt rows not all kept: {f}")
    aa, rr = a[p0:], r[p0:]
    ka += int(aa.sum()); kr += int(rr.sum()); tot += aa.size
    full_a += int(aa.all(axis=(1, 2)).sum()); full_r += int(rr.all(axis=(1, 2)).sum())
    d = f.split("_")[1]
    s = per_dom.setdefault(d, [0, 0, 0]); s[0] += int(aa.sum()); s[1] += int(rr.sum()); s[2] += aa.size
say(shapes, "(1b) shapes equal")
say(sub, "(2) every q1u_0_live keep is inside q1u_0.9_live's")
say(prompt, "(3) prompt rows keep all 10 in both")
say(ka < kr, f"(4) kept share of answer-row slots: q1u_0_live {ka / tot:.4f} < q1u_0.9_live {kr / tot:.4f}")
for d, (x, y, n) in sorted(per_dom.items()):
    print(f"  {d}: kept share {x / n:.4f} vs {y / n:.4f}")
print(f"  answer rows keeping all 10 (bonus / undrafted rows): q1u_0_live {full_a}, q1u_0.9_live {full_r}")
print("MASK CHECK " + ("PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
