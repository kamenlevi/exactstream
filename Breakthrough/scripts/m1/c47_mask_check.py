#!/usr/bin/env python3
"""Bit-check of the qom_0_live_c47 masks (NewIdeas' builder dm_oea_masks.py, arm qom_0_live_c47 = QOM at theta 0 with a
47-slot base pool per layer, Fit12's call pool) before the map arm runs (Breakthrough, Mon 9 Nov 2026 23:4x):
(1) the same 36 window files and shapes as qom_0_live_c51 (~/Breakthrough/data/m1/masks); (2) every q1u_0_live keep
(~/Breakthrough/data/dm_masks/q1u_0_live) is inside c47's: QOM = Q1-UNION + resident experts at the same theta, so this holds
by construction; (3) prompt rows keep all 10. Prints the answer-row kept shares of q1u_0_live, c47 and c51 (c47 is expected
at or below c51: a smaller pool keeps fewer residents; LRU on a different touch stream gives no strict inclusion, so this
is information). Exit 0 only if (1)-(3) hold.
Usage (BigMoE venv): python -I -B c47_mask_check.py
"""
import json, os, sys
import numpy as np

C47 = os.path.expanduser("~/Breakthrough/data/m1/masks/qom_0_live_c47")
C51 = os.path.expanduser("~/Breakthrough/data/m1/masks/qom_0_live_c51")
Q1U = os.path.expanduser("~/Breakthrough/data/dm_masks/q1u_0_live")
base = json.load(open(os.path.expanduser("~/FableGod/decision_map/out/baseline.json")))
ls = lambda d: sorted(f for f in os.listdir(d) if f.endswith(".npz")) if os.path.isdir(d) else []
f47, f51, fq = ls(C47), ls(C51), ls(Q1U)
ok = True


def say(c, m):
    global ok
    ok = ok and bool(c)
    print(("PASS " if c else "FAIL ") + m, flush=True)


say(f47 == f51 and len(f47) == 36, f"(1) window files: c47 {len(f47)}, c51 {len(f51)}; same names: {f47 == f51}")
say(fq == f47, f"(2a) q1u_0_live has the same 36 files ({len(fq)})")
shapes = sub = prompt = True
k = {"q1u": [0, 0], "c47": [0, 0], "c51": [0, 0]}
for f in f47:
    a = np.load(os.path.join(C47, f))["keep"]
    r = np.load(os.path.join(C51, f))["keep"] if f in f51 else None
    q = np.load(os.path.join(Q1U, f))["keep"] if f in fq else None
    if (r is not None and r.shape != a.shape) or (q is not None and q.shape != a.shape):
        shapes = False; print(f"  shape {f}: c47 {a.shape}, c51 {None if r is None else r.shape}, q1u {None if q is None else q.shape}")
        continue
    if q is not None and (q & ~a).any():
        sub = False; print(f"  q1u_0_live keeps outside c47's: {f}: {int((q & ~a).sum())}")
    # rows before the walk: row a - 1 (the last prompt row) is the first check's anchor A, whose logits verify the first
    # answer token (dm_q1u_masks: checks are rows A .. A + m), so it is masked like any check row. FIX Mon 23:5x: p0 was
    # a, which failed every window, NewIdeas' scored q1u_0.9_live masks included.
    p0 = int(base[f]["a"]) - 1 if f in base else None
    if p0 is None:
        prompt = False; print(f"  no baseline row for {f}"); continue
    if not a[:p0].all():
        prompt = False; print(f"  prompt rows not all kept: {f}")
    for nm, m in (("c47", a), ("c51", r), ("q1u", q)):
        if m is not None:
            k[nm][0] += int(m[p0:].sum()); k[nm][1] += m[p0:].size
say(shapes, "(1b) shapes equal")
say(sub, "(2) every q1u_0_live keep is inside qom_0_live_c47's (QOM = Q1-UNION + residents)")
say(prompt, "(3) prompt rows keep all 10")
sh = {nm: (x / n if n else float("nan")) for nm, (x, n) in k.items()}
print(f"  answer-row kept shares: q1u_0_live {sh['q1u']:.4f} <= c47 {sh['c47']:.4f}; c51 {sh['c51']:.4f} "
      f"(c47 {'<=' if sh['c47'] <= sh['c51'] else '>'} c51)", flush=True)
print("MASK CHECK " + ("PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
