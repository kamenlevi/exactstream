#!/usr/bin/env python3
"""ADAPTK-SWITCH (NewIdeas, Tue 17 Nov; the time is the commit's): the build budget for a 16 <-> 32 row SWITCH.
If the engine pays s ms whenever a check's row count differs from the previous check's (a pipeline / buffer change),
how large may s be before adaptive K's gain falls under the gate's +3 % (and under +5 %)? Reuses adaptk_lossy.py's
data and walk (exact acceptance on E14 costs at the frozen setting: tau 0.9, KERN2, L 0.66; plus CC(1) on v4pass).
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_switch.py
"""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import adaptk_lossy as AL  # noqa: E402

A16, OV, R32, C16, C32 = 62.49, 4.0, 1.4372, 1.025, 1.04
SETTINGS = {"e14 exact (gate)": ("ex", C16 * A16 + OV, R32 * C32 * A16 + OV, 0.66),
            "v4pass CC(1)": ("cc",) + AL.COSTS["v4pass"]}
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def walk_sw(W, O, acc, choose, T16, T32, L, s):
    kept = tm = 0.0; nsw = nch = 0
    for X in W:
        o = O[X["fi"]]; n = X["n"]; p = 1; prev = None
        while p < n:
            r = choose(o, p); a = int(o[acc + str(r)][p])
            if a < 0: r, a, h = 16, 0, 0.0
            else: h = o["h" + str(r)][p]
            sw = prev is not None and r != prev
            kept += a + 1; tm += (T16 if r == 16 else T32) + L * h + (s if sw else 0.0)
            nsw += sw; nch += 1; prev = r; p += a + 1
    return 1000 * kept / tm, nsw, nch

say("ADAPTK-SWITCH: tau 0.9, KERN2; s ms paid per change of row count between consecutive checks")
data = {}
for st, (tag, pub) in AL.SETS.items():
    W = AL.load_cells(tag, pub); data[st] = (W, AL.anchor_arrays(tag, W))
pol = lambda o, p: 16 if max(15, o["npop"][0.9][p]) <= 15 else 32
for name, (acc, T16, T32, L) in SETTINGS.items():
    for st, (W, O) in data.items():
        f16 = walk_sw(W, O, acc, lambda o, p: 16, T16, T32, L, 0)[0]
        f32 = walk_sw(W, O, acc, lambda o, p: 32, T16, T32, L, 0)[0]
        fb = max(f16, f32)
        row = []
        for s in (0, 0.5, 1, 2, 4, 8):
            t, nsw, nch = walk_sw(W, O, acc, pol, T16, T32, L, s)
            row.append(f"s {s}: {100 * (t / fb - 1):+.1f} %")
        _, nsw, nch = walk_sw(W, O, acc, pol, T16, T32, L, 0)
        say(f"[{name}] {st}: switches {nsw} of {nch} checks ({100 * nsw / nch:.0f} %) | " + "; ".join(row))
open(os.path.join(HERE, "adaptk_switch.txt"), "w").write("\n".join(lines) + "\n")
