"""LIP-k for larger k and SRRIP-4 (insert_policy125.py functions), plus per-prompt-half robustness: first half vs second
half of the passes. usage: insert_policy125_sweep.py   (CPU, 1 core)"""
import sys
sys.argv = [sys.argv[0]]
import numpy as np
import insert_policy125 as I                               # runs its own table first (kept for the record)
MB, P = I.MB, I.P
half = P // 2
for C in (64, 96, 128):
    out = []
    for name, fn in [("lru", lambda s: I.lip(s, C, 0))] + [(f"lip-{k}", (lambda k: lambda s: I.lip(s, C, k))(k)) for k in (4, 8, 16, 32)] + \
                    [("srrip-4", lambda s: I.srrip(s, C, 4))]:
        a = sum(fn(s[:half]) for s in I.seqs) / half * MB
        b = sum(fn(s[half:]) for s in I.seqs) / (P - half) * MB
        out.append(f"{name} {a:6.1f}/{b:6.1f}")
    print(f"C={C:3d} MB/pass (1st half / 2nd half)  " + " | ".join(out), flush=True)
