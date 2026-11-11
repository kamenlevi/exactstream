"""Fit12 pos-2 follow-up (Tue 27 Oct 2026; PLAN.md 'pos-2 follow-up'): compare the prefill logits that fit_cmp.py saved
(F12_SAVE_LOGITS) for a prompt prefilled twice in one process, pos 1 vs pos 2, in arms s1 (the fold stack) and s0 (stock).
Usage: [BigMoE venv] fit_pos2.py DIR   (CPU, numpy only)"""
import os, sys
import numpy as np

d = sys.argv[1]
moved = {}
for arm in ("s1", "s0"):
    a, b = (np.load(os.path.join(d, f"{arm}_pos{i}.npy")) for i in (1, 2))
    msg = f"[pos2b] {arm}: shapes {a.shape} / {b.shape}"
    if a.shape == b.shape:
        a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        nd = int((a != b).sum())
        moved[arm] = nd > 0
        top_a, top_b = np.argsort(-a2[-1])[:10], np.argsort(-b2[-1])[:10]
        msg += (f"; {nd} of {a.size} values differ, max |diff| {np.abs(a - b).max():.3g} (logit range {a.min():.1f}..{a.max():.1f})"
                f"; argmax equal in every row {bool((a2.argmax(-1) == b2.argmax(-1)).all())}; last row bitwise equal "
                f"{bool((a2[-1] == b2[-1]).all())}; last-row top-10 ids equal {bool((top_a == top_b).all())}")
    else:
        moved[arm] = True
    print(msg, flush=True)
p1, p0 = np.load(os.path.join(d, "s1_pos1.npy")), np.load(os.path.join(d, "s0_pos1.npy"))
print(f"[pos2b] pos 1, s1 vs s0: {'bitwise equal' if p1.shape == p0.shape and (p1 == p0).all() else 'differ'}", flush=True)
print(f"[pos2b] verdict: {'PRE-EXISTING (stock moves too)' if moved.get('s0') else ('FOLD (only the fold stack moves)' if moved.get('s1') else 'NONE (neither moves)')}", flush=True)
if sys.argv[2:] == ["ids"]:                              # pos2c (27 Oct): which vocab ids differ, and which way
    for arm in ("s0", "s1"):
        a, b = (np.load(os.path.join(d, f"{arm}_pos{i}.npy")).reshape(-1) for i in (1, 2))
        ix = np.nonzero(a != b)[0]
        print(f"[pos2c] {arm}: ids {ix.tolist()}", flush=True)
        print(f"[pos2c] {arm}: pos 1 {[f'{x:.4g}' for x in a[ix]]}", flush=True)
        print(f"[pos2c] {arm}: pos 2 {[f'{x:.4g}' for x in b[ix]]}", flush=True)
        print(f"[pos2c] {arm}: masked (< -1e29) pos 1 {int((a < -1e29).sum())}, pos 2 {int((b < -1e29).sum())}; top-10 pos 1 "
              f"{np.argsort(-a)[:10].tolist()}, pos 2 {np.argsort(-b)[:10].tolist()}", flush=True)
