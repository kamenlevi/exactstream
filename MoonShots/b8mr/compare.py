"""Checks for the B8 multi-row phase B (data/b8mr/tf*_ARM.npz from run.py tf1 / tfs):
  tf1: mr1 == wait bit for bit (the multi-row code on 1-row calls vs bt_b12's own 1-row late path)
  tfs4: force == ref bit for bit (phase-B machinery vs synchronous late slots, same numerics)
  KL views: mr (S=4) vs wait (S=4: verify waits every forced rank), mr (S=4) vs wait (S=1 steps: the 1-row late path,
  same semantics, different batch shapes and residency history)
Exit 1 if a bit-identity check fails. usage: compare.py"""
import os, sys
import numpy as np
D = os.path.expanduser("~/MoonShots/data/b8mr")
L = lambda n: np.load(os.path.join(D, n + ".npz"))["logits"].astype(np.float64) if os.path.exists(os.path.join(D, n + ".npz")) else None


def kl(p, q):
    lp = p - p.max(-1, keepdims=True); lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = q - q.max(-1, keepdims=True); lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


bad = 0
for a, b in (("tf1_mr1", "tf1_wait"), ("tfs4_force", "tfs4_ref")):
    x, y = L(a), L(b)
    if x is None or y is None:
        print(f"{a} vs {b}: missing"); continue
    same = bool((x == y).all())
    print(f"{a} vs {b}: {'BIT-IDENTICAL' if same else 'DIFFER'} (max |dlogit| {np.abs(x - y).max():.3g}, "
          f"positions differing {int((np.abs(x - y).max(-1) > 0).sum())} of {len(x)})")
    bad += not same
for a, b in (("tfs4_mr", "tfs4_wait"), ("tfs4_mr", "tf1_wait"), ("tfs4_wait", "tf1_wait"), ("tfs4_force", "tfs4_wait")):
    x, y = L(a), L(b)
    if x is None or y is None:
        continue
    k = kl(y, x)
    print(f"KL({b} || {a}): mean {k.mean():.5f}, max {k.max():.4f}; top-1 agreement {(x.argmax(-1) == y.argmax(-1)).mean():.3f}")
sys.exit(1 if bad else 0)
