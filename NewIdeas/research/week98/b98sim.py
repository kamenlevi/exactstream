#!/usr/bin/env python3
"""BREAK98 delivery simulator (NewIdeas, Tue 17 Nov; the time is the commit's). Prices an INFORMATION-DELIVERY PATTERN in
kept per 32-row check (chain), from the cells only, so any channel idea can be screened in minutes.

Guess sources per (anchor p, depth d), all from cells_t30b_{tag}.npz via certify98.windows (rt: 1 right / 0 wrong / -1):
  own[p, d]  = the head's depth-d guess from anchor p (carried state; the walk's tokens teacher-forced), = the real chain;
  lab[p, d]  = own[p + d, 0], the depth-0 guess of anchor p + d = the guess WITH the true 125B state (Lunatic's LAB line).
Patterns:
  point(S)       the true state arrives at depths in S only; the carried state is NOT refreshed (lab at d in S, else own);
  refresh(k)     every k-th depth the carried state is replaced by the true one: the guess at depth d uses own[p + r, d - r]
                 with r = the last refresh depth <= d (r = 0 is the anchor itself, so refresh(32) = the real chain and
                 refresh(1) = LAB);
  refresh_at(D)  one refresh at depth D (D = 0: the real chain);
  random(pi)     each depth independently gets the true state with probability pi (point delivery), Monte Carlo.
Self-check: refresh(32) must equal the real chain-32 (13.77 HE+ / 6.92 MBPP) and refresh(1) LAB (20.77 / 12.72).
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98sim.py
"""
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98  # noqa: E402

CE = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
DEP = 31


def load(tag):
    return [w[0].astype(np.int8) for w in C98.windows(f"{CE}/cells_t30b_{tag}.npz")]


def shifted(rt, r):
    """out[p, d] = rt[p + r, d - r] for d >= r (the guess of anchor p + r at depth d - r), -1 where it doesn't exist"""
    n = rt.shape[0]
    out = np.full(rt.shape, -1, np.int8)
    if r == 0:
        return rt.copy()
    if r >= DEP or r >= n:
        return out
    out[: n - r, r:] = rt[r:, : DEP - r]
    return out


def mix(rt, src_of_depth):
    """src_of_depth[d] = r: depth d uses the guess of anchor p + r at depth d - r"""
    out = np.full(rt.shape, -1, np.int8)
    cache = {}
    for d in range(DEP):
        r = int(src_of_depth[d])
        if r not in cache:
            cache[r] = shifted(rt, r)
        out[:, d] = cache[r][:, d]
    out[rt < 0] = -1                      # keep the walk's own end-of-window masking
    return out


def kept(windows, src_of_depth):
    ks = []
    for rt in windows:
        ks += C98.kept_walk(mix(rt, src_of_depth), C98.R0)
    return float(np.mean(ks))


def refresh(k):
    return [(d // k) * k for d in range(DEP)]


def refresh_at(D):
    return [D if (D and d >= D) else 0 for d in range(DEP)]


def point(S):
    return [d if d in S else 0 for d in range(DEP)]


def random_point(windows, pi, draws=20, seed=17):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        ks = []
        for rt in windows:
            lab = shifted_lab(rt)
            m = rng.random(rt.shape) < pi
            g = np.where(m, lab, rt).astype(np.int8)
            g[rt < 0] = -1
            ks += C98.kept_walk(g, C98.R0)
        vals.append(np.mean(ks))
    return float(np.mean(vals)), float(np.std(vals))


def shifted_lab(rt):
    return mix(rt, list(range(DEP)))


def main():
    for tag in ("heplus24", "mbppbank"):
        W = load(tag)
        print(f"== {tag}: {len(W)} windows")
        print(f"  self-check: refresh(32) = real chain {kept(W, refresh(32)):.2f}; refresh(1) = LAB {kept(W, refresh(1)):.2f}")
        print("  refresh every k depths (the carried state replaced by the true one): " +
              ", ".join(f"k={k} {kept(W, refresh(k)):.2f}" for k in (1, 2, 3, 4, 6, 8, 12, 16)))
        print("  one refresh at depth D: " + ", ".join(f"D={D} {kept(W, refresh_at(D)):.2f}" for D in (2, 4, 8, 12, 16, 24)))
        print("  true state at depths >= D only, no refresh (point): " +
              ", ".join(f"D>={D} {kept(W, point(set(range(D, DEP)))):.2f}" for D in (1, 4, 8, 16)))
        print("  true state at depths < D only, no refresh (point): " +
              ", ".join(f"D<{D} {kept(W, point(set(range(0, D)))):.2f}" for D in (4, 8, 16)))
        print("  random point delivery with probability pi: " +
              ", ".join(f"pi={p} {random_point(W, p)[0]:.2f}" for p in (0.25, 0.5, 0.75, 0.9)))


if __name__ == "__main__":
    main()
