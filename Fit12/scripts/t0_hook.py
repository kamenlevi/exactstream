"""Fit12 T0 hook for EvalSuite's same-scores runs (Mon 2 Nov 2026; PLAN.md 'T0'). Read-only use from EvalSuite:
  sys.path.insert(0, os.path.expanduser("~/Fit12/scripts")); import t0_hook
  rt = t0_hook.load(0.10)                      # bt_b12 as imported by the caller (the live file), two-pass patch first
  toks, finish = t0_hook.gen(rt, ids, cap, sampler, seed, theta)   # like evalsuite.decode._gen_*; greedy only
  t0_hook.STATS                                # the last row: steps, pass2, rate, lsha, pf_s, dec_s
theta: 0.8 (code) / 0.95 (chat), Breakthrough's PLAN_12GB_TWOPASS.md section 4. The process env is the caller's (the suite's
SHIP_AR + B12_ENV for the live file); the MTP head is not loaded (T0 has no drafter); stops at EOS 248044 / 248046."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t0_twopass as T

STATS = {}


def load(frac=0.10):
    import bt_b12 as B
    T.apply(B)                                   # before load_b12: the decode block is built there
    rt, _ = B.load_b12(frac, mtp=False)
    return rt


def gen(rt, ids, cap, sampler, seed, theta):
    assert sampler.temperature == 0.0, "T0 rows are greedy only"
    import bt_b12 as B
    toks, fin, st = T.generate(B, rt.model if hasattr(rt, "model") else rt, list(ids), cap, theta)
    STATS.clear()
    STATS.update(st)
    return toks, fin
