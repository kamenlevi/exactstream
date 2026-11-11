"""Fit12 T2 hook for EvalSuite's same-scores runs (Mon 2 Nov 2026; PLAN.md 'T1 + T2', 'T2s'). The two-pass MTP check on
B12 at 12 GB (FAST track). Read-only use from EvalSuite:
  sys.path.insert(0, os.path.expanduser("~/Fit12/scripts")); import t2_hook
  rt = t2_hook.load(0.10)                      # bt_b12 as imported by the caller (the live file): T1 patch first, MTP head on
  toks, finish = t2_hook.gen(rt, ids, cap, sampler, seed, theta, skip=(2, 8))   # like evalsuite.decode._gen_mtp
  t2_hook.STATS                                # the last row: passes, pass2, rows1, rows2, skipped, refused, lsha, tok_s
The process env is the caller's (the suite's MTP env for the live file, as fit_t2.py sets it). Each row's state comes from
bt_b12.b12_prefill_state (exact prefill, the fold's MOV), then t2_twopass.generate. theta 0.8 code / 0.95 chat; skip=None
gives the plain two-pass check (the T2 gates' semantics); greedy only; stops at EOS 248044 / 248046."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_twopass as T

STATS = {}


def load(frac=0.10):
    import bt_b12 as B
    T.apply(B)                                   # before load_b12: the decode block is built there
    rt, _ = B.load_b12(frac, mtp=True)
    return rt


def gen(rt, ids, cap, sampler, seed, theta, skip=(2, 8)):
    assert sampler.temperature == 0.0, "T2 rows are greedy only"
    import bt_b12 as B
    from bigmoe_v3f.fuse import reset_generation_state
    model = rt.model if hasattr(rt, "model") else rt
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()
    state = B.b12_prefill_state(model, list(ids), 512, 4096)
    toks, st = T.generate(B, model, list(ids), cap, theta, state, skip=skip)
    STATS.clear()
    STATS.update(st)
    toks = [int(x) for x in toks]
    for j, x in enumerate(toks):
        if x in (248044, 248046):
            return toks[:j + 1], "eos"
    return toks[:cap], "length"
