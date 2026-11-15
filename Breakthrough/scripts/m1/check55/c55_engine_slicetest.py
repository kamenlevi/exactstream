"""c55_engine_slicetest.py (Breakthrough, Sun 15 Nov 2026): runs c55_engine.py's code paths (cold, window with all 7 arms,
pass K, finish with fresh windows, dump, report) on Leap's C6 slice with a stand-in for r57's globals (no gate: stats()
returns zeros, set_arm / gate_req are no-ops). A function test before the unattended engine run at ~04:15 Mon, not a
measurement. Usage: [venv] python -I -B c55_engine_slicetest.py   (MEM ~21 GB; share lock)"""
import os
import sys
import types

for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("V32_MOE1", "1"), ("HF_HUB_OFFLINE", "1"), ("MTPLX_NGRAM_PREWARM", "off"), ("R57_G32TIME", "2"),
               ("R57_G32TIME_WARM", "1"), ("C55_KROUNDS", "1"), ("R57_G32TEST", "dd__mc_27921"),
               ("C55_FRESH", "dd__mc_27623,dd__mc_27904")):
    os.environ.setdefault(k_, v_)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtlprof as MP

MP.load(2)
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
NL = int(os.environ.get("C55_ST_NL", "12"))
orig = mu.load_model


def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a:
        a[0] = True
    else:
        kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        model.language_model.model.layers = model.language_model.model.layers[:NL]
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(os.path.expanduser("~/BigMoE/models/bare"), mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
import mtplx.cache_state as CS

_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (4 if 2 <= int(rows) <= 32 else 0)
model = rt.model
import ref2

ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x"):
    ref2.CFG[k_] = True
import v32

v32.install(ref2)
import c55_engine as E


def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]


def prefill(ids, want_w=False, chunk=1024):
    cache = model.make_cache()
    lg = model(mx.array([ids]), cache=cache)
    mx.eval(lg, states(cache))
    return cache, None, None, None


OUT = os.path.expanduser(os.environ.get("C55_ST_OUT", "~/Breakthrough/data/m1/check55/engine_slicetest"))
os.makedirs(OUT, exist_ok=True)
OVC = [0]                                                # Mon 00:4x: a stand-in for the ext's overflow_count(); from call
OVAT = int(os.environ.get("C55_ST_OVF_AT", "0") or 0)    # C55_ST_OVF_AT on it reads 1, so the check whose second read that
                                                         # is gets flagged (even calls = a check's after-read): tests the drop


def ovf_count():
    OVC[0] += 1
    return 1 if OVAT and OVC[0] >= OVAT else 0


G = dict(model=model, ref2=ref2, v32=v32, CS=CS, Q=Q, states=states, stats=lambda: np.zeros(6), set_arm=lambda a: None,
         prefill=prefill, PFCHUNK=1024, SW_=[], BG=types.SimpleNamespace(MAXK=320, ext=types.SimpleNamespace(overflow_count=ovf_count)),
         BS=types.SimpleNamespace(GATE_N=[0]),
         say=lambda s: print(s, flush=True), OUT=OUT, TAG="c55slice")
for key in os.environ["R57_G32TEST"].split(","):
    bz = np.load(os.path.expanduser(f"~/datasets/deepdraft/bank/{key}.npz"))
    gids, gen = [int(t) for t in bz["ids"]], int(bz["gen"])
    P, _a, _b, _c = prefill(gids[:gen])
    S0 = CS.snapshot_untrimmable_cache_lazy(P)
    E.cold(G, key, P, S0, gids, gen)
    E.window(G, key, P, S0, gids, gen, 0, first=(key == os.environ["R57_G32TEST"].split(",")[0]))
E.finish(G)
print("SLICETEST DONE", flush=True)
