"""v3 tiers + deferral, quality (NewIdeas' gating number for Kamen, 2026-10-22): runs NewIdeas' tf_defer125.py UNCHANGED with
the Eval suite's plan tiers underneath. A hook on bigmoe.loader.load_streamed (Breakthrough's fork125bt, read-only) calls
evalsuite.model.install(model) + configure(plan4433333333, prefill=False) right after loading: the base SwitchGLU that the
scorer's block calls for its per-rank outputs then computes ranks 0-1 exact (4-bit) and ranks 2-9 at MSB-3 on decode calls
(exact prefill), i.e. BigMoE v3's deployed numerics; the scorer's own MoE block (deferral) replaces the suite's block patch.
VAR as tf_defer125 plus a "_v3" token (ignored by its parser, keeps the output files apart): dh_w4_v3, dh_w10_v3 (= v3 alone).
usage: tf_defer_v3.py VAR FRAC SETS EVERY   (GPU, ~24 GB at C=128; quiet lock)"""
import os, runpy, sys
os.environ["EVALSUITE_FRUGAL"] = "1"
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
SCRIPT = os.path.expanduser("~/NewIdeas/scripts/tf_defer125.py")
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
sys.path.insert(2, os.path.expanduser("~/EvalSuite"))
import bigmoe.loader as BL

_orig = BL.load_streamed


def _load_with_v3_tiers(*a, **kw):
    rt, st = _orig(*a, **kw)
    from evalsuite import model as M
    M.install(rt.model)
    M.configure(M.variant("plan4433333333"), prefill=False)
    print("[tf_defer_v3] Eval suite plan4433333333 tiers on the decode calls (exact prefill)", flush=True)
    return rt, st


BL.load_streamed = _load_with_v3_tiers
sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
