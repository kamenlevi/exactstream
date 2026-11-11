"""Round 32k follow-up: export one 4-row verify forward's graph (unevaluated) under the CURRENT environment flags (pass the
reference's BIGMOE_* / MTPLX_* variables in the environment) and measure its dependent-kernel chain with
critical_chain125.py. As kcount125.py (BigMoE) but 4 rows (the MTP verify shape) inside attention_phase("decode_verify").
usage: [ENV...] export_pass_graph125.py [FRAC] [OUT.dot]   (GPU-light: builds a graph, one prefill; model-size memory)"""
import os, subprocess, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
out_path = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/MoonShots/data/pass_graph125_ref.dot")
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
m = rt.model
ids = rt.tokenizer.encode("def parse(path):\n    rows = []\n    with open(path) as f:")
cache = m.make_cache()
lg = m(mx.array([ids[:-4]]), cache=cache)
mx.eval(lg)                                                         # prefill (evaluated)
try:
    from mtplx.attention_context import attention_phase
    ctx = attention_phase("decode_verify")
except Exception:                                                   # older MTPLX: plain 4-row forward
    import contextlib
    ctx = contextlib.nullcontext()
with ctx:
    out = m(mx.array([ids[-4:]]), cache=cache)                      # one 4-row verify forward, NOT evaluated
out = out[0] if isinstance(out, tuple) else out
mx.export_to_dot(out_path, out)
flags = {k: v for k, v in os.environ.items() if k.startswith(("BIGMOE_", "MTPLX_"))}
print("exported", out_path, "flags:", flags, flush=True)
mx.eval(out)
print(subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "critical_chain125.py"), out_path],
                     capture_output=True, text=True).stdout, flush=True)
