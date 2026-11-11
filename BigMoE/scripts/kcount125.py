"""Kernel count of one decode step on the streamed 125B: build the step's graph (no eval), export it with
mx.export_to_dot, count primitives by name, total and per layer. -> data/kcount125.txt"""
import os, sys, re, collections
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(float(sys.argv[1] if len(sys.argv) > 1 else 0.5) * 512))
m = rt.model
ids = rt.tokenizer.encode("def parse(path):\n    rows = []")
cache = m.make_cache() if hasattr(m, "make_cache") else None
lg = m(mx.array([ids]), cache=cache); mx.eval(lg)                       # prefill (evaluated)
out = m(mx.array([[ids[-1]]]), cache=cache)                              # one decode step, NOT evaluated
out = out[0] if isinstance(out, tuple) else out
path = os.path.expanduser("~/BigMoE/data/step125.dot")
mx.export_to_dot(path, out)
txt = open(path).read()
labels = re.findall(r'label\s*=\s*"([A-Za-z_][A-Za-z0-9_]*)', txt)
ops = collections.Counter(l for l in labels if not l.startswith(("array", "constant")))
total = sum(ops.values()); nl = len(m.language_model.model.layers)
lines = [f"one decode step: {total} primitives ({total / nl:.0f} per layer over {nl} layers)"]
lines += [f"  {n:6d}  {name}" for name, n in ops.most_common(25)]
mx.eval(out)
print("\n".join(lines))
open(os.path.expanduser("~/BigMoE/data/kcount125.txt"), "w").write("\n".join(lines) + "\n")
