"""Round 27b: how many kernels does one 4-row 125B forward launch? Builds the graph of a 4-token forward (BigMoE's
streamed runtime, eager MTPLX as BigMoE runs it) without evaluating it, writes it with mx.export_to_dot and counts the
primitive nodes by type; per layer = total / 48. With ~3-7 us per dependent kernel (measured: 6.3 us/op at 1000 dependent
elementwise ops, 13.7 at 200), the count says how much of the ~1 ms per layer is dependency latency.
usage: pass_graph125.py [FRAC]   (GPU, BigMoE-size memory: quiet lock)"""
import collections, os, re, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
model = rt.model
ids = rt.tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 4)[:32]
cache = model.make_cache()
out = model(mx.array([ids[:28]]), cache=cache)                 # a short prefill so the caches exist
mx.eval(out if not isinstance(out, tuple) else out[0])
y = model(mx.array([ids[28:32]]), cache=cache)                 # the 4-row forward, not evaluated yet
y = y[0] if isinstance(y, tuple) else y
path = os.path.expanduser("~/MoonShots/data/pass_graph125.dot")
mx.export_to_dot(path, y)
mx.eval(y)
st.gate.check()
txt = open(path).read()
labels = re.findall(r'label\s*=\s*"([^"]*)"', txt)
prims = [l for l in labels if not re.match(r"^[\d\s,()x]*$", l)]
nodes = collections.Counter(re.sub(r"[\s(].*$", "", l) for l in prims)
total = sum(nodes.values())
print(f"{total} labelled nodes in the 4-row forward graph (~{total / 48:.0f} per layer incl. head/embedding)")
for name, n in nodes.most_common(40):
    print(f"  {name:32s} {n:6d}  ({n / 48:5.1f} per layer)")
