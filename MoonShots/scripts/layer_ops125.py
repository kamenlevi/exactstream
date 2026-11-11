"""Round 29 map: which primitives does each layer of a 4-row 125B forward launch? Captures every DecoderLayer's output
array during one (unevaluated) 4-token forward on BigMoE's streamed runtime, exports each to DOT and diffs the node sets
of consecutive layers, so each layer's own nodes are counted by primitive type (GDN vs QSA vs PLE layer). Free view ops
(Reshape, Broadcast, ExpandDims, Squeeze, Flatten, Unflatten, Slice, Split, Transpose*) are listed apart from the rest.
usage: layer_ops125.py [FRAC]   (GPU, BigMoE-size memory: quiet lock)"""
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
TM = model.language_model
LAYERS = TM.model.layers
OUT = []
for L, layer in enumerate(LAYERS):
    cls = type(layer)
    if not getattr(cls, "_l29_wrapped", False):
        orig = cls.__call__

        def wrapped(self, *a, __orig=orig, **kw):
            y = __orig(self, *a, **kw)
            OUT.append((self, y))
            return y
        cls.__call__ = wrapped
        cls._l29_wrapped = True
ids = rt.tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 4)[:32]
cache = model.make_cache()
out = model(mx.array([ids[:28]]), cache=cache)
mx.eval(out if not isinstance(out, tuple) else out[0])
OUT.clear()
y = model(mx.array([ids[28:32]]), cache=cache)
y = y[0] if isinstance(y, tuple) else y
lay = {id(l): i for i, l in enumerate(LAYERS)}
arrs = [(lay.get(id(s)), (o[0] if isinstance(o, tuple) else o)) for s, o in OUT]


def nodes(a, path):
    mx.export_to_dot(path, a)
    txt = open(path).read()
    return dict(re.findall(r'\{\s*(\d+)\s*\[label\s*=\s*"([^"]*)"', txt))


FREE = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "Slice", "Split", "Transpose", "AsStrided"}
tmp = os.path.expanduser("~/MoonShots/data/_layer.dot")
prev = set()
per = {}
for L, a in arrs:
    n = nodes(a, tmp)
    mine = {k: v for k, v in n.items() if k not in prev and not re.match(r"^[\d\s,()x]*$", v)}
    prev |= set(n)
    per[L] = collections.Counter(re.sub(r"[\s(].*$", "", v) for v in mine.values())
mx.eval(y)
st.gate.check()
types = ["ple" if getattr(LAYERS[L], "ple", None) is not None else ("gdn" if LAYERS[L].is_linear else "qsa") for L in range(len(LAYERS))]
for kind in ("gdn", "qsa", "ple"):
    Ls = [L for L in per if L is not None and types[L] == kind and L > 0]
    if not Ls:
        continue
    tot = collections.Counter()
    for L in Ls:
        tot.update(per[L])
    k_all = {k: v / len(Ls) for k, v in tot.items()}
    kern = sum(v for k, v in k_all.items() if k not in FREE)
    print(f"\n{kind.upper()} layers ({len(Ls)}): {sum(k_all.values()):.0f} nodes per layer, ~{kern:.0f} non-view (kernel) nodes:")
    print("  " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(k_all.items(), key=lambda kv: -kv[1]) if k not in FREE))
    print("  views: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(k_all.items(), key=lambda kv: -kv[1]) if k in FREE))
