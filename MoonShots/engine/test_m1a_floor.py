"""M1a calibration: the natural numerics floor of the exact reference on the same text. MTPLX's own forward (exact sync
MoE, rle/ref.py) at M=1 steps vs M=4 chunks (different kernel paths), and the engine at M=4 vs MTPLX at M=4, on the same
N tokens; mean / median / p99 KL per position and top-1 agreement. If MTPLX-vs-MTPLX is close to engine-vs-MTPLX, the engine
sits at the floor.
usage: test_m1a_floor.py [n_tokens=64]   (GPU, ~10 GB, ~2 min; lock)"""
import os, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from rle.model import Engine
from rle import ref
from bigmoe.loader import load_streamed
from bigmoe.gated import ext

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 64
MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, 48, mtp=False)
model = rt.model
ref.install(ext)
eng = Engine(model, ext)
ids = rt.tokenizer.encode(open(os.path.expanduser("~/MoonShots/FINDINGS.md")).read()[:8000])[:NT]


def lp(x):
    x = (x[0] if isinstance(x, tuple) else x)
    x = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def run_ref(M):
    c, out = model.make_cache(), []
    for s in range(0, NT, M):
        out.append(lp(model(mx.array([ids[s:s + M]]), cache=c)))
        mx.eval(out[-1])
    return mx.concatenate(out)


def run_eng(M):
    st_, out = eng.new_state(), []
    for s in range(0, NT, M):
        out.append(lp(eng.forward(mx.array([ids[s:s + M]]), st_)))
        mx.eval(out[-1])
    return mx.concatenate(out)


def cmp(name, a, b):
    kl = np.array((mx.exp(a) * (a - b)).sum(-1))
    ag = np.array(mx.argmax(a, -1) == mx.argmax(b, -1)).mean()
    print(f"{name:34s} KL mean {kl.mean():.2e} median {np.median(kl):.2e} p99 {np.percentile(kl, 99):.2e}; top-1 agree {ag:.3f}",
          flush=True)


r4 = run_ref(4)
r1 = run_ref(1)
e4 = run_eng(4)
cmp(f"MTPLX M=1 vs MTPLX M=4 ({NT} tok)", r4, r1)
cmp(f"engine M=4 vs MTPLX M=4 ({NT} tok)", r4, e4)
cmp(f"engine M=4 vs MTPLX M=1 ({NT} tok)", r1, e4)
