"""M1a model-level check: the roofline engine (rle.model.Engine) vs MTPLX's own forward on the real 125B (BigMoE loader,
exact env, C=16 slots/layer), both with exact experts through synchronous residency (the reference's SparseMoeBlock is
patched like the KLD probes: all 10 ranks ensured, pool SwitchGLU, top-10 renormalized, gated shared expert). 32 real tokens
in 4-row chunks (verify-like): per-position KL(ref || engine), top-1 agreement, max |dlogit|. Expect KL ~1e-3 or below.
usage: test_m1a_model.py [n_tokens=32] [M=4]   (GPU, ~6 GB, ~1-2 min; share lock)"""
import os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from rle.model import Engine, K
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from mtplx.models import qwen4_exp
from mtplx.attention_math import attention_gate

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MD = os.path.expanduser("~/BigMoE/models/bare")
t0 = time.time()
rt, st = load_streamed(MD, 48, mtp=False)            # a 4-row chunk needs up to 40 experts resident at once
model = rt.model
print(f"loaded in {time.time() - t0:.0f} s", flush=True)


from rle import ref
ref.install(ext)
eng = Engine(model, ext)
tok = rt.tokenizer if hasattr(rt, "tokenizer") else None
text = open(os.path.expanduser("~/MoonShots/FINDINGS.md")).read()[:4000]
ids = (tok.encode(text) if tok is not None else list(range(1000, 1000 + NT + 1)))[:NT]
ref_cache = model.make_cache()
est = eng.new_state()
kls, agree, dmax = [], [], 0.0
for c in range(0, NT, M):
    chunk = mx.array([ids[c:c + M]])
    lr = model(chunk, cache=ref_cache)
    lr = (lr[0] if isinstance(lr, tuple) else lr)[0].astype(mx.float32)
    le = eng.forward(chunk, est).astype(mx.float32)
    lr = lr - mx.logsumexp(lr, axis=-1, keepdims=True)
    le = le - mx.logsumexp(le, axis=-1, keepdims=True)
    kl = (mx.exp(lr) * (lr - le)).sum(-1)
    mx.eval(kl)
    kls += np.array(kl).tolist()
    agree += np.array(mx.argmax(lr, -1) == mx.argmax(le, -1)).tolist()
    dmax = max(dmax, mx.abs(lr - le).max().item())
    print(f"tokens {c}-{c + chunk.shape[1] - 1}: KL {np.array(kl).mean():.2e}, top-1 agree {np.array(agree[-chunk.shape[1]:]).mean():.2f}",
          flush=True)
kls = np.array(kls)
print(f"M={M}, {NT} tokens: mean KL {kls.mean():.2e} (p99 {np.percentile(kls, 99):.2e}), top-1 agreement {np.mean(agree):.3f}, "
      f"max |dlogprob| {dmax:.3f} ({time.time() - t0:.0f} s)", flush=True)
print("PASS" if kls.mean() < 5e-3 and np.mean(agree) > 0.9 else "CHECK", flush=True)
