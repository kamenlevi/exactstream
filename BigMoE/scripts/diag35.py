"""Where do streamed runs differ from the reference, and where does decode time go?
usage: diag35.py FRAC [ref]   -> prints prefill-logit hash, then decode tok/s for 100 tokens
FRAC=1.0: every layer fully resident through the streamer (no gates) -- isolates loader setup.
FRAC=0.999: C=255 (gate on every layer, almost no misses) -- isolates the gate cost."""
import os, sys, time, hashlib
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
import mlx_lm.utils as mu
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
frac = float(sys.argv[1]); ref = len(sys.argv) > 2
if ref:
    model, tok = load(MD)
else:
    from bigmoe.stream import ExpertStreamer
    st = ExpertStreamer(MD, max(1, int(frac * 256)))
    orig = mu.load_model
    def lazy(path, *a, **kw):
        a = list(a); a[0] = True
        m, cfg = orig(path, *a, **kw); st.install(m); mx.eval(m.parameters()); return m, cfg
    mu.load_model = lazy; model, tok = load(MD); mu.load_model = orig
prompt = open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()
ids = mx.array(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False))
from mlx_lm.models import cache as cm
c = cm.make_prompt_cache(model)
lg = model(ids[None], cache=c)[0, -1].astype(mx.float32); mx.eval(lg)
h = hashlib.sha1(np.array(lg).tobytes()).hexdigest()[:10]
# 1-token decode logits from that state
lg2 = model(mx.argmax(lg).reshape(1, 1), cache=c)[0, -1].astype(mx.float32); mx.eval(lg2)
h2 = hashlib.sha1(np.array(lg2).tobytes()).hexdigest()[:10]
from bigmoe.gated import ext
ext.reset_stats()
out, ts = [], []
for i, (t, _) in enumerate(generate_step(ids, model, max_tokens=100, sampler=make_sampler(temp=0.0))):
    ts.append(time.perf_counter())
s = ext.stats() if not ref else {}
if not ref: print("fast-path layer passes:", st.gate.fast_hits(), "of gated", s["gated"])
g = max(1, s.get("gated", 0))
print(f"{'ref' if ref else f'C={int(frac*256)}'}: prefill-logits {h}, next-step logits {h2}, decode {99/(ts[-1]-ts[0]):.1f} tok/s"
      + (f" | gated {s['gated']}, loader seen->signal {s['proc_s']/g*1e6:.1f} us/call, idle {s['cpu_wait_s']/g*1e6:.0f} us/call, misses {s['misses']}" if s else ""), flush=True)
