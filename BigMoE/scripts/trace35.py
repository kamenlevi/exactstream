"""Timeline of the exact gate in the real 35B model (BIGMOE_TRACE=1). Per (layer, pass):
  S, E   GPU start/end of the command buffer that begins with layer L's wait
  seen   CPU loader sees layer L's routing row;  sig  CPU signals layer L
Reports medians of: sig - S (buffer waiting on the CPU), E - sig (buffer runs after the
signal), seen(L+1) - E(L) (end of that buffer -> CPU sees the next row), and the full
per-layer period seen(L+1) - seen(L)."""
import os, sys, time
os.environ["BIGMOE_TRACE"] = "1"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
import mlx_lm.utils as mu
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from bigmoe.stream import ExpertStreamer
from bigmoe.gated import ext
MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
st = ExpertStreamer(MD, int(float(sys.argv[1]) * 256) if len(sys.argv) > 1 else 255)
orig = mu.load_model
def lazy(path, *a, **kw):
    a = list(a); a[0] = True
    m, cfg = orig(path, *a, **kw); st.install(m); mx.eval(m.parameters()); return m, cfg
mu.load_model = lazy; model, tok = load(MD); mu.load_model = orig
ids = mx.array(tok.apply_chat_template([{"role": "user", "content": "Write a haiku about SSDs."}], add_generation_prompt=True))
for i, _ in enumerate(generate_step(ids, model, max_tokens=12, sampler=make_sampler(temp=0.0))): pass
ext.trace()                                   # drop warm-up
for i, _ in enumerate(generate_step(ids, model, max_tokens=40, sampler=make_sampler(temp=0.0))): pass
time.sleep(0.2)
tr = ext.trace()
gpu = {(int(L), int(q)): (a, b) for L, q, k, a, b in tr if k == 0}
cpu = {(int(L), int(q)): (a, b) for L, q, k, a, b in tr if k == 1}
enc = {(int(L), int(q)): (a, b) for L, q, k, a, b in tr if k == 2}
e2s, e2g, sc2s = [], [], []
for key, (S, E) in gpu.items():
    if key in enc and key in cpu:
        te, tsch = enc[key]; e2s.append(cpu[key][1] - te); e2g.append(S - te); sc2s.append(S - tsch)
w, r, v, per = [], [], [], []
for (L, q), (S, E) in gpu.items():
    if (L, q) in cpu:
        seen, sig = cpu[(L, q)]
        w.append(sig - S); r.append(E - sig)
        nxt = cpu.get((L + 1, q))
        if nxt: v.append(nxt[0] - E); per.append(nxt[0] - seen)
md = lambda a: np.median(a) * 1e6 if a else float("nan")
p10 = lambda a: np.percentile(a, 10) * 1e6 if a else float("nan")
line = (f"n={len(w)}: buffer waits on CPU (sig-S) median {md(w):.0f} us (p10 {p10(w):.0f}); runs after signal (E-sig) {md(r):.0f} us; "
        f"buffer end -> CPU sees next row {md(v):.0f} us (p10 {p10(v):.0f}); per-layer period {md(per):.0f} us")
line += (f" | gate encoded -> CPU signal {md(e2s):.0f} us, gate encoded -> GPU start {md(e2g):.0f} us, "
         f"scheduled-handler -> GPU start {md(sc2s):.0f} us")
print(line)
open(os.path.expanduser("~/BigMoE/data/trace35.txt"), "a").write(line + "\n")
