"""Layer-major prefill on the SSD-streamed 35B (BigMoE's ExpertStreamer + PF1 staging): bit identity and speed.
usage: bench_lmp35.py ref|lmp|pf3 [FRAC] [PROMPT_TOKENS] [N]
  ref  chunk-major prefill (mlx-lm generate_step's loop, 2,048-token chunks) with PF1 staging
  lmp  layer-major prefill (lmp.prefill_mlxlm) with PF1 staging kept per layer
Each run is a fresh process (cold pools). Writes the post-prefill cache arrays to the scratch dir; the second mode compares
them bitwise with the first, plus the sha of N greedy tokens. Prints prefill seconds and expert bytes staged."""
import hashlib, os, sys, time
os.environ.setdefault("BIGMOE_PF_PREFILL", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

mode = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55
PT = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
N = int(sys.argv[4]) if len(sys.argv) > 4 else 32
MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
OUT = os.environ.get("LMP_OUT", "/private/tmp/claude-501/-Users-kamen-MoonShots/3c6677ac-9553-4d46-9a7d-3f8008b167c0/scratchpad/lmp35")
os.makedirs(OUT, exist_ok=True)

from bigmoe.stream import ExpertStreamer, _STAGE
import lmp
st = ExpertStreamer(MD, round(frac * 256))
orig = mu.load_model


def lazy(path, *a, **kw):
    a = list(a)
    if a: a[0] = True
    else: kw["lazy"] = True
    m, cfg = orig(path, *a, **kw)
    st.install(m)
    mx.eval(m.parameters())
    return m, cfg


mu.load_model = lazy
model, tok = load(MD)
mu.load_model = orig

text = ""
for p in ("~/SpecDecode/FINDINGS.md", "~/MoEPrefetch/FINDINGS.md", "~/BigMoE/FINDINGS.md", "~/SpecDecode/server.py",
          "~/SpecDecode/specdec.py", "~/BigMoE/bigmoe/stream.py"):
    text += open(os.path.expanduser(p)).read() + "\n\n"
ids = tok.encode(text)[:PT]
prompt = mx.array(ids)
cache = make_prompt_cache(model)
staged0 = _STAGE["t"].get("staged_experts", 0)

t0 = time.perf_counter()
if mode == "ref":
    rest = prompt
    for a, b in lmp._spans(prompt.size, 2048):          # mlx-lm generate_step's prefill loop, verbatim semantics
        model(prompt[a:b][None], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
    rest = prompt[lmp._spans(prompt.size, 2048)[-1][1]:]
elif mode == "pf3":
    rest = lmp.prefill_mlxlm_pf3(model, prompt, cache, step=2048)
    mx.eval([c.state for c in cache])
else:
    rest = lmp.prefill_mlxlm(model, prompt, cache, step=2048)
    mx.eval([c.state for c in cache])
t_pf = time.perf_counter() - t0
staged = _STAGE["t"].get("staged_experts", 0) - staged0

arrs = {}
for i, c in enumerate(cache):
    for j, a in enumerate(c.state if isinstance(c.state, (list, tuple)) else [c.state]):
        if isinstance(a, mx.array):
            arrs[f"L{i}_{j}"] = np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)
np.savez(os.path.join(OUT, f"cache_{mode}.npz"), **arrs)

out = []
for t, _ in generate_step(rest, model, prompt_cache=cache, max_tokens=N, sampler=make_sampler(temp=0.0)):
    out.append(int(t))
sha = hashlib.sha1(str(out).encode()).hexdigest()[:10]
print(f"{mode}: {len(ids)} prompt tokens, prefill {t_pf:.1f} s ({len(ids)/t_pf:.0f} tok/s), experts staged {staged}, "
      f"peak {mx.get_peak_memory()/1e9:.1f} GB, {N} greedy tokens sha {sha}", flush=True)

other = os.path.join(OUT, f"cache_{'ref' if mode == 'lmp' else 'lmp'}.npz")    # pf3 compares with lmp
if os.path.exists(other):
    o = np.load(other)
    same = [k for k in arrs if k in o and np.array_equal(arrs[k], o[k])]
    print(f"cache arrays bit-identical to the other mode: {len(same)}/{len(arrs)}"
          + ("" if len(same) == len(arrs) else f"  first differing: {[k for k in arrs if k not in same][:5]}"), flush=True)
