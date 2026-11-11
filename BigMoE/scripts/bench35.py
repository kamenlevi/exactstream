"""Real-model check on Qwen3.6-35B-A3B 4-bit (on disk, fits in RAM, so exactness can be checked
against the fully resident model). Greedy decode.
usage: bench35.py ref|stream [FRAC] [N] [ALLOC]
  ref     stock mlx-lm, all weights resident
  stream  routed experts streamed from SSD, FRAC of 256 slots per layer (exact gate)
  ALLOC   'uniform' (default) or 'ends': first/last layers fully resident, rest shares the budget
-> data/runs35/<tag>.json, one summary line"""
import json, os, sys, time, hashlib
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import mlx_lm.utils as mu
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
mode = sys.argv[1]; frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200; alloc = sys.argv[4] if len(sys.argv) > 4 else "uniform"
E, NL = 256, 40
t0 = time.perf_counter()
st = None
if mode == "ref":
    model, tok = load(MD)
else:
    from bigmoe.stream import ExpertStreamer
    if alloc == "ends":   # ~alloc.py result at 60%: early + late layers full, middle gets the rest
        full = set(range(0, 11)) | {35, 36, 38, 39}
        rest = round((frac * E * NL - E * len(full)) / (NL - len(full)))
        C = [E if L in full else rest for L in range(NL)]
    else:
        C = round(frac * E)
    st = ExpertStreamer(MD, C)
    orig = mu.load_model
    def lazy(path, *a, **kw):
        a = list(a)
        if a: a[0] = True                    # load_model(path, lazy, ...) positional
        else: kw["lazy"] = True
        m, cfg = orig(path, *a, **kw)
        st.install(m)
        mx.eval(m.parameters())
        return m, cfg
    mu.load_model = lazy
    model, tok = load(MD)
    mu.load_model = orig
t_load = time.perf_counter() - t0
prompt = open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)
ids = mx.array(ids)
sampler = make_sampler(temp=0.0)
from bigmoe.gated import ext
out, stamps = [], []
t0 = time.perf_counter()
for i, (t, _) in enumerate(generate_step(ids, model, max_tokens=N, sampler=sampler)):
    out.append(int(t)); stamps.append(time.perf_counter())
    if i == 0 and st: ext.reset_stats()
dec = (stamps[-1] - stamps[0]); n_dec = len(out) - 1
s = ext.stats() if st else {}
tag = f"{mode}_{frac:.2f}_{alloc}_N{N}" if st else f"ref_N{N}"
res = dict(mode=mode, frac=frac, alloc=alloc, N=N, tokens=out, t_first=stamps[0] - t0, decode_tok_s=n_dec / dec,
           load_s=t_load, peak_gb=mx.get_peak_memory() / 1e9, pools_gb=st.resident_gb() if st else None, stats=s,
           text=tok.decode(out)[:300])
os.makedirs(os.path.expanduser("~/BigMoE/data/runs35"), exist_ok=True)
json.dump(res, open(os.path.expanduser(f"~/BigMoE/data/runs35/{tag}.json"), "w"), indent=1)
h = hashlib.sha1(str(out).encode()).hexdigest()[:10]
extra = ""
if st:
    per = max(1, n_dec)
    extra = (f" | slow calls/token {s['slow_calls']/per:.1f} at {s['slow_s']/max(1,s['slow_calls'])*1e6:.0f} us CPU each"
             f" | per token: misses {s['misses']/per:.1f}, stalled layers {s['stalled']/per:.1f}, io {s['io_s']/per*1e3:.2f} ms;"
             f" pools {st.resident_gb():.1f} GB")
if st and s.get("pf_issued"):
    print(f"   prefetch: issued {s['pf_issued']}, used {s['pf_used']}, waited {s['pf_waited_s']*1e3:.1f} ms total", flush=True)
print(f"{tag}: decode {n_dec/dec:.1f} tok/s, first token {stamps[0]-t0:.1f}s, peak {mx.get_peak_memory()/1e9:.1f} GB, "
      f"sha {h}{extra} | {tok.decode(out)[:50]!r}", flush=True)
