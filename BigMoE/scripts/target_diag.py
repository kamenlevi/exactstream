"""Where does decode time go on the 125B? Tiny prompt, a few decode tokens, loader timeline on.
Prints: time per token, gated calls per token, fast-path share, CPU time per call, and the median gap
between consecutive layers' routing rows (GPU time per layer as the CPU sees it)."""
import os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ["BIGMOE_TRACE"] = "1"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
ids = list(rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}], add_generation_prompt=True, enable_thinking=False))
ids = ids if isinstance(ids[0], int) else ids["input_ids"]
stamps = []
def cb(t):
    if not stamps: ext.reset_stats(); ext.trace()
    stamps.append(time.perf_counter())
G.generate_ar(rt, ids, max_tokens=40, sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0), token_callback=cb)
s = ext.stats(); _all = ext.trace(); tr = [e for e in _all if e[2] == 1]
SLOW = set((int(e[0]), int(e[1])) for e in _all if e[2] == 4)
ntok = len(stamps) - 1
print(f"{(stamps[-1]-stamps[0])/ntok*1e3:.0f} ms/token over {ntok} tokens; gated calls/token {s['gated']/ntok:.1f}; "
      f"fast-path hits {st.gate.fast_hits()}; CPU per call {s['proc_s']/max(1,s['gated'])*1e6:.0f} us; "
      f"misses/token {s['misses']/ntok:.1f}; io {s['io_s']/ntok*1e3:.1f} ms/token")
ev = {}
for e in tr: ev[(int(e[0]), int(e[1]))] = (e[3], e[4])
qs = sorted({q for _, q in ev})[1:-1]
NL = max(L for L, _ in ev) + 1
cpu_slow, gap_fast, gap_slow, between = [], [], [], []
for q in qs:
    cs = gf = gs = 0.0
    for L in range(NL):
        if (L, q) not in ev: continue
        seen, rel = ev[(L, q)]
        if (L, q) in SLOW: cs += rel - seen
        if L > 0 and (L - 1, q) in ev:
            g = seen - ev[(L - 1, q)][1]
            if (L - 1, q) in SLOW: gs += g
            else: gf += g
    if (NL - 1, q) in ev and (0, q + 1) in ev:
        between.append(ev[(0, q + 1)][0] - ev[(NL - 1, q)][1])
    cpu_slow.append(cs); gap_fast.append(gf); gap_slow.append(gs)
m = lambda a: np.median(a) * 1e3 if a else float("nan")
print(f"per token (median over {len(qs)}): loader CPU on slow calls {m(cpu_slow):.1f} ms | GPU gaps after fast layers "
      f"{m(gap_fast):.1f} ms | after slow layers {m(gap_slow):.1f} ms | token-to-token (last layer -> next layer 0) {m(between):.1f} ms "
      f"| slow layers/token {len(SLOW)/max(1,len(qs)+2):.1f}")
