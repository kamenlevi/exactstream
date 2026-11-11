"""MTP through MTPLX on the streamed 125B: per verify pass, time inside the trunk (first -> last layer's
routing row as the loader sees it) versus the wall time between passes (draft head, sampling, MTPLX)."""
import os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"), ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)   # MTPLX server defaults: commit rejected MTP rounds without a 2nd trunk forward
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ["BIGMOE_TRACE"] = "1"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
ids = list(rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                            add_generation_prompt=True, enable_thinking=False))
ids = ids if isinstance(ids[0], int) else ids["input_ids"]
stamps = []
def cb(t):
    if not stamps: ext.trace()
    stamps.append(time.perf_counter())
G.generate_mtpk(rt, ids, max_tokens=48, sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0), speculative_depth=2, token_callback=cb, mtp_history_policy="committed")
tr = [e for e in ext.trace() if e[2] == 1]
by = {}
for L, q, k, a, b in tr:
    by.setdefault(int(q), []).append((a, b))
qs = sorted(by)[2:-1]
inside = [max(b for _, b in by[q]) - min(a for a, _ in by[q]) for q in qs]
between = [min(a for a, _ in by[q2]) - max(b for _, b in by[q1]) for q1, q2 in zip(qs, qs[1:])]
cpu = [sum(b - a for a, b in by[q]) for q in qs]
line = (f"passes {len(qs)}: trunk (first->last layer) median {np.median(inside)*1e3:.1f} ms, CPU loader time per pass "
        f"{np.median(cpu)*1e3:.1f} ms, gap between passes median {np.median(between)*1e3:.1f} ms; "
        f"tokens {len(stamps)}, wall {(stamps[-1]-stamps[0])*1e3/max(1,len(stamps)-1):.0f} ms/token")
print(line)
open(os.path.expanduser("~/BigMoE/data/mtp_diag.txt"), "a").write(line + "\n")
