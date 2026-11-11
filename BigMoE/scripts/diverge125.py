"""Why do MTP tokens differ from plain greedy on the streamed 125B? Find the first divergence and
the plain model's top-1/top-2 logit gap at that position (a full forward over the prefix).
A near-tie (gap << 1) means a benign numerics flip (verify width changes kernels); a large gap = bug.
usage: diverge125.py [FRAC] [N] [K]"""
import os, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"), ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)   # MTPLX server defaults: commit rejected MTP rounds without a 2nd trunk forward
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
from bigmoe.loader import load_streamed
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
ids = list(rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                            add_generation_prompt=True, enable_thinking=False))
ids = ids if isinstance(ids[0], int) else ids["input_ids"]
def toks(res):
    t = getattr(res, "tokens", None) or getattr(res, "token_ids", None) or (res.get("tokens") if isinstance(res, dict) else None)
    return list(t if t is not None else res[0])
plain = toks(G.generate_ar(rt, ids, max_tokens=N, sampler=greedy))
mtp = toks(G.generate_mtpk(rt, ids, max_tokens=N, sampler=greedy, speculative_depth=K, mtp_history_policy="committed"))
i = next((j for j in range(min(len(plain), len(mtp))) if plain[j] != mtp[j]), None)
lines = [f"plain {len(plain)} tok, mtp k={K} {len(mtp)} tok, first divergence at {i}"]
if i is not None:
    lg = rt.model(mx.array([ids + plain[:i]]))
    lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1].astype(mx.float32)
    top = np.sort(np.array(lg))[::-1][:2]
    lines.append(f"plain-model logits at the divergence: top1 {top[0]:.3f}, top2 {top[1]:.3f}, gap {top[0]-top[1]:.3f}; "
                 f"plain chose {plain[i]} ({rt.tokenizer.decode([plain[i]])!r}), mtp chose {mtp[i]} ({rt.tokenizer.decode([mtp[i]])!r}); "
                 f"forward argmax {int(mx.argmax(lg).item())}")
print("\n".join(lines))
open(os.path.expanduser("~/BigMoE/data/diverge125.txt"), "a").write("\n".join(lines) + "\n")
