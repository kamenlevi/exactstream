"""D-17 MTPDEEP capture (MTP Head, Thu 10-29; REPORT 'D-17'): a copy of ~/Chat1000/scripts/agent125.py (read-only
source; same runtime) without the route hook. spec125.generate with max_draft=0 (one token per pass = plain greedy)
and k_mtp=15, so the native MTP head drafts a 15-token chain at EVERY position from the true prefix. Prompts: D-05's 8
code + 8 gsm + 8 ifeval + 8 multi (data/d05_prompts.jsonl, chat-templated ids, thinking off). Saves data/d17/<key>.npz:
out (greedy tokens), mtp (T x 15; mtp[p][j] guesses out[p + j]), prompt ids. Resumable (skips saved keys).
Usage: [BigMoE venv] python sticky/d17_mtpchain.py FRAC N_TOKENS"""
import json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
import spec125

MD = os.path.expanduser("~/BigMoE/models/bare")
OUT = os.path.expanduser("~/TrainLab/data/d17"); os.makedirs(OUT, exist_ok=True)
frac, N = float(sys.argv[1]), int(sys.argv[2])
tc = json.load(open(os.path.join(MD, "config.json"))); tc = tc.get("text_config", tc)
P = [json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d05_prompts.jsonl"))]
t0 = time.time()
rt, st = load_streamed(MD, max(16, round(frac * tc["num_experts"])), mtp=True)
print(f"[d17] loaded {time.time() - t0:.0f}s (C={max(16, round(frac * tc['num_experts']))}); {len(P)} prompts", flush=True)
spec125.generate(rt.model, rt.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True,
                 enable_thinking=False), 8, k_mtp=3, max_draft=0, lookup=0)   # warm-up
for r in P:
    key = r["key"].replace("/", "_")
    path = os.path.join(OUT, key + ".npz")
    if os.path.exists(path):
        continue
    ids = [int(t) for t in r["ids"]]
    s = {"trace": []}
    t1 = time.time()
    out, s = spec125.generate(rt.model, ids, N, k_mtp=15, max_draft=0, lookup=0, stats=s)
    st.gate.check()
    T = len(out)
    mtp = np.full((T, 15), -1, np.int64)
    pos = 0
    for x in s["trace"]:              # one token per pass: the pass at position pos drafted mtp for out[pos], out[pos+1], ...
        m = x.get("mtp") or []
        if pos < T and m:
            mtp[pos, :min(15, len(m))] = m[:15]
        pos += x["n"] + 1
    np.savez_compressed(path, out=np.array(out), mtp=mtp, prompt=np.array(ids), domain=r["domain"])
    print(f"[d17] {r['domain']} {key}: prompt {len(ids)}, {T} tokens in {time.time() - t1:.0f}s; mtp depth-1 hit "
          f"{np.mean(mtp[:-1, 0] == np.array(out)[:-1]) if T > 1 else 0:.3f}", flush=True)
print(f"[d17] done in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
