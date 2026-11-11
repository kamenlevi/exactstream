"""Verify-pass routing trace for the wrong-path expert study (round 22). The 125B on BigMoE's streamed runtime, MTPLX
generate_mtpk with committed MTP history (the product setting, round 21), depth 3, greedy, BigMoE's MTPLX flags
(capture-commit: one gated trunk pass per round, no repair pass). Records every decode/verify trunk pass through the
gate (Gate.route_and_gate, wrapped): its seq and, per layer, the routing ids of every row (row 0 = the committed token,
rows 1.. = drafts), and every token_callback (seq at the time, tokens committed). Offline, each round's accepted drafts =
committed tokens - 1, so rows after the first rejected draft are labelled wrong-path. The routing does not depend on the
cache, so cache policies can be compared exactly on this trace (scripts/wrongpath_sim.py).
Output: data/verify_trace125.npz (ids [passes, layers, rows, k] int16, -1 padded; rows per pass; seq; prompt index)
        data/verify_trace125.json (callbacks, per-prompt token counts, MTP stats)
usage: verify_trace125.py [FRAC] [MAXTOK]   (GPU, BigMoE-size memory: exclusive window)"""
import json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
os.environ.setdefault("MTPLX_CONTEXT_COPY", "0")    # prompt-lookup rounds can exceed the 40-id gate row (MAXK) and
                                                   # take the unrecorded sync path; this study is about MTP verify rows
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe import fuse
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 400
MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
K = 10
REC = []                                    # (seq, L, ids) per gated call
gate = st.gate
_orig = gate.route_and_gate


def hook(L, ids, seq=None, preds=None, counts=None):
    REC.append((gate.seq if seq is None else seq, L, ids))
    return _orig(L, ids, seq=seq, preds=preds, counts=counts)


gate.route_and_gate = hook
lines = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
PROMPTS = [
    open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
    "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + lines[0]["question"],
    "Explain how a refrigerator works to a curious 10 year old, in about 150 words.",
    "Explain in detail how TCP congestion control works: slow start, congestion avoidance, fast retransmit and fast "
    "recovery, and how CUBIC and BBR differ. Use headings and a short example.",
    "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + lines[1]["question"],
]
smp = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
CB, META, passes = [], [], []
for pi, text in enumerate(PROMPTS):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    fuse.reset_generation_state(rt.model)
    REC.clear()

    def cb(toks, pi=pi):
        CB.append((pi, int(gate.seq), len(toks) if isinstance(toks, list) else 1))
    t0 = time.perf_counter()
    res = G.generate_mtpk(rt, ids, max_tokens=MAXTOK, sampler=smp, speculative_depth=3, seed=0,
                          mtp_history_policy="committed", token_callback=cb)
    S = res.stats
    # group this prompt's gated calls by pass (seq); rows from the ids' size
    byseq = {}
    for seq, L, a in REC:
        byseq.setdefault(int(seq), {})[int(L)] = np.array(a.reshape(-1, K), dtype=np.int16)
    for seq in sorted(byseq):
        d = byseq[seq]
        passes.append((pi, seq, d))
    META.append(dict(prompt=pi, prompt_tokens=len(ids), tokens=len(res.tokens), verify_calls=S.verify_calls,
                     accepted_by_depth=list(S.accepted_by_depth), drafted_by_depth=list(S.drafted_by_depth),
                     gated_passes=len(byseq), seconds=round(time.perf_counter() - t0, 1)))
    print(json.dumps(META[-1]), flush=True)
st.gate.check()
NL = 48
P = len(passes)
R = max(max(a.shape[0] for a in d.values()) for _, _, d in passes)
arr = np.full((P, NL, R, K), -1, np.int16)
rows = np.zeros(P, np.int16); seqs = np.zeros(P, np.int64); pidx = np.zeros(P, np.int16); full = np.zeros(P, np.int16)
for i, (pi, seq, d) in enumerate(passes):
    pidx[i], seqs[i], full[i] = pi, seq, len(d)
    for L, a in d.items():
        arr[i, L, :a.shape[0]] = a
        rows[i] = max(rows[i], a.shape[0])
np.savez_compressed(os.path.expanduser("~/MoonShots/data/verify_trace125.npz"), ids=arr, rows=rows, seq=seqs, prompt=pidx,
                    layers_seen=full)
json.dump(dict(callbacks=CB, meta=META, K=K, frac=frac, maxtok=MAXTOK), open(os.path.expanduser("~/MoonShots/data/verify_trace125.json"), "w"))
print(f"{P} gated passes, rows per pass {np.bincount(rows).tolist()}, passes with all 48 layers {int((full == NL).sum())}, "
      f"{len(CB)} callbacks", flush=True)
