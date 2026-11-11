"""Round 23: can the MTP draft head predict the trunk's experts for the next verify pass (cross-pass prefetch lead time)?
SP-MoE (arXiv 2510.10302) feeds a separate, layer-aligned draft model's attention outputs to the target's gates
(top-1 ~88 %) and prefetches while drafting. Here the drafter is the 125B's own 1-layer MTP head: its output state
(the widened [hc*d] stream it chains on) for draft step j sits at the position of verify row j-1. During drafting we
apply every trunk layer's MoE read (mlp hyper-connection mix of the hc-normed stream) and router to it and keep the top
PRED experts per layer; the verify pass then reveals the actual top-10 of every row (gate hook, as verify_trace125.py).
Baselines from the same trace: the previous position's experts at the same layer, and the previous pass's union.
Recall@10/20 by layer group is the measure; if deep layers are predictable a whole pass ahead, their misses can load
during drafting and the first layers of the verify pass.
Output: data/mtp_route_pred125.npz/json.   usage: mtp_route_pred125.py [FRAC] [MAXTOK]   (GPU, exclusive window)"""
import json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
os.environ.setdefault("MTPLX_CONTEXT_COPY", "0")           # keep every round an MTP verify pass of <= 4 rows
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from bigmoe.loader import load_streamed
from bigmoe import fuse
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 300
MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
TM = rt.model.language_model
LAYERS = TM.model.layers
NL, K, PRED = len(LAYERS), 10, 20
HC, D = TM.args.hc_count, TM.args.hidden_size
gate = st.gate
REC, MTPP = [], []                       # gate calls (seq, L, ids); MTP predictions (seq, step, [NL, PRED] ids)
_orig = gate.route_and_gate


def hook(L, ids, seq=None, preds=None, counts=None):
    REC.append((gate.seq if seq is None else seq, L, ids))
    return _orig(L, ids, seq=seq, preds=preds, counts=counts)


gate.route_and_gate = hook


def predict(h):
    """top-PRED experts of every trunk layer's router on the stream h [.., hc*d] (each layer's own MoE read)."""
    out = []
    for layer in LAYERS:
        hcm = layer.mlp_hyper_connection
        normed = hcm.hc_norm(h)
        w = nn.silu(hcm.input_mix_weight_down(normed) / HC)
        w = mx.sigmoid(hcm.input_mix_weight_up(w))
        mixed = (w.reshape(*w.shape[:-1], HC, D) * normed.reshape(*normed.shape[:-1], HC, D)).mean(axis=-2)
        lg = layer.mlp.gate(mixed).reshape(-1, 512)[-1]
        out.append(mx.argsort(-lg.astype(mx.float32))[:PRED])
    return mx.stack(out)


mtp = TM.mtp
_fr = mtp.fuse_and_run
STEP = [0, -1]                                          # draft step within the current round, seq it belongs to


def fr_hook(widened, tok_emb, cache):
    h = _fr(widened, tok_emb, cache)
    s = int(gate.seq)
    if STEP[1] != s:
        STEP[0], STEP[1] = 0, s
    STEP[0] += 1
    p = predict(h)
    mx.eval(p)
    MTPP.append((s, STEP[0], np.array(p, dtype=np.int16)))
    return h


object.__setattr__(mtp, "fuse_and_run", fr_hook)
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
CB, META, passes, preds = [], [], [], []
for pi, text in enumerate(PROMPTS):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    fuse.reset_generation_state(rt.model)
    REC.clear(); MTPP.clear()

    def cb(toks, pi=pi):
        CB.append((pi, int(gate.seq), len(toks) if isinstance(toks, list) else 1))
    t0 = time.perf_counter()
    res = G.generate_mtpk(rt, ids, max_tokens=MAXTOK, sampler=smp, speculative_depth=3, seed=0,
                          mtp_history_policy="committed", token_callback=cb)
    S = res.stats
    byseq = {}
    for seq, L, a in REC:
        byseq.setdefault(int(seq), {})[int(L)] = np.array(a.reshape(-1, K), dtype=np.int16)
    for seq in sorted(byseq):
        passes.append((pi, seq, byseq[seq]))
    for s, j, p in MTPP:
        preds.append((pi, s, j, p))
    META.append(dict(prompt=pi, tokens=len(res.tokens), verify_calls=S.verify_calls, accepted_by_depth=list(S.accepted_by_depth),
                     gated_passes=len(byseq), mtp_steps=len(MTPP), seconds=round(time.perf_counter() - t0, 1)))
    print(json.dumps(META[-1]), flush=True)
st.gate.check()
P = len(passes)
R = max(max(a.shape[0] for a in d.values()) for _, _, d in passes)
arr = np.full((P, NL, R, K), -1, np.int16)
rows = np.zeros(P, np.int16); seqs = np.zeros(P, np.int64); pidx = np.zeros(P, np.int16)
for i, (pi, seq, d) in enumerate(passes):
    pidx[i], seqs[i] = pi, seq
    for L, a in d.items():
        arr[i, L, :a.shape[0]] = a
        rows[i] = max(rows[i], a.shape[0])
pp = np.stack([p for (_, _, _, p) in preds]) if preds else np.zeros((0, NL, PRED), np.int16)
np.savez_compressed(os.path.expanduser("~/MoonShots/data/mtp_route_pred125.npz"), ids=arr, rows=rows, seq=seqs, prompt=pidx,
                    pred=pp, pred_prompt=np.array([a for (a, _, _, _) in preds], np.int16),
                    pred_seq=np.array([b for (_, b, _, _) in preds], np.int64), pred_step=np.array([c for (_, _, c, _) in preds], np.int16))
json.dump(dict(callbacks=CB, meta=META, K=K, PRED=PRED), open(os.path.expanduser("~/MoonShots/data/mtp_route_pred125.json"), "w"))
print(f"{P} gated passes, {len(preds)} MTP draft steps with predictions", flush=True)
