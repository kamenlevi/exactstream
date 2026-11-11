"""Agentic turns on the 125B (Chat1000, 2026-10-21 22:20): the raw material for tau and union of exact speculative
decoding on real Claude Code turns. For each turn (SpecDecode route4 prompts: real turns from Kamen's sessions, already
tokenized; the 125B tokenizer is identical), SpecDecode's spec125.generate runs with max_draft=0 (one token per pass =
plain greedy, exact) and k_mtp=15, so the native MTP head drafts a 15-token chain at EVERY position from the true prefix.
A hook on the MoE blocks records each decoded token's top-10 experts per main layer (MTP layer excluded).
Saves data/agent125/<turn>.npz: out (greedy tokens), mtp (T x 15 drafts), routes (T x 48 x 10), prompt ids.
Offline: agent_analyze.py (tau for MTP / lookup / hybrid at K 8-32, unions, projections).
Usage: [BigMoE venv] agent125.py FRAC N_TOKENS [turn ...]"""
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
from mtplx.models import qwen4_exp as _q

MD = os.path.expanduser("~/BigMoE/models/bare")
RUN = os.environ.get("TL_MTP_RUN", "")
OUT = os.path.expanduser("~/TrainLab/data/agent125_" + (RUN or "stock")); os.makedirs(OUT, exist_ok=True)
frac, N = float(sys.argv[1]), int(sys.argv[2])
TURNS = sys.argv[3:] or ["edit00", "edit03", "edit06", "edit09", "bash17", "bash19", "bash21", "text23", "text25", "text27",
                         "write11", "write14"]
tc = json.load(open(os.path.join(MD, "config.json"))); tc = tc.get("text_config", tc)
NL = tc["num_hidden_layers"]


class REC:
    on = False
    buf = {}


_orig = _q.SparseMoeBlock.__call__
def _traced(self, x):
    bm = getattr(self.switch_mlp, "_bm", None)
    if REC.on and bm is not None and x.shape[1] == 1 and bm["L"] < NL:
        g = self.gate(x)[0, 0].astype(mx.float32)
        REC.buf.setdefault(bm["L"], []).append(mx.argsort(-g, axis=-1)[:10].astype(mx.int16))
    return _orig(self, x)
_q.SparseMoeBlock.__call__ = _traced

t0 = time.time()
rt, st = load_streamed(MD, max(16, round(frac * tc["num_experts"])), mtp=True)
print(f"loaded {time.time() - t0:.0f}s", flush=True)
if RUN:                                           # TrainLab: the fine-tuned head in place of the stock one
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/TrainLab_root"))
    os.environ["TL_INFER"] = "1"              # keep the fused (bit-identical) HC prefill read
    import mtp125 as M
    _head = rt.model.language_model.mtp
    _ck = os.path.expanduser(f"~/TrainLab/ckpt/{RUN}.mtp125")
    if json.load(open(_ck + ".json")).get("native"):          # native run: the head's own modules were trained
        M.native_prepare(_head)
        _head.load_weights(_ck + ".safetensors", strict=False)
        mx.eval(_head.parameters())
    else:
        _P = M.Trainable(_head); _P.load_weights(_ck + ".safetensors")
        M.install(_head, _P)
    print(f"installed {RUN}", flush=True)
spec125.generate(rt.model, rt.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True,
                 enable_thinking=False), 8, k_mtp=3, max_draft=0, lookup=0)   # warm-up
for name in TURNS:
    d = np.load(os.path.expanduser(f"~/SpecDecode/data/route4/{name}.npz"))
    ids = [int(t) for t in d["ids"]]
    REC.buf = {}; REC.on = True
    s = {"trace": []}
    t1 = time.time()
    out, s = spec125.generate(rt.model, ids, N, k_mtp=15, max_draft=0, lookup=0, stats=s)
    REC.on = False
    st.gate.check()
    tr = s["trace"]
    T = len(out)
    mtp = np.full((T, 15), -1, np.int64)
    pos = 0
    for r in tr:                      # one token per pass: the pass at position pos drafted mtp for out[pos], out[pos+1], ...
        m = r.get("mtp") or []
        if pos < T and m:
            mtp[pos, :min(15, len(m))] = m[:15]
        pos += r["n"] + 1
    layers = sorted(REC.buf)
    nrows = min(len(REC.buf[l]) for l in layers) if layers else 0
    routes = np.stack([np.array(mx.stack(REC.buf[l][:nrows])) for l in layers], 1) if layers else np.zeros((0, 0, 10))
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), out=np.array(out), mtp=mtp, routes=routes, prompt=np.array(ids),
                        layers=np.array(layers))
    print(f"{name}: prompt {len(ids)}, {T} tokens in {time.time() - t1:.0f}s (prefill {s.get('t_prefill', 0):.0f}s), "
          f"routes {routes.shape}, mtp depth-1 hit {np.mean(mtp[:-1, 0] == np.array(out)[:-1]) if T > 1 else 0:.3f}", flush=True)
print(f"done in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
