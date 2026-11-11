"""D-02a capture (MTP Head, Thu 10-29; NewIdeas 15:5x; REPORT 'D-02a'): /Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts/dfl2_extract.py (read-only copy,
same runtime and hooks: MTPLX prefill on BigMoE's streamed loader, compiled paths off) plus the 125B's top-32 next-token
ids / log-probs per row, computed as /Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts/dfl2c_extract.py does. Per position:
- f: the mean of the 4 hyper-connection streams after zero-based layers 5/15/25/35/45 (DFlash2's taps), fp16;
- top1 (= tk[:, 0]), tk / lp: the 125B's top-32 ids (sorted, best first) and log-probs (fp16) for the NEXT token.
Sources: the Eval suite's tf_gen (the 125B's own greedy continuations; tf_gsm / tf_ifeval / tf_multi / tf_code) and
Chat1000's 12 agentic turns (agent125). Kept rows: from 2055 before the first generated token (the drafter's window is
2048) to the end; the last token is not prefilled. Same fields as ~/Leap/data/dfl2d_* (f, top1, ids, off, gen), so
sticky/d02b_walk.py reads them, plus tk / lp.
Output ~/TrainLab/data/d02a/<key>.npz. Resumable; stops under 110 GB free disk; passes wait while PAUSE exists.
Env: D02A_SRC=d17 (D-17: data/d17's prompt + out instead), D02A_ONLY = comma-separated key prefixes (e.g. tf_gsm,tf_ifeval), D02A_OUT, D02A_C (resident experts per layer).
Usage: [BigMoE venv] python sticky/d02a_extract.py"""
import json, os, shutil, sys, time, glob
for _k, _v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
               ("MTPLX_COMPILED_GDN", "0"), ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ[_k] = _v
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))      # pausewait (read-only import)
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
import pausewait   # quiet windows: passes wait at their boundary while PAUSE exists (RULES.md, 10-24)
from mtplx.models import qwen4_exp as Q
from mtplx import generation as G

TAPS, TOPK = (5, 15, 25, 35, 45), 32
KEEP = 2055
MD = os.path.expanduser("~/BigMoE/models/bare")
OUT = os.path.expanduser(os.environ.get("D02A_OUT", "~/TrainLab/data/d02a")); os.makedirs(OUT, exist_ok=True)
REC = {"on": False, "mean": {}, "tk": [], "lp": []}
MEAN_OF = {}

_layer_call = Q.DecoderLayer.__call__
def _layer(self, hidden, **kw):
    out = _layer_call(self, hidden, **kw)
    i = MEAN_OF.get(id(self))
    if REC["on"] and i is not None:
        B, L, W = out.shape
        x = out.reshape(B, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16)
        mx.async_eval(x)
        REC["mean"].setdefault(i, []).append(x)
    return out
Q.DecoderLayer.__call__ = _layer

_text_call = Q.Qwen4ExpTextModel.__call__
def _text(self, inputs, cache=None, input_embeddings=None):
    out = _text_call(self, inputs, cache, input_embeddings)
    if REC["on"]:
        h = out[0]
        for i in range(0, h.shape[0], 256):
            lg = HEAD(h[i:i + 256]).astype(mx.float32)
            top = mx.argpartition(-lg, TOPK - 1, axis=-1)[:, :TOPK]
            v = mx.take_along_axis(lg, top, axis=-1)
            o = mx.argsort(-v, axis=-1)
            top, v = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(v, o, axis=-1)
            lp = (v - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float16)
            mx.async_eval(top, lp)
            REC["tk"].append(top.astype(mx.int32)); REC["lp"].append(lp)
    return out
Q.Qwen4ExpTextModel.__call__ = _text

items = []
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    items.append((r["key"].replace("/", "_"), list(r["prompt_ids"]) + list(r["completion_ids"]), len(r["prompt_ids"])))
for f in sorted(glob.glob(os.path.expanduser("~/Chat1000/data/agent125/*.npz"))):
    z = np.load(f)
    items.append(("agent_" + os.path.basename(f)[:-4], [int(x) for x in z["prompt"]] + [int(x) for x in z["out"]],
                  int(z["prompt"].shape[0])))
if os.environ.get("D02A_SRC") == "d17":              # D-17: the texts of the MTP-chain decode (data/d17: prompt + out)
    items = []
    for f in sorted(glob.glob(os.path.expanduser("~/TrainLab/data/d17/*.npz"))):
        z = np.load(f)
        items.append((os.path.basename(f)[:-4], [int(x) for x in z["prompt"]] + [int(x) for x in z["out"]], int(z["prompt"].shape[0])))
for l in (open(os.path.expanduser("~/TrainLab/data/mgsm_items.jsonl")) if os.environ.get("D02A_SRC") != "d17" else []):   # D-20 / D-22: MGSM
    r = json.loads(l)                                # (3 per language, the exact ref's answers)
    items.append((r["key"], list(r["prompt_ids"]) + list(r["completion_ids"]), len(r["prompt_ids"])))
if os.environ.get("D02A_ONLY"):
    pre = tuple(os.environ["D02A_ONLY"].split(","))
    items = [it for it in items if it[0].startswith(pre)]

t0 = time.time()
rt, st = load_streamed(MD, int(os.environ.get("D02A_C", "51")), mtp=False)
model = rt.model if hasattr(rt, "model") else rt
HEAD = model.language_model.lm_head
layers = model.language_model.model.layers
for i, L in enumerate(TAPS):
    MEAN_OF[id(layers[L])] = i
print(f"[d02a-x] loaded {time.time() - t0:.0f}s; {len(items)} sequences, {sum(len(x[1]) for x in items)} tokens", flush=True)
tot = 0
for k, (key, ids, gen) in enumerate(items):
    path = os.path.join(OUT, key + ".npz")
    if os.path.exists(path):
        continue
    if shutil.disk_usage(os.path.expanduser("~")).free / 1e9 < 110:
        print(f"[d02a-x] free disk under 110 GB: stopping at {k}/{len(items)} (resumable)", flush=True)
        sys.exit(2)
    t1 = time.time()
    REC["mean"], REC["tk"], REC["lp"] = {}, [], []
    REC["on"] = True
    G.restore_or_prefill_prompt_state(rt, ids, base_hidden_variant=None, mtp_history_policy="cycle", capture_hidden=False)
    REC["on"] = False
    st.gate.check()
    N = len(ids)
    f = [mx.concatenate(REC["mean"][i]) for i in range(len(TAPS))]
    tk, lp = mx.concatenate(REC["tk"]), mx.concatenate(REC["lp"])
    n = {int(x.shape[0]) for x in f} | {int(tk.shape[0])}
    if n == {N - 1} or (n == {N, N - 1} and int(tk.shape[0]) == N - 1):
        N -= 1                                   # the last row is not prefilled: its features are never needed
        f, tk, lp = [x[:N] for x in f], tk[:N], lp[:N]
        n = {N}
    assert n == {N}, f"{key}: captured rows {n}, want {N}"
    a = max(0, gen - KEEP)
    tk, lp = np.array(tk[a:]), np.array(lp[a:])
    np.savez(path, f=np.stack([np.array(x[a:]) for x in f], 1), top1=tk[:, 0].copy(), tk=tk, lp=lp,
             ids=np.array(ids[a:N], np.int32), off=a, gen=gen)
    tot += N
    mx.clear_cache()
    agree = float((tk[gen - 1 - a:N - 1 - a, 0] == np.array(ids[gen:N])).mean()) if N > gen else float("nan")
    print(f"[d02a-x] {k + 1}/{len(items)} {key}: {N} tokens in {time.time() - t1:.1f}s ({tot / (time.time() - t0):.0f} "
          f"tok/s overall); prefill top-1 = stored continuation on {agree:.3f}", flush=True)
print(f"[d02a-x] done in {time.time() - t0:.0f}s, {tot} tokens, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
