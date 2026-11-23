"""Leap L2 step 1 (2026-10-23): the 125B's DFlash2 target features, teacher-forced, adapted from Chat1000's
scripts/wc_extract.py (same runtime and hooks: MTPLX prefill on BigMoE's streamed loader, compiled paths off so every row
passes the hooks). Per position: the mean of the 4 hyper-connection streams after zero-based layers 5/15/25/35/45 (as the
incoai capture patch: "completed, zero-based layer outputs, averaged over the four hyper-connection streams"), fp16, and
the 125B's own prefill top-1. Sources: Eval suite tf_gen (own greedy continuations) and Chat1000's 12 agentic turns.
Kept rows: from 2055 positions before the first generated token to the end (the drafter's window is 2048).
Output ~/Leap/data/dfl2/<key>.npz: f (n x 5 x 2560 fp16), top1 (n), ids (n), off (absolute position of row 0), gen.
Resumable; stops under 110 GB free disk. Usage: [BigMoE venv] dfl2_extract.py"""
import json, os, shutil, sys, time, glob
for _k, _v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
               ("MTPLX_COMPILED_GDN", "0"), ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ[_k] = _v
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
import pausewait   # quiet windows: passes wait at their boundary while PAUSE exists (RULES.md, 10-24)
from mtplx.models import qwen4_exp as Q
from mtplx import generation as G

TAPS = (5, 15, 25, 35, 45)
KEEP = int(os.environ.get("DFL2_KEEP", "2055"))
MD = os.path.expanduser("~/BigMoE/models/bare")
OUT = os.path.expanduser(os.environ.get("DFL2_OUT", "~/Leap/data/dfl2")); os.makedirs(OUT, exist_ok=True)
REC = {"on": False, "mean": {}, "out": []}
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
        top = mx.concatenate([mx.argmax(HEAD(h[i:i + 512]), axis=-1).astype(mx.int32) for i in range(0, h.shape[0], 512)])
        mx.async_eval(top)
        REC["out"].append(top)
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
if os.environ.get("DFL2_ONLY"):
    items = [it for it in items if it[0].startswith(os.environ["DFL2_ONLY"])]
if os.environ.get("DFL2_LIMIT"):
    items = items[:int(os.environ["DFL2_LIMIT"])]

t0 = time.time()
rt, st = load_streamed(MD, 51, mtp=False)
model = rt.model if hasattr(rt, "model") else rt
HEAD = model.language_model.lm_head
layers = model.language_model.model.layers
for i, L in enumerate(TAPS):
    MEAN_OF[id(layers[L])] = i
print(f"[dfl2x] loaded {time.time() - t0:.0f}s; {len(items)} sequences, {sum(len(x[1]) for x in items)} tokens", flush=True)
tot = 0
for k, (key, ids, gen) in enumerate(items):
    path = os.path.join(OUT, key + ".npz")
    if os.path.exists(path):
        continue
    if shutil.disk_usage(os.path.expanduser("~")).free / 1e9 < 110:
        print(f"[dfl2x] free disk under 110 GB: stopping at {k}/{len(items)} (resumable)", flush=True)
        sys.exit(2)
    t1 = time.time()
    REC["mean"], REC["out"] = {}, []
    REC["on"] = True
    G.restore_or_prefill_prompt_state(rt, ids, base_hidden_variant=None, mtp_history_policy="cycle", capture_hidden=False)
    REC["on"] = False
    st.gate.check()
    N = len(ids)
    f = [mx.concatenate(REC["mean"][i]) for i in range(len(TAPS))]
    top = mx.concatenate(REC["out"])
    n = {int(x.shape[0]) for x in f} | {int(top.shape[0])}
    if n == {N - 1} or (n == {N, N - 1} and int(top.shape[0]) == N - 1):
        N -= 1                                   # the last row is not prefilled: its features are never needed
        f, top = [x[:N] for x in f], top[:N]
        n = {N}
    assert n == {N}, f"{key}: captured rows {n}, want {N}"
    a = max(0, gen - KEEP)
    np.savez(path, f=np.stack([np.array(x[a:]) for x in f], 1), top1=np.array(top[a:]), ids=np.array(ids[a:N], np.int32),
             off=a, gen=gen)
    tot += N
    mx.clear_cache()
    agree = float((np.array(top)[gen - 1:N - 1] == np.array(ids[gen:N])).mean()) if N > gen else float("nan")
    print(f"[dfl2x] {k + 1}/{len(items)} {key}: {N} tokens in {time.time() - t1:.1f}s ({tot / (time.time() - t0):.0f} tok/s "
          f"overall); prefill top-1 = stored continuation on {agree:.3f}", flush=True)
print(f"[dfl2x] done in {time.time() - t0:.0f}s, {tot} tokens, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
