"""EARLYTAP extractor (MTP Head, Mon 11-02; REPORT 'EARLYTAP'): a copy of /Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts/dfl2c_extract.py (read-only source).
TX_MODE=train (default): L1-20's windows (~/Leap/data/l120_windows.jsonl) -> three folders data/taps_h5 / taps_h4 /
taps_h3, each in l120's format (h = hidden_norm(fc(taps)) as bf16 bits, tk, lp, ids, asst, split) with taps {5..45},
{5..35} (tap 45 zeroed), {5..25} (35, 45 zeroed) through the same frozen fc / hidden_norm. TX_MODE=walk: the first 100
MBPP capture files (data/dlog/d1102mbpp, sorted) -> data/taps_mbpp_walk/<name>.npz with the raw taps (f [n, 5, 2560]
f16, ids, off 0, gen), the walk's format. Original docstring:
Leap L3 step 1 (2026-10-23): the 125B's features and labels on the agentic training windows (dfl2c_data.py), same
runtime and hooks as dfl2_extract.py (Chat1000's wc_extract.py pattern). Per position:
- h: the drafter's own context input, hidden_norm(fc(taps)) with taps = the mean of the 4 hyper-connection streams after
  zero-based layers 5/15/25/35/45, computed exactly as the official path does (bf16), stored as bf16 bits (uint16);
- tk / lp: the 125B's top-32 next-token ids (sorted, best first) and log-probs (fp16) = the regeneration-free labels.
The last token is not prefilled, so rows are 0..N-2 (ids keeps all N). Also logs, per window, how often the 125B's
greedy token equals the session text on assistant tokens (the on-path rate).
Output ~/Leap/data/dfl2c/<key>.npz. Resumable; stops under 110 GB free disk. Usage: [BigMoE venv] dfl2c_extract.py
L4 (Sat 10-24): DFL2C_IN / DFL2C_OUT pick the windows file and output folder. For windows with a "gen" field (prompt +
the 125B's output), rows the trainer never reads (before gen - 2047; labels before gen - 1) are zeros, saved compressed."""
import json, os, shutil, sys, time
for _k, _v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
               ("MTPLX_COMPILED_GDN", "0"), ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ[_k] = _v
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_third_party"))
sys.path.insert(2, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts"))   # pausewait (read-only import; the original found it as its own folder)
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
import pausewait   # quiet windows: passes wait at their boundary while PAUSE exists (RULES.md, 10-24)
from mtplx.models import qwen4_exp as Q
from mtplx import generation as G
import dflash_model_mlx as D

TAPS, TOPK = (5, 15, 25, 35, 45), 32
MD = os.path.expanduser("~/BigMoE/models/bare")
MODE = os.environ.get("TX_MODE", "train")
TD = os.path.expanduser("~/TrainLab/data")
OUTS = {v: os.path.join(TD, f"taps_{v}") for v in ("h5", "h4", "h3", "raw")} if MODE == "train" else {"walk": os.path.join(TD, "taps_mbpp_walk")}
for o in OUTS.values():
    os.makedirs(o, exist_ok=True)
KEEP = {"h5": (1, 1, 1, 1, 1), "h4": (1, 1, 1, 1, 0), "h3": (1, 1, 1, 0, 0)}
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

if MODE == "train":
    rows = [json.loads(l) for l in open(os.path.expanduser("~/Leap/data/l120_windows.jsonl"))]
else:                                                    # walk: the same 100 MBPP answers as MATCH (sorted names)
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(TD, "dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]:
        z = np.load(p)
        rows.append(dict(key=os.path.basename(p)[:-4], ids=[int(x) for x in z["ids"]], gen=int(z["gen"]),
                         asst=[int(x) for x in z["asst"]], split="walk"))
if os.environ.get("DFL2C_LIMIT"):
    rows = rows[:int(os.environ["DFL2C_LIMIT"])]
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")      # only fc + hidden_norm are used here
t0 = time.time()
rt, st = load_streamed(MD, int(os.environ.get("TX_C", "51")), mtp=False)   # C = resident experts per layer: speed (not re-checked bit-equal)
model = rt.model if hasattr(rt, "model") else rt
HEAD = model.language_model.lm_head
layers = model.language_model.model.layers
for i, L in enumerate(TAPS):
    MEAN_OF[id(layers[L])] = i
print(f"[dfl2c-x] loaded {time.time() - t0:.0f}s; {len(rows)} windows, {sum(len(r['ids']) for r in rows)} tokens", flush=True)
tot, on_n, on_d = 0, 0, 0
for k, r in enumerate(rows):
    paths = {v: os.path.join(o, r["key"] + ".npz") for v, o in OUTS.items()}
    if all(os.path.exists(p_) for p_ in paths.values()):
        continue
    if shutil.disk_usage(os.path.expanduser("~")).free / 1e9 < 114:
        print(f"[dfl2c-x] free disk under 110 GB: stopping at {k}/{len(rows)} (resumable)", flush=True)
        sys.exit(2)
    ids, t1 = r["ids"], time.time()
    REC["mean"], REC["tk"], REC["lp"] = {}, [], []
    REC["on"] = True
    G.restore_or_prefill_prompt_state(rt, ids, base_hidden_variant=None, mtp_history_policy="cycle", capture_hidden=False)
    REC["on"] = False
    st.gate.check()
    f = [mx.concatenate(REC["mean"][i]) for i in range(len(TAPS))]
    tk, lp = mx.concatenate(REC["tk"]), mx.concatenate(REC["lp"])
    n = len(ids) - 1
    assert {int(x.shape[0]) for x in f} | {int(tk.shape[0])} <= {n, n + 1}, (r["key"], [x.shape for x in f], tk.shape)
    f, tk, lp = [x[:n] for x in f], tk[:n], lp[:n]
    taps5 = mx.stack(f, axis=1)                      # [n, 5, 2560] f16: the 5 taps
    tk_np, ids_np, asst = np.array(tk), np.array(ids, np.int32), np.array(r["asst"], np.uint8)
    lp_np = np.array(lp)
    if MODE == "walk":                               # the walk's format: raw taps (f), ids, off 0, gen
        np.savez(paths["walk"], f=np.array(taps5), ids=ids_np, off=np.int64(0), gen=np.int64(r["gen"]), top1=tk_np[:, 0])
    else:
        np.savez(paths["raw"], f=np.array(taps5), tk=tk_np, lp=lp_np, ids=ids_np, asst=asst, split=r["split"])   # kept for Fable's
        for v, keep in KEEP.items():                 # a dropped tap = zeros before the frozen fc (= its columns removed)
            taps = (taps5 * mx.array(keep, dtype=taps5.dtype)[None, :, None]).reshape(n, -1)
            h = mx.concatenate([draft.hidden_norm(draft.fc(taps[i:i + 1024].astype(mx.bfloat16))) for i in range(0, n, 1024)])
            mx.eval(h)
            hb = (np.array(h.astype(mx.float32)).view(np.uint32) >> 16).astype(np.uint16)    # bf16 bits, exact
            np.savez(paths[v], h=hb, tk=tk_np, lp=lp_np, ids=ids_np, asst=asst, split=r["split"])
        h = None
    m = asst[1:n + 1].astype(bool)
    hit = tk_np[:, 0] == ids_np[1:n + 1]
    on_n += int(hit[m].sum()); on_d += int(m.sum()); tot += len(ids)
    del f, tk, lp, taps5
    mx.clear_cache()
    print(f"[dfl2c-x] {k + 1}/{len(rows)} {r['key']} ({r['split']}): {len(ids)} tokens in {time.time() - t1:.1f}s; "
          f"125B greedy = text on assistant tokens {hit[m].mean() if m.any() else float('nan'):.3f} "
          f"({tot / (time.time() - t0):.0f} tok/s overall)", flush=True)
print(f"[dfl2c-x] done in {time.time() - t0:.0f}s, {tot} tokens; on-path rate (assistant) {on_n / max(1, on_d):.3f}; "
      f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
