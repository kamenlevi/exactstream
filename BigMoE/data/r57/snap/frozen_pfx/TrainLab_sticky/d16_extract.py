"""D-16 HCFULL capture (MTP Head, Thu 10-29; REPORT 'D-16'): sticky/d02a_extract.py (same runtime and hooks) saving
only the 125B's final PRE-MIXER WIDENED stream (Qwen4ExpTextModel._last_widened: 4 x 2560 = 10,240 per position, the
native MTP head's own input), fp16, row r = position r. Sources (D16_ONLY = comma-separated key prefixes):
- l120 windows (~/Leap/data/l120_windows.jsonl, L1-20's training set): all rows from position 0 (off 0);
- the Eval suite's tf_gen (tf_code / tf_gsm walk sets) and Chat1000's agent125 turns: rows from 2055 before the
  first generated token, as ~/Leap/data/dfl2d_* and data/d02a store their taps.
Output ~/TrainLab/data/d16w/<key>.npz: wd (rows x 10240 fp16), off, ids. Resumable; stops under 110 GB free disk.
Usage: [BigMoE venv] python sticky/d16_extract.py"""
import json, os, shutil, sys, time, glob
for _k, _v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
               ("MTPLX_COMPILED_GDN", "0"), ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ[_k] = _v
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))      # pausewait (read-only import)
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
import pausewait   # quiet windows: passes wait at their boundary while PAUSE exists (RULES.md, 10-24)
from mtplx.models import qwen4_exp as Q
from mtplx import generation as G

TAPS, TOPK = (5, 15, 25, 35, 45), 32
KEEP = 2055
MD = os.path.expanduser("~/BigMoE/models/bare")
OUT = os.path.expanduser(os.environ.get("D16_OUT", "~/TrainLab/data/d16w")); os.makedirs(OUT, exist_ok=True)
REC = {"on": False, "wd": []}
MEAN_OF = {}

_text_call = Q.Qwen4ExpTextModel.__call__
def _text(self, inputs, cache=None, input_embeddings=None):
    out = _text_call(self, inputs, cache, input_embeddings)
    if REC["on"]:
        w = self._last_widened[0].astype(mx.float16)
        mx.async_eval(w)
        REC["wd"].append(w)
    return out
Q.Qwen4ExpTextModel.__call__ = _text

items = []
for l in open(os.path.expanduser("~/Leap/data/l120_windows.jsonl")):
    r = json.loads(l)
    items.append(("l120:" + r["key"], list(r["ids"]), None))
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    items.append((r["key"].replace("/", "_"), list(r["prompt_ids"]) + list(r["completion_ids"]), len(r["prompt_ids"])))
for f in sorted(glob.glob(os.path.expanduser("~/Chat1000/data/agent125/*.npz"))):
    z = np.load(f)
    items.append(("agent_" + os.path.basename(f)[:-4], [int(x) for x in z["prompt"]] + [int(x) for x in z["out"]],
                  int(z["prompt"].shape[0])))
for l in open(os.path.expanduser("~/TrainLab/data/mgsm_items.jsonl")):   # D-20 / D-22: MGSM, 3 per language, the exact ref's answers
    r = json.loads(l)
    items.append((r["key"], list(r["prompt_ids"]) + list(r["completion_ids"]), len(r["prompt_ids"])))
if os.environ.get("D16_ONLY"):
    pre = tuple(os.environ["D16_ONLY"].split(","))
    items = [it for it in items if it[0].startswith(pre)]

t0 = time.time()
rt, st = load_streamed(MD, int(os.environ.get("D16_C", "51")), mtp=False)
model = rt.model if hasattr(rt, "model") else rt
print(f"[d16-x] loaded {time.time() - t0:.0f}s; {len(items)} sequences, {sum(len(x[1]) for x in items)} tokens", flush=True)
tot = 0
for k, (key, ids, gen) in enumerate(items):
    name = key.split(":", 1)[1] if key.startswith("l120:") else key
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path):
        continue
    if shutil.disk_usage(os.path.expanduser("~")).free / 1e9 < 110:
        print(f"[d16-x] free disk under 110 GB: stopping at {k}/{len(items)} (resumable)", flush=True)
        sys.exit(2)
    t1 = time.time()
    REC["wd"] = []
    REC["on"] = True
    G.restore_or_prefill_prompt_state(rt, ids, base_hidden_variant=None, mtp_history_policy="cycle", capture_hidden=False)
    REC["on"] = False
    st.gate.check()
    wd = mx.concatenate(REC["wd"])
    N = len(ids)
    n = int(wd.shape[0])
    assert n in (N, N - 1), f"{key}: captured rows {n}, want {N} or {N - 1}"
    a = 0 if gen is None else max(0, gen - KEEP)
    np.savez(path, wd=np.array(wd[a:]), off=a, ids=np.array(ids[a:], np.int32))
    tot += N
    mx.clear_cache()
    print(f"[d16-x] {k + 1}/{len(items)} {name}: {N} tokens, {n - a} rows in {time.time() - t1:.1f}s "
          f"({tot / (time.time() - t0):.0f} tok/s overall)", flush=True)
print(f"[d16-x] done in {time.time() - t0:.0f}s, {tot} tokens, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
