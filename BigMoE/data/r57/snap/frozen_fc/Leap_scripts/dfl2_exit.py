"""QUEUE L1-09 (Leap, Sat 2026-10-24): early-exit verify screen, teacher-forced, same runtime and hooks as dfl2_extract.py.
An exit after zero-based layer 35 / 39 (= after 36 / 40 of 48 layers) applies the model's own final hyper-connection
mixer (the text model's last step) and LM head to that layer's widened stream, exactly as the full model does after
layer 47. Per completion position: the exit's top-1 and max probability, and the full model's.
Reported per set (tf_code, tf_gsm from EvalSuite's tf_gen; the 12 agentic turns): top-1 agreement with the full model
overall, and on positions where the FULL model's p >= .9 (NewIdeas' line: kill if < 0.98 at layer 40 on code), and where
the EXIT's own p >= .9 (what a real gate can see), with the share of positions each gate admits.
Usage: [BigMoE venv] [DFL2X_EXITS=35,39] [DFL2X_N=0] [DFL2X_OUT=...] dfl2_exit.py   -> ~/Leap/data/exit_result.txt
Control (Sat 06:05): DFL2X_EXITS=43,45,46,47 DFL2X_N=6: the exit after layer 47 must equal the full model (agree 1.000)."""
import json, os, sys, time, glob
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

EXITS = tuple(int(x) for x in os.environ.get("DFL2X_EXITS", "35,39").split(","))   # zero-based; 47 = the full model
NMAX = int(os.environ.get("DFL2X_N", "0"))          # > 0: only the first N sequences of each set (the control run)
OUT = os.path.expanduser(os.environ.get("DFL2X_OUT", "~/Leap/data/exit_result.txt"))
MD =os.path.expanduser("~/BigMoE/models/bare")
REC = {"on": False, "exit": {}, "fin": []}
EXIT_OF = {}

_layer_call = Q.DecoderLayer.__call__
def _layer(self, hidden, **kw):
    out = _layer_call(self, hidden, **kw)
    i = EXIT_OF.get(id(self))
    if REC["on"] and i is not None:
        x = out[0]; mx.async_eval(x)
        REC["exit"].setdefault(i, []).append(x)
    return out
Q.DecoderLayer.__call__ = _layer

_text_call = Q.Qwen4ExpTextModel.__call__
def _text(self, inputs, cache=None, input_embeddings=None):
    out = _text_call(self, inputs, cache, input_embeddings)
    if REC["on"]:
        x = out[0]; mx.async_eval(x)
        REC["fin"].append(x)
    return out
Q.Qwen4ExpTextModel.__call__ = _text

items = []
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    if r["key"].startswith(("tf_code", "tf_gsm")):
        if NMAX and sum(1 for x in items if x[0] == r["key"].split("/")[0]) >= NMAX:
            continue
        items.append((r["key"].split("/")[0], list(r["prompt_ids"]) + list(r["completion_ids"]), len(r["prompt_ids"])))
for f in sorted(glob.glob(os.path.expanduser("~/Chat1000/data/agent125/*.npz"))):
    if NMAX and sum(1 for x in items if x[0] == "agent") >= NMAX:
        break
    z = np.load(f)
    items.append(("agent", [int(x) for x in z["prompt"]] + [int(x) for x in z["out"]], int(z["prompt"].shape[0])))

t0 = time.time()
rt, st = load_streamed(MD, 51, mtp=False)
model = rt.model if hasattr(rt, "model") else rt
HEAD = model.language_model.lm_head
TEXT = model.language_model.model
for L in EXITS:
    EXIT_OF[id(TEXT.layers[L])] = L
print(f"[exit] loaded {time.time() - t0:.0f}s; {len(items)} sequences, {sum(len(x[1]) for x in items)} tokens", flush=True)

def top_p(h):
    out_t, out_p = [], []
    for i in range(0, h.shape[0], 256):
        lg = HEAD(h[i:i + 256]).astype(mx.float32)
        p = mx.softmax(lg, axis=-1)
        t = mx.argmax(lg, axis=-1)
        out_t.append(t); out_p.append(mx.take_along_axis(p, t[:, None], axis=-1)[:, 0])
    t, p = mx.concatenate(out_t), mx.concatenate(out_p)
    mx.eval(t, p)
    return np.array(t), np.array(p)

res = {}
for k, (cls, ids, gen) in enumerate(items):
    t1 = time.time()
    REC["exit"], REC["fin"] = {}, []
    REC["on"] = True
    G.restore_or_prefill_prompt_state(rt, ids, base_hidden_variant=None, mtp_history_policy="cycle", capture_hidden=False)
    REC["on"] = False
    st.gate.check()
    fin = mx.concatenate(REC["fin"])
    n = min(fin.shape[0], len(ids) - 1)              # row r predicts token r + 1; the last token predicts nothing
    lo = gen - 1                                     # rows lo..n-1 predict the completion tokens
    tf, pf = top_p(fin[lo:n])
    r = res.setdefault(cls, {"pos": 0, "f": [], "e": {L: [] for L in EXITS}})
    r["f"].append((tf, pf, np.array(ids[gen:n + 1])))
    for L in EXITS:
        ex = mx.concatenate(REC["exit"][L])[lo:n][None]
        te, pe = top_p(TEXT.hyper_connection_mixer(ex)[0])
        r["e"][L].append((te, pe))
    r["pos"] += n - lo
    del fin, REC["exit"]; REC["exit"] = {}
    mx.clear_cache()
    print(f"[exit] {k + 1}/{len(items)} {cls}: {n - lo} positions in {time.time() - t1:.1f}s; full top-1 = stored token on "
          f"{np.mean(tf == r['f'][-1][2]):.3f}", flush=True)

lines = []
for cls, r in res.items():
    tf = np.concatenate([x[0] for x in r["f"]]); pf = np.concatenate([x[1] for x in r["f"]])
    for L in EXITS:
        te = np.concatenate([x[0] for x in r["e"][L]]); pe = np.concatenate([x[1] for x in r["e"][L]])
        ag = te == tf
        s = f"{cls:8s} exit after {L + 1} layers: agree {ag.mean():.3f} (n {len(ag)})"
        for thr in (0.9, 0.99):
            m = pf >= thr; me = pe >= thr
            s += (f" | full p>={thr}: {ag[m].mean() if m.any() else float('nan'):.3f} on {m.mean():.0%}"
                  f" | exit p>={thr}: {ag[me].mean() if me.any() else float('nan'):.3f} on {me.mean():.0%}")
        lines.append(s); print(s, flush=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"[exit] done in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
