"""R-49 POM12 (NewIdeas Thu 18:3x): the 12 GB line's own decode routing, for the CPU price screen (scripts/pom12.py).
The live bt_b12 at its defaults (fold-exact B12LL, MTP 3:3:0 as the suite decodes), fold_check's env. Every trunk layer's
router (mlp.gate, 48 layers) is wrapped at run time: on decode / verify calls (<= 8 rows; prefills are not recorded) it
records each row's top-20 experts and their probabilities (fp32 softmax), plus the tokens emitted before the call, so each
verify pass's rows and how many of them were kept can be rebuilt (a pass = layers 0..47 in order). Numerics untouched (the
recorder only reads the router's output). Prompts in order, each only if it fits before R49_DEADLINE (epoch s): p8k (up to
512 tokens, stops at EOS), the suite's agent rows bash17 / edit00 / text23 / write16 (their 128-token caps), then math
prealgebra/1640 (its cap). One npz per prompt in data/r49/: calls_L, calls_S, calls_E, ids [rows, 20] int16, probs
[rows, 20] fp16, tokens, n_prompt.
Usage: [BigMoE venv] [R49_DEADLINE=epoch] r49_capture.py"""
import json, os, sys, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
OUTD = os.path.expanduser("~/Breakthrough/data/r49")
os.makedirs(OUTD, exist_ok=True)
DL = float(os.environ.get("R49_DEADLINE", "0")) or float("inf")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import bt_b12 as B
t0 = time.perf_counter()
rt, st = B.load_b12(0.10, mtp=True)
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
_TM = model.language_model.model if hasattr(model, "language_model") else model.model
REC = {"on": False, "L": [], "S": [], "E": [], "ids": [], "p": [], "emitted": 0}


class GateRec(nn.Module):
    """the router, unchanged; on decode / verify calls it also keeps each row's top-20 (lazy; evaluated per prompt)"""
    def __init__(self, inner, L):
        super().__init__()
        self.inner = inner
        self.rec_layer = L

    def __call__(self, x):
        g = self.inner(x)
        if REC["on"]:
            S = x.size // x.shape[-1]
            if S <= 8:
                p = mx.softmax(g.reshape(S, g.shape[-1]).astype(mx.float32), axis=-1)
                top = mx.argsort(-p, axis=-1)[:, :20]
                REC["L"].append(self.rec_layer); REC["S"].append(S); REC["E"].append(REC["emitted"])
                REC["ids"].append(top.astype(mx.int16)); REC["p"].append(mx.take_along_axis(p, top, axis=-1).astype(mx.float16))
        return g


n_moe = 0
for i, l in enumerate(_TM.layers):
    if hasattr(getattr(l, "mlp", None), "gate"):
        l.mlp.gate = GateRec(l.mlp.gate, i)
        n_moe += 1
assert n_moe == 48, n_moe
print(f"r49: bt_b12 {B.__file__}, engine {B.V3F}, C_m {B.CFG['cm']} C_l {B.CFG['cl']}, {n_moe} routers wrapped, load "
      f"{time.perf_counter() - t0:.1f} s", flush=True)
EOS = {248044, 248046}
ES = os.path.expanduser("~/EvalSuite/")
AG = {json.loads(l)["key"]: json.loads(l) for l in open(ES + "data/items/dec_agent.jsonl")}
MA = {json.loads(l)["key"]: json.loads(l) for l in open(ES + "data/items/dec_math.jsonl")}
SECS = {json.loads(l)["key"]: json.loads(l)["seconds"] for l in open(ES + "results/b12llfx/decode.jsonl")}


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
JOBS = [("p8k", lambda: chat_ids(HEAD + open(os.path.expanduser("~/Breakthrough/data/prompts/ctx32k.txt")).read()[:26000]
                                 + "\n```"), 512, 75)]
for k in ("agent/bash17", "agent/edit00", "agent/text23", "agent/write16"):
    JOBS.append((k.replace("/", "_"), (lambda k=k: list(AG[k]["prompt_ids"])), AG[k]["max_tokens"], SECS[k]))
k = "math500/test/prealgebra/1640.json"
JOBS.append(("math_prealgebra_1640", lambda: list(MA[k]["prompt_ids"]), MA[k]["max_tokens"], SECS[k]))
for name, get_ids, cap, est in JOBS:
    need = 1.3 * est + 20
    if time.time() + need > DL:
        print(f"r49: {name} skipped (needs ~{need:.0f} s, {DL - time.time():.0f} s left)", flush=True)
        continue
    ids = get_ids()
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    for key in ("L", "S", "E", "ids", "p"):
        REC[key] = []
    REC["emitted"] = 0
    t1 = time.perf_counter()
    state = B.b12_prefill_state(model, list(ids), 512, 4096)
    REC["on"] = True

    def on_tok(t):
        REC["emitted"] += 1
    out, s = spec125.generate(model, list(ids), cap, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state,
                              stats={}, stop_ids=EOS, on_token=on_tok)
    REC["on"] = False
    st.gate.check()
    mx.eval(REC["ids"], REC["p"])
    out = [int(x) for x in out]
    np.savez_compressed(os.path.join(OUTD, f"{name}.npz"), calls_L=np.array(REC["L"], np.int16),
                        calls_S=np.array(REC["S"], np.int8), calls_E=np.array(REC["E"], np.int32),
                        ids=np.concatenate([np.array(a) for a in REC["ids"]]),
                        probs=np.concatenate([np.array(a) for a in REC["p"]]), tokens=np.array(out, np.int32),
                        n_prompt=len(ids))
    nc = len(REC["L"])
    print(f"r49: {name} RESULT prompt {len(ids)} | {len(out)} tokens, {REC['emitted']} emitted, {nc} router calls = "
          f"{nc / 48:.1f} passes, {sum(REC['S']) / max(nc, 1):.2f} rows / call | {time.perf_counter() - t1:.0f} s "
          f"(est {est:.0f})", flush=True)
    del state
    mx.clear_cache()
print("r49: done", flush=True)
