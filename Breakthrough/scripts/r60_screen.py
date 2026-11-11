"""R-60 MTPDROP screen (NewIdeas Fri 23:1x; lines in SCOREBOARD; exact by construction: drafts only): MTP Head's trained
MTP head (D-22 b3) in bt_b12's MTP path (spec125 3:3:0) at the live defaults (fold-fast, decision 5), one process per arm:
  R60_ARM=stock  the head as shipped (4-bit dense parts)
  R60_ARM=d22    B12_MTP_HEAD=~/TrainLab/ckpt/d22_b3.safetensors in bf16      (R60_CKPT overrides the file)
  R60_ARM=d22q8  the same in 8-bit g64 (B12_MTP_HEAD_BITS=8)
Prompts: p2k / p8k / write16 (cand_check's, 256 tokens, write16 stops at EOS) + the first R60_MATH (8) math rows of the
Eval suite (dec_math items, stop at EOS; tokens vs results/b12llff2, the fold-fast reference for short prompts). Per
prompt: tokens (sha), verify passes, tokens per pass, decode s, lifetime peak. Timings are "shared GPU, rough".
Usage: [BigMoE venv] BT_DIR=<candidate dir> R60_ARM=stock|d22|d22q8 r60_screen.py -> data/r60/<arm>.jsonl"""
import ctypes, hashlib, json, os, sys, threading, time

ARM = os.environ.get("R60_ARM", "stock")
CK = os.path.expanduser(os.environ.get("R60_CKPT", "~/TrainLab/ckpt/d22_b3.safetensors"))
for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_CM", "B12_PFR_MIN", "B12_MODE", "B12_M01",
          "BIGMOE_M01", "B12_FRS_NOCYCLE", "MTPLX_NGRAM_HOT_MB", "B12_MTP_HEAD", "B12_MTP_HEAD_BITS", "EVALSUITE_FRUGAL",
          "EVALSUITE_FAST24", "B12_W", "B12_CM", "B12_CL", "BIGMOE_PF_PREFILL", "BIGMOE_PF_FULLSCR"):
    os.environ.pop(k, None)                          # the candidate's own defaults (fold-fast), as the suite's live process
os.environ.update({"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0",
                   "BIGMOE_PF_GROUP": "64", "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1",
                   "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80", "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1",
                   "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1", "MTPLX_DEFER_REPAIR_EVAL": "1",
                   "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1", "B12_TRIMS": "1",
                   "B12_PFR": "1", "B12_ALLWAIT": "0", "EVALSUITE_SHIP": "1", "MTPLX_NGRAM_PREWARM": "off",
                   "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0", "MTPLX_COMPILE_AR_FORWARD": "0",
                   "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"})
if ARM != "stock":
    os.environ["B12_MTP_HEAD"] = CK
    if ARM == "d22q8":
        os.environ["B12_MTP_HEAD_BITS"] = "8"
NMATH = int(os.environ.get("R60_MATH", "8"))
OUTD = os.path.expanduser("~/Breakthrough/data/r60")
os.makedirs(OUTD, exist_ok=True)


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def ru():                                            # (footprint GB, lifetime peak GB)
    r = _RU()
    assert _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) == 0
    return r.u7 / 1e9, r.u28 / 1e9


import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__))
BTD = os.environ.get("BT_DIR") or HERE
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, BTD)
BTSHA = hashlib.sha256(open(os.path.join(BTD, "bt_b12.py"), "rb").read()).hexdigest()[:10]
import bt_b12 as B
assert os.path.dirname(os.path.abspath(B.__file__)) == os.path.abspath(BTD), (B.__file__, BTD)
t0 = time.perf_counter()
rt, st = B.load_b12(0.10, mtp=True)
t_load = time.perf_counter() - t0
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
assert (ARM == "stock") == ("mtp_head" not in B.CFG), (ARM, B.CFG.get("mtp_head"))
print(f"[r60 {ARM}] bt_b12 {BTSHA} engine {B.V3F} mode {B.MODE} | MTP head {B.CFG.get('mtp_head', 'stock')} | load "
      f"{t_load:.1f} s | footprint {ru()[0]:.3f} GB, lifetime {ru()[1]:.3f}", flush=True)
EOS = {248044, 248046}
ES = os.path.expanduser("~/EvalSuite/")
R = os.path.expanduser("~/Breakthrough/")
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
W16 = next(json.loads(l) for l in open(ES + "data/items/dec_agent.jsonl") if json.loads(l)["key"] == "agent/write16")
ITEMS = {it["key"]: it for it in map(json.loads, open(ES + "data/items/dec_math.jsonl"))}
REFM = [r for r in map(json.loads, open(ES + "results/b12llff2/decode.jsonl")) if r["set"] == "math"][:NMATH]
JOBS = [("p2k", chat_ids(HEAD + open(R + "scripts/tf_frugal_var.py").read()[:7600] + "\n```"), 256, False, None),
        ("p8k", chat_ids(HEAD + open(R + "data/prompts/ctx32k.txt").read()[:26000] + "\n```"), 256, False, None),
        ("write16", list(W16["prompt_ids"]), 256, True, None)]
JOBS += [(r["key"], list(ITEMS[r["key"]]["prompt_ids"]), ITEMS[r["key"]]["max_tokens"], True, r["completion_ids"]) for r in REFM]
with open(os.path.join(OUTD, f"{ARM}.jsonl"), "w") as f:
    pass
for name, ids, cap, stop, ref in JOBS:
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()
    tp = time.perf_counter()
    state = B.b12_prefill_state(model, ids, 512, 4096)
    mx.eval(state["logits"])
    tp = time.perf_counter() - tp
    td = time.perf_counter()
    out, s = spec125.generate(model, ids, cap, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state,
                              stats={}, **({"stop_ids": EOS} if stop else {}))
    td = time.perf_counter() - td
    out = [int(x) for x in out]
    if stop:
        for j, x in enumerate(out):
            if x in EOS:
                out = out[:j + 1]
                break
    row = dict(arm=ARM, prompt=name, n_prompt=len(ids), ntok=len(out), sha=sha(out), passes=int(s.get("passes", 0)),
               tok_per_pass=round(len(out) / max(1, s.get("passes", 1)), 4), prefill_s=round(tp, 2), decode_s=round(td, 2),
               ref_equal=(out == ref) if ref is not None else None, lifetime=round(ru()[1], 3), bt_b12=BTSHA,
               head=B.CFG.get("mtp_head", "stock"))
    with open(os.path.join(OUTD, f"{ARM}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[r60 {ARM}] {name} RESULT {len(out)} tok, sha {row['sha']}, {row['passes']} passes = {row['tok_per_pass']:.3f} "
          f"per pass | prefill {tp:.1f} s, decode {td:.1f} s ({len(out) / td:.2f} tok/s) | ref equal {row['ref_equal']} | "
          f"lifetime {row['lifetime']:.3f} GB", flush=True)
    del state
    mx.clear_cache()
print(f"[r60 {ARM}] done", flush=True)
