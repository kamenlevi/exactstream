"""Fit12 R-53 SCRATCH12 A/B (Fri 30 Oct 2026; PLAN.md 'R-53 A/B pre-registration', written before any run): does reusing the
prefill scratch (/Users/kamen/Fit12/v3n53: persistent group buffers, group outputs scattered in place, the MTP head's takes
through MLX's cache) speed the live 12 GB prefill, bit-exact? One process per arm: F12_V3F = the engine tree (A =
~/Breakthrough/v3n, B = /Users/kamen/Fit12/v3n53), bt_b12 23b42972ef pinned (F12_BT, read-only, its own defaults: v3n +
records3m, fold-exact), the suite's b12llfx env, MTP on. Prompts F12_PROMPTS (p2k = PFR's p2k from a frozen copy of its
source, memory only: <= 4,096 tokens keeps the pools resident, as the suite's agent rows; p8k = PFR's p8k, p16k = the first
16,385 tokens of ctx64k.txt): each prefilled by B.b12_prefill_state (timed), hashed (every layer's KV, the MTP cache, h and the
draft logits, as uint16 / uint32 views in <= 4M-element slices so the hash adds ~no footprint), then 32 spec125 3:3:0
tokens. Footprint peaks per phase from a 20 ms sampler. Usage: [BigMoE venv] fit_scr53.py TAG"""
import ctypes, hashlib, os, sys, threading, time

POP = ("EVALSUITE_FRUGAL", "EVALSUITE_SHIP", "EVALSUITE_FAST24", "B12_LATE3", "B12_W", "B12_NORM", "BIGMOE_LOOP_GUARD", "B12_CM",
       "B12_CL", "B12_ENGINE", "B12_M01", "BIGMOE_M01", "B12_TRIMS", "B12_PFR", "B12_ALLWAIT", "B12_CACHE_MB", "B12_FOLD",
       "B12_FOLD_CM", "B12_FOLD_ROWS", "B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV", "BIGMOE_PF_GROUP",
       "BIGMOE_ASYNC_EVERY", "BIGMOE_PF_PREFILL", "BIGMOE_PF_FULLSCR", "MTPLX_NGRAM_HOT_MB")
ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "EVALSUITE_SHIP": "1",  # SHIP_AR
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1",              # B12_ENV b12ll
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64",                               # FOLD_SHIP
       "B12_FOLD": "1", "B12_PFR_GRID": "keep",                                     # b12llfx; engine = bt_b12's default (v3n)
       "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80", "BIGMOE_FRSPEC": "1",                # SHIP_MTP
       "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0",
       "MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0",                # env.py
       "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"}
if os.environ.get("F12_LIVE") == "1":            # R-53D (Sun 1 Nov): the live file's own mode defaults (fast: fold pieces,
    for k in ("B12_FOLD", "B12_PFR_GRID"):       # grid balanced, PFR_MIN 512), as fit_creep's live arm
        ENV.pop(k, None)
for k in POP:
    os.environ.pop(k, None)
os.environ.update(ENV)
TAG = sys.argv[1] if len(sys.argv) > 1 else "r53"
BT = os.environ.get("F12_BT", os.path.expanduser("~/Fit12/pinned/b23/bt_b12.py"))
V3F = os.path.expanduser(os.environ["F12_V3F"])


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def fp():                                            # (now, lifetime peak) physical footprint, GB
    ru = _RU()
    return (ru.u7 / 1e9, ru.u28 / 1e9) if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else (float("nan"),) * 2


PH = {"name": "load"}
PK = {}                                              # phase -> [max footprint, max MLX active, samples]
PZ = set()                                           # R-53C: phases that saw the quiet PAUSE file (heat pauses): void
PAUSEF = os.path.expanduser("~/.local/share/quiet/PAUSE")


def _sampler():
    while True:
        f, a = fp()[0], mx.get_active_memory() / 1e9
        d = PK.setdefault(PH["name"], [0.0, 0.0, 0])
        d[0], d[1], d[2] = max(d[0], f), max(d[1], a), d[2] + 1
        if os.path.exists(PAUSEF):
            PZ.add(PH["name"])
        time.sleep(0.02)


sha = hashlib.sha256(open(BT, "rb").read()).hexdigest()[:10]
assert sha in ("23b42972ef", "677c6558b6", "8db71cc537"), f"bt_b12 at {BT} is {sha}, not 23b42972ef (R-53), 677c6558b6 (R-53C on v3o) or 8db71cc537 (R-53D, live)"
import numpy as np
import mlx.core as mx
threading.Thread(target=_sampler, daemon=True).start()
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, os.path.dirname(BT))
import bt_b12 as B
assert os.path.realpath(B.__file__) == os.path.realpath(BT), B.__file__
B.V3F = V3F
t0 = time.perf_counter()
rt, st = B.load_b12(0.10, mtp=True)
tl = time.perf_counter() - t0
import spec125
import bigmoe_v3f
from bigmoe_v3f import tier as T
from bigmoe_v3f.fuse import reset_generation_state
assert os.path.realpath(bigmoe_v3f.__file__).startswith(os.path.realpath(V3F) + "/"), bigmoe_v3f.__file__
model, tok = rt.model, rt.tokenizer
R53 = getattr(T, "R53", None)
n0, p0 = fp()
print(f"[{TAG}] bt_b12 {sha}, engine {os.path.dirname(os.path.dirname(bigmoe_v3f.__file__))} (R-53 patch "
      f"{'on: a ' + str(int(T._R53A)) + ' b ' + str(int(T._R53B)) if R53 is not None else 'absent'}), BIGMOE_M01 "
      f"{os.environ.get('BIGMOE_M01')}, loaded in {tl:.0f} s: footprint {n0:.2f} GB, lifetime peak {p0:.2f} GB, MLX active "
      f"{mx.get_active_memory() / 1e9:.2f} GB", flush=True)


def _upd(h, a):                                      # exact bits, in <= 4M-element slices along the longest axis
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    elif a.dtype == mx.float32:
        a = a.view(mx.uint32)
    if a.ndim == 0 or a.size <= (1 << 22):
        h.update(np.array(a).tobytes()); return
    ax = max(range(a.ndim), key=lambda i: a.shape[i])
    step = max(1, ((1 << 22) * a.shape[ax]) // a.size)
    for s in range(0, a.shape[ax], step):
        sl = [slice(None)] * a.ndim
        sl[ax] = slice(s, s + step)
        h.update(np.array(a[tuple(sl)]).tobytes())


def _arrs(c):
    s = c.state if hasattr(c, "state") else []
    return [a for a in (s if isinstance(s, (list, tuple)) else [s]) if isinstance(a, mx.array)]


def st_hash(state):                                  # (KV of every layer + the MTP cache, h, draft logits)
    kv = hashlib.sha1()
    m = state.get("mtp_cache")
    m = list(m) if isinstance(m, (list, tuple)) else ([] if m is None else [m])
    for c in list(state["cache"]) + m:
        for a in _arrs(c):
            _upd(kv, a)
    hs = []
    for key in ("h", "logits"):
        h = hashlib.sha1()
        _upd(h, state[key])
        hs.append(h.hexdigest()[:10])
    return kv.hexdigest()[:10], hs[0], hs[1]


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
PROMPTS = {"p2k": lambda: chat_ids(HEAD + open(os.path.expanduser("~/Fit12/data/prompts/p2k_src.txt")).read() + "\n```"),
           "p8k": lambda: chat_ids(HEAD + open(os.path.expanduser("~/Breakthrough/data/prompts/ctx32k.txt")).read()[:26000] + "\n```"),
           "p16k": lambda: tok.encode(open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read())[:16385]}
import json                                         # SG12 (Sun 1 Nov): M-5d's binding math rows, the suite's own prompt ids
_MI = {it["key"]: it for it in map(json.loads, open(os.path.expanduser("~/EvalSuite/data/items/dec_math.jsonl")))}
for _p, _key in {"m10": "number_theory/459", "m13": "prealgebra/1761", "m3": "algebra/694", "m15": "algebra/509"}.items():
    PROMPTS[_p] = lambda key=f"math500/test/{_key}.json": list(_MI[key]["prompt_ids"])
_HI = {it["key"]: it for it in map(json.loads, open(os.path.expanduser("~/EvalSuite/data/items/dec_humaneval.jsonl")))}
for _n in ("156", "45", "42", "8", "68"):              # SGB (Mon 2 Nov): HT1's HumanEval+ rows, the suite's own prompt ids
    PROMPTS[f"h{_n}"] = lambda key=f"humaneval/HumanEval/{_n}": list(_HI[key]["prompt_ids"])
for P in os.environ.get("F12_PROMPTS", "p2k,p8k,p16k").split(","):
    ids = list(PROMPTS[P]())
    n = len(ids) - 1
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    c0 = {k: R53[k] for k in ("alloc", "reuse", "drop", "scat", "mtp")} if R53 is not None else {}
    mx.reset_peak_memory()
    PH["name"] = f"{P} prefill"
    t0 = time.perf_counter()
    state = B.b12_prefill_state(model, ids, 512, 4096)
    mx.eval(state["logits"], state["h"])
    tp = time.perf_counter() - t0
    pk = mx.get_peak_memory() / 1e9
    PH["name"] = f"{P} hash"
    th = time.perf_counter()
    kvh, hh, lh = st_hash(state)
    th = time.perf_counter() - th
    PH["name"] = f"{P} decode"
    tg = time.perf_counter()
    out, s = spec125.generate(model, ids, 32, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
    st.gate.check()
    tg = time.perf_counter() - tg
    out = [int(x) for x in out]
    fd = fp()[0]
    cnt = {k: R53[k] - c0[k] for k in ("alloc", "reuse", "drop", "scat", "mtp")} if R53 is not None else {}
    print(f"[{TAG}] {P}: prefill {n} tokens in {tp:.2f} s = {n / tp:.0f} tok/s | MLX peak {pk:.2f} GB | hash {th:.1f} s | "
          f"decode 32 in {tg:.2f} s | after decode: footprint {fd:.2f} GB, MLX active {mx.get_active_memory() / 1e9:.2f} GB, "
          f"cache {mx.get_cache_memory() / 1e9:.3f} GB | R-53 counts {cnt} | PF grouped {T.PF['grouped']} single "
          f"{T.PF['single']} fallback {T.PF['fallback']} | SG12 {getattr(T, 'SG12', 'absent')} | SGB {getattr(T, 'SGB', 'absent')}", flush=True)
    print(f"[{TAG}] {P} RESULT kv {kvh} h {hh} logits {lh} sha32 {hashlib.sha1(str(out).encode()).hexdigest()[:10]} "
          f"prefill {tp:.2f} s after {fd:.2f} GB paused {int(P + ' prefill' in PZ)}", flush=True)
    del state
    mx.clear_cache()
    PH["name"] = "idle"
    time.sleep(0.1)
time.sleep(0.1)
for ph, (f, a, n_) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB, peak MLX active {a:.2f} GB ({n_} samples)", flush=True)
srv = max([v[0] for k, v in PK.items() if k.endswith(("prefill", "decode"))] or [float("nan")])
print(f"[{TAG}] RESULT serving peak {srv:.2f} GB lifetime peak {fp()[1]:.2f} GB", flush=True)
print(f"[{TAG}] done", flush=True)
