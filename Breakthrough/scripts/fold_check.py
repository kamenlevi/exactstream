"""Breakthrough FOLD check (NewIdeas Tue 10-27: fold Fit12's exact MTP-path stack into bt_b12 behind B12_FOLD=1, run it
after Fit12's R-01 passes, then an Eval-suite verdict). The harness is Fit12's scripts/fit_cmp.py reduced to the
shipped API: per prompt, B.b12_prefill_state + spec125.generate (3:3:0, greedy) exactly as the Eval suite's
decode._gen_mtp calls them, with every piece coming from bt_b12 itself (B12_FOLD: PFR on the MTP prefill, C_m per
prompt, the QSA row target, MOV, ALLWAIT inside generate). Same prompts (p2k, p8k = fit_cmp's), hashes (first-step
logits from the state, before and after the decode; sha32 / sha<N> of the tokens; decode-end trunk cache and MTP head
cache, as fit_cmp's kv_hash) and warm rate (tokens 33..N) as fit_cmp, so the RESULT lines compare with Fit12's.
Expected with ALLWAIT (bits independent of C_m): Fit12's PFM pf1 / AW a51 hashes (p2k sha256 a0774ec635, kv 37028a6545,
head 8563204e11; p8k logits 7d045d7b7d, sha256 ed67050746, kv 5a760388d6, head f59755cf27).
Usage: [BigMoE venv] B12_ENGINE=v3m [B12_FOLD=1] [FC_PROMPTS=p2k,p8k] [FC_NDEC=256] fold_check.py TAG"""
import ctypes, hashlib, os, sys, threading, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1"}
for k, v in ENV.items():                             # fit_cmp's env (minus its B12_PFR=0: here bt_b12's own PFR runs)
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "fold"
NDEC = int(os.environ.get("FC_NDEC", "256"))


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def fp():                                            # (now, lifetime peak) physical footprint, GB
    ru = _RU()
    return (ru.u7 / 1e9, ru.u28 / 1e9) if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else (float("nan"),) * 2


import numpy as np
import mlx.core as mx
PH = {"name": "load"}
PK = {}


def _sampler():
    while True:
        f = fp()[0]
        d = PK.setdefault(PH["name"], [0.0, 0])
        d[0], d[1] = max(d[0], f), d[1] + 1
        time.sleep(0.02)


threading.Thread(target=_sampler, daemon=True).start()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
if os.environ.get("BT_DIR"):                         # a bt_b12 under test, before it replaces the live one
    sys.path.insert(0, os.environ["BT_DIR"])
import bt_b12 as B
print(f"bt_b12 from {B.__file__}", flush=True)
rt, st = B.load_b12(0.10, mtp=True)
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
CAPT = []


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
R = os.path.expanduser("~/Breakthrough/")
PROMPTS = {"p2k": lambda: HEAD + open(R + "scripts/tf_frugal_var.py").read()[:7600] + "\n```",     # = fit_cmp's
           "p8k": lambda: HEAD + open(R + "data/prompts/ctx32k.txt").read()[:26000] + "\n```",
           "p16k": lambda: HEAD + open(R + "data/prompts/ctx32k.txt").read()[:49000] + "\n```",
           "p32k": lambda: HEAD + open(R + "data/prompts/ctx32k.txt").read()[:98000] + "\n```"}


def kv_hash(cache):
    full = hashlib.sha1()
    for c in (cache if isinstance(cache, (list, tuple)) else [cache]):
        h = hashlib.sha1()
        for a in (c.state if hasattr(c, "state") else []):
            if isinstance(a, mx.array):
                h.update(np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes())
        full.update(h.digest())
    return full.hexdigest()[:10]


sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
lgh = lambda a: hashlib.sha1(np.array(a.astype(mx.float32)).tobytes()).hexdigest()[:10]
print(f"[{TAG}] engine {B.V3F}, B12_FOLD {int(getattr(B, 'FOLD', [False])[0])}, B12_PFR {os.environ.get('B12_PFR', '1')}, load C_m {B.CFG['cm']} "
      f"C_l {B.CFG['cl']} | footprint {fp()[0]:.2f} GB", flush=True)
for pos, P in enumerate(os.environ.get("FC_PROMPTS", "p2k,p8k").split(","), 1):
    ids = chat_ids(PROMPTS[P]())
    n = len(ids) - 1
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    PH["name"] = f"{pos}:{P} prefill"
    t0 = time.perf_counter()
    state = B.b12_prefill_state(model, ids, 512, 4096)
    mx.eval(state["logits"])
    tp = time.perf_counter() - t0
    lpre = lgh(state["logits"])
    if not CAPT:                                     # capture the decode's own cache + head (after the fold's wrapper)
        _rs = spec125.restore_state

        def _cap(model_, state_, with_head=True):
            r = _rs(model_, state_, with_head=with_head)
            CAPT.append(r)
            return r
        spec125.restore_state = _cap
        CAPT.append(None)
    PH["name"] = f"{pos}:{P} decode"
    TT, aw0 = [], B.AW_N[0]
    tg = time.perf_counter()
    out, s = spec125.generate(model, ids, NDEC, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state,
                              stats={}, on_token=lambda t: TT.append(time.perf_counter()))
    st.gate.check()
    lpk = fp()[1]
    PH["name"] = f"{pos}:{P} hash"
    dc, hd = CAPT[-1]
    kh, hh = kv_hash(dc), kv_hash(hd.cache) if hd is not None else "none"
    warm = (len(TT) - 32) / (TT[-1] - TT[31]) if len(TT) > 33 else float("nan")
    print(f"[{TAG}] {P} RESULT logits {lgh(state['logits'])} sha32 {sha(out[:32])} sha{len(out)} {sha(out)} kv {kh} head {hh} "
          f"| n {n} C_m {B.CFG['cm']} | warm tok/s {warm:.2f} (tokens 33-{len(TT)}) | lifetime peak {lpk:.2f} GB | "
          f"prefill {tp:.1f} s, decode peak {PK[f'{pos}:{P} decode'][0]:.2f} GB | PFR {B.PFR_LOG[-1] if B.PFR_LOG else '-'} | "
          f"state keys after {sorted(k for k in state if not k.startswith('_'))} | ALLWAIT calls {B.AW_N[0] - aw0} | "
          f"lpre {lpre}", flush=True)
    CAPT[-1:] = [None]
    del state, dc, hd
    mx.clear_cache()
    PH["name"] = "idle"
for ph, (f, n_) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB ({n_} samples)", flush=True)
print(f"[{TAG}] done", flush=True)
