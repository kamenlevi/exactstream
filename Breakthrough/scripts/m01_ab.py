"""Breakthrough R-48 M01-12 (NewIdeas Thu 17:2x): one arm of the M-01 A/B on the 12 GB line, one process per arm.
A = B12_ENGINE=v3m (records3), S = B12_ENGINE=v3n BIGMOE_M01=1 (BigMoE's records3m, the per-group fields re-packed; the
ext unpacks them). The live bt_b12 at its defaults (fold-exact), fold_check's env; in order p2k and p8k (fold_check's
method: b12_prefill_state + spec125.generate 3:3:0, greedy, M01_NDEC tokens) then the Eval suite's agent/write16
(w16_bisect's method: its prompt_ids, stops at {248044, 248046}). Per prompt: hashes as fold_check (first-step logits
from the state, sha32 / sha<N> of the tokens, decode-end trunk cache and MTP head cache), rusage disk bytes read
(ri_diskio_bytesread) in the prefill and in the decode, decode tok/s (all tokens, and tokens 33+ as fold_check's warm
rate), lifetime peak footprint. Writes one JSON line per prompt to $M01_OUT/<TAG>.jsonl.
Expected at 256 tokens on both arms (31b63af's fold-exact regression): p2k 98c6bc1c0d 82995cdb4e a0774ec635 37028a6545
8563204e11, p8k 61b2520e58 0b0d91cd43 ed67050746 5a760388d6 f59755cf27; write16 = the suite's b12llfoff row.
Usage: [BigMoE venv] B12_ENGINE=v3m|v3n [BIGMOE_M01=1] [M01_NDEC=256] [M01_PROMPTS=p2k,p8k,write16] m01_ab.py TAG"""
import ctypes, hashlib, json, os, sys, threading, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1"}
for k, v in ENV.items():                             # = fold_check's env (w16_bisect's adds only bt_b12's own defaults)
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "m01"
NDEC = int(os.environ.get("M01_NDEC", "256"))
OUTD = os.path.expanduser(os.environ.get("M01_OUT", "~/Breakthrough/data/m01"))
ENGINE, M01 = os.environ.get("B12_ENGINE", "v3m"), bool(os.environ.get("BIGMOE_M01"))
assert M01 == (ENGINE == "v3n"), f"arm mix-up: B12_ENGINE={ENGINE} BIGMOE_M01={os.environ.get('BIGMOE_M01')}"
os.makedirs(OUTD, exist_ok=True)


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def ru():                                            # (footprint GB, lifetime peak GB, disk bytes read) of this process
    r = _RU()
    assert _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) == 0
    return r.u7 / 1e9, r.u28 / 1e9, r.u16


import numpy as np
import mlx.core as mx
PH = {"name": "load"}
PK = {}


def _sampler():
    while True:
        f = ru()[0]
        d = PK.setdefault(PH["name"], [0.0, 0])
        d[0], d[1] = max(d[0], f), d[1] + 1
        time.sleep(0.02)


threading.Thread(target=_sampler, daemon=True).start()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
assert not os.environ.get("BT_DIR"), "R-48 runs on the live bt_b12"
BTSHA = hashlib.sha256(open(os.path.join(HERE, "bt_b12.py"), "rb").read()).hexdigest()[:10]
import bt_b12 as B
d_load0 = ru()[2]
t_load = time.perf_counter()
rt, st = B.load_b12(0.10, mtp=True)
t_load = time.perf_counter() - t_load
import spec125
import bigmoe_ext as EXT
from bigmoe_v3f.fuse import reset_generation_state
m01_on = getattr(EXT, "m01_on", lambda: False)()
assert m01_on == M01 and EXT.__file__.startswith(B.V3F), (EXT.__file__, m01_on)
model, tok = rt.model, rt.tokenizer
CAPT = []
EOS = {248044, 248046}


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
R = os.path.expanduser("~/Breakthrough/")
ES = os.path.expanduser("~/EvalSuite/")
W16 = next(json.loads(l) for l in open(ES + "data/items/dec_agent.jsonl") if json.loads(l)["key"] == "agent/write16")
W16REF = next(json.loads(l)["completion_ids"] for l in open(ES + "results/b12llfoff/decode.jsonl")
              if json.loads(l)["key"] == "agent/write16")
PROMPTS = {"p2k": lambda: chat_ids(HEAD + open(R + "scripts/tf_frugal_var.py").read()[:7600] + "\n```"),   # = fold_check's
           "p8k": lambda: chat_ids(HEAD + open(R + "data/prompts/ctx32k.txt").read()[:26000] + "\n```"),
           "write16": lambda: list(W16["prompt_ids"])}


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
_TM = model.language_model.model if hasattr(model, "language_model") else model.model
tier_dirs = sorted({os.path.basename(os.path.dirname(l.mlp.switch_mlp._bm["tier"]["path"])) for l in _TM.layers
                    if "tier" in getattr(getattr(getattr(l, "mlp", None), "switch_mlp", None), "_bm", {})})
assert tier_dirs == (["records3m"] if M01 else ["records3"]), tier_dirs
print(f"[{TAG}] engine {B.V3F} ext {EXT.__file__} m01_on {m01_on} | bt_b12 {BTSHA} | records {tier_dirs} | B12_FOLD "
      f"{int(getattr(B, 'FOLD', [False])[0])} | load C_m {B.CFG['cm']} C_l {B.CFG['cl']} | load {t_load:.1f} s, disk "
      f"{(ru()[2] - d_load0) / 1e6:.0f} MB | footprint {ru()[0]:.2f} GB", flush=True)
rows = []
for pos, P in enumerate(os.environ.get("M01_PROMPTS", "p2k,p8k,write16").split(","), 1):
    ids = PROMPTS[P]()
    n = len(ids) - 1
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    PH["name"] = f"{pos}:{P} prefill"
    d0, t0 = ru()[2], time.perf_counter()
    state = B.b12_prefill_state(model, ids, 512, 4096)
    mx.eval(state["logits"])
    tp, d1 = time.perf_counter() - t0, ru()[2]
    lpre = lgh(state["logits"])
    cm_pre = B.CFG["cm"]
    if not CAPT:                                     # capture the decode's own cache + head (after the fold's wrapper)
        _rs = spec125.restore_state

        def _cap(model_, state_, with_head=True):
            r = _rs(model_, state_, with_head=with_head)
            CAPT.append(r)
            return r
        spec125.restore_state = _cap
        CAPT.append(None)
    PH["name"] = f"{pos}:{P} decode"
    TT = []
    kw = {"stop_ids": EOS} if P == "write16" else {}
    tg = time.perf_counter()
    out, s = spec125.generate(model, ids, NDEC, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state,
                              stats={}, on_token=lambda t: TT.append(time.perf_counter()), **kw)
    st.gate.check()
    td, d2 = time.perf_counter() - tg, ru()[2]
    lpk = ru()[1]
    PH["name"] = f"{pos}:{P} hash"
    dc, hd = CAPT[-1]
    kh, hh = kv_hash(dc), kv_hash(hd.cache) if hd is not None else "none"
    out = [int(x) for x in out]
    warm = (len(TT) - 32) / (TT[-1] - TT[31]) if len(TT) > 33 else float("nan")
    ref = None
    if P == "write16":
        ref = next((i for i, (a, b) in enumerate(zip(out, W16REF)) if a != b), None)
    row = dict(tag=TAG, prompt=P, n=n, engine=os.path.basename(B.V3F), m01=m01_on, bt_b12=BTSHA, records=tier_dirs,
               logits=lgh(state["logits"]), lpre=lpre, sha32=sha(out[:32]), shaN=sha(out), ntok=len(out), kv=kh, head=hh,
               cm=B.CFG["cm"], cm_pre=cm_pre, cl=B.CFG["cl"], prefill_s=round(tp, 2), decode_s=round(td, 3),
               tps=round(len(out) / td, 3), warm=round(warm, 3), disk_prefill=d1 - d0, disk_decode=d2 - d1,
               dec_bpt=(d2 - d1) / max(1, len(out)), lifetime_peak=round(lpk, 3),
               decode_peak=round(PK[f"{pos}:{P} decode"][0], 3), w16_first_diff_vs_b12llfoff=ref,
               w16_ref_len=len(W16REF) if P == "write16" else None)
    rows.append(row)
    with open(os.path.join(OUTD, f"{TAG}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{TAG}] {P} RESULT logits {row['logits']} sha32 {row['sha32']} sha{len(out)} {row['shaN']} kv {kh} head {hh} | "
          f"n {n} C_m {row['cm']} C_l {row['cl']} | decode {len(out)} tok {td:.2f} s = {row['tps']:.2f} tok/s, warm "
          f"{warm:.2f} | disk prefill {row['disk_prefill'] / 1e6:.0f} MB, decode {row['disk_decode'] / 1e6:.0f} MB = "
          f"{row['dec_bpt'] / 1e6:.2f} MB/tok | lifetime peak {lpk:.3f} GB | prefill {tp:.1f} s"
          + (f" | vs b12llfoff first diff {ref} (ref {len(W16REF)} tok)" if P == "write16" else ""), flush=True)
    CAPT[-1:] = [None]
    del state, dc, hd
    mx.clear_cache()
    PH["name"] = "idle"
for ph, (f, n_) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB ({n_} samples)", flush=True)
dd = sum(r["disk_decode"] for r in rows); nt = sum(r["ntok"] for r in rows)
print(f"[{TAG}] TOTAL decode {dd / 1e6:.0f} MB / {nt} tok = {dd / nt / 1e6:.3f} MB/tok; prefill "
      f"{sum(r['disk_prefill'] for r in rows) / 1e6:.0f} MB; process disk {ru()[2] / 1e6:.0f} MB; lifetime peak "
      f"{ru()[1]:.3f} GB", flush=True)
print(f"[{TAG}] done", flush=True)
