"""Breakthrough L2-11 (NewIdeas PORT GO 12:30): the A/B before bt_b12's B12_PFR becomes the default. One arm per process,
on the v3l engine, AR path, shipped env (B12LL, no lane): the first PA_N tokens of data/prompts/ctx64k.txt prefilled by
B.b12_prefill(model, ids[:-1], cache, 512, 4096) (B12_PFR=1: bt_b12 releases the trunk pools and picks the 512-aligned
chunks itself), then 32 greedy one-row decode steps. Exact = the same KV hash after the prefill, first-step logits hash
and 32-token sha as the B12_PFR=0 arm. Harness = Fit12's scripts/fit_pfr64.py (same KV hash, phase sampler and RESULT
line, so the hashes compare with Fit12's r8 / r64 on v3p), with the release / restore inside bt_b12 instead.
R-20 KV8M (Tue 10-27): PA_KV8=kv:64 runs b12_kv8's 8-bit KV; PA_KVHASH=0 skips the KV hash (clean lifetime peak).
Usage: [BigMoE venv] B12_ENGINE=v3l B12_PFR=0|1 PA_N=8474|65000 [PA_KV8=kv:64] [PA_KVHASH=0] pfr_ab.py TAG"""
import ctypes, hashlib, json, os, sys, threading, time

ENV = {"B12_CACHE_MB": "64", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_PF_GROUP": "64", "BIGMOE_MTP_C": "32",
       "BIGMOE_ASYNC_EVERY": "1", "B12_LATE3": "2,10", "B12_NORM": "mass"}          # pk64.py's shipped env
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "pfr_ab"
N, DEC = int(os.environ.get("PA_N", "65000")), int(os.environ.get("PA_DEC", "32"))


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def fp():                                            # (now, lifetime peak) physical footprint, GB
    ru = _RU()
    return (ru.u7 / 1e9, ru.u28 / 1e9) if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else (float("nan"),) * 2


import numpy as np
import mlx.core as mx
PH = {"name": "load"}
PK = {}                                              # phase -> [max footprint, max MLX active, samples]


def _sampler():
    while True:
        f, a = fp()[0], mx.get_active_memory() / 1e9
        d = PK.setdefault(PH["name"], [0.0, 0.0, 0])
        d[0], d[1], d[2] = max(d[0], f), max(d[1], a), d[2] + 1
        time.sleep(0.02)


threading.Thread(target=_sampler, daemon=True).start()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("BT_DIR"):                         # a bt_b12 under test, before it replaces the live one
    sys.path.insert(0, os.environ["BT_DIR"])
import bt_b12 as B
print(f"bt_b12 from {B.__file__}", flush=True)
rt, st = B.load_b12(0.10, mtp=False)
KV8 = os.environ.get("PA_KV8", "")                   # R-20 KV8M: "kv:64" = b12_kv8's 8-bit KV (kv8_long's arm); "" = bf16
if KV8:
    import b12_kv8 as K
    K.install()
    K.ON[0] = True
    K.CONF.update(part=KV8.split(":")[0], gs=int(KV8.split(":")[1]))
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
ids = tok.encode(open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read())[:N + 1]
assert len(ids) == N + 1, len(ids)
time.sleep(0.1)
PH["name"] = "idle"


def kv_hash(cache):
    """(sha of the whole cache, per-layer prefixes): every array of each layer's state, bf16 hashed as exact float32"""
    full, per = hashlib.sha1(), []
    for c in cache:
        h = hashlib.sha1()
        for a in (c.state if hasattr(c, "state") else []):
            if isinstance(a, mx.array):
                h.update(np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes())
        full.update(h.digest())
        per.append(h.hexdigest()[:6])
    return full.hexdigest()[:10], per


n = len(ids) - 1
n0, p0 = fp()
print(f"[{TAG}] engine {B.ENGINE}, AR path, N {n}, B12_PFR {os.environ.get('B12_PFR', '0')}, C_m {B.CFG['cm']} C_l "
      f"{B.CFG['cl']}, pools {st.resident_gb():.2f} GB | after load: active {mx.get_active_memory() / 1e9:.2f} GB, "
      f"footprint now {n0:.2f}, lifetime peak {p0:.2f} GB", flush=True)
reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
mx.clear_cache()
mx.reset_peak_memory()
PH["name"] = "prefill"
tw = t0 = time.perf_counter()
cache = model.make_cache()
B.b12_prefill(model, mx.array([ids[:-1]]), cache, 512, 4096)
mx.eval([a for c in cache for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
tp = time.perf_counter() - t0
pk = mx.get_peak_memory() / 1e9
print(f"[{TAG}] prefill {n} tokens in {tp:.1f} s = {n / tp:.0f} tok/s | PFR {B.PFR_LOG or '-'} (tokens, s_moe, GB, release s, "
      f"restore s) | MLX peak {pk:.2f} GB | after prefill footprint {fp()[0]:.2f} GB, MLX active "
      f"{mx.get_active_memory() / 1e9:.2f} GB", flush=True)
PH["name"] = "kvhash"
t0 = time.perf_counter()
if os.environ.get("PA_KVHASH", "1") == "1":         # R-20: off, so the lifetime peak has no float32 hash copies in it
    kvh, kvl = kv_hash(cache)
else:
    kvh, kvl = "-", []
th = time.perf_counter() - t0
print(f"[{TAG}] KV {KV8 or 'bf16'} after prefill: sha {kvh} ({th:.2f} s, not in the timings) | per layer {' '.join(kvl)}", flush=True)
PH["name"] = "decode"
y, out, lhash, TT = ids[-1], [], None, []
td = time.perf_counter()
for i in range(DEC):
    lg = model(mx.array([[y]]), cache=cache)
    lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
    if i == 0:
        lhash = hashlib.sha1(np.array(lg.astype(mx.float32)).tobytes()).hexdigest()[:10]
    y = int(mx.argmax(lg).item())
    out.append(y)
    TT.append(time.perf_counter())
st.gate.check()
f2 = fp()
sha8 = hashlib.sha1(json.dumps(out[:8]).encode()).hexdigest()[:10]
sha32 = hashlib.sha1(json.dumps(out).encode()).hexdigest()[:10]
print(f"[{TAG}] decode {DEC}: time to first token {TT[0] - tw - th:.2f} s (first step {TT[0] - td:.2f} s), steps 2-{DEC} "
      f"{(DEC - 1) / (TT[-1] - TT[0]):.2f} tok/s | after decode footprint {f2[0]:.2f} GB | process lifetime peak "
      f"{f2[1]:.2f} GB | sha {sha8} (first 8; logits {lhash}), {sha32} ({DEC})", flush=True)
print(f"[{TAG}] RESULT kv {kvh} logits {lhash} sha32 {sha32} sha8 {sha8}", flush=True)
PH["name"] = "end"
time.sleep(0.1)
for ph, (f, a, n_) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB, peak MLX active {a:.2f} GB ({n_} samples)", flush=True)
print(f"[{TAG}] done", flush=True)
