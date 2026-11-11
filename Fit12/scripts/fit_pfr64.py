"""Fit12 PFR T2 (2026-10-25; pre-registered in PLAN.md 'PFR T2 pre-registration'): PFR at 64K on the AR path.
Mirrors Breakthrough's pk64.py (read-only reference): B12LL loaded without the MTP head, the first F12_N (65000) tokens of
~/Breakthrough/data/prompts/ctx64k.txt prefilled by B.b12_prefill(model, ids[:-1], cache, 512, s_moe) (F12_LANE=1: inside
mtplx attention_phase("prefill"), the L2-10 lane), then F12_DEC (32) greedy one-row decode steps. The sha of the first 8
tokens is pk64's token sha (plain cfa3402e3e, lane 56c4e038e8). F12_PFR=1: fit_pfr.py's release()/restore() around the
prefill, balanced MoE chunks <= F12_PFR_CAP (16384); restore() first frees the prefill's cached buffers (gc, clear_cache).
Prints prefill s, per-phase footprint peaks, the process lifetime peak, time to first token, decode tok/s, KV bytes.
A/B adds (NewIdeas 03:03): the KV hash at the end of the prefill, before any restore (sha1 over every layer's cache state,
bf16 as exact float32; per-layer prefixes too), so prefill math and decode state separate; F12_LOGITS=path.npy saves the
first-step logits; a last "RESULT kv .. logits .. sha32 .. sha8 .." line for the runner to compare.
Usage: [BigMoE venv] fit_pfr64.py TAG"""
import ctypes, gc, hashlib, json, math, os, sys, threading, time
from contextlib import nullcontext

ENV = {"B12_CACHE_MB": "64", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_PF_GROUP": "64", "BIGMOE_MTP_C": "32",
       "BIGMOE_ASYNC_EVERY": "1", "B12_LATE3": "2,10", "B12_NORM": "mass"}          # day_pk64b.sh + pk64.py
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "pfr64"
N, DEC = int(os.environ.get("F12_N", "65000")), int(os.environ.get("F12_DEC", "32"))
SMOE = int(os.environ.get("F12_SMOE", "4096"))
PFR, LANE = os.environ.get("F12_PFR") == "1", os.environ.get("F12_LANE") == "1"
CAP = int(os.environ.get("F12_PFR_CAP", "16384"))
BAL = os.environ.get("F12_PFR_BAL", "1") == "1"      # 0: chunks of exactly CAP (the shipped chunking when CAP = 4096)


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
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
import bt_b12 as B
if os.environ.get("F12_V3F"):
    B.V3F = os.path.expanduser(os.environ["F12_V3F"])
rt, st = B.load_b12(0.10, mtp=False)
from bigmoe_v3f import tier as T
from bigmoe_v3f.stream import PIECES
from bigmoe_v3f.gated import ext
from bigmoe_v3f.fuse import reset_generation_state
from mtplx.attention_context import attention_phase
model, tok = rt.model, rt.tokenizer
assert not PFR or hasattr(ext, "reset_layer"), f"F12_PFR=1 needs an engine with reset_layer (B.V3F = {B.V3F})"
ids = tok.encode(open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read())[:N + 1]
assert len(ids) == N + 1, len(ids)
time.sleep(0.1)
PH["name"] = "idle"


def release(st):
    """drop the trunk tier pools; returns what restore() needs"""
    info, nb = [], 0
    for sw in st._tier_sws:
        bm = sw._bm
        t = bm["tier"]
        L, Ls = bm["L"], t["side"]
        sb = [(tuple(p.shape), p.dtype) for p in bm["pools"]]
        ss = [(tuple(p.shape), p.dtype) for p in st.pools[Ls]]
        nb += sum(p.nbytes for p in bm["pools"]) + sum(p.nbytes for p in st.pools[Ls])
        ph = [mx.zeros((1, *s[1:]), d) for s, d in sb]
        for (proj, field), arr in zip(PIECES, ph):
            setattr(getattr(sw, proj), field, arr)
        bm["pools"] = ph
        st.pools[L] = ph
        st.pools[Ls] = [mx.zeros((1, *s[1:]), d) for s, d in ss]
        t["base"] = t["sideP"] = None
        info.append((sw, L, Ls, sb, ss))
    gc.collect()
    mx.clear_cache()
    return info, nb


def restore(st, info):
    gc.collect()
    mx.clear_cache()
    nb = 0
    for sw, L, Ls, sb, ss in info:
        bm = sw._bm
        t = bm["tier"]
        P = [mx.zeros(s, d) for s, d in sb]
        S = [mx.zeros(s, d) for s, d in ss]
        mx.eval(*P, *S)
        for (proj, field), arr in zip(PIECES, P):
            setattr(getattr(sw, proj), field, arr)
        bm["pools"] = P
        st.pools[L], st.pools[Ls] = P, S
        t["base"] = {p: {f: P[i] for i, (pp, f) in enumerate(T._MSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}
        t["sideP"] = {p: {f: S[i] for i, (pp, f) in enumerate(T._LSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}
        ext.reset_layer(L, P)
        ext.reset_layer(Ls, S)
        nb += sum(p.nbytes for p in P) + sum(p.nbytes for p in S)
    ext.invalidate_tables()
    return nb


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


_tier_call = T.tier_call


def _no_decode(*a, **kw):
    raise RuntimeError("PFR: a tier layer took the decode path while its pools were released")


n = len(ids) - 1
n0, p0 = fp()
print(f"[{TAG}] engine {B.V3F}, AR path (no MTP head), N {n}, lane {int(LANE)}, PFR {int(PFR)} (cap {CAP}, balanced {int(BAL)}), s_moe {SMOE} "
      f"when off, C_m {B.CFG['cm']} C_l {B.CFG['cl']}, pools {st.resident_gb():.2f} GB | after load: active "
      f"{mx.get_active_memory() / 1e9:.2f} GB, footprint now {n0:.2f}, lifetime peak {p0:.2f} GB", flush=True)
reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
mx.clear_cache()
sm = SMOE
tw = time.perf_counter()                             # wall clock: release + prefill + restore + decode
if PFR:
    k = math.ceil(n / CAP)
    sm = math.ceil(n / k) if BAL else CAP            # balanced chunks (none takes the decode path), or exactly CAP
    assert n - (k - 1) * sm > 8, (n, k, sm)
    PH["name"] = "release"
    t0 = time.perf_counter()
    info, nb = release(st)
    T.tier_call = _no_decode
    print(f"[{TAG}] released {nb / 1e9:.2f} GB of trunk pools in {time.perf_counter() - t0:.2f} s; footprint now "
          f"{fp()[0]:.2f} GB, MLX active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
mx.reset_peak_memory()
PH["name"] = "prefill"
t0 = time.perf_counter()
cache = model.make_cache()
with (attention_phase("prefill") if LANE else nullcontext()):
    B.b12_prefill(model, mx.array([ids[:-1]]), cache, 512, sm)
    mx.eval([a for c in cache for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
tp = time.perf_counter() - t0
pk = mx.get_peak_memory() / 1e9
f1 = fp()[0]
print(f"[{TAG}] prefill {n} tokens in {tp:.1f} s = {n / tp:.0f} tok/s (s_moe {sm}) | MLX peak {pk:.2f} GB | after prefill "
      f"footprint {f1:.2f} GB, MLX active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
PH["name"] = "kvhash"
t0 = time.perf_counter()
kvh, kvl = kv_hash(cache) if os.environ.get("F12_KVHASH", "1") == "1" else ("-", [])   # 0: T2's memory conditions
th = time.perf_counter() - t0
print(f"[{TAG}] KV after prefill (before any restore): sha {kvh} ({th:.2f} s, not in the timings) | per layer {' '.join(kvl)}",
      flush=True)
if PFR:
    T.tier_call = _tier_call
    PH["name"] = "restore"
    t0 = time.perf_counter()
    nb2 = restore(st, info)
    del info
    print(f"[{TAG}] re-created {nb2 / 1e9:.2f} GB of pools in {time.perf_counter() - t0:.2f} s (released {nb / 1e9:.2f}); "
          f"footprint now {fp()[0]:.2f} GB, MLX active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
    assert nb2 == nb, (nb2, nb)
PH["name"] = "decode"
y, out, lhash, TT = ids[-1], [], None, []
td = time.perf_counter()
for i in range(DEC):
    lg = model(mx.array([[y]]), cache=cache)
    lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
    if i == 0:
        l32 = np.array(lg.astype(mx.float32))
        lhash = hashlib.sha1(l32.tobytes()).hexdigest()[:10]
        if os.environ.get("F12_LOGITS"):
            np.save(os.environ["F12_LOGITS"], l32)
    y = int(mx.argmax(lg).item())
    out.append(y)
    TT.append(time.perf_counter())
st.gate.check()
f2 = fp()
kvb = sum(c.kv.nbytes for c in cache if hasattr(c, "kv")) / 1e9
sha8 = hashlib.sha1(json.dumps(out[:8]).encode()).hexdigest()[:10]
sha32 = hashlib.sha1(json.dumps(out).encode()).hexdigest()[:10]
print(f"[{TAG}] decode {DEC}: time to first token {TT[0] - tw - th:.2f} s (first step {TT[0] - td:.2f} s), steps 2-{DEC} "
      f"{(DEC - 1) / (TT[-1] - TT[0]):.2f} tok/s | after decode footprint {f2[0]:.2f} GB | "
      f"process lifetime peak {f2[1]:.2f} GB | KV {kvb:.2f} GB | sha {sha8} (first 8; logits {lhash}), {sha32} ({DEC})",
      flush=True)
print(f"[{TAG}] RESULT kv {kvh} logits {lhash} sha32 {sha32} sha8 {sha8}", flush=True)
PH["name"] = "end"
time.sleep(0.1)
for ph, (f, a, n_) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB, peak MLX active {a:.2f} GB ({n_} samples)", flush=True)
print(f"[{TAG}] done", flush=True)
