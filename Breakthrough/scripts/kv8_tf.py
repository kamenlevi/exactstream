"""Queue L2-06 screen (Breakthrough, 2026-10-25): 8-bit KV (b12_kv8) vs bf16 KV at long context, on the B12LL decode path
(B12_LATE3=2,10, mass: every top-10 expert computed, so residency cannot move the logits). Per context N: the decoupled
prefill of the first N tokens of data/prompts/ctx64k.txt, then the next M tokens teacher-forced in S-row chunks (verify
width). Arms run in one process (bf16 first, then 8-bit) on the same tokens; per position KL(bf16 || 8-bit) over the
full vocabulary, top-1 agreement, ms per chunk, QSA KV bytes and MLX active memory after the prefill.
usage: kv8_tf.py   (env KV8_N=16384,32768 KV8_M=256 KV8_S=4)   -> a line per N in data/kv8/results.txt"""
import os, sys, time

os.environ.setdefault("B12_LATE3", "2,10")
os.environ.setdefault("B12_NORM", "mass")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NS = [int(x) for x in os.environ.get("KV8_N", "16384,32768").split(",")]
M, S = int(os.environ.get("KV8_M", "256")), int(os.environ.get("KV8_S", "4"))
ARMS = os.environ.get("KV8_ARMS", "bf16,kv:64").split(",")      # arm 0 = the reference; "part:gs" = b12_kv8 CONF
import bt_b12 as B
rt, st = B.load_b12(0.10)
import numpy as np
import mlx.core as mx
import b12_kv8 as K
K.install()
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
ids_all = tok.encode(open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read())
print(f"kv8_tf: {len(ids_all)} tokens available; N {NS}, M {M}, S {S}, 8-bit group {K.GS}", flush=True)
assert max(NS) + M <= len(ids_all)


import ctypes


class _RU(ctypes.Structure):                              # rusage_info_v4: uuid, then 64-bit counters (as bt_b12)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def footprint():
    """(physical footprint now, lifetime max) in GB: Activity Monitor's Memory"""
    ru = _RU()
    if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) != 0:
        return float("nan"), float("nan")
    return ru.u7 / 1e9, ru.u28 / 1e9


def kv_bytes(cache):
    return sum(c.kv.nbytes for c in cache if hasattr(c, "kv"))


def run(N, arm):
    arm, _, sa = arm.partition("@")                     # "@256": the decoupled prefill's attention chunk (a benign
    sa = int(sa or 512)                                 # runtime perturbation: the noise floor of this metric)
    K.ON[0] = arm != "bf16"
    if K.ON[0]:
        part, gs = arm.split(":")
        K.CONF.update(part=part, gs=int(gs))
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()
    mx.clear_cache()
    cache = model.make_cache()
    t0 = time.perf_counter()
    B.b12_prefill(model, mx.array([ids_all[:N]]), cache, s_attn=sa)
    mx.eval([c.state for c in cache if hasattr(c, "state")])
    tp = time.perf_counter() - t0
    act, kvb = mx.get_active_memory() / 1e9, kv_bytes(cache) / 1e9
    print(f"kv8_tf {arm}@{sa} N={N}: after prefill active {act:.2f} GB, QSA KV {kvb:.3f} GB, MLX peak "
          f"{mx.get_peak_memory() / 1e9:.2f} GB, footprint now / lifetime max {footprint()[0]:.2f} / {footprint()[1]:.2f} GB",
          flush=True)
    lps, tc = [], []
    for i in range(0, M, S):
        t1 = time.perf_counter()
        lg = model(mx.array([ids_all[N + i:N + i + S]]), cache=cache)
        lg = lg[0] if isinstance(lg, tuple) else lg
        lp = lg[0].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        mx.eval(lp)
        tc.append(time.perf_counter() - t1)
        lps.append(np.array(lp))
    st.gate.check()
    print(f"kv8_tf {arm}@{sa} N={N}: after {M} teacher tokens active {mx.get_active_memory() / 1e9:.2f} GB, footprint now / "
          f"lifetime max {footprint()[0]:.2f} / {footprint()[1]:.2f} GB, ms/chunk {1e3 * float(np.median(tc[2:])):.0f}", flush=True)
    del cache
    return np.concatenate(lps), dict(prefill_s=tp, active_gb=act, kv_gb=kvb, ms_chunk=1e3 * float(np.median(tc[2:])))


os.makedirs(os.path.expanduser("~/Breakthrough/data/kv8"), exist_ok=True)
if len(ARMS) == 1:                                      # memory arm alone, in its own process
    for N in NS:
        run(N, ARMS[0])
    sys.exit(0)
for N in NS:
    a, ia = run(N, ARMS[0])
    p = np.exp(a)
    for arm in ARMS[1:]:
        b, ib = run(N, arm)
        kld = (p * (a - b)).sum(-1)
        agree = (a.argmax(-1) == b.argmax(-1)).mean()
        line = (f"N={N} M={len(kld)} S={S} {ARMS[0]} vs {arm}: KLD mean {kld.mean():.6f} p50 {np.median(kld):.6f} "
                f"p99 {np.quantile(kld, 0.99):.6f} max {kld.max():.6f} | top-1 agree {agree:.4f} | QSA KV {ia['kv_gb']:.3f} -> "
                f"{ib['kv_gb']:.3f} GB | active {ia['active_gb']:.2f} -> {ib['active_gb']:.2f} GB | ms/chunk {ia['ms_chunk']:.0f} "
                f"-> {ib['ms_chunk']:.0f} | prefill {ia['prefill_s']:.0f} -> {ib['prefill_s']:.0f} s")
        print(line, flush=True)
        with open(os.path.expanduser("~/Breakthrough/data/kv8/results.txt"), "a") as f:
            f.write(time.strftime("%m-%d %H:%M ") + line + "\n")
