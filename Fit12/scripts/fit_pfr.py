"""Fit12 PFR T1 (2026-10-25; pre-registered in PLAN.md 'PFR pre-registration'): Prefill with the decode pools Released.
A copy of fit_pf.py (one mode per process, honest footprints, per-phase peak sampler, 32 spec125 3:3:0 decode tokens and
their sha per prompt), on an engine tree with the PFR calls (F12_V3F, e.g. /Users/kamen/Fit12/v3p: ext.reset_layer,
ext.invalidate_tables). bt_b12 is imported read-only (its folded trims and env as shipped; no fit_trim on top).
F12_PFR=1: before each prompt's prefill the 48 trunk tier pools (MSB-3 base + LSB-1 sidecar; the MTP head keeps its pool,
which its stream-path prefill gathers from) are dropped: module fields -> 1-slot placeholders, every Python reference
cleared, mx.clear_cache(); tier.tier_call (the only decode path of a tier layer) raises while they are gone, so no gate
pass can write through the stale pointers. The prefill runs with balanced MoE chunks of <= F12_PFR_CAP tokens (one chunk
up to the cap), then the pools are re-created (mx.zeros, same shapes), ext.reset_layer swaps each layer's pointers and
empties its slots, and ext.invalidate_tables() drops every published residency table. Decode then starts cold.
F12_PFR=0: the shipped path (s_moe F12_SMOE, default 4096). Slots: B12_CM / B12_CL. Prompts: F12_PROMPTS (p2k, p8k).
Usage: [BigMoE venv] fit_pfr.py TAG"""
import ctypes, gc, hashlib, math, os, sys, threading, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "pfr"
SMOE = int(os.environ.get("F12_SMOE", "4096"))
PFR = os.environ.get("F12_PFR") == "1"
CAP = int(os.environ.get("F12_PFR_CAP", "16384"))


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def fp():                                            # (now, lifetime peak) physical footprint, GB
    ru = _RU()
    return (ru.u7 / 1e9, ru.u28 / 1e9) if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else (float("nan"),) * 2


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
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import bt_b12 as B
if os.environ.get("F12_V3F"):
    B.V3F = os.path.expanduser(os.environ["F12_V3F"])
rt, st = B.load_b12(float(os.environ.get("F12_FRAC", "0.10")), mtp=True)
import spec125
from bigmoe_v3f import tier as T
from bigmoe_v3f.stream import PIECES
from bigmoe_v3f.gated import ext
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
assert not PFR or hasattr(ext, "reset_layer"), f"F12_PFR=1 needs an engine with reset_layer (B.V3F = {B.V3F})"
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


_tier_call = T.tier_call


def _no_decode(*a, **kw):
    raise RuntimeError("PFR: a tier layer took the decode path while its pools were released")


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
PROMPTS = {"p2k": lambda: HEAD + open(os.path.expanduser("~/Breakthrough/scripts/tf_frugal_var.py")).read()[:7600] + "\n```",
           "p8k": lambda: HEAD + open(os.path.expanduser("~/Breakthrough/data/prompts/ctx32k.txt")).read()[:26000] + "\n```"}
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
n0, p0 = fp()
print(f"[{TAG}] engine {B.V3F}, PFR {int(PFR)} (cap {CAP}), s_moe {SMOE} when off, C_m {B.CFG['cm']} C_l {B.CFG['cl']}, "
      f"pools {st.resident_gb():.2f} GB | after load: active {mx.get_active_memory() / 1e9:.2f} GB, footprint now {n0:.2f}, "
      f"lifetime peak {p0:.2f} GB", flush=True)
for P in os.environ.get("F12_PROMPTS", "p2k,p8k").split(","):
    ids = chat_ids(PROMPTS[P]())
    n = len(ids) - 1
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    sm = SMOE
    tw = time.perf_counter()                                     # this prompt's wall clock (release + prefill + restore + decode)
    if PFR:
        k = math.ceil(n / CAP)
        sm = math.ceil(n / k)                                    # balanced chunks: none takes the decode path
        assert n - (k - 1) * sm > 8, (n, k, sm)
        PH["name"] = f"{P} release"
        t0 = time.perf_counter()
        info, nb = release(st)
        T.tier_call = _no_decode
        print(f"[{TAG}] {P}: released {nb / 1e9:.2f} GB of trunk pools in {time.perf_counter() - t0:.2f} s; footprint now "
              f"{fp()[0]:.2f} GB, MLX active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
    mx.reset_peak_memory()
    PH["name"] = f"{P} prefill"
    t0 = time.perf_counter()
    state = B.b12_prefill_state(model, ids, 512, sm)
    mx.eval(state["logits"])
    tp = time.perf_counter() - t0
    pk = mx.get_peak_memory() / 1e9
    if PFR:
        T.tier_call = _tier_call
        PH["name"] = f"{P} restore"
        t0 = time.perf_counter()
        nb2 = restore(st, info)
        del info
        print(f"[{TAG}] {P}: re-created {nb2 / 1e9:.2f} GB of pools in {time.perf_counter() - t0:.2f} s (released {nb / 1e9:.2f}); "
              f"footprint now {fp()[0]:.2f} GB", flush=True)
        assert nb2 == nb, (nb2, nb)
    PH["name"] = f"{P} decode"
    TT = []                                                      # perf_counter at each emitted token (NewIdeas 23:33: TTFT)
    tg = time.perf_counter()
    out, s = spec125.generate(model, ids, 32, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={},
                              on_token=lambda t: TT.append(time.perf_counter()))
    st.gate.check()
    print(f"[{TAG}] {P}: prefill {n} tokens in {tp:.2f} s = {n / tp:.0f} tok/s (s_moe {sm}) | MLX peak {pk:.2f} GB, process "
          f"lifetime peak {fp()[1]:.2f} GB, now {fp()[0]:.2f} GB | decode 32: {s['tok_s']:.1f} tok/s, sha {sha(out)}", flush=True)
    print(f"[{TAG}] {P}: time to first token {TT[0] - tw:.2f} s (before decode {tg - tw:.2f}, state copy {s['t_prefill']:.2f}, "
          f"first pass {TT[0] - tg - s['t_prefill']:.2f}); tokens 1 -> {len(TT)} in {TT[-1] - TT[0]:.2f} s; prompt wall "
          f"{TT[-1] - tw:.2f} s", flush=True)
    del state
    mx.clear_cache()
    PH["name"] = "idle"
    time.sleep(0.1)
time.sleep(0.1)
for ph, (f, a, n_) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB, peak MLX active {a:.2f} GB ({n_} samples)", flush=True)
print(f"[{TAG}] done", flush=True)
