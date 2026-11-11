"""Fit12 CMP (Mon 26 Oct 2026; PLAN.md 'Rapid screens, batch 1'): per-prompt C_m screen = fit_pfr.py (PFR T1) with PFR-a's
512-aligned chunks, F12_NDEC greedy tokens (default 96), a first-step logits hash, decode-end state hashes (trunk cache +
MTP head cache, captured from spec125.restore_state) and a warm decode rate (tokens 33..N); B12_PFR=0 so bt_b12 adds no
PFR of its own. One RESULT line per prompt. Batch 2 (R-01..R-03) adds prompts c<L>_<i> (conf_prompts.py), F12_NDECS
(tokens per prompt position, else F12_NDEC), F12_TGT=m (mtplx qsa_rows_target(n + tokens + m) around prefill + decode) and
per-position phases; the RESULT line gains pos / clean peak / first128 / kv cap fields (hash fields unchanged).
Below: fit_pfr.py's docstring.
Fit12 PFR T1 (2026-10-25; pre-registered in PLAN.md 'PFR pre-registration'): Prefill with the decode pools Released.
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
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1", "B12_PFR": "0"}
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
if os.environ.get("F12_BTC") == "1":                          # Fit12 experiment copy of bt_b12 (F12_ALLWAIT)
    sys.path.insert(0, os.path.expanduser("~/Fit12/btc"))
import bt_b12 as B
print(f"bt_b12 from {B.__file__}", flush=True)
if os.environ.get("F12_V3F"):
    B.V3F = os.path.expanduser(os.environ["F12_V3F"])
rt, st = B.load_b12(float(os.environ.get("F12_FRAC", "0.10")), mtp=True)
import spec125
import conf_prompts as CP
from mtplx.models import qwen4_exp as Q
HCN = [0]
if os.environ.get("F12_HCX") == "1":     # R-14 price: every HC read + write runs twice (the copy on hyper + 0, added x 0)
    _gr_call, _Z = Q.GatedResidual.__call__, mx.array(0.0, dtype=mx.bfloat16)

    def _gr2(self, hyper_input, pending=None):
        HCN[0] += 1
        r = _gr_call(self, hyper_input, pending)
        e = _gr_call(self, hyper_input + _Z, None if pending is None else (pending[0] + _Z, pending[1]))
        return (r[0] + _Z * e[0],) + tuple(r[1:]) if isinstance(r, tuple) else r + _Z * e
    Q.GatedResidual.__call__ = _gr2
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
           "p8k": lambda: HEAD + open(os.path.expanduser("~/Breakthrough/data/prompts/ctx32k.txt")).read()[:26000] + "\n```",
           "p32k": lambda: HEAD + open(os.path.expanduser("~/Breakthrough/data/prompts/ctx32k.txt")).read()[:98000] + "\n```",
           "p64k": lambda: HEAD + open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read()[:199000] + "\n```"}
NDEC = int(os.environ.get("F12_NDEC", "96"))
NDECS = [int(x) for x in os.environ.get("F12_NDECS", "").split(",") if x]   # tokens per prompt position (else NDEC)
TGT = int(os.environ.get("F12_TGT", "0"))      # R-02 KVC: > 0 = qsa_rows_target(n + tokens + TGT) around prefill + decode
import numpy as np
_RS, CAPT = spec125.restore_state, []


def _rs(model, state, with_head=True):                        # keep the decode's own cache and head
    if os.environ.get("F12_MOVE") == "1":                    # MOV: hand the prefill arrays over, no second copy
        cache, head = state.pop("cache"), None
        if with_head:
            head = spec125._MTPHead(model)
            head.cache = state.pop("mtp_cache")
            head.logits, head.h = state["logits"], state["h"]
        r = (cache, head)
    else:
        r = _RS(model, state, with_head=with_head)
    CAPT.append(r)
    return r


spec125.restore_state = _rs


PER = {}


def kv_hash(cache, key="kv"):
    """sha of every array in each layer's state, bf16 hashed as exact float32 (fit_pfr64.kv_hash); per-layer 6-char
    prefixes (with each array's shape) kept in PER[key] for the LAYERS line"""
    full, per = hashlib.sha1(), []
    for c in (cache if isinstance(cache, (list, tuple)) else [cache]):
        h, shp = hashlib.sha1(), []
        for a in (c.state if hasattr(c, "state") else []):
            if isinstance(a, mx.array):
                h.update(np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes())
                shp.append("x".join(map(str, a.shape)))
        full.update(h.digest())
        per.append(f"{h.hexdigest()[:6]}[{type(c).__name__}:{','.join(shp)}]")
    PER[key] = per
    return full.hexdigest()[:10]


def qcap(cache):
    """'allocated/used' rows of the first QSA attention buffer in a cache list (or one cache)"""
    for c in (cache if isinstance(cache, (list, tuple)) else [cache]):
        kv = getattr(c, "kv", None)
        if kv is not None and getattr(kv, "keys", None) is not None:
            return f"{kv.keys.shape[2]}/{kv.offset}"
    return "none"

sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
n0, p0 = fp()
print(f"[{TAG}] engine {B.V3F}, PFR {int(PFR)} (cap {CAP}), s_moe {SMOE} when off, C_m {B.CFG['cm']} C_l {B.CFG['cl']}, "
      f"pools {st.resident_gb():.2f} GB | after load: active {mx.get_active_memory() / 1e9:.2f} GB, footprint now {n0:.2f}, "
      f"lifetime peak {p0:.2f} GB", flush=True)
for pos, P in enumerate(os.environ.get("F12_PROMPTS", "p2k,p8k").split(","), 1):
    ids = chat_ids(CP.conf_text(P) if CP.is_conf(P) else PROMPTS[P]())
    n = len(ids) - 1
    nd, PP = (NDECS[pos - 1] if pos <= len(NDECS) else NDEC), f"{pos}:{P}"
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    sm = SMOE
    tw = time.perf_counter()                                     # this prompt's wall clock (release + prefill + restore + decode)
    if PFR:
        k = math.ceil(n / CAP)
        sm = n if k == 1 else 512 * math.ceil(n / (k * 512))     # PFR-a: 512-aligned chunks (b12_prefill asserts it)
        assert n - (k - 1) * sm > 8, (n, k, sm)
        PH["name"] = f"{PP} release"
        t0 = time.perf_counter()
        info, nb = release(st)
        T.tier_call = _no_decode
        print(f"[{TAG}] {P}: released {nb / 1e9:.2f} GB of trunk pools in {time.perf_counter() - t0:.2f} s; footprint now "
              f"{fp()[0]:.2f} GB, MLX active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
    tcm = Q.qsa_rows_target(n + nd + TGT) if TGT else None
    if tcm is not None:
        tcm.__enter__()
    mx.reset_peak_memory()
    PH["name"] = f"{PP} prefill"
    t0 = time.perf_counter()
    state = B.b12_prefill_state(model, ids, 512, sm)
    mx.eval(state["logits"])
    tp = time.perf_counter() - t0
    pk = mx.get_peak_memory() / 1e9
    PH["name"] = f"{PP} lpre"                       # 27 Oct: the same logits hash before the decode (own phase, not in clean peak)
    lpre = hashlib.sha1(np.array(state["logits"].astype(mx.float32)).tobytes()).hexdigest()[:10]
    if os.environ.get("F12_SAVE_LOGITS"):              # pos-2 follow-up: the prefill logits themselves, per position
        np.save(os.path.join(os.environ["F12_SAVE_LOGITS"], f"{TAG}_pos{pos}.npy"), np.array(state["logits"].astype(mx.float32)))
    if PFR:
        T.tier_call = _tier_call
        PH["name"] = f"{PP} restore"
        t0 = time.perf_counter()
        nb2 = restore(st, info)
        del info
        print(f"[{TAG}] {P}: re-created {nb2 / 1e9:.2f} GB of pools in {time.perf_counter() - t0:.2f} s (released {nb / 1e9:.2f}); "
              f"footprint now {fp()[0]:.2f} GB", flush=True)
        assert nb2 == nb, (nb2, nb)
    PH["name"] = f"{PP} decode"
    TT = []                                                      # perf_counter at each emitted token (NewIdeas 23:33: TTFT)
    mx.reset_peak_memory()
    tg = time.perf_counter()
    CAPT.clear()
    out, s = spec125.generate(model, ids, nd, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={},
                              on_token=lambda t: TT.append(time.perf_counter()))
    st.gate.check()
    if tcm is not None:
        tcm.__exit__(None, None, None)
    md = mx.get_peak_memory() / 1e9
    cp = max((PK[k][0] for k in (f"{PP} {x}" for x in ("release", "prefill", "restore", "decode")) if k in PK), default=-1.0)
    f128 = 128 / (TT[127] - tg) if len(TT) >= 128 else float("nan")
    print(f"[{TAG}] {P}: prefill {n} tokens in {tp:.2f} s = {n / tp:.0f} tok/s (s_moe {sm}) | MLX peak {pk:.2f} GB, process "
          f"lifetime peak {fp()[1]:.2f} GB, now {fp()[0]:.2f} GB | decode {len(out)}: {s['tok_s']:.1f} tok/s, sha {sha(out)}", flush=True)
    warm = (len(TT) - 32) / (TT[-1] - TT[31]) if len(TT) > 33 else float('nan')
    lpk = fp()[1]
    PH["name"] = f"{PP} hash"
    lh = hashlib.sha1(np.array(state["logits"].astype(mx.float32)).tobytes()).hexdigest()[:10]
    dc, hd = CAPT[-1] if CAPT else (None, None)
    NH = os.environ.get("F12_NOHASH") == "1"            # GROW (27 Oct): no kv / head hashing (does it cause the multi-prompt growth?)
    kh = "skip" if NH else (kv_hash(dc) if dc is not None else "none")
    hh = "skip" if NH else (kv_hash(hd.cache, "head") if hd is not None and getattr(hd, "cache", None) is not None else "none")
    kc = qcap(dc) if dc is not None else "none"
    hc = qcap(hd.cache) if hd is not None and getattr(hd, "cache", None) is not None else "none"
    if os.environ.get("F12_LAYERS") == "1":
        for k_, v_ in PER.items():
            print(f"[{TAG}] {P} LAYERS {k_} " + " ".join(f"{i}:{x}" for i, x in enumerate(v_)), flush=True)
    CAPT.clear(); del dc, hd
    print(f"[{TAG}] {P} RESULT logits {lh} sha32 {sha(out[:32])} sha{len(out)} {sha(out)} kv {kh} head {hh} | n {n} C_m "
          f"{B.CFG['cm']} C_l {B.CFG['cl']} | warm tok/s {warm:.2f} (tokens 33-{len(TT)}), all {s['tok_s']:.2f} | lifetime peak "
          f"{lpk:.2f} GB before the hash | pos {pos} | clean peak {cp:.2f} GB (decode {PK[f'{PP} decode'][0]:.2f}, MLX decode "
          f"{md:.2f}) | first128 {f128:.2f} tok/s | kv cap {kc} head cap {hc}"
          f"{f' | HC calls doubled {HCN[0]}' if os.environ.get('F12_HCX') == '1' else ''} | lpre {lpre}", flush=True)
    print(f"[{TAG}] {P}: time to first token {TT[0] - tw:.2f} s (before decode {tg - tw:.2f}, state copy {s['t_prefill']:.2f}, "
          f"first pass {TT[0] - tg - s['t_prefill']:.2f}); tokens 1 -> {len(TT)} in {TT[-1] - TT[0]:.2f} s; prompt wall "
          f"{TT[-1] - tw:.2f} s", flush=True)
    del state
    mx.clear_cache()
    print(f"[{TAG}] {P}: after pos {pos}: MLX active {mx.get_active_memory() / 1e9:.3f} GB, cache {mx.get_cache_memory() / 1e9:.3f} GB, "
          f"footprint {fp()[0]:.3f} GB", flush=True)        # GROW (27 Oct): what stays between prompts
    PH["name"] = "idle"
    time.sleep(0.1)
time.sleep(0.1)
for ph, (f, a, n_) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB, peak MLX active {a:.2f} GB ({n_} samples)", flush=True)
print(f"[{TAG}] done", flush=True)
