"""Fit12 DF1 (2026-10-23): the native DFlash2 drafter on B12LL at 12 GB (code / math).
B12LL (bt_b12.load_b12(0.10) + B12_LATE3=2,10 + mass) is loaded WITHOUT the MTP head. The drafter
(incoai/Qwen3.8-Flash-Next-DFlash2, on disk since Leap L2; Leap's vendored z-lab model_mlx.py) is quantised to 8 bits unless
F12_DQ=16, and bound to B12's own embedding and lm_head. Target features = the mean of the 4 hyper-connection streams after
zero-based layers 5/15/25/35/45 (Leap's capture), taken in every forward (prompt prefill in 512-token chunks, verify passes).
Loop = spec125.generate's verify / commit (snapshot -> verify_capture_scope -> commit_verified_window; repair forward on a
refused commit), with drafts = DFlash2's 7-token block, the first F12_VW - 1 of them verified. Greedy.
Switches: F12_VW (8 | 6 | 4), F12_KC (10 for S = 8), F12_BUDGET (b16 / b24: BigMoE's per-pass budget hook).
Arms F12_ARMS="tfc0,tfc1,tfg0,..." (Eval suite tf_gen items); BT_N tokens per item after a 16-token warm-up on the first.
Oracle timing (2026-10-24, Leap's T16/T8 ask): F12_ORACLE=TAG takes the drafts from TAG's saved greedy outputs (F12_SAVE dir;
no drafter), so every pass verifies F12_VWS rows (e.g. 8,16; alternating order per item) on real continuation tokens. Rows
per pass above MAXK // KC take v3's exact tier path (bt_b12 block), so a same-path ratio needs BIGMOE_MAXK=70 (SMAX 7).
Usage: [BigMoE venv] fit_dfl2.py TAG"""
import contextlib, ctypes, hashlib, json, os, sys, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MAXK": "80", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64",
       "BIGMOE_ASYNC_EVERY": "1", "HF_HUB_OFFLINE": "1",
       "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_DEFER_REPAIR_EVAL": "1"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "df"
VW = int(os.environ.get("F12_VW", "8"))
ORC = os.environ.get("F12_ORACLE", "")
VWS = [int(v) for v in os.environ.get("F12_VWS", str(VW)).split(",")]
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(2, os.path.expanduser("~/Fit12/scripts"))
sys.path.insert(3, os.path.expanduser("~/Leap/third_party"))
import bt_b12 as B
B.KC = int(os.environ.get("F12_KC", "10" if VW > 4 else "20"))
assert ORC or VW * B.KC <= int(os.environ["BIGMOE_MAXK"]), (VW, B.KC)
import fit_trim                                # L2-02/03: F12_EMBED=1 (streamed embed), F12_HCPRE=1 (hc8 before the eval)
fit_trim.from_env(B)
rt, st = B.load_b12(float(os.environ.get("F12_FRAC", "0.10")), mtp=False)   # F12_FRAC: residency (0.375 = C_m 192)
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import spec125
from mtplx.models import qwen4_exp as Q
from bigmoe_v3f.gated import ext
from bigmoe_v3f.fuse import reset_generation_state
import dflash_model_mlx as D
model = rt.model
BUD = os.environ.get("F12_BUDGET", "")
if BUD:
    import fit_budget
    print(f"fit12: budget {BUD}: {fit_budget.install(model, BUD)} routers", flush=True)

DQ, WIN = int(os.environ.get("F12_DQ", "8")), 0
if not ORC:
    draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    if DQ < 16:
        nn.quantize(draft, group_size=64, bits=DQ, class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
    draft.bind(model)
    mx.eval(draft.parameters())
    MASK = int(draft.config.mask_token_id)
    WIN = int(draft.config.sliding_window)
print(f"fit12: {'oracle drafts from ' + ORC if ORC else f'DFlash2 {DQ}-bit, window {WIN}'}, verify width {VWS if ORC else VW}, "
      f"KC {B.KC}, MAXK {os.environ['BIGMOE_MAXK']}, active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)

TAPS = (5, 15, 25, 35, 45)
TM = model.language_model.model
TAPID = {id(TM.layers[i]): j for j, i in enumerate(TAPS)}
CAP = {}
_lc = Q.DecoderLayer.__call__


def _layer(self, hidden, **kw):
    out = _lc(self, hidden, **kw)
    j = TAPID.get(id(self))
    if j is not None:
        b_, l_, w_ = out.shape
        CAP[j] = out.reshape(b_, l_, 4, w_ // 4).astype(mx.float32).mean(axis=2)[0]
    return out
Q.DecoderLayer.__call__ = _layer


def feats():
    return mx.concatenate([CAP[j] for j in range(len(TAPS))], axis=-1).astype(mx.bfloat16)   # [rows, 12800]


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def _ru():
    ru = _RU()
    return ru if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else None


TFK = {"tfc0": "tf_code/HumanEval/103", "tfc1": "tf_code/HumanEval/10", "tfc2": "tf_code/HumanEval/153",
       "tfg0": "tf_gsm/600", "tfg1": "tf_gsm/601", "tfg2": "tf_gsm/602"}
TF = {}
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    if r["key"] in TFK.values():
        TF[r["key"]] = [int(t) for t in r["prompt_ids"]]
CHAT = {"code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),     # as fit_b12.PROMPTS
        "lru": lambda: ("Implement a thread-safe LRU cache class in Python with get and put methods and a fixed capacity, "
                        "using only the standard library, then show a short usage example."),
        "csv": lambda: ("Write a Python script that reads a CSV file of transactions (customer_id, amount, date), computes "
                        "each customer's total spend per month, and prints the five biggest spenders of the latest month.")}


def ids_of(P):
    if P in TFK:
        return TF[TFK[P]]
    t = rt.tokenizer.apply_chat_template([{"role": "user", "content": CHAT[P]()}], add_generation_prompt=True,
                                         enable_thinking=False)
    return list(t if isinstance(t[0], int) else t["input_ids"])


def generate(ids, n, vw=VW, orc=None):
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    cache, P, F = model.make_cache(), len(ids) - 1, []
    for i in range(0, P, 512):                                # prompt minus its last token, features per chunk
        CAP.clear()
        model(mx.array([ids[i:min(i + 512, P)]]), cache=cache, emit_logits=False)
        f = feats()
        mx.eval(f, [c.state for c in cache if hasattr(c, "state")])
        F.append(f)
    mx.clear_cache()
    if orc is None:
        start = max(0, P - (WIN - 1))
        ctx = mx.concatenate(F)[start:] if len(F) > 1 else F[0][start:]
        dc = draft.make_cache()
        for c in dc:
            c.offset = start
    y, a, out = int(ids[-1]), P, []                           # a = absolute position of the anchor y
    s = dict(passes=0, refused=0, t_draft=0.0, t_verify=0.0, acc=[], tv=[])
    t0 = time.perf_counter()
    while len(out) < n:
        td = time.perf_counter()
        if orc is None:
            path, _, _ = draft.propose(mx.array([[y] + [MASK] * 7]), ctx[None], dc, 0.0, logits_start=1)
            mx.eval(path)
            if (trim := dc[0].offset - a) > 0:                # as the official loop (0 in steady state)
                D._trim_recent_cache(dc, trim)
            d = [int(t) for t in np.array(path[0]).tolist()][:vw - 1][:max(0, n - len(out) - 1)]
        else:                                                 # oracle: the saved greedy continuation
            if out != orc[:len(out)]:
                s["diverged"] = len(out)
                break
            d = orc[len(out):len(out) + vw - 1][:max(0, n - len(out) - 1)]
        tv = time.perf_counter()
        s["t_draft"] += tv - td
        toks = [y] + d
        S = len(toks)
        snap = None
        if S > 1:
            snap = spec125.snapshot_untrimmable_cache_lazy(cache)
            spec125.ensure_eager_window_capacity(cache, S)
        CAP.clear()
        with (model.verify_capture_scope() if S > 1 else contextlib.nullcontext()):
            lg, _ = spec125._split(model(mx.array([toks]), cache=cache, return_hidden=True))
        g = mx.argmax(lg[0].astype(mx.float32), axis=-1)
        f = feats()
        mx.eval(g, f)
        g = g.tolist()
        k = 0
        while k < len(d) and d[k] == g[k]:
            k += 1
        new = d[:k] + [g[k]]
        if S > 1:
            if not model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=S):
                s["refused"] += 1
                spec125.rollback_after_verify(cache, snap, verified_tokens=S)
                CAP.clear()
                model(mx.array([toks[:k + 1]]), cache=cache, return_hidden=True)
                f = feats()
                mx.eval(f)
            model.clear_verify_capture(cache)
        s["t_verify"] += time.perf_counter() - tv
        s["tv"].append((S, time.perf_counter() - tv))
        ctx = f[:k + 1]                                       # features of y and the k accepted drafts
        s["passes"] += 1; s["acc"].append(k)
        out += new[:n - len(out)]
        y, a = new[-1], a + k + 1
        if STOP & set(new):                                   # end of turn (Leap 04:44: post-turn tokens dilute tok/pass)
            out = out[:min(i for i, t in enumerate(out) if t in STOP) + 1]
            break
    s["time"] = time.perf_counter() - t0
    st.gate.check()
    return out, s


sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
STOP = {248046, 248044} if os.environ.get("F12_STOP", "1") == "1" else set()   # <|im_end|>, <|endoftext|>
N = int(os.environ.get("BT_N", "160"))
arms = os.environ.get("F12_ARMS", "tfc0,tfc1,tfc2,tfg0,tfg1,tfg2").split(",")
SAVE = os.environ.get("F12_SAVE")
if ORC:                                                       # Leap's T16/T8: per-pass time at each verify width
    ref = {P: json.load(open(os.path.join(os.path.expanduser(SAVE), f"{ORC}_{P}.json"))) for P in arms}
    for w in VWS:
        generate(ids_of(arms[0]), 2 * w, vw=w, orc=ref[arms[0]])   # warm-up both shapes
    T = {w: [] for w in VWS}
    for i, P in enumerate(arms):
        ids = ids_of(P)
        for w in (VWS if i % 2 == 0 else VWS[::-1]):
            e0, r0 = ext.stats(), _ru()
            out, s = generate(ids, N, vw=w, orc=ref[P])
            e1, r1 = ext.stats(), _ru()
            full = [t for S_, t in s["tv"] if S_ == w]
            T[w] += full
            np_ = max(1, s["passes"])
            disk = (r1.u16 - r0.u16) if r0 and r1 else 0
            print(f"[{TAG}] {P} k={w}: {len(out) / s['time']:.1f} tok/s, {len(out) / np_:.2f} tok/pass, {np_} passes "
                  f"({len(full)} full) | full pass {1e3 * np.mean(full):.1f} ms (median {1e3 * np.median(full):.1f}) | reads "
                  f"{(e1['misses'] - e0['misses']) / np_:.1f}, stall {1e3 * (e1['slow_s'] - e0['slow_s']) / np_:.1f} ms, SSD "
                  f"{disk / 1e6 / np_:.0f} MB per pass | refused {s['refused']} | diverged {s.get('diverged', '-')} | peak "
                  f"footprint {(r1.u28 / 1e9) if r1 else float('nan'):.2f} GB | sha {sha(out)}", flush=True)
    m = {w: 1e3 * float(np.mean(T[w])) for w in VWS}
    print(f"[{TAG}] ALL: full-pass mean " + ", ".join(f"T{w} {m[w]:.1f} ms ({len(T[w])} passes)" for w in VWS)
          + (f" | T{VWS[1]}/T{VWS[0]} = {m[VWS[1]] / m[VWS[0]]:.3f}" if len(VWS) > 1 else ""), flush=True)
    sys.exit(0)
generate(ids_of(arms[0]), 16)                                 # warm-up
tot = dict(tok=0, time=0.0, passes=0)
for P in arms:
    ids = ids_of(P)
    e0, d0, r0 = ext.stats(), ext.defer_stats(), _ru()
    out, s = generate(ids, N)
    e1, d1, r1 = ext.stats(), ext.defer_stats(), _ru()
    np_ = max(1, s["passes"])
    disk = (r1.u16 - r0.u16) if r0 and r1 else 0
    hist = np.bincount(np.array(s["acc"]), minlength=VW)
    tot["tok"] += len(out); tot["time"] += s["time"]; tot["passes"] += np_
    print(f"[{TAG}] {P}: {len(out) / s['time']:.1f} tok/s, {len(out) / np_:.2f} tok/pass, {np_} passes | per pass: verify "
          f"{1e3 * s['t_verify'] / np_:.1f} ms, draft {1e3 * s['t_draft'] / np_:.1f} ms, waited {(d1[1] - d0[1]) / np_:.1f}, "
          f"reads {(e1['misses'] - e0['misses']) / np_:.1f}, stall {1e3 * (e1['slow_s'] - e0['slow_s']) / np_:.1f} ms, SSD "
          f"{disk / 1e6 / np_:.0f} MB | refused {s['refused']} | accepted hist {hist.tolist()} | peak footprint "
          f"{(r1.u28 / 1e9) if r1 else float('nan'):.2f} GB | sha {sha(out)}", flush=True)
    if SAVE:
        json.dump(out, open(os.path.join(os.path.expanduser(SAVE), f"{TAG}_{P}.json"), "w"))
print(f"[{TAG}] ALL: {tot['tok'] / tot['time']:.1f} tok/s, {tot['tok'] / tot['passes']:.2f} tok/pass over {len(arms)} items", flush=True)
