"""D-23 SWITCHDEF (MTP Head, Sun 11-01; NewIdeas 08:1x; RANKED_45GB #5 on the 45 GB DEFAULT): the drafter switch on the
default's own live loop. Harness = a copy of Leap's /Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts/k69_live.py (the 45 GB default, its R57_PRESET=45:
REF2-T tiers 4433333333 / 3c on the R-55 gate, MAXK 160, ext build_next as Leap runs it, + L1-14 truncation t 0.3 + ref2's
defaults: REF2b key buckets, GDNX), read-only use of their engines, plus D-20's MTP chain (sticky/d20_switch45.py:
spec125._MTPHead semantics; the head's history = each prefill chunk's widened stream, then each check's kept rows).
Arms per prompt, interleaved, order alternating by prompt, after one discarded warm-up per arm:
  dfl   the default as it ships: DFlash2 L1-20, 15 drafts cut where the running confidence < 0.3 (arm gate+t)
  mtp7  the native MTP chain with head b3 (D-22): 7 drafts, 8-row checks, on the same gate
The switch (DFlash2 on code, MTP K7 on math / chat / agent, NewIdeas' type rule) decides up front, so it is computed from
the two arms (NewIdeas 08:1x). Exactness: every token of both arms equal (REF2-T is its own reference), and == EvalSuite's
REF2-T AR (results/r56c/ar) where a file exists. Logged per check for a later CPU fit (a truncation rule for the MTP
chain): rows, kept, check and draft ms, and the drafter's per-step confidences (dfl: the selector's, before the cut;
mtp7: each step's top-1 probability; computing them adds ~20 small kernels to each MTP draft).
PASS (pre-registered 11-01, before the run): the switch >= +10 % whole-set tok/s over dfl with every token equal;
KILL < +4 %; BETWEEN otherwise. Per type as information (code stays on dfl by construction).
Usage: [BigMoE venv] D23_KEYS=k1,k2,... d23_switchdef.py [C=224] [CAP=256]  (D23_HEAD=b3 default)
Resumable: one line per (key, arm, head, C) in ~/TrainLab/data/d23_runs.jsonl; done ones are skipped."""
import json, os, re, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.pop("BIGMOE_MTP_STREAM", None)              # the MTP head resident, as in D-20
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["TL_INFER"] = "1"                           # mtp125: keep MTPLX's fused HC read
os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"   # k69_live R57_PRESET=45 (REF2-T)
os.environ.setdefault("BIGMOE_TIER_CL", "112")
os.environ.pop("BIGMOE_TIER_COMBINE", None)
os.environ.setdefault("BIGMOE_MAXK", "160")
os.environ.setdefault("BIGMOE_EXT_BUILD", "build_next")   # as Leap's R-69L / R-69b runners
assert os.environ.get("BIGMOE_MAXK") == "160", "see Usage"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF
from tokenizers import Tokenizer

TD = os.path.expanduser("~/TrainLab/data")
OUT = os.path.join(TD, "d23_runs.jsonl")
TXT_OUT = os.path.join(TD, "d23_switchdef.txt")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 224
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
HEAD = os.environ.get("D23_HEAD", "b3")
ARMS = ("dfl", "mtp7")
TRUNC = 0.3
R, K = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = os.environ.get("D23_KEYS", "humaneval/HumanEval/16,math500/test/prealgebra/1640.json,ifeval/2853,agent/bash18").split(",")
BUDGET, HARD = 680, 740                                  # s of running time: no new run / no new check after these
ACT = [0.0]                                              # running time without SIGSTOP freezes (Leap's ticker)
def _tick():
    last = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        ACT[0] += min(now - last, 3.0)
        last = now
threading.Thread(target=_tick, daemon=True).start()
el = lambda: ACT[0]
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

# ---------------------------------------------------------------- NewIdeas' type rule (scripts/type_rule.py, via D-20)
TOK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
CODE = re.compile(r"(^|\n)\s*(def |class |import |from \w+ import|#include|fn |func |public |const |let |var )|```|\breturn\b|\w+\(.*\)\s*:|>>>")
MATH = re.compile(r"\\boxed|\\frac|\\sqrt|\$[^$]+\$|\b(solve|compute|calculate|how many|how much|what is the (sum|product|value|area|probability))\b"
                  r"|\d+\s*[-+*/^=]\s*\d+", re.I)
TOOLS = re.compile(r"<tools>|\"type\":\s*\"function\"|tool_call|<function|\bfunctions?\b.*\bparameters\b", re.I | re.S)
def rule(item):
    full = item.get("prompt") or TOK.decode(item["prompt_ids"], skip_special_tokens=False)
    user = item.get("user") or full
    if TOOLS.search(full):
        return "agent"
    if CODE.search(user) or CODE.search(full[-4000:]):
        return "code"
    if MATH.search(user):
        return "math"
    return "chat"

# ---------------------------------------------------------------- the default's model, with the MTP head loaded resident
import bigmoe.loader as BL
_ls = BL.load_streamed
BL.load_streamed = lambda md, C_, mtp=False: _ls(md, C_, mtp=True)
t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
BL.load_streamed = _ls
model = rt.model
assert getattr(model.language_model, "mtp", None) is not None, "no MTP head loaded"
import ref2
import bigmoe.stream as BS
import bigmoe.gated as BG
import bigmoe.tier as _BT
from mtplx import cache_state as CS
Q = ref2.Q
assert BG.MAXK == 160 and BG.XLINES == 4, (BG.MAXK, BG.XLINES)
TXT = model.language_model.model
lay = TXT.layers
SW_ = [l.mlp.switch_mlp for l in lay]
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/TrainLab_root"))
import mtp125 as M
T_ = M.Trainable(model.language_model.mtp)
T_.load_weights(os.path.expanduser(f"~/TrainLab/ckpt/d22_{HEAD}.safetensors"))
M.install(model.language_model.mtp, T_, dtype=mx.bfloat16)
say(f"D-23 SWITCHDEF: the 45 GB default (REF2-T tiers {os.environ['BIGMOE_TIER']} / {os.environ['BIGMOE_TIER_GLU']}, C_l "
    f"{os.environ['BIGMOE_TIER_CL']}; gate MAXK {BG.MAXK}, ext {BG._BUILD}; TIER_COMBINE {_BT._COMBINE}, OOO {_BT._OOO}; "
    f"truncation t {TRUNC}) at C {C} with the MTP head {HEAD}; loaded in {time.time() - t0:.0f} s, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active; arms {ARMS}; CAP {CAP}")

def set_arm(arm):                                        # k69_live's: the prefill as shipped ('slot'), the loop on the gate
    BS.GATE_N[0] = 40 if arm == "slot" else 160
    BS._PF_RANKED = False
    for sw in SW_:
        sw._bm.pop("next", None)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def stats():
    s = st.stats()
    return np.array([s["misses"], s["io_s"]], dtype=np.float64)

# ---------------------------------------------------------------- taps for the drafter (Leap's S45 harness)
TAP = {"on": False, "x": [[] for _ in TAPS]}
TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    out = _layer_call(self, *a, **kw)
    i = TAP_OF.get(id(self))
    if TAP["on"] and i is not None:
        o = out if isinstance(out, mx.array) else out[0]
        B, L, W = o.shape
        TAP["x"][i].append(o.reshape(B, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
    return out
Q.DecoderLayer.__call__ = _layer
def tap_reset():
    TAP["x"] = [[] for _ in TAPS]
def taps():
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
draft.bind(model)
mx.eval(draft.parameters())
MASK, SWIN = int(draft.config.mask_token_id), int(draft.config.sliding_window)

def head2s(hid):
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)

def chain(blk, unary, cand, hp):
    sel = draft.candidate_selector
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)

def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))

# ---------------------------------------------------------------- the native MTP head (D-20's, + each step's top-1 p)
class MTPHead:
    def __init__(self):
        self.cache = model.make_mtp_cache()
        self.logits = self.h = None

    def append(self, hidden, next_ids):
        """commit pairs (hidden row i, next_ids[i]); the last one runs through mtp_forward for the next draft"""
        if hidden.shape[1] > 1:
            model.mtp_update_cache(hidden[:, :-1], mx.array([next_ids[:-1]]), mtp_cache=self.cache)
        self.logits, self.h = model.mtp_forward(hidden[:, -1:], mx.array([next_ids[-1:]]), mtp_cache=self.cache,
                                                return_hidden=True)

    def draft(self, k):
        l0 = self.logits[0, -1].astype(mx.float32)
        toks, ps, h = [mx.argmax(l0)], [mx.softmax(l0).max()], self.h
        for _ in range(k - 1):
            lgx, h = model.mtp_forward(h, toks[-1].reshape(1, 1), mtp_cache=self.cache, return_hidden=True)
            lx = lgx[0, -1].astype(mx.float32)
            toks.append(mx.argmax(lx)); ps.append(mx.softmax(lx).max())
        mx.eval(toks, ps)
        if k > 1:
            self.cache[0].trim(k - 1)                    # the chain's own entries are not committed history
        return [int(t.item()) for t in toks], [float(p.item()) for p in ps]

_midloop = Q._prefill_midloop_eval_layers
def ref_on(on):
    if on:
        ref2.apply()
        Q._prefill_midloop_eval_layers = lambda rows: _midloop(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
    else:
        ref2.remove()
        Q._prefill_midloop_eval_layers = _midloop

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def prefill(ids, want_w, chunk=1024):
    """k69_live's prefill (chunks of 1024 on the PF1 path, taps), plus each chunk's widened stream for the MTP head"""
    pf1(True)
    cache, lg, ws = model.make_cache(), None, []
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        if want_w:
            w = TXT._last_widened
            assert w.shape[1] == min(chunk, len(ids) - a), w.shape
            mx.eval(w)
            ws.append((a, w))
    TAP["on"] = False
    pf1(False)
    return cache, lg[0, -1], taps(), ws

def run(ids, arm, cap):
    while os.path.exists(PAUSE):
        time.sleep(5)
    mtp = arm.startswith("mtp")
    KM = int(arm[3:]) if mtp else 0
    ref2.reset_counts()
    set_arm("slot")                                      # the prefill as shipped in every arm
    ref_on(True)
    p0, g0 = BG.PAUSED["pause_s"], BG.PAUSED["n_gap"]
    t1 = time.perf_counter()
    cache, last, T, ws = prefill(ids, mtp)
    pf = time.perf_counter() - t1
    P = len(ids)
    tok = int(mx.argmax(last).item())
    hist_t, pend, head, dcache, ctx = 0.0, None, None, None, None
    if mtp:                                              # the prompt's pairs (W[u], token u+1); the last pair uses tok
        ref2.remove()
        t2 = time.perf_counter()
        head, full = MTPHead(), list(ids) + [tok]
        for a_, w in ws:
            head.append(w, full[a_ + 1:a_ + 1 + w.shape[1]])
            mx.eval(head.logits)
        hist_t = time.perf_counter() - t2
        ref2.apply()
    else:
        dcache = draft.make_cache()
        start = max(0, P - (SWIN - 1))
        for c in dcache:
            c.offset = start
        ctx = T[start:P]
    a = P
    set_arm("gate")
    out, cyc, dr_t, ck_t, kept, rows, sm, wl, cfs, cut = [tok], [], [], [], [], [], [], [], [], False
    td = c_prev = time.perf_counter()
    while len(out) < cap and tok not in EOS:
        if el() > HARD:
            cut = True
            break
        c0 = time.perf_counter()
        if out[1:]:
            cyc.append(c0 - c_prev)
        c_prev = c0
        ref2.remove()
        if mtp:
            if pend is not None:                         # the last check's kept rows -> the head's history
                head.append(*pend)
            dr, cf = head.draft(KM)
        else:
            pth, cfa = propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)
            dr, cf = np.array(pth[0]).tolist(), np.array(cfa[0]).tolist()
            dr = dr[:int((np.cumprod(np.array(cf)) >= TRUNC).sum())]   # Leap's S45c: L1-14 running-confidence truncation
            if (trim := dcache[0].offset - a) > 0:
                DF._trim_recent_cache(dcache, trim)
        ref2.apply()
        s0 = stats()
        c1 = time.perf_counter()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        if not mtp:
            tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([[tok] + dr]), cache=cache)
        TAP["on"] = False
        am = mx.argmax(lg[0], axis=-1)
        wv = TXT._last_widened if mtp else None
        tp = None if mtp else taps()
        mx.eval(am, *([wv[:, :len(dr) + 1]] if mtp else [tp]))
        am = am.tolist()
        k = 0
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        c2 = time.perf_counter()
        sm.append(stats() - s0)
        dr_t.append(c1 - c0); ck_t.append(c2 - c1); kept.append(k + 1); rows.append(len(dr) + 1); wl.append(c2 - c0)
        cfs.append([round(float(x), 4) for x in cf])
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
                new = new[:j + 1]
                break
        if mtp:                                          # rows 0..len(new)-1 hold the pairs (W[a + i], new[i])
            pend = (wv[:, :len(new)], new)
        else:
            ctx = tp[:k + 1]
        out += new
        tok, a = out[-1], a + k + 1
    mx.eval(states(cache))
    dec = time.perf_counter() - td
    bad = ""
    try:
        st.gate.check()
    except RuntimeError as e:
        bad = f" INVALID ({e})"
    if BG.PAUSED["pause_s"] - p0 > 0.5 or BG.PAUSED["n_gap"] > g0:
        bad += f" PAUSED ({BG.PAUSED['pause_s'] - p0:.1f} s waited, {BG.PAUSED['n_gap'] - g0} gaps)"
    set_arm("slot")
    ref_on(False)
    del cache, dcache, head
    mx.clear_cache()
    sm = np.array(sm) if sm else np.zeros((1, 2))
    n = min(len(out), cap)
    return dict(out=out[:cap], n=n, tok_s=n / dec, pf=pf, hist=hist_t, checks=len(kept),
                kept=float(np.mean(kept)) if kept else 0.0, rows=float(np.mean(rows)) if rows else 0.0,
                cyc=1e3 * med(cyc), dr=1e3 * med(dr_t), ck=1e3 * med(ck_t), miss=float(np.median(sm[:, 0])),
                io=1e3 * float(np.median(sm[:, 1])), tpsw=sum(kept[2:]) / max(sum(wl[2:]), 1e-9), cut=cut, bad=bad,
                pc=[[int(r_), int(k_), round(1e3 * c_, 2), round(1e3 * d_, 2)] for r_, k_, c_, d_ in zip(rows, kept, ck_t, dr_t)],
                cf=cfs)

# ---------------------------------------------------------------- driver: interleaved arms per prompt, resumable
SETOF = {"humaneval": "humaneval", "math500": "math", "gsm8k": "math", "ifeval": "ifeval", "multi": "multi", "agent": "agent"}
items = {}
for key in PICK:
    items.update({it["key"]: it for it in ED.load("dec_" + SETOF[key.split("/")[0]]) if it["key"] == key})
assert all(k in items for k in PICK), [k for k in PICK if k not in items]
AR = os.path.expanduser("~/EvalSuite/results/r56c/ar")
def ar_toks(key):
    f = os.path.join(AR, key.replace("/", "__") + ".json")
    return json.load(open(f))["toks"] if os.path.exists(f) else None
done = {}
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l)
        if not (d["cut"] or d["paused"] or d["invalid"]):   # an unclean run is re-run on the next ticket
            done[(d["key"], d["arm"], d["head"], d["C"])] = d
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

it0 = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
for arm in ARMS:                                         # warm-up, discarded (S45-T's method)
    r = run(list(it0["prompt_ids"]), arm, 64)
    say(f"  warm-up (discarded): gsm8k/708 [{arm}] {r['n']} tokens, {r['tok_s']:.1f} tok/s{r['bad']}")
for n_p, key in enumerate(PICK):
    it = items[key]
    typ = rule(it)
    for arm in (ARMS if n_p % 2 == 0 else ARMS[::-1]):
        if (key, arm, HEAD, C) in done:
            continue
        if el() > BUDGET:
            say(f"  time budget: stopped before {key} [{arm}] (resumable)")
            break
        r = run(list(it["prompt_ids"]), arm, CAP)
        want = ar_toks(key)
        ar_ok = None if want is None else same(r["out"], want) >= min(len(want), r["n"])
        rec = dict(key=key, arm=arm, head=HEAD, C=C, type=typ, ar_ok=ar_ok, paused="PAUSED" in r["bad"],
                   invalid="INVALID" in r["bad"], **{k_: v for k_, v in r.items() if k_ != "bad"})
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not (rec["cut"] or rec["paused"] or rec["invalid"]):
            done[(key, arm, HEAD, C)] = rec
        hs = f" + MTP history {r['hist']:.2f} s" if arm != "dfl" else ""
        say(f"  {key} [{typ}] {arm}: {r['n']} tokens, {r['tok_s']:.1f} tok/s (steady {r['tpsw']:.1f}), kept {r['kept']:.2f} of "
            f"{r['rows']:.1f} rows, median cycle {r['cyc']:.1f} ms (draft {r['dr']:.1f} + check {r['ck']:.1f}); misses {r['miss']:.0f}, "
            f"SSD {r['io']:.1f} ms per check; prefill {r['pf']:.1f} s{hs}; == REF2-T AR: {ar_ok}"
            f"{'; cut' if r['cut'] else ''}{r['bad']}")
    else:
        continue
    break

# ---------------------------------------------------------------- read-out over everything saved at this C and head
D = {}
for d in done.values():
    if d["head"] == HEAD and d["C"] == C and d["key"] in PICK:
        D.setdefault(d["key"], {})[d["arm"]] = d
both = [k for k in PICK if all(a in D.get(k, {}) for a in ARMS)]
clean = [k for k in both if not any(D[k][a]["paused"] or D[k][a]["invalid"] or D[k][a]["cut"] for a in ARMS)]
eq = all(D[k]["dfl"]["out"] == D[k]["mtp7"]["out"] for k in both)
say(f"D-23 read-out (C {C}, head {HEAD}): {len(both)} prompts with both arms, {len(clean)} clean (no pause / invalid / cut); "
    f"every token equal across arms: {eq}; == REF2-T AR where a file exists: "
    f"{[(k, D[k][a]['ar_ok']) for k in both for a in ARMS if D[k][a]['ar_ok'] is not None]}")
for t in ("code", "math", "chat", "agent"):
    ks = [k for k in clean if D[k]["dfl"]["type"] == t]
    if ks:
        a = np.mean([D[k]["dfl"]["tok_s"] for k in ks]); b = np.mean([D[k]["mtp7"]["tok_s"] for k in ks])
        g = float(np.exp(np.mean([np.log(D[k]["mtp7"]["tok_s"] / D[k]["dfl"]["tok_s"]) for k in ks])))
        say(f"  {t:6s} ({len(ks)}): mean tok/s dfl {a:.1f} -> mtp7 {b:.1f} ({100 * (b / a - 1):+.1f} %), geo-mean ratio {g:.3f}, "
            f"mtp7 faster on {sum(D[k]['mtp7']['tok_s'] > D[k]['dfl']['tok_s'] for k in ks)}; kept per check "
            f"{np.mean([D[k]['dfl']['kept'] for k in ks]):.2f} of {np.mean([D[k]['dfl']['rows'] for k in ks]):.1f} rows -> "
            f"{np.mean([D[k]['mtp7']['kept'] for k in ks]):.2f} of 8")
if clean:
    tsum = lambda pick: sum(D[k]["dfl"]["n"] / D[k][pick(k)]["tok_s"] for k in clean)
    n = sum(D[k]["dfl"]["n"] for k in clean)
    t_d, t_m, t_s = tsum(lambda k: "dfl"), tsum(lambda k: "mtp7"), tsum(lambda k: "dfl" if D[k]["dfl"]["type"] == "code" else "mtp7")
    g = t_d / t_s - 1
    v = "FAIL (a token differs across arms)" if not eq else "PASS" if g >= 0.10 else "KILL" if g < 0.04 else "BETWEEN"
    say(f"  whole set ({len(clean)} prompts, {n} tokens): dfl (the default) {n / t_d:.1f} tok/s, mtp7 everywhere {n / t_m:.1f}, "
        f"the switch {n / t_s:.1f} = {100 * g:+.1f} % over the default -> D-23 verdict (PASS >= +10 %, KILL < +4 %): {v}"
        f"{'' if len(clean) >= 16 else ' (PARTIAL: fewer than 16 clean prompts)'}")
say(f"  footprint now {footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB; {time.time() - t0:.0f} s")
open(TXT_OUT, "a").write("\n".join(lines) + "\n")
