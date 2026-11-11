"""D-23c MTPTRUNC (MTP Head, Sun 11-01): D-23 with the MTP chain cut by the default's own running-confidence rule.
D-23 (data/d23_switchdef.txt) was a KILL (-2.7 %): the 45 GB default's DFlash2 arm verifies only the rows whose running
selector confidence stays >= 0.3 (L1-14, chat 3.9 rows), while mtp7 always verified 8. D-23b's CPU replay of D-23's own
checks (sticky/d23_trunc.py, data/d23_trunc.txt) put the MTP chain with the same cut at chat +28 % / agent +6 % over dfl
with all 7 steps drafted, chat +50 % / agent +7 % if the chain stops drafting at the cut; the switch (dfl on code + math,
MTP on chat + agent) +12 % / +19 % whole set. This run measures it.
Harness = d23_switchdef.py (a copy of Leap's k69_live.py, the 45 GB default R57_PRESET=45 + D-20's MTP chain), unchanged
except the MTP arms:
  dfl    the default as it ships (DFlash2 L1-20, 15 drafts cut where the running confidence < 0.3)
  mtp7e  MTP K7 with head b3, early exit: each step is evaluated, drafting stops at the first step whose running product
         of top-1 p falls below 0.3; the drafts kept are the leading ones at >= 0.3 (the dfl arm's rule)
  mtp7t  (optional, D23C_ARMS) all 7 steps drafted as in D-23, then the same cut on the verified rows
Prompts: D-23's 6 chat + 4 agent (code and math stay on dfl in the switch; their dfl times are D-23's, same harness and C).
PASS (pre-registered 11-01 before the run): the switch >= +10 % whole-set tok/s over dfl (D-23's code + math dfl runs + this
run's chat + agent dfl runs) with every token equal; KILL < +4 %; BETWEEN otherwise. Per type as information.
v2 (11-01 13:5x, still before any run; D-24, data/d24_read.txt): the MTP chain's cap matters once the cut trims the tail.
Teacher-forced on held-out answers (head b5), K15 + cut vs K7 + cut: code +14.5 %, gsm +4.2 %, mgsm +2.3 %, chat +0.1-0.6 %;
D-23's agent runs saturated K7 (edit02 7.03 of 8 kept). Leaf branches at uncertain depths: -5 .. +0.2 % (no tree arm).
So the run is: arms dfl and mtp15e (K15, the same cut and early exit), head b5 (>= b4 on every held-out set, = b3 on code;
D-22 b5), all 18 of D-23's prompts in one 30-min quiet ticket (D23_BUDGET 1500 s); every number from this run.
PASS (v2, pre-registered before the run): the switch S1 (dfl on code + math, mtp15e on chat + agent) >= +10 % whole-set
tok/s over dfl with every token equal; KILL < +4 %; BETWEEN otherwise. Reported too: S2 (dfl on code only, mtp15e on math /
chat / agent) and mtp15e everywhere, per type, and D-23's numbers on the same prompts (session drift).
v2b (NewIdeas 13:5x, before any run): on the CURRENT 45 GB default, R57_PRESET=45 with OOO + TIER_COMBINE on (BigMoE 3faa0dc,
EvalSuite R-56d exact), via BigMoE's routing-scores hook copied verbatim (r57_screen.py = Leap's k73_live.py); OOO calls are
logged after the warm-ups (0 = not engaged: then the run is the pre-OOO path and the result says so). D23_NEWDEF=0 = D-23's
preset. Runs carry pre = '45+ooo' / '45'; D-23's numbers (pre-OOO) are shown beside, not used.
Usage: [BigMoE venv] D23_KEYS=k1,k2,... d23c_mtptrunc.py [C=224] [CAP=256]  (D23_HEAD=b5, D23C_ARMS=dfl,mtp15e default;
D23_BUDGET / D23_HARD seconds of running time, default 680 / 740; D23_ORDER=sep: each arm its own pass over the prompts,
the first arm alternating within each set, instead of the arms back to back on each prompt)
Resumable: one line per (key, arm, head, C) in ~/TrainLab/data/d23c_runs.jsonl; done ones are skipped."""
import json, os, re, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.pop("BIGMOE_MTP_STREAM", None)              # the MTP head resident, as in D-20
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["TL_INFER"] = "1"                           # mtp125: keep MTPLX's fused HC read
os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"   # k69_live R57_PRESET=45 (REF2-T)
os.environ.setdefault("BIGMOE_TIER_CL", "112")
NEWDEF = os.environ.get("D23_NEWDEF", "1") == "1"      # v2b (NewIdeas 13:5x): the CURRENT 45 GB default = R57_PRESET=45 with
if NEWDEF:                                             # OOO + TIER_COMBINE (BigMoE 3faa0dc, EvalSuite R-56d exact), engaged
    os.environ["BIGMOE_TIER_COMBINE"] = "1"; os.environ["BIGMOE_OOO"] = "1"   # by the scores hook below (R-69D)
else:                                                  # D23_NEWDEF=0: D-23's pre-OOO preset
    os.environ.pop("BIGMOE_TIER_COMBINE", None); os.environ.pop("BIGMOE_OOO", None)
PRE = "45+ooo" if NEWDEF else "45"
os.environ.setdefault("BIGMOE_MAXK", "160")
os.environ.setdefault("BIGMOE_EXT_BUILD", "build_next")   # as Leap's R-69L / R-69b runners
assert os.environ.get("BIGMOE_MAXK") == "160", "see Usage"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF
from tokenizers import Tokenizer

TD = os.path.expanduser("~/TrainLab/data")
OUT = os.environ.get("D23_RUNS", os.path.join(TD, "d23c_runs.jsonl"))              # D-23x: its own files
TXT_OUT = os.environ.get("D23_TXT", os.path.join(TD, "d23c_mtptrunc.txt"))
TRACE = bool(os.environ.get("D23_TRACE")) and bool(os.environ.get("BIGMOE_TRACE"))   # D-23x: reads of the first checks
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 224
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
HEAD = os.environ.get("D23_HEAD", "b5")
ARMS = tuple(os.environ.get("D23C_ARMS", "dfl,mtp15e").split(","))
TRUNC = 0.3
R, K = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = os.environ.get("D23_KEYS", "ifeval/2853,agent/bash18,multi/bg1").split(",")
BUDGET, HARD = int(os.environ.get("D23_BUDGET", "680")), int(os.environ.get("D23_HARD", "740"))   # s of running time: no new run / no new check after these
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
if NEWDEF:                                               # BigMoE's scores hook, verbatim from ~/BigMoE/scripts/r57_screen.py (= Leap's
    from mtplx.models import qwen4_exp as _QX            # k73_live.py): EM.load's block does the sorted top-k and MLX's combine itself,
    _ev_block = _QX.SparseMoeBlock.__call__              # so tier_call never got the scores and TIER_COMBINE / OOO never engaged; this
                                                         # hands the combine to tier_call at 1-16 rows (R-69: bitwise == MLX's combine)
    def _fused_block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if (bm is None or "tier" not in bm or not EM.STATE.sorted or not (_BT._COMBINE and _BT._GLU == "3c")
                or x.ndim < 2 or int(np.prod(x.shape[:-1])) > 16):
            return _ev_block(self, x)                    # prefill, the resident MTP head (no _bm) and every other call: as is
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        if EM.STATE.capture is not None:
            EM.STATE.capture.setdefault(bm["L"], []).append(inds)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)   # built first: OOO's spins wait behind it
        bm["tier"]["scores"], bm["tier"]["shared"] = scores, shared
        try:
            y = self.switch_mlp(x, inds)
        finally:
            bm["tier"].pop("scores", None); bm["tier"].pop("shared", None)
        if y.ndim == inds.ndim + 1:                      # a path that left the scores unused: MLX's combine, as EvalSuite
            y = (y * scores[..., None]).sum(axis=-2)
        return y + shared
    _QX.SparseMoeBlock.__call__ = _fused_block
assert BG.MAXK == 160 and BG.XLINES == 4, (BG.MAXK, BG.XLINES)
TXT = model.language_model.model
lay = TXT.layers
SW_ = [l.mlp.switch_mlp for l in lay]
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/TrainLab_root"))
import mtp125 as M
T_ = M.Trainable(model.language_model.mtp)
T_.load_weights(os.path.expanduser(f"~/TrainLab/ckpt/d22_{HEAD}.safetensors"))
M.install(model.language_model.mtp, T_, dtype=mx.bfloat16)
say(f"D-23c MTPTRUNC: the 45 GB default (REF2-T tiers {os.environ['BIGMOE_TIER']} / {os.environ['BIGMOE_TIER_GLU']}, C_l "
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

def trsum(rd):
    """D-23x: one trace_reads() batch ([L, e, tag 0 demand / 1 prefetch / 2 joint, t0, t1]) -> counts by pool (base MSB-3 =
    ext layer L < side0, LSB-1 sidecar = side0 + L) and tag, read seconds, wall span, and per-layer base / sidecar counts"""
    S0 = getattr(st, "_side0", None) or 10 ** 9
    o = [0] * 6 + [0.0, 0.0]
    NL_ = S0 if S0 < 1000 else 64
    lb, ls = [0] * NL_, [0] * NL_
    t0s, t1s = [], []
    for L, e, tag, t0, t1 in rd:
        side = int(L) >= S0
        o[(3 if side else 0) + min(int(tag), 2)] += 1
        o[6] += t1 - t0; t0s.append(t0); t1s.append(t1)
        Lr = int(L) - (S0 if side else 0)
        if 0 <= Lr < NL_:
            (ls if side else lb)[Lr] += 1
    o[7] = (max(t1s) - min(t0s)) if t0s else 0.0
    return [o[0], o[1], o[2], o[3], o[4], o[5], round(o[6], 4), round(o[7], 4), lb, ls]

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

    def draft_cut(self, k, t):
        """D-23c early exit: evaluate each step and stop at the first one whose running product of top-1 p is < t (that
        step's p is still logged); returns the leading drafts at >= t (draft()'s tokens and the dfl arm's cut rule)"""
        l0 = self.logits[0, -1].astype(mx.float32)
        tk, p = mx.argmax(l0), mx.softmax(l0).max()
        mx.eval(tk, p)
        toks, ps, h, run, n_fw = [int(tk.item())], [float(p.item())], self.h, float(p.item()), 0
        while run >= t and len(toks) < k:
            lgx, h = model.mtp_forward(h, mx.array([[toks[-1]]]), mtp_cache=self.cache, return_hidden=True)
            n_fw += 1
            lx = lgx[0, -1].astype(mx.float32)
            tk, p = mx.argmax(lx), mx.softmax(lx).max()
            mx.eval(tk, p)
            toks.append(int(tk.item())); ps.append(float(p.item())); run *= ps[-1]
        if n_fw:
            self.cache[0].trim(n_fw)
        return toks[:int((np.cumprod(np.array(ps)) >= t).sum())], ps

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
    KM = int(re.match(r"mtp(\d+)", arm).group(1)) if mtp else 0
    MODE = arm[len(f"mtp{KM}"):] if mtp else ""            # '' = D-23's chain, 't' = cut rows, 'e' = early exit
    ref2.reset_counts()
    if TRACE:
        BG.ext.trace_reads(); BG.ext.trace(); BG.ext.trace_cov()   # D-23x: drop the trace before this prompt
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
    trc = [["pf"] + trsum(BG.ext.trace_reads())] if TRACE else None   # D-23x: reads since the last call (prefill + history)
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
            if MODE == "e":
                dr, cf = head.draft_cut(KM, TRUNC)
            else:
                dr, cf = head.draft(KM)
                if MODE == "t":
                    dr = dr[:int((np.cumprod(np.array(cf)) >= TRUNC).sum())]
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
        if TRACE and len(kept) <= 6:                     # D-23x: this check's reads (outside the timed window)
            trc.append([len(kept)] + trsum(BG.ext.trace_reads()))
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
                pc=[[int(r_), int(k_), round(1e3 * c_, 2), round(1e3 * d_, 2), int(m_[0]), round(1e3 * m_[1], 2)]   # + misses, io ms
                    for r_, k_, c_, d_, m_ in zip(rows, kept, ck_t, dr_t, sm.tolist() if len(sm) == len(rows) else [[0, 0]] * len(rows))],
                cf=cfs, **({"tr": trc} if TRACE else {}))

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
        if not (d["cut"] or d["paused"] or d["invalid"]) and d.get("pre", "45") == PRE:   # unclean: re-run; other preset: not ours
            done[(d["key"], d["arm"], d["head"], d["C"])] = d
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

it0 = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
for arm in ARMS:                                         # warm-up, discarded (S45-T's method)
    r = run(list(it0["prompt_ids"]), arm, 64)
    say(f"  warm-up (discarded): gsm8k/708 [{arm}] {r['n']} tokens, {r['tok_s']:.1f} tok/s{r['bad']}")
say(f"  preset {PRE}: TIER_COMBINE {_BT._COMBINE}, OOO {_BT._OOO}, OOO calls after the warm-ups {_BT.OOO_STATS['calls']}"
    f"{' (NOT ENGAGED: the run is the pre-OOO path, bitwise equal per R-69)' if NEWDEF and not _BT.OOO_STATS['calls'] else ''}")
if os.environ.get("D23_ORDER") == "sep":                # D-23d: the second arm on a prompt decodes the same tokens on the experts
    nth, jt = {}, {}                                     # the first just loaded (17 of D-23c''s 18 prompts: fewer misses, ~12-20 %
    for key in PICK:                                     # faster), so each arm gets its own pass over the set, the first arm
        s_ = key.split("/")[0]; jt[key] = nth.get(s_, 0); nth[s_] = jt[key] + 1   # alternating within each set
    SCHED = [(key, ARMS[(jt[key] + p) % len(ARMS)]) for p in range(len(ARMS)) for key in PICK]
else:
    SCHED = [(key, arm) for n_p, key in enumerate(PICK) for arm in (ARMS if n_p % 2 == 0 else ARMS[::-1])]
for key, arm in SCHED:
    it = items[key]
    typ = rule(it)
    if (key, arm, HEAD, C) in done:
        continue
    if el() > BUDGET:
        say(f"  time budget: stopped before {key} [{arm}] (resumable)")
        break
    r = run(list(it["prompt_ids"]), arm, CAP)
    want = ar_toks(key)
    ar_ok = None if want is None else same(r["out"], want) >= min(len(want), r["n"])
    rec = dict(key=key, arm=arm, head=HEAD, C=C, pre=PRE, type=typ, ar_ok=ar_ok, paused="PAUSED" in r["bad"],
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

# ---------------------------------------------------------------- read-out over everything saved at this C and head (v2)
D = {}
for d in done.values():
    if d["head"] == HEAD and d["C"] == C and d["key"] in PICK:
        D.setdefault(d["key"], {})[d["arm"]] = d
both = [k for k in PICK if all(a in D.get(k, {}) for a in ARMS)]
clean = [k for k in both if not any(D[k][a]["paused"] or D[k][a]["invalid"] or D[k][a]["cut"] for a in ARMS)]
eq = all(D[k][a]["out"] == D[k]["dfl"]["out"] for k in both for a in ARMS)
say(f"D-23c read-out (C {C}, head {HEAD}, cut t {TRUNC}): {len(both)} prompts with all arms, {len(clean)} clean; every token "
    f"equal across arms: {eq}")
D23 = {}                                                 # D-23's runs (same harness and C; head b3): drift check
for l in open(os.path.join(TD, "d23_runs.jsonl")):
    d = json.loads(l)
    if d["C"] == C and not (d["cut"] or d["paused"] or d["invalid"]):
        D23.setdefault(d["key"], {})[d["arm"]] = d
MT = [a for a in ARMS if a != "dfl"]
for t in ("code", "math", "chat", "agent"):
    ks = [k for k in clean if D[k]["dfl"]["type"] == t]
    if not ks:
        continue
    a = np.mean([D[k]["dfl"]["tok_s"] for k in ks])
    for m in MT:
        b = np.mean([D[k][m]["tok_s"] for k in ks])
        g = float(np.exp(np.mean([np.log(D[k][m]["tok_s"] / D[k]["dfl"]["tok_s"]) for k in ks])))
        say(f"  {t:6s} ({len(ks)}): mean tok/s dfl {a:.1f} -> {m} {b:.1f} ({100 * (b / a - 1):+.1f} %), geo-mean ratio {g:.3f}, "
            f"{m} faster on {sum(D[k][m]['tok_s'] > D[k]['dfl']['tok_s'] for k in ks)}; kept {np.mean([D[k]['dfl']['kept'] for k in ks]):.2f} "
            f"of {np.mean([D[k]['dfl']['rows'] for k in ks]):.1f} rows -> {np.mean([D[k][m]['kept'] for k in ks]):.2f} of "
            f"{np.mean([D[k][m]['rows'] for k in ks]):.1f}; draft {np.mean([D[k]['dfl']['dr'] for k in ks]):.1f} -> "
            f"{np.mean([D[k][m]['dr'] for k in ks]):.1f} ms, check {np.mean([D[k]['dfl']['ck'] for k in ks]):.1f} -> "
            f"{np.mean([D[k][m]['ck'] for k in ks]):.1f} ms (medians per prompt)")
    if all({"dfl", "mtp7"} <= set(D23.get(k, {})) for k in ks):
        say(f"           D-23 (b3, pre-OOO preset, 8-row mtp7, same prompts): dfl {np.mean([D23[k]['dfl']['tok_s'] for k in ks]):.1f}, "
            f"mtp7 {np.mean([D23[k]['mtp7']['tok_s'] for k in ks]):.1f}")
if clean:
    n = sum(D[k]["dfl"]["n"] for k in clean)
    tt = lambda pick: sum(D[k]["dfl"]["n"] / D[k][pick(k)]["tok_s"] for k in clean)
    t_d = tt(lambda k: "dfl")
    for m in MT:
        typ = lambda k: D[k]["dfl"]["type"]
        t_s1 = tt(lambda k: "dfl" if typ(k) in ("code", "math") else m)
        t_s2 = tt(lambda k: "dfl" if typ(k) == "code" else m)
        t_m = tt(lambda k: m)
        g = t_d / t_s1 - 1
        v = "FAIL (a token differs across arms)" if not eq else "PASS" if g >= 0.10 else "KILL" if g < 0.04 else "BETWEEN"
        say(f"  whole set ({len(clean)} prompts, {n} tokens): dfl (the default) {n / t_d:.1f} tok/s; S1 (dfl on code + math, {m} "
            f"on chat + agent) {n / t_s1:.1f} = {100 * g:+.1f} %; S2 (dfl on code only) {n / t_s2:.1f} = {100 * (t_d / t_s2 - 1):+.1f} %; "
            f"{m} everywhere {n / t_m:.1f} = {100 * (t_d / t_m - 1):+.1f} % -> D-23c verdict on S1 (PASS >= +10 %, KILL < +4 %): {v}"
            f"{'' if len(clean) >= 16 else ' (PARTIAL: fewer than 16 clean prompts)'}")
say(f"  preset {PRE}: OOO calls {_BT.OOO_STATS['calls']}; footprint now {footprint_now_gb():.1f} GB, lifetime peak "
    f"{footprint_peak_gb():.1f} GB; {time.time() - t0:.0f} s")
open(TXT_OUT, "a").write("\n".join(lines) + "\n")
