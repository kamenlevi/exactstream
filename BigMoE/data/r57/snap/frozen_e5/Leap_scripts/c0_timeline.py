"""C0 / NEW-2 TIMELINE (Leap, Mon 2026-11-02; ROUTE_350 block C step C0, DAY9 condition 5): a per-layer GPU timeline
of the 45 GB default's 16-row check, measured in the live loop. A copy of BigMoE's scripts/r57_screen.py at e3210bf
(read-only use; outputs to ~/Leap/data/c0) with probes at the checks in C0_AT (1-based; default 1,2,5,10,15,30). At a
probe check, after the real forward (whose logits decide the tokens as usual), the cache is rolled back
(mtplx rollback_after_verify) and the check is re-run:
- warm repeats of the same rows (all experts now resident), unsynced as live (SB): warm wall and graph-build ms;
- one synced pass: every layer's pieces evaluated in turn (hc_a = the attention HC read, gdn / att = the mixer, hc_m = the
  MLP HC read, moe_pre = router + shared expert, experts = the tier GLU call with its combine, moe_post = the tail,
  hc_w = the residual write, ple), each split into CPU build ms and eval ms (GPU + one sync, calibrated);
- a row sweep (C0_SWEEP rows of the untruncated draft path: a cold pass, then two warm passes) and synced passes at 1
  and 16 rows, for the per-row slope per piece;
- a last forward of the real rows, whose logits must equal the real check's bitwise (then the commit runs as usual).
Probe time and probe misses are taken out of the run's cycle, check and SSD numbers. Every token vs REF2-T as usual.
Usage: [BigMoE venv] R57_PRESET=45 BIGMOE_EXT_BUILD=build_next R57_REPS=1 R57_WARM=1 R57_KEYS=humaneval/HumanEval/68,
       multi/bg0 c0_timeline.py C CAP -> ~/Leap/data/c0/<tag>_screen.txt + <tag>_probes.json
Original docstring follows.
R-57 PREFETCH16 (NewIdeas Fri 2026-10-30 00:55, from Leap's S45; with R-55): the exact live REF2 loop at C 192 misses
413-779 experts per 16-row check (111-194 ms of SSD reads per check, serial with the ~185 ms warm compute). With R-55's
wide gate the 16-row check takes the sync-free gate, so BigMoE's look-ahead predictor can ride on it: at layer L the GPU
applies layer L+1's router to layer L's MoE input (qwen4_exp: as is), ranks the experts by their best row log-prob
(BIGMOE_PF_RANKED, one unique list for the 16 rows) and posts the top-B with L's ids; the loader reserves and reads the
non-resident ones (PF_JIT, PF_QD 4, up to PF_VICTIM in flight per layer) while L computes. Reads only: bits unchanged.
EvalSuite's REF env refuses BIGMOE_PREFETCH, so the predictors are attached here after the load (stream.py's own
wiring, ratio None), and switched per arm.
Harness = Leap's S45 (/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts/r40s3k_screen.py speed mode): EvalSuite's REF harness, C 192, PF1 for prefills only,
REF2 + SB, DFlash2 L1-20 drafting live (greedy, keep = accepted + 1), up to CAP tokens per run.
Arms (one process, BIGMOE_MAXK=160, BIGMOE_PRED_LINES=4): slot = the shipped exact path (16-row calls on _sync_call, Leap's
slot call); gate = R-55 (stream.GATE_N 160); gate+pf = R-55 + the look-ahead prefetch (B = PF_B ranked ids for L+1).
HumanEval/68 slot gate gate+pf gate+pf gate slot, prealgebra/1640 the reverse.
Per run: tok/s, median cycle, check + commit, misses and demand SSD ms (the loader's io_s) per check, tokens vs
EvalSuite's REF2 row and vs the other arms; since C45 (NewIdeas 11-01 11:5x) also the gate service of calls with misses
(slow_s: post seen -> release, the GPU's spin bound; OOO overlaps part of it) and the drafter's ms (the SSD idles); since
NewIdeas 11-01 15:3x (M-17 LAYSZ45) the whole loop's SSD MB (ri_diskio_bytesread) and misses per kept token.
Lines (NewIdeas 00:55): R-57 PASS = demand SSD ms per check -50 % or better AND check + commit -25 % or better (gate+pf vs
gate, medians over the prompts), every token == REF2; KILL = the cycle under -10 %. R-55 in the live loop (gate vs slot):
information.
Usage: [BigMoE venv] BIGMOE_MAXK=160 BIGMOE_PRED_LINES=4 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PF_VICTIM=24
       BIGMOE_EXT_BUILD=build_next r57_screen.py [C=192] [CAP=256] [PF_B=96] -> data/r57/r57_screen.txt"""
import ctypes, json, os, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.setdefault("HF_HUB_OFFLINE", "1")
> (a quote from the project owner, kept private)
if CHATSW:                                               # ~/TrainLab/INTEGRATION_CHATSWITCH.md): prompts the type rule calls chat
    os.environ["TL_INFER"] = "1"                         # draft on the 125B's MTP head (d22 b5, K15, cut t, early exit); code /
    os.environ.pop("BIGMOE_MTP_STREAM", None)            # math / agent stay on DFlash2. Head resident (D-20 / D-23d); TL_INFER:
                                                         # mtp125 keeps MTPLX's fused HC read (EvalSuite's env pins the rest)
if os.environ.get("R57_PRESET") == "45":                 # THE 45 GB DEFAULT (NewIdeas 11-01 06:1x): REF2-T tiers 4433333333 / 3c
    os.environ.setdefault("R57_TIER", "1")               # + the R-55 gate (MAXK default 160) + L1-14 truncation t 0.3 + ref2's
    os.environ.setdefault("R57_ARMS", "gate+t")          # defaults (REF2b key buckets, GDNX); REF2b 4-bit exact = R57_PRESET unset
    os.environ.setdefault("R57_TRUNC", "0.3")
    os.environ.setdefault("R57_COMBINE", "1")            # + OOO + TIER_COMBINE (NewIdeas 11-01 11:4x on EvalSuite's R-56d PASS,
    os.environ.setdefault("BIGMOE_TIER_COMBINE", "1")    # exact); the no-OOO control: R57_COMBINE=0 BIGMOE_OOO=0
    os.environ.setdefault("BIGMOE_OOO", "1")
    # NewIdeas 11-02 01:3x on C45 (all exact): ADOPT MoonShots' M-17 per-layer pool sizes (L: -17.2 % SSD MB per kept token,
    # +8.3 % steady at C 224) from the md5-pinned data/seed45/laysz45_<C>.json (uniform pools at other C; R57_LAYSZ=0 = the
    # uniform control) and ADOPT Leap's K-73 exact-safe stack (+2.4 %; off with R57_COMBINE=0, which it needs).
    _LS = {"224": "3098a50550e27a1ebb2075e3caf86960", "256": "ee76e2a7390c18f8d53b539a3ba9700a",
           "300": "cd0df64a980d217d75a53a10d600ddce", "320": "f0a3997c88379d01cdc2b39175c39307"}
    _C = sys.argv[1] if len(sys.argv) > 1 else "256"     # the 45 GB default's C since 11-02 (41.7 GB lifetime, C50)
    if os.environ.get("R57_LAYSZ", "1") == "1" and "BIGMOE_TIER_SIZES" not in os.environ and _C in _LS:
        import hashlib
        _p = os.path.expanduser(f"~/BigMoE/data/seed45/laysz45_{_C}.json")
        assert hashlib.md5(open(_p, "rb").read()).hexdigest() == _LS[_C], f"{_p}: md5 differs from the pinned one"
        os.environ["BIGMOE_TIER_SIZES"] = _p
    if os.environ.get("R57_COMBINE") == "1":
        os.environ.setdefault("K73_EXACT", "1")
TIER = os.environ.get("R57_TIER") == "1"                 # S45-T (NewIdeas 18:2x): REF2-T on the gate at C's RAM (M-08 sizing)
if TIER:
    os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"
    os.environ.setdefault("BIGMOE_TIER_CL", "112")       # C_m follows from C's 4-bit bytes (stream.py's budget split)
    if os.environ.get("R57_COMBINE") != "1":             # R-69 (Leap): TIER_COMBINE (+ BIGMOE_OOO) is bitwise == this path;
        os.environ.pop("BIGMOE_TIER_COMBINE", None)      # R57_COMBINE=1 keeps it for the OOO arm
_PF_ARM = "gate+pf" in os.environ.get("R57_ARMS", "slot,gate,gate+pf")
os.environ.setdefault("BIGMOE_MAXK", "160")             # the default since R-55's adoption (gated.py); explicit for the asserts
assert os.environ.get("BIGMOE_MAXK") == "160", "see Usage"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
if os.environ.get("R57_SPARE"):                          # R-57b: reserved prefetch slots per layer (EvalSuite forces 0); the
    os.environ["BIGMOE_SPARE"] = os.environ["R57_SPARE"]  # ext reads it when bigmoe loads (EM.load), so set it here
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

OUT = os.path.expanduser("~/Leap/data/c0"); os.makedirs(OUT, exist_ok=True)
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else (256 if os.environ.get("R57_PRESET") == "45" else 192)
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
PF_B = int(sys.argv[3]) if len(sys.argv) > 3 else 96
R, K = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = (("humaneval", "humaneval/HumanEval/68"), ("math", "math500/test/prealgebra/1640.json"),
        ("humaneval", "humaneval/HumanEval/91"), ("math", "math500/test/algebra/694.json"),        # S45's other prompts
        ("math", "math500/test/algebra/1282.json"), ("math", "gsm8k/708"))
# R-55L (the orchestrator 17:0x: headline the steady-state cycle; a clean slot re-run settles code): R57_ARMS=slot,gate
# R57_PROMPTS=6 R57_TAG=r55l runs only those arms on S45's six prompts and writes data/r57/r55l_screen.txt
ARMS_ = tuple(os.environ.get("R57_ARMS", "slot,gate,gate+pf").split(","))
PICK = PICK[:int(os.environ.get("R57_PROMPTS", "2"))]
SETOF = {"humaneval": "humaneval", "math500": "math", "gsm8k": "math", "ifeval": "ifeval", "multi": "multi", "agent": "agent"}
if os.environ.get("R57_KEYS"):                           # an explicit prompt list ("set/key,..."; e.g. D-23d's chat prompts)
    PICK = tuple((SETOF[k.split("/")[0]], k) for k in os.environ["R57_KEYS"].split(","))
TAG = os.environ.get("R57_TAG", "r57")
REPS = int(os.environ.get("R57_REPS", "2"))
TRUNC = float(os.environ.get("R57_TRUNC", "0.3"))
S10B_DIR = os.environ.get("R57_S10B_DIR")               # with BIGMOE_S10B=1 (R-57b data for Fit12): per run the gated calls'
if S10B_DIR:                                            # ids + MoE inputs (bigmoe/s10b.py) -> <dir>/<prompt>_<arm>_<n>.npz
    os.makedirs(S10B_DIR, exist_ok=True)        # arm gate+t: drafts cut where the running confidence < TRUNC              # runs per arm per prompt: 2 = ABBA order, 1 = one each
HARD = 600                                               # s of running time: no new check after this (job stop at 660)
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

t0 = time.time()
if CHATSW:                                               # EvalSuite's load asks for no MTP head: load it resident (d23c_mtptrunc)
    import bigmoe.loader as BL
    _ls = BL.load_streamed
    BL.load_streamed = lambda md, C_, mtp=False: _ls(md, C_, mtp=True)
rt, st = EM.load(EM.variant("ref"), C=C)
if CHATSW:
    BL.load_streamed = _ls
model = rt.model
import ref2
import bigmoe.stream as BS
import bigmoe.gated as BG
from mtplx import cache_state as CS
Q = ref2.Q
assert BG.MAXK == 160 and BG.XLINES == 4 and (not _PF_ARM or PF_B <= 30 + 31 * BS._PRED_LINES), (PF_B, BS._PRED_LINES)
if TIER and os.environ.get("R57_COMBINE") == "1":       # R-69 (Leap R-69L/R-69D): EM.load's block does the sorted top-k
    import bigmoe.tier as _BT                            # and MLX's combine itself, so tier_call never got the scores and
    from mtplx.models import qwen4_exp as _QX            # TIER_COMBINE / OOO never engaged. This copy of EvalSuite's
    _ev_block = _QX.SparseMoeBlock.__call__              # sorted block (evalsuite/model.py block) hands the combine to
                                                         # tier_call at 1-16 rows (R-69: bitwise == MLX's combine there)
    def _fused_block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if (bm is None or "tier" not in bm or not EM.STATE.sorted or not (_BT._COMBINE and _BT._GLU == "3c")
                or x.ndim < 2 or int(np.prod(x.shape[:-1])) > 16):
            return _ev_block(self, x)                    # prefill and every other call: EvalSuite's block as is
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
if os.environ.get("K73_EXACT") == "1":                  # K-73 (Leap 11-01, /Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts/k73_live.py, verbatim): the exact-safe
    assert TIER and os.environ.get("R57_COMBINE") == "1", "K73_EXACT runs on R57_PRESET=45"   # stack on the 45 GB default:
    from router_fused import router_fused                # ref2 att_rope + idx_prep (R-72), hc_write (R-73); the MoE block with
    from glue_fused import moe_tail, shared_cat          # R-70's fused router and R-73's fused tail (y + sigmoid(g) * s);
                                                         # + R-74 (Leap 12:1x): the shared expert's gate / up + its gate as one call
    for _k in ("att_rope", "idx_prep", "hc_write"):
        ref2.CFG[_k] = True
    import draft_fused                                   # R-76 (Leap 12:3x): DFlash2's grouped conv as one kernel + the mask
    draft_fused.install()                                # memoized (drafts bitwise identical; 649 -> 364 kernels per propose)
    _fb = _QX.SparseMoeBlock.__call__
    K73N = [0]
    def _exact_block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if (bm is None or "tier" not in bm or not EM.STATE.sorted or not (_BT._COMBINE and _BT._GLU == "3c")
                or x.ndim < 2 or int(np.prod(x.shape[:-1])) > 16 or not self.norm_topk_prob):
            return _fb(self, x)
        K73N[0] += 1
        inds, scores = router_fused(self.gate(x), self.top_k)   # R-70: bitwise == softmax .. argsort .. take chain
        if EM.STATE.capture is not None:
            EM.STATE.capture.setdefault(bm["L"], []).append(inds)
        g, s = shared_cat(self, x)                              # R-74: == shared_expert_gate(x), shared_expert(x), bitwise
        bm["tier"]["scores"], bm["tier"]["shared"] = scores, s   # OOO's spins wait behind the shared expert, as before
        try:
            y = self.switch_mlp(x, inds)
        finally:
            bm["tier"].pop("scores", None); bm["tier"].pop("shared", None)
        if y.ndim == inds.ndim + 1:
            y = (y * scores[..., None]).sum(axis=-2)
        return moe_tail(y, g, s, exp="precise")                  # R-73: == y + sigmoid(g) * s, bitwise
    _QX.SparseMoeBlock.__call__ = _exact_block
MTP_HEAD, MTP_K = os.environ.get("R57_MTP_HEAD", "b5"), int(os.environ.get("R57_MTP_K", "15"))
_MTP_MD5 = {"b5": "1f8de68a468b6ae16a9f42c62cbca39f"}   # D-23d's head, pinned like the M-17 sizes
if CHATSW:                                               # the head's trained parts over the shipped MTP layer (D-22 b5: attention,
    import hashlib                                       # router, shared expert; experts, norms, lm_head as shipped)
    assert getattr(model.language_model, "mtp", None) is not None, "no MTP head loaded"
    _hp = os.path.expanduser(f"~/TrainLab/ckpt/d22_{MTP_HEAD}.safetensors")
    assert MTP_HEAD not in _MTP_MD5 or hashlib.md5(open(_hp, "rb").read()).hexdigest() == _MTP_MD5[MTP_HEAD], f"{_hp}: md5"
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/TrainLab_root"))
    import mtp125 as _M                                  # its import-time env defaults = EvalSuite's forced ones (no-ops here)
    _TH = _M.Trainable(model.language_model.mtp)
    _TH.load_weights(_hp)
    _M.install(model.language_model.mtp, _TH, dtype=mx.bfloat16)   # touches the head's own modules only
    del _TH
    import re
    from tokenizers import Tokenizer                     # NewIdeas' type rule (TrainLab scripts/type_rule.py, verbatim from
    _TKZ = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))   # d23c_mtptrunc.py rule())
    _CODE = re.compile(r"(^|\n)\s*(def |class |import |from \w+ import|#include|fn |func |public |const |let |var )|```|\breturn\b|\w+\(.*\)\s*:|>>>")
    _MATH = re.compile(r"\\boxed|\\frac|\\sqrt|\$[^$]+\$|\b(solve|compute|calculate|how many|how much|what is the (sum|product|value|area|probability))\b"
                       r"|\d+\s*[-+*/^=]\s*\d+", re.I)
    _TOOLS = re.compile(r"<tools>|\"type\":\s*\"function\"|tool_call|<function|\bfunctions?\b.*\bparameters\b", re.I | re.S)
def rule(item):
    full = item.get("prompt") or _TKZ.decode(item["prompt_ids"], skip_special_tokens=False)
    user = item.get("user") or full
    if _TOOLS.search(full):
        return "agent"
    if _CODE.search(user) or _CODE.search(full[-4000:]):
        return "code"
    if _MATH.search(user):
        return "math"
    return "chat"
TXT = model.language_model.model
lay = TXT.layers
NL = len(lay)
SW_ = [l.mlp.switch_mlp for l in lay]
class _Corr:                                             # R-57b: Fit12's logit correction on the L+1 router (raw @ W + b),
    def __init__(self, r):                               # a callable in the router's place, so stream.py stays unchanged
        self.r, self.W, self.b = r, None, None
    def __call__(self, x):
        lg = self.r(x)
        return lg if self.W is None else lg.astype(mx.float32) @ self.W + self.b
CORR = {L: _Corr(lay[L + 1].mlp.gate) for L in range(NL - 1)}
NEXT = {L: [(CORR[L] if os.environ.get("R57_CORR") else lay[L + 1].mlp.gate, None, PF_B)] for L in range(NL - 1)
        if not SW_[L]._bm["full"] and not SW_[L + 1]._bm["full"]}   # stream.py's wiring for qwen4_exp (ratio None)
def corr_for(key):
    """Fit12's held-out 2-fold maps: each prompt gets the map fit on the other three (R57_CORR = the maps' directory)"""
    if not os.environ.get("R57_CORR"):
        return None
    d = os.environ["R57_CORR"]
    f = "corr_fit68_91_1282.npz" if any(k in key for k in ("1640", "694", "708")) else "corr_fit1640_694_708.npz"
    z = np.load(os.path.join(d, f))
    for L, c in CORR.items():
        T = L + 1
        c.W = mx.array(z[f"W{T}"].astype(np.float32)); c.b = mx.array(z[f"b{T}"].astype(np.float32))
    mx.eval([c.W for c in CORR.values()] + [c.b for c in CORR.values()])
    return f
say(f"R-57 harness{' (REF2-T tiers ' + os.environ['BIGMOE_TIER'] + ', C_l ' + os.environ['BIGMOE_TIER_CL'] + ')' if TIER else ''}: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; REF2 + SB; gate rows MAXK {BG.MAXK} (ext {BG._BUILD}), PRED_LINES {BS._PRED_LINES} (ranked list <= "
    f"{30 + 31 * BS._PRED_LINES}), PF_B {PF_B}, PF_JIT {os.environ.get('BIGMOE_PF_JIT')}, PF_QD {os.environ.get('BIGMOE_PF_QD')}, "
    f"PF_VICTIM {os.environ.get('BIGMOE_PF_VICTIM')}; {len(NEXT)} look-ahead layers; CAP {CAP}")

PRIOR = os.environ.get("R57_PRIOR")                      # SEED45: BIGMOE_EVICT_PRIOR's popularity (data/seed45/prior45.npz),
def prior_for(key):                                      # each prompt held out (its "loo_" map: built on the other five)
    if not PRIOR:
        return None
    z = np.load(os.path.expanduser(PRIOR))
    k = "loo_" + key.replace("/", "_").replace(".json", "")
    k = k if k in z.files else "all"
    for L, sw in enumerate(SW_):
        if not sw._bm["full"]:
            BG.ext.set_prior(sw._bm["L"], z[k][L].astype(np.float32).tolist())
    return k

def set_arm(arm):
    BS.GATE_N[0] = 40 if arm == "slot" else 160
    BS._PF_RANKED = arm == "gate+pf"
    for L, sw in enumerate(SW_):
        if arm == "gate+pf" and L in NEXT:
            sw._bm["next"] = NEXT[L]
        else:
            sw._bm.pop("next", None)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def stats():                                             # misses, demand SSD busy (load batches' wall time), gate service of
    s = st.stats()                                       # calls with misses (post seen -> release: the GPU's spin bound;
    return np.array([s["misses"], s["io_s"], s["slow_s"], s["slow_calls"]], dtype=np.float64)   # OOO overlaps part of it)

class _RU4(ctypes.Structure):                            # rusage_info_v4: a 16-byte uuid, then uint64 fields
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]
_LIBC = ctypes.CDLL(None)

def disk_read_bytes():                                   # ri_diskio_bytesread (u16): this process's SSD reads. The ext reads
    ru = _RU4()                                          # F_NOCACHE, so every load counts (checked 11-01: 210 MB -> +209.7 MB)
    return ru.u16 if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else 0

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

class MTPHead:                                           # the chat switch's drafter: D-20's native MTP chain + D-23c's early exit
    def __init__(self):                                  # (/Users/kamen/BigMoE/data/r57/snap/frozen_e5/TrainLab_sticky/d23c_mtptrunc.py, arm mtp15e, verbatim)
        self.cache = model.make_mtp_cache()
        self.logits = self.h = None

    def append(self, hidden, next_ids):
        """commit pairs (hidden row i, next_ids[i]); the last one runs through mtp_forward for the next draft"""
        if hidden.shape[1] > 1:
            model.mtp_update_cache(hidden[:, :-1], mx.array([next_ids[:-1]]), mtp_cache=self.cache)
        self.logits, self.h = model.mtp_forward(hidden[:, -1:], mx.array([next_ids[-1:]]), mtp_cache=self.cache,
                                                return_hidden=True)

    def draft_cut(self, k, t):
        """evaluate each step and stop at the first one whose running product of top-1 p is < t; returns the leading
        drafts at >= t (the DFlash2 path's cut rule) and every evaluated step's p"""
        l0 = self.logits[0, -1].astype(mx.float32)
        tk, p = mx.argmax(l0), mx.softmax(l0).max()
        mx.eval(tk, p)
        toks, ps, h, run_, n_fw = [int(tk.item())], [float(p.item())], self.h, float(p.item()), 0
        while run_ >= t and len(toks) < k:
            lgx, h = model.mtp_forward(h, mx.array([[toks[-1]]]), mtp_cache=self.cache, return_hidden=True)
            n_fw += 1
            lx = lgx[0, -1].astype(mx.float32)
            tk, p = mx.argmax(lx), mx.softmax(lx).max()
            mx.eval(tk, p)
            toks.append(int(tk.item())); ps.append(float(p.item())); run_ *= ps[-1]
        if n_fw:
            self.cache[0].trim(n_fw)                     # the chain's own entries are not committed history
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

# ---------------------------------------------------------------- C0 probes (Leap, Mon 2026-11-02)
C0_AT = {int(x) for x in os.environ.get("C0_AT", "1,2,5,10,15,30").split(",") if x}
C0_SWEEP = tuple(int(x) for x in os.environ.get("C0_SWEEP", "1,2,4,8,12,16").split(",") if x)
PROBES, CUR = [], [None]
SEG = {"on": False, "rec": [], "L": None}
LIDX = {id(l): i for i, l in enumerate(lay)}
HCK = {}
for _l in lay:
    HCK[id(_l.attn_hyper_connection)] = "hc_a"
    HCK[id(_l.mlp_hyper_connection)] = "hc_m"
def _arrs(o):
    if isinstance(o, mx.array):
        return [o]
    if isinstance(o, (list, tuple)):
        return [a for x in o for a in _arrs(x)]
    return []
def _rec(kind, t0, t1, t2):
    SEG["rec"].append((kind, SEG["L"], 1e3 * (t1 - t0), 1e3 * (t2 - t1)))
def _mseg(kind, fn):                                     # a method: build, then eval its outputs
    def w(self, *a, **kw):
        if not SEG["on"]:
            return fn(self, *a, **kw)
        t0 = time.perf_counter()
        out = fn(self, *a, **kw)
        t1 = time.perf_counter()
        mx.eval(_arrs(out))
        _rec(kind(self) if callable(kind) else kind, t0, t1, time.perf_counter())
        return out
    return w
def _fseg(kind, fn):                                     # a module function
    def w(*a, **kw):
        if not SEG["on"]:
            return fn(*a, **kw)
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        t1 = time.perf_counter()
        mx.eval(_arrs(out))
        _rec(kind, t0, t1, time.perf_counter())
        return out
    return w
def _lseg(fn):                                           # the whole layer (its pieces nest inside)
    def w(self, *a, **kw):
        if not SEG["on"]:
            return fn(self, *a, **kw)
        SEG["L"] = LIDX.get(id(self))
        t0 = time.perf_counter()
        out = fn(self, *a, **kw)
        t1 = time.perf_counter()
        mx.eval(_arrs(out))
        _rec("layer", t0, t1, time.perf_counter())
        SEG["L"] = None
        return out
    return w
def _xseg(fn):                                           # the experts call: its inputs (router + shared) first
    def w(self, x, indices, *a, **kw):
        if not SEG["on"]:
            return fn(self, x, indices, *a, **kw)
        t0 = time.perf_counter()
        t_ = getattr(self, "_bm", {}).get("tier", {})
        mx.eval([x, indices] + [v for k_, v in t_.items() if k_ in ("scores", "shared") and isinstance(v, mx.array)])
        t1 = time.perf_counter()
        _rec("moe_pre", t0, t0, t1)
        out = fn(self, x, indices, *a, **kw)
        t2 = time.perf_counter()
        mx.eval(_arrs(out))
        _rec("experts", t1, t2, time.perf_counter())
        return out
    return w
def seg_install():
    saved = []
    def put(obj, name, new):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, new)
    put(Q.DecoderLayer, "__call__", _lseg(Q.DecoderLayer.__call__))
    for T_ in {type(_l.attn_hyper_connection) for _l in lay}:
        put(T_, "__call__", _mseg(lambda s_: HCK.get(id(s_), "hc_f"), T_.__call__))
    for T_ in {type(_l.linear_attn) for _l in lay if _l.is_linear}:
        put(T_, "__call__", _mseg("gdn", T_.__call__))
    for T_ in {type(_l.self_attn) for _l in lay if not _l.is_linear}:
        put(T_, "__call__", _mseg("att", T_.__call__))
    for T_ in {type(_l.mlp) for _l in lay}:
        put(T_, "__call__", _mseg("moe", T_.__call__))
    for T_ in {type(_l.mlp.switch_mlp) for _l in lay}:
        put(T_, "__call__", _xseg(T_.__call__))
    for T_ in {type(_l.ple) for _l in lay if "ple" in _l}:
        put(T_, "__call__", _mseg("ple", T_.__call__))
    put(Q, "_hyper_residual_write", _fseg("hc_w", Q._hyper_residual_write))
    return saved
def seg_remove(saved):
    for obj, name, old in reversed(saved):
        setattr(obj, name, old)
def sync_ms():
    a = mx.array([1.0]); mx.eval(a)
    ts = []
    for _ in range(300):
        t0 = time.perf_counter(); b = a + 1; mx.eval(b); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))
def b16eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b, equal_nan=True).item())
def probe(cache, snap, real, full, lg_ref, n):
    """C0: re-run check n (rows `real`) on the rolled-back cache: warm repeats, one synced pass, the row sweep on
    `full` (the untruncated draft path), synced passes at 1 and 16 rows; the last forward is the real rows again"""
    tA, sA = time.perf_counter(), stats()
    last = [len(real)]
    def fwd(rows, synced=False):
        CS.rollback_after_verify(cache, snap, last[0])
        mx.eval(states(cache))                           # the restored GDN states' copies, out of the timed window
        s0 = stats()
        saved = None
        if synced:
            saved = seg_install(); SEG["rec"] = []; SEG["on"] = True
        try:
            t0 = time.perf_counter()
            with ref2.verify_scope(model):
                lg_ = model(mx.array([rows]), cache=cache)
            t1 = time.perf_counter()
            mx.eval(lg_)
            t2 = time.perf_counter()
        finally:
            if synced:
                SEG["on"] = False; seg_remove(saved)
        last[0] = len(rows)
        d = stats() - s0
        return lg_, dict(rows=len(rows), build=1e3 * (t1 - t0), wall=1e3 * (t2 - t0), miss=int(d[0]), io=1e3 * float(d[1]),
                         seg=list(SEG["rec"]) if synced else None)
    res = dict(key=CUR[0], check=n, rows=len(real))
    w = []
    for _ in range(2):
        lg_, r_ = fwd(real)
        r_["equal"] = b16eq(lg_, lg_ref)
        w.append(r_)
    res["warm"] = w
    res["synced_real"] = fwd(real, True)[1]
    sw = {}
    for r in C0_SWEEP:
        if r > len(full):
            continue
        sw[r] = [fwd(full[:r])[1] for _ in range(3)]     # cold (rows past the real check's), then two warm
    res["sweep"] = sw
    res["synced_1"] = fwd(full[:1], True)[1]
    res["synced_16"] = fwd(full[:16], True)[1] if len(full) >= 16 else None
    lg_, r_ = fwd(real)
    res["final_equal"] = b16eq(lg_, lg_ref)
    res["wall_s"] = time.perf_counter() - tA
    res["stats"] = (stats() - sA).tolist()
    return res

def prefill(ids, want_w=False, chunk=1024):
    pf1(True)
    cache, lg, ws = model.make_cache(), None, []
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        if want_w:                                       # the chat switch: each chunk's widened stream for the MTP head
            w = TXT._last_widened
            assert w.shape[1] == min(chunk, len(ids) - a), w.shape
            mx.eval(w)
            ws.append((a, w))
    TAP["on"] = False
    pf1(False)
    return cache, lg[0, -1], taps(), ws

def run(ids, arm, name=None, drafter="dfl"):
    while os.path.exists(PAUSE):
        time.sleep(5)
    mtp = drafter == "mtp"
    set_arm("slot")                                      # the prefill as shipped in every arm
    ref_on(True)
    p0, g0 = BG.PAUSED["pause_s"], BG.PAUSED["n_gap"]
    mx.reset_peak_memory()                               # C45 memory anatomy: MLX peak in prefill vs decode, cache, footprint
    cache, last, T, ws = prefill(ids, mtp)
    mem = [mx.get_peak_memory() / 1e9, footprint_peak_gb()]
    mx.reset_peak_memory()
    P = len(ids)
    dcache = head = pend = None
    if not mtp:
        dcache = draft.make_cache()
        start = max(0, P - (SWIN - 1))
        for c in dcache:
            c.offset = start
        ctx = T[start:P]
    a = P
    tok = int(mx.argmax(last).item())
    set_arm(arm)
    res0 = None
    if S10B_DIR:
        BS._S10B.drop()                                  # only the live loop's checks + commit replays
        res0 = np.array([BS.ext.lookup(L, list(range(512))) for L in range(NL)], np.int32)   # pools at loop start
    out, cyc, ck_t, kept, sm, wl, cut, drf = [tok], [], [], [], [], [], False, []
    rows_, cb_ = [], []                                  # per check: verified rows, SSD bytes (the per-row cost split, 11-01 lead)
    rb0, sl0 = disk_read_bytes(), stats()                # the whole loop's SSD bytes + misses (M-17's MB per kept token)
    td = c_prev = time.perf_counter()
    hist = 0.0
    if mtp:                                              # the prompt's pairs (W[u], token u + 1), the last one with tok; in the
        ref2.remove()                                    # timed window (D-23d kept its 0.03-0.34 s out), as DFlash2's window
        head, full = MTPHead(), list(ids) + [tok]        # prefill sits in its first propose
        for a_, w in ws:
            head.append(w, full[a_ + 1:a_ + 1 + w.shape[1]])
            mx.eval(head.logits)
        ref2.apply()
        hist = time.perf_counter() - td
    del ws
    while len(out) < CAP and tok not in EOS:
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
            dr, _ = head.draft_cut(MTP_K, TRUNC)
        else:
            pth, cf = propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)
            dr = np.array(pth[0]).tolist()
            full_ = [tok] + list(dr)                     # C0: the untruncated path for the row sweep
            if arm == "gate+t":                          # Leap's S45c: L1-14 running-confidence truncation
                dr = dr[:int((np.cumprod(np.array(cf[0])) >= TRUNC).sum())]
            if (trim := dcache[0].offset - a) > 0:
                DF._trim_recent_cache(dcache, trim)
        ref2.apply()
        s0, b0_ = stats(), disk_read_bytes()
        c1 = time.perf_counter()
        drf.append(c1 - c0)                              # the drafter's share of the cycle: the SSD idles (R-70D's window)
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        if not mtp:
            tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([[tok] + dr]), cache=cache)
        TAP["on"] = False
        am = mx.argmax(lg[0], axis=-1)
        if mtp:                                          # the MTP head's history needs the verified rows' widened stream
            wv = TXT._last_widened
            mx.eval(am, wv[:, :len(dr) + 1])
        else:
            tp = taps()
            mx.eval(am, tp)
        am = am.tolist()
        k = 0
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        pdt, pst = 0.0, 0.0
        if not mtp and len(ck_t) + 1 in C0_AT:           # C0: probe this check (its time and misses come out below)
            tq = time.perf_counter()
            pr = probe(cache, snap, [tok] + dr, full_, lg, len(ck_t) + 1)
            pdt, pst = time.perf_counter() - tq, np.array(pr["stats"])
            PROBES.append(pr)
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        c2 = time.perf_counter() - pdt
        c_prev += pdt; td += pdt
        sm.append(stats() - s0 - pst)
        rows_.append(len(dr) + 1); cb_.append(disk_read_bytes() - b0_)
        ck_t.append(c2 - c1); kept.append(k + 1); wl.append(c2 - c0)
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
                new = new[:j + 1]
                break
        out += new
        if mtp:
            pend = (wv[:, :len(new)], new)               # rows 0..len(new)-1 hold the pairs (W[a + i], new[i])
        else:
            ctx = tp[:k + 1]
        tok, a = out[-1], a + k + 1
    mx.eval(states(cache))
    dec = time.perf_counter() - td
    rb, ml, kt = disk_read_bytes() - rb0, float((stats() - sl0)[0]), max(sum(kept), 1)
    mem += [mx.get_peak_memory() / 1e9, mx.get_active_memory() / 1e9, mx.get_cache_memory() / 1e9, footprint_now_gb(),
            footprint_peak_gb()]
    if S10B_DIR and name:
        np.savez(os.path.join(S10B_DIR, name + ".npz"), **BS._S10B.flush(), tokens=np.array(out, np.int32), res0=res0)
    bad = ""
    try:
        st.gate.check()
    except RuntimeError as e:
        bad = f" INVALID ({e})"
    if BG.PAUSED["pause_s"] - p0 > 0.5 or BG.PAUSED["n_gap"] > g0:
        bad += f" PAUSED ({BG.PAUSED['pause_s'] - p0:.1f} s waited, {BG.PAUSED['n_gap'] - g0} gaps)"
    set_arm("slot")
    ref_on(False)
    del cache, dcache, head, pend
    mx.clear_cache()
    sm = np.array(sm)
    return dict(out=out[:CAP], n=min(len(out), CAP), tps=min(len(out), CAP) / dec, drafter=drafter, hist=hist,
                cyc=1e3 * med(cyc), ck=1e3 * med(ck_t),
                kept=float(np.mean(kept)), checks=len(kept), miss=float(np.median(sm[:, 0])), io=1e3 * float(np.median(sm[:, 1])),
                svc=1e3 * float(np.median(sm[:, 2])), slow=float(np.median(sm[:, 3])), drf=1e3 * med(drf), mem=mem,
                rb=rb, ml=ml, kt=kt, mb=rb / 1e6 / kt, mpk=ml / kt,   # per kept token: SSD MB and misses, the whole loop
                pc=dict(rows=rows_, kept=kept, ck=[1e3 * x for x in ck_t], drf=[1e3 * x for x in drf[:len(ck_t)]],
                        miss=sm[:, 0].tolist(), io=(1e3 * sm[:, 1]).tolist(), svc=(1e3 * sm[:, 2]).tolist(), mb=[x / 1e6 for x in cb_]),
                tpsw=sum(kept[2:]) / max(sum(wl[2:]), 1e-9), cut=cut, bad=bad)   # tpsw: steady state, checks >= 2

REFTOK, REFARM = {}, None                                # R57_REFTOK=<tokens.json>[:arm]: every run vs those tokens
if os.environ.get("R57_REFTOK"):
    _f, _, REFARM = os.environ["R57_REFTOK"].partition(":")
    REFTOK, REFARM = json.load(open(os.path.expanduser(_f))), REFARM or "gate+t"
REFROWS = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
RES, ok_all, valid, TOK = {}, True, True, {}
PC = []                                                  # per-check rows / ms / misses / MB of every run -> {TAG}_checks.json
if S10B_DIR:                                             # every layer's router, dequantized (Fit12's lead-1 forecast at every L)
    W = {}
    for T_ in range(1, NL):
        g = lay[T_].mlp.gate
        w = (mx.dequantize(g.weight, g.scales, getattr(g, "biases", None), g.group_size, g.bits,
                           **({"mode": g.mode} if hasattr(g, "mode") else {})) if hasattr(g, "scales") else g.weight)
        W[f"w{T_}"] = np.array(w.astype(mx.float16))
    np.savez(os.path.join(S10B_DIR, "routers.npz"), **W)
    say(f"  routers.npz: {len(W)} layers, {sum(v.nbytes for v in W.values()) / 1e6:.0f} MB")
if os.environ.get("R57_WARM") == "1":                    # warm-up, discarded (NewIdeas: S45-T's method)
    it = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
    CAP, cap0 = 64, CAP
    prior_for(it["key"])
    for dr_ in (("dfl", "mtp") if CHATSW else ("dfl",)):   # the chat switch warms its MTP chain too (D-23d: one per arm)
        r = run(list(it["prompt_ids"]), ARMS_[0], drafter=dr_)
        say(f"  warm-up (discarded): gsm8k/708 [{ARMS_[0]}, {dr_}] {r['n']} tokens, {r['tps']:.1f} tok/s{r['bad']}; MLX peak "
            f"prefill {r['mem'][0]:.1f} / decode {r['mem'][2]:.1f} GB, lifetime peak {r['mem'][1]:.1f} -> {r['mem'][6]:.1f} GB")
    CAP = cap0
RUNS = []                                                # per run: type, drafter, tok/s, kept, ms, misses, MB -> {TAG}_runs.json
for n_p, (s, key) in enumerate(PICK):
    it = next(x for x in ED.load("dec_" + s) if x["key"] == key)
    ids, want = list(it["prompt_ids"]), REFROWS.get(key, {}).get("completion_ids", [])
    typ = rule(it) if CHATSW else ""
    drafter = "mtp" if typ == "chat" else "dfl"          # the chat switch; without R57_CHATSW every prompt drafts on DFlash2
    cf = corr_for(key)
    pk = prior_for(key)
    if pk:
        say(f"  {key}: eviction prior {pk} x {os.environ.get('BIGMOE_EVICT_PRIOR')} (held out)")
    if cf:
        say(f"  {key}: forecast correction {cf} (held out)")
    arms = (ARMS_ + ARMS_[::-1]) if REPS == 2 else ARMS_
    arms = arms if n_p % 2 == 0 else arms[::-1]
    for arm in arms:
        if el() > HARD:
            say(f"  time budget: stopped before {key} [{arm}]")
            break
        CUR[0] = key
        r = run(ids, arm, f"{key.replace('/', '_').replace('.json', '')}_{arm}_{len(TOK.get(key, {}).get(arm, []))}", drafter)
        PC.append(dict(key=key, arm=arm, paused="PAUSED" in r["bad"], **r["pc"]))
        RUNS.append(dict(key=key, arm=arm, type=typ, bad=r["bad"], cut=r["cut"], P=len(ids),
                         **{k_: r[k_] for k_ in ("drafter", "hist", "n", "tps", "tpsw", "kept", "checks", "cyc", "ck", "drf",
                                                 "miss", "io", "mb", "mpk", "mem")}))
        m = min(len(want), r["n"])
        agree = same(r["out"], want)
        ok = (agree >= m or TIER) and not r["cut"]          # REF2-T is its own reference (tokens compared across runs)
        TOK.setdefault(key, {}).setdefault(arm, []).append(r["out"])
        note = ""
        if REFTOK:                                       # REF2-T's own reference (R57_REFTOK, e.g. s45t_B gate+t)
            rt_ = REFTOK.get(key, {}).get(REFARM, [None])[0]
            a_t, m_t = (same(r["out"], rt_), min(len(rt_), r["n"])) if rt_ is not None else (0, 1)
            ok = ok and a_t >= m_t
            note = f"; tokens == REFTOK {a_t} / {m_t}"
        ok_all &= ok
        valid &= not r["bad"]
        if "PAUSED" not in r["bad"]:                     # a paused run's clock is wrong: out of the aggregates
            RES.setdefault(key, {}).setdefault(arm, []).append(r)
        sw_ = (f" [{typ}: {drafter}" + (f", history {r['hist']:.2f} s" if drafter == "mtp" else "") + "]") if CHATSW else ""
        say(f"  {key} [{arm}]{sw_}: {r['n']} tokens in {r['checks']} checks (kept {r['kept']:.2f}), {r['tps']:.1f} tok/s "
            f"(steady state, checks >= 2: {r['tpsw']:.1f}), median "
            f"cycle {r['cyc']:.1f} ms, check + commit {r['ck']:.1f} ms; per check (medians) {r['miss']:.0f} misses, demand "
            f"SSD {r['io']:.1f} ms, gate service {r['svc']:.1f} ms ({r['slow']:.0f} calls with misses), draft {r['drf']:.1f} "
            f"ms; per kept token SSD {r['mb']:.1f} MB, {r['mpk']:.1f} misses; tokens == REF2 row {agree} / {m}{note}"
            f"{'; cut' if r['cut'] else ''}{r['bad']}")
        mm = r["mem"]
        say(f"    mem: MLX peak prefill {mm[0]:.1f} / decode {mm[2]:.1f} GB; at the end active {mm[3]:.1f}, cache {mm[4]:.1f}, "
            f"footprint {mm[5]:.1f} GB; lifetime peak after prefill {mm[1]:.1f} -> after decode {mm[6]:.1f} GB")
eq_arms = all(len({tuple(r["out"]) for runs in v.values() for r in runs}) == 1 for v in RES.values())
def agg(arm, f):
    return med([med([f(r) for r in v[arm]]) for v in RES.values() if arm in v])
for arm in ARMS_:
    say(f"  {arm}: tok/s {agg(arm, lambda r: r['tps']):.1f} (steady state {agg(arm, lambda r: r['tpsw']):.1f}), cycle "
        f"{agg(arm, lambda r: r['cyc']):.1f} ms, check + commit {agg(arm, lambda r: r['ck']):.1f} ms, misses "
        f"{agg(arm, lambda r: r['miss']):.0f}, demand SSD {agg(arm, lambda r: r['io']):.1f} ms per check (medians over valid "
        f"runs, then prompts)")
    say(f"  {arm} where the time goes (medians per check): check + commit {agg(arm, lambda r: r['ck']):.1f} ms, demand SSD "
        f"busy {agg(arm, lambda r: r['io']):.1f} ms ({agg(arm, lambda r: r['io'] / r['ck']):.0%} of the check), gate service on "
        f"misses {agg(arm, lambda r: r['svc']):.1f} ms (the GPU's spin bound; OOO overlaps part of it), draft "
        f"{agg(arm, lambda r: r['drf']):.1f} ms (SSD idle)")
    rs_ = [r for v in RES.values() for r in v.get(arm, [])]
    if rs_:                                              # M-17 / C45 sizing: bytes and misses over all kept tokens of the arm
        kt_ = sum(r["kt"] for r in rs_)
        say(f"  {arm} per kept token (pooled over valid runs): SSD {sum(r['rb'] for r in rs_) / 1e6 / kt_:.1f} MB, misses "
            f"{sum(r['ml'] for r in rs_) / kt_:.1f} ({kt_} kept tokens; median per prompt {agg(arm, lambda r: r['mb']):.1f} MB, "
            f"{agg(arm, lambda r: r['mpk']):.1f} misses)")
if "slot" in ARMS_ and "gate" in ARMS_:
    for key, v in RES.items():
        if "slot" in v and "gate" in v:
            f = lambda arm, k: med([r[k] for r in v[arm]])
            say(f"    {key}: steady-state tok/s slot {f('slot', 'tpsw'):.1f} -> gate {f('gate', 'tpsw'):.1f} "
                f"({f('gate', 'tpsw') / f('slot', 'tpsw') - 1:+.0%}); whole run {f('slot', 'tps'):.1f} -> {f('gate', 'tps'):.1f}; "
                f"median cycle {f('slot', 'cyc'):.0f} -> {f('gate', 'cyc'):.0f} ms")
    cy_g, cy_s = agg("gate", lambda r: r["cyc"]), agg("slot", lambda r: r["cyc"])
    w_g, w_s = agg("gate", lambda r: r["tpsw"]), agg("slot", lambda r: r["tpsw"])
    say(f"R-55 live (gate vs slot): median cycle {cy_g:.1f} vs {cy_s:.1f} ms ({cy_g / cy_s - 1:+.1%}); steady-state tok/s "
        f"{w_g:.1f} vs {w_s:.1f} ({w_g / w_s - 1:+.1%}); whole-run tok/s {agg('gate', lambda r: r['tps']):.1f} vs "
        f"{agg('slot', lambda r: r['tps']):.1f}; tokens equal across every run of a prompt: {eq_arms}; every token == REF2: "
        f"{ok_all}; all runs valid: {valid}")
if "gate+pf" in ARMS_:
    io_g, io_p = agg("gate", lambda r: r["io"]), agg("gate+pf", lambda r: r["io"])
    ck_g, ck_p = agg("gate", lambda r: r["ck"]), agg("gate+pf", lambda r: r["ck"])
    cy_g, cy_p = agg("gate", lambda r: r["cyc"]), agg("gate+pf", lambda r: r["cyc"])
    d_io, d_ck, d_cy = io_p / io_g - 1, ck_p / ck_g - 1, cy_p / cy_g - 1
    v = ("FAIL (a token left REF2's row)" if not ok_all
         else "PASS" if d_io <= -0.5 and d_ck <= -0.25 else "KILL" if d_cy > -0.10 else "BETWEEN")
    say(f"R-57 verdict (PASS: demand SSD per check -50 % AND check + commit -25 %, gate+pf vs gate, every token == REF2; "
        f"KILL: cycle under -10 %): {v} (SSD {d_io:+.1%}, check + commit {d_ck:+.1%}, cycle {d_cy:+.1%})")
if TIER:
    import bigmoe.tier as _BT
    say(f"  tier flags: TIER_COMBINE {_BT._COMBINE}, OOO {_BT._OOO} ({_BT.OOO_STATS['calls']} OOO calls); EVICT_PRIOR "
        f"{os.environ.get('BIGMOE_EVICT_PRIOR')}; REFTOK {os.environ.get('R57_REFTOK')}: every token equal {ok_all}")
if os.environ.get("K73_EXACT") == "1":
    _c = ref2.counts()
    say(f"  K-73 engagement: exact MoE block calls {K73N[0]}; ref2 att_rope {_c['att_rope']}, idx_prep {_c['idx_prep']}, "
        f"hc_write {_c['hc_write']} (counted at graph build)")
if CHATSW:
    _nt = {t: sum(1 for x in RUNS if x["type"] == t) for t in ("chat", "code", "math", "agent")}
    say(f"  chat switch: MTP head d22_{MTP_HEAD} (md5 pinned) resident, K{MTP_K}, cut t {TRUNC} with early exit; type rule "
        f"runs: {_nt} (chat -> MTP, the rest -> DFlash2)")
say(f"  footprint now {footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB; {time.time() - t0:.0f} s")
# ---------------------------------------------------------------- C0 summary
SYNC = sync_ms()
KINDS = ("hc_a", "gdn", "att", "hc_m", "moe_pre", "experts", "moe_post", "hc_w", "ple", "glue", "hc_f")
def kinds(seg):
    """exclusive ms per piece kind (sum over layers) and the CPU build share, one sync taken off every record"""
    by = {}
    for kind, L, b, e in seg:
        by.setdefault(L, {}).setdefault(kind, []).append((b, e))
    tot, cpu = {k_: 0.0 for k_ in KINDS}, {k_: 0.0 for k_ in KINDS}
    inc = lambda v: sum(b + e for b, e in v)
    for L, d in by.items():
        if L is None:
            for b, e in d.get("hc_f", []):
                tot["hc_f"] += b + e - SYNC; cpu["hc_f"] += b
            continue
        for k_ in ("hc_a", "gdn", "att", "hc_m", "moe_pre", "experts", "hc_w", "ple"):
            for b, e in d.get(k_, []):
                tot[k_] += b + e - SYNC; cpu[k_] += b
        moe = inc(d.get("moe", [])) - inc(d.get("moe_pre", [])) - inc(d.get("experts", []))
        tot["moe_post"] += moe - SYNC * len(d.get("moe", []))
        kids = sum(inc(d.get(k_, [])) for k_ in ("hc_a", "gdn", "att", "hc_m", "moe", "hc_w", "ple"))
        tot["glue"] += inc(d.get("layer", [])) - kids - SYNC * len(d.get("layer", []))
    return tot, cpu
def fmt(t):
    return ", ".join(f"{k_} {t[k_]:.1f}" for k_ in KINDS if abs(t[k_]) >= 0.05)
say(f"C0 timeline: {len(PROBES)} probe checks; one eval sync costs {SYNC * 1e3:.0f} us (taken off every synced record)")
for pr in PROBES:
    pc = next((x for x in PC if x["key"] == pr["key"]), None)
    j = pr["check"] - 1
    live = (f"live check + commit {pc['ck'][j]:.1f} ms, {pc['miss'][j]:.0f} misses, demand SSD {pc['io'][j]:.1f} ms, draft "
            f"{pc['drf'][j]:.1f} ms" if pc and j < len(pc["ck"]) else "live: n/a")
    wm = pr["warm"]
    say(f"  {pr['key']} check {pr['check']} ({pr['rows']} rows): {live}; warm repeats {wm[0]['wall']:.1f} / {wm[1]['wall']:.1f} ms "
        f"(build {wm[0]['build']:.1f} / {wm[1]['build']:.1f}, misses {wm[0]['miss']} / {wm[1]['miss']}, logits == real "
        f"{wm[0]['equal'] and wm[1]['equal']}); last forward == real {pr['final_equal']}; probe {pr['wall_s']:.1f} s")
    for tag in ("synced_real", "synced_1", "synced_16"):
        x = pr.get(tag)
        if not x:
            continue
        t, c = kinds(x["seg"])
        rest = x["wall"] - sum(b + e for kind, L, b, e in x["seg"] if kind in ("layer", "hc_f")) - SYNC
        say(f"    {tag} ({x['rows']} rows, misses {x['miss']}): sum of pieces {sum(t.values()):.1f} ms + top-level (embed, "
            f"norm, head) {rest:.1f} vs synced wall {x['wall']:.1f}; by piece (ms per check): {fmt(t)}; CPU build in it "
            f"{sum(c.values()):.1f}")
    sw = pr["sweep"]
    if sw:
        rr = sorted(sw)
        wv = [float(np.median([sw[r][1]["wall"], sw[r][2]["wall"]])) for r in rr]
        say("    sweep (rows: warm wall ms [cold pass misses, warm misses]): " + "; ".join(
            f"{r}: {w_:.1f} [{sw[r][0]['miss']}, {sw[r][2]['miss']}]" for r, w_ in zip(rr, wv)))
        r2 = [(r, w_) for r, w_ in zip(rr, wv) if r >= 2]
        if len(r2) >= 2:
            sl = np.polyfit([a for a, _ in r2], [b for _, b in r2], 1)
            say(f"    warm slope over 2-16 rows: {sl[0]:.2f} ms per row, intercept {sl[1]:.1f} ms")
    if pr.get("synced_1") and pr.get("synced_16"):
        t1_, _ = kinds(pr["synced_1"]["seg"]); t16, _ = kinds(pr["synced_16"]["seg"])
        say("    per added row (synced 16 vs 1 rows, /15): " + ", ".join(f"{k_} {(t16[k_] - t1_[k_]) / 15:.2f}" for k_ in KINDS
                                                                        if abs(t16[k_] - t1_[k_]) >= 0.05))
try:
    json.dump(dict(sync_ms=SYNC, probes=PROBES), open(os.path.join(OUT, f"{TAG}_probes.json"), "w"))
except Exception as e:
    say(f"  probes.json not written: {e}")
json.dump(TOK, open(os.path.join(OUT, f"{TAG}_tokens.json"), "w"))
try:                                                     # never let the side log cost a run
    json.dump(PC, open(os.path.join(OUT, f"{TAG}_checks.json"), "w"))
    json.dump(RUNS, open(os.path.join(OUT, f"{TAG}_runs.json"), "w"))
except Exception as e:
    say(f"  checks.json / runs.json not written: {e}")
open(os.path.join(OUT, f"{TAG}_screen.txt"), "w").write("\n".join(lines) + "\n")
