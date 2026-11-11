"""Where does frugal routing's KLD come from? (Breakthrough, 2026-10-20 evening.) The Eval suite measured teacher-forced
step KLD 0.057 for frugal at 10 % (ref step noise 0.0018; accepted ar4 0.0096), 0.115 on ifeval and 0.092 on multilingual.
This runs the Eval suite's own one-token-per-call scorer (evalsuite.stages.step_forward + engine.var_head against its
cached reference, read-only import) on a subset of its own-output sequences, for frugal variants on fork125bt:
  VAR  nw  kc  renorm   meaning
  base  2  20  yes      today's frugal (bt_frugal.py): missing ranks 2-9 replaced by resident ranks 10-19, renormalised
  nosub 2  10  yes      missing ranks 2-9 dropped, renormalised over the used
  noren 2  10  no       dropped, NOT renormalised (weights stay p_r / sum of the top-10)
  subnr 2  20  no       substitutes at their own router weight, no renormalisation
  w4    4  20  yes      wait for ranks 0-3
  sere  2  80  no       SERE-style similar-resident substitute at the missing expert's weight (serer: renormalised)
  mNN   *  20  yes      wait for the ranks covering NN % of the top-10 gate mass (at least rank 0)
  exact 10 10  yes      every top-10 waited (harness check: should sit near the step noise)
usage: tf_frugal_var.py VAR FRAC SETS EVERY   e.g. base 0.10 tf_ifeval,tf_multi 3   -> data/tfvar/<VAR>_<FRAC>.txt line
Env as the suite's frugal runs (EVALSUITE_FRUGAL=1: plain exact streamer flags, no speed fusions)."""
import json, os, sys, time
os.environ["EVALSUITE_FRUGAL"] = "1"
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
for k in ("MTPLX_NGRAM_PREWARM",):
    os.environ.setdefault(k, "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed            # fork125bt (imported before evalsuite puts ~/BigMoE on sys.path)
from fused_route import topk_probs
from bigmoe.gated import ext
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import stages as SG, engine as EN

VAR, frac, SETS, EVERY = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4])
CFG = {"base": (2, 20, True), "nosub": (2, 10, True), "noren": (2, 10, False), "subnr": (2, 20, False),
       "w4": (4, 20, True), "w3": (3, 20, True), "exact": (10, 10, True)}
# VAR = head[_mod...]: head in CFG | mNN (wait for the ranks covering NN % of the top-10 mass) | bNN (base selection,
# partial renormalisation beta = NN/100: w = p / ((1 - beta) * P10 + beta * P_used)) | sere (SERE-style: missing rank ->
# most output-similar resident expert among the router's top-80, data/simtab D/N tables, only if D[u, v] < N[u], at u's
# own gate weight) | serer (= sere_r). Mods: nr (no renorm), r (renorm), mNN (wait by mass, for sere).
parts = VAR.split("_")
head = "sere" if parts[0] == "serer" else parts[0]
if parts[0] == "serer":
    parts.append("r")
NW, KC, RENORM, THETA, BETA, SERE = 2, 20, True, None, None, False
BILD_TAU = None
DENSE = []                # dense-diet modifiers (hc8 / hc4 / lm3), applied after load
ADA = None                # adaptive theta (head 'ada'); LAST holds the previous step's next-token entropy
LAST = {"ent": 99.0}
MODE = {"exact": False}   # set while a BiLD re-run recomputes a token with every top-10 expert
TAILBITS = None    # tK: waited rank >= 2 experts that were not resident are computed from their MSB-K planes
BAND = None        # (a, b): apply the variant only in layers a..b-1; every other layer waits for its whole top-10 (exact)
PRE = None         # pre2 (TrainLab's pre-routed frugal, 2026-10-21): in layers [a, b) (BT_PRE_BAND, default 8,40) the two
PREPICK = {}       # waited experts are layer L's router (+ MoEPrefetch's correction) on layer L-1's MoE input, top-2; the
                   # rest = the true router's best outside them, resident only (frugal substitution); weights = the TRUE
                   # router's probabilities over the chosen 10, renormalised; other layers plain frugal
LATE = None        # lAtB: ranks A..B-1 are read; those cold before the call add their output half a layer late (written
STASH = {}         # into the HC streams after layer L+1's attention, NewIdeas' dh semantics); l1t2 = frugal waiting for
                   # rank 0 only (a cold rank 1 lands late); l2t4 at 5 % = frugal + cold ranks 2-3 late, not dropped
if head in CFG:
    NW, KC, RENORM = CFG[head]
elif head.startswith("m"):
    NW, THETA = 0, int(head[1:]) / 100
elif head.startswith("b") and head[1:].isdigit():
    BETA, RENORM = int(head[1:]) / 100, False
elif head == "sere":
    SERE, NW, KC, RENORM = True, 2, 80, False
    SIM = {}
elif head == "ada":                # wait by mass with theta = lo if the previous step's entropy <= tau else hi
    NW, KC, RENORM, THETA = 0, 20, False, 0.5
    ADA = {"lo": 0.5, "hi": 0.9, "tau": 0.1}
elif head == "bild":               # BiLD-style fallback: frugal (drop, no renorm) step; if its next-token entropy > tau,
    NW, KC, RENORM = 2, 10, False  # roll the token back and recompute it with the whole top-10 waited (mod eNN: tau = NN/100)
    BILD_TAU = 0.10
else:
    raise SystemExit(f"unknown variant {VAR}")
for p in parts[1:]:
    if p == "nr": RENORM = False
    elif p == "r": RENORM = True
    elif p.startswith("m"): NW, THETA = 0, int(p[1:]) / 100
    elif p.startswith("t"): TAILBITS = int(p[1:])
    elif p.startswith("L") and "T" in p: BAND = tuple(int(v) for v in p[1:].split("T"))   # LaTb: variant only in layers a..b-1
    elif p.startswith("e") and BILD_TAU is not None: BILD_TAU = int(p[1:]) / 100
    elif ADA is not None and p.startswith("lo"): ADA["lo"] = int(p[2:]) / 100
    elif ADA is not None and p.startswith("hi"): ADA["hi"] = int(p[2:]) / 100
    elif ADA is not None and p.startswith("e"): ADA["tau"] = int(p[1:]) / 100
    elif p.startswith("S"): SETS = [{"g": "tf_gsm", "c": "tf_code", "i": "tf_ifeval", "m": "tf_multi"}[ch] for ch in p[1:]]
    elif p in ("hc8", "hc4", "lm3"): DENSE.append(p)   # dense diet (TASKS.md): HC mixers 8 / 4-bit, lm_head 3-bit
    elif p.startswith("l") and "t" in p and p[1:].replace("t", "").isdigit(): LATE = tuple(int(v) for v in p[1:].split("t"))
    elif p == "pre2": PRE = tuple(int(v) for v in os.environ.get("BT_PRE_BAND", "8,40").split(","))
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K, NLAY = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"]
C = max(16, round(frac * E))
rt, st = load_streamed(MD, C, mtp=False)
MEM0 = mx.get_active_memory() / 1e9
if DENSE:
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    if "hc8" in DENSE or "hc4" in DENSE:   # MoonShots' patches/hc8 (read-only import): GatedResidual mixers + PLE K/V
        sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc8"))
        import hc8 as _hc8
        print("dense: hc", _hc8.install(rt.model, bits=8 if "hc8" in DENSE else 4), flush=True)
    if "lm3" in DENSE:                     # lm_head 4-bit -> 3-bit (affine, group 64) from its dequantised weights
        TM = getattr(rt.model, "language_model", rt.model)
        for name, mod in TM.named_modules():
            if name.endswith("lm_head") and isinstance(mod, nn.QuantizedLinear):
                W = mx.dequantize(mod.weight, mod.scales, mod.biases, mod.group_size, mod.bits)
                lin = nn.Linear(W.shape[1], W.shape[0], bias=False); lin.weight = W
                q = nn.QuantizedLinear.from_linear(lin, 64, 3)
                parent = TM
                for part in name.split(".")[:-1]:
                    parent = getattr(parent, part)
                before = sum(a.nbytes for _, a in tree_flatten(mod.parameters())) / 1e6
                setattr(parent, name.split(".")[-1], q)
                print(f"dense: {name} 4-bit -> 3-bit, {before:.0f} -> "
                      f"{sum(a.nbytes for _, a in tree_flatten(q.parameters())) / 1e6:.0f} MB", flush=True)
        mx.eval(TM.parameters())
    mx.clear_cache()
if PRE is not None:   # layer L+1's routers and MoEPrefetch's d=1 correction (indexed by the source layer)
    _TMP = rt.model.language_model.model if hasattr(rt.model, "language_model") else rt.model.model
    ROUT = [l.mlp.gate for l in _TMP.layers]
    _z = np.load(os.path.expanduser(os.environ.get("BIGMOE_PF_CORR", "~/MoEPrefetch/data/corr125_r128.npz")))
    CORR = [(mx.array(_z["A"][i].astype(np.float32)), mx.array(_z["B"][i].astype(np.float32)), mx.array(_z["mu"][i]),
             mx.array(_z["delta"][i])) for i in range(_z["A"].shape[0])]
print(f"memory after load: active {MEM0:.2f} GB, after dense changes {mx.get_active_memory() / 1e9:.2f} GB "
      f"(expert pools {st.resident_gb():.2f} GB)", flush=True)
from mtplx.models import qwen4_exp
from mtplx.attention_math import attention_gate
orig_call = qwen4_exp.SparseMoeBlock.__call__
STAT = {"used": 0.0, "mass": 0.0, "n": 0, "w": 0.0, "wn": 0, "calls": 0, "full": 0, "low": 0}
def _make_mask(n):
    """MSB-n of every 4-bit code in a packed uint32 word: (codes with the low 4-n bits cleared, per-word sum of them)."""
    low = (1 << (4 - n)) - 1
    keep = mx.array(int(("%X" % (0xF & ~low)) * 8, 16), dtype=mx.uint32)

    @mx.compile
    def f(w):
        lsum = None
        for j in range(8):
            part = (w >> (4 * j)) & low
            lsum = part if lsum is None else lsum + part
        return w & keep, lsum
    return f


_MASK = {n: _make_mask(n) for n in (1, 2, 3)}


class P:   # bias-corrected MSB-n tier of a 4-bit slot pool (planes125's math, here for n = 1..3)
    @staticmethod
    def _tier_proj(lin, h, sl, idx, n):
        w, s, b = lin.weight[sl], lin.scales[sl], lin.biases[sl]
        gs = lin.group_size
        wm, lsum = _MASK[n](w)
        off = lsum.reshape(*s.shape, gs // 8).sum(-1).astype(mx.float32) / gs
        bias = (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)
        return mx.gather_qmm(h, wm, s, bias, rhs_indices=idx, transpose=True, group_size=gs, bits=4)


SYNCGATE = os.environ.get("BT_SYNCGATE", "1") == "1"   # residency on the CPU, no GPU spin gate: survives the silent-
                                                       # night thermostat's SIGSTOP (a stopped loader times out spin gates)


def sync_slots(L, idl, nw, nk):
    """frugal residency without the GPU gate: (slots before, slots after) for the ids idl (-1 = absent). Loads the first nw
    ids (blocking), LRU-touches the resident ones among the first nk, never reads the others (BIGMOE_SPARE=0 semantics)."""
    pre = np.array(ext.lookup(L, idl))
    STAT["ld"] = STAT.get("ld", 0) + int((pre[:nw] < 0).sum()); STAT["c1"] = STAT.get("c1", 0) + 1   # measured reads
    need = idl[:nw] + [e for r, e in enumerate(idl[:nk]) if r >= nw and pre[r] >= 0]
    ext.ensure(L, mx.array(np.array(need, np.int32)))
    return pre, np.array(ext.lookup(L, idl))


def route(bm, ids, nw, nk):
    if not SYNCGATE:
        return None, bm["gate"].route_and_gate(bm["L"], ids[0], nw=nw, nk=nk).reshape(1, KC).astype(mx.int32)
    pre, after = sync_slots(bm["L"], [int(v) for v in np.array(ids[0])], nw, nk)
    return pre, mx.array(after.astype(np.int32)).reshape(1, KC)


def block(self, x):
    sw = self.switch_mlp
    bm = getattr(sw, "_bm", None)
    if bm is None or x.shape[1] != 1 or bm["full"]:
        return orig_call(self, x)
    if bm["L"] == bm["first"]:
        bm["gate"].new_pass()
    if PRE is not None and not MODE["exact"]:
        L = bm["L"]
        if PRE[0] <= L + 1 < PRE[1] and L < len(CORR):              # pre-pick layer L+1's two waited experts now
            x1 = x.reshape(1, -1)
            A, Bm, mu, dl = CORR[L]
            lg = ROUT[L + 1](x1).astype(mx.float32) + ((x1.astype(mx.float32) - mu) @ A) @ Bm + dl
            PREPICK[L + 1] = [int(v) for v in np.array(mx.argpartition(-lg, kth=1, axis=-1)[0, :2])]
        if L in PREPICK:
            return pre_block(self, sw, bm, x, PREPICK.pop(L))
    ids, pk = topk_probs(self.gate(x)[0], KC)                       # [1, KC]
    nw = NW
    if MODE["exact"] or (BAND is not None and not (BAND[0] <= bm["L"] < BAND[1])):   # exact: all top-10 waited
        _, slots = route(bm, ids, K, K)
        res = slots >= 0
        sel = res & (mx.cumsum(res.astype(mx.int32), axis=-1) <= K)
        w = pk * sel; w = w / mx.maximum(w.sum(axis=-1, keepdims=True), 1e-9)
        safe = mx.where(res, slots, slots[:, :1]).astype(mx.uint32)
        y = (type(sw).__mro__[1].__call__(sw, x, safe[None]) * w[None][..., None].astype(x.dtype)).sum(axis=-2)
        return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))
    if THETA is not None:                                          # CPU sync per call: a quality probe, not a speed path
        th = THETA if ADA is None else (ADA["lo"] if LAST["ent"] <= ADA["tau"] else ADA["hi"])
        p = np.array(pk[0, :K].astype(mx.float32)); c = np.cumsum(p) / p.sum()
        nw = max(1, int(np.searchsorted(c, th - 1e-9)) + 1)
        STAT["w"] += nw; STAT["wn"] += 1
    if LATE is not None:                                           # late ranks are read too (only their timing differs)
        nw = max(nw, LATE[1])
    pre0 = None
    if TAILBITS and not SYNCGATE:                                  # residency before the call (CPU, no loading)
        pre0 = np.array(ext.lookup(bm["L"], [int(v) for v in np.array(ids[0, :K])]))
    pre_s, slots = route(bm, ids, nw, K)
    pre = pre0 if pre0 is not None else (pre_s[:K] if pre_s is not None else None)
    if SERE:
        return sere_block(self, sw, bm, x, ids, pk, slots)
    res = slots >= 0
    sel = res & (mx.cumsum(res.astype(mx.int32), axis=-1) <= K)      # first K resident in rank order
    w = pk * sel
    p10 = pk[:, :K].sum(axis=-1, keepdims=True) if self.norm_topk_prob else 1.0
    if BETA is not None:
        w = w / ((1 - BETA) * p10 + BETA * w.sum(axis=-1, keepdims=True))
    elif RENORM:
        w = w / mx.maximum(w.sum(axis=-1, keepdims=True), 1e-9)
    else:
        w = w / p10
    safe = mx.where(res, slots, slots[:, :1]).astype(mx.uint32)      # rank 0 is always resident (waited)
    base = type(sw).__mro__[1]
    yk = base.__call__(sw, x, safe[None])                             # [1, 1, KC, D]
    if TAILBITS:
        loaded = np.zeros(KC, bool); loaded[:K] = pre < 0; loaded[nw:] = False   # read this call: missing waited ranks
        low = loaded.copy(); low[:2] = False                          # ranks 0-1 stay 4-bit
        STAT["calls"] += 1; STAT["full"] += int(loaded[:2].sum()); STAT["low"] += int(low.sum())
        if low.any():
            xe = mx.expand_dims(x, (-2, -3)); sl = safe.reshape(-1)
            idx = mx.arange(KC).astype(mx.uint32).reshape(1, 1, KC)
            h = sw.activation(P._tier_proj(sw.up_proj, xe, sl, idx, TAILBITS), P._tier_proj(sw.gate_proj, xe, sl, idx, TAILBITS))
            ylow = P._tier_proj(sw.down_proj, h, sl, idx, TAILBITS).squeeze(-2)
            yk = mx.where(mx.array(low)[None, None, :, None], ylow.astype(yk.dtype), yk)
    if LATE is not None and bm["L"] < NLAY - 1:                    # cold ranks A..B-1: out of this layer's sum, into the
        r = np.arange(KC)                                          # HC streams after layer L+1's attention (layer_call)
        lm = (r >= LATE[0]) & (r < LATE[1]) & (np.concatenate([pre, np.zeros(KC - len(pre), int)]) < 0)
        if lm.any():
            lmx = mx.array(lm)[None]
            STASH[bm["L"]] = (yk * (w * lmx)[None][..., None].astype(x.dtype)).sum(axis=-2)
            w = w * ~lmx
            STAT["late"] = STAT.get("late", 0) + int(lm.sum())
    STAT["blk"] = STAT.get("blk", 0) + 1
    y = (yk * w[None][..., None].astype(x.dtype)).sum(axis=-2)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


def pre_block(self, sw, bm, x, pre):
    """pre2: the pre-picked two are read and waited for (in an engine: issued a layer earlier); the rest = the true router's
    best outside them, resident only, up to 10 in all; weights = the true probabilities over the chosen, renormalised."""
    L = bm["L"]
    p = np.array(mx.softmax(self.gate(x)[0].astype(mx.float32), axis=-1, precise=True)[0])   # true router, all experts
    order = np.argsort(-p)
    rest = [int(e) for e in order[:KC + 2] if int(e) not in pre][:KC - 2]
    idl = list(pre) + rest
    _, after = sync_slots(L, idl, 2, K)                            # reads the pre-picks; LRU-touches resident ranks < 10
    use = [0, 1] + [i for i in range(2, len(idl)) if after[i] >= 0][:K - 2]
    ids_u = [idl[i] for i in use]
    w = p[ids_u].astype(np.float64); w = w / w.sum()
    top2, top10 = set(int(e) for e in order[:2]), set(int(e) for e in order[:K])
    STAT["pre_n"] = STAT.get("pre_n", 0) + 2
    STAT["pre_top2"] = STAT.get("pre_top2", 0) + sum(e in top2 for e in pre)
    STAT["pre_top10"] = STAT.get("pre_top10", 0) + sum(e in top10 for e in pre)
    b = L // 8                                                     # per 8-layer band: true rank 0 / rank 1 among the picks
    PB = STAT.setdefault("pre_band", {})
    c = PB.setdefault(b, [0, 0, 0])
    c[0] += 1; c[1] += int(order[0]) in pre; c[2] += int(order[1]) in pre
    sl = [int(after[i]) for i in use] + [int(after[0])] * (K - len(use))
    w = np.concatenate([w, np.zeros(K - len(use))]).astype(np.float32)
    base = type(sw).__mro__[1]
    y = (base.__call__(sw, x, mx.array(sl, mx.uint32).reshape(1, 1, K)) * mx.array(w)[None, None, :, None].astype(x.dtype)).sum(axis=-2)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


def sere_block(self, sw, bm, x, ids, pk, slots):
    L = bm["L"]
    if L not in SIM:
        d = os.path.expanduser("~/Breakthrough/data/simtab")
        SIM[L] = (np.load(f"{d}/D{L}.npy").astype(np.float32), np.load(f"{d}/N{L}.npy"))
    Dm, Nm = SIM[L]
    i, sl, p = np.array(ids[0]), np.array(slots[0]), np.array(pk[0].astype(mx.float32))
    w10 = p[:K] / p[:K].sum()
    res = [(int(e), int(s)) for e, s in zip(i, sl) if s >= 0]
    used, outs, ws = set(), [], []
    for r in range(K):
        if sl[r] >= 0:
            used.add(int(i[r])); outs.append(int(sl[r])); ws.append(w10[r]); continue
        u = int(i[r])
        cand = [(Dm[u, e], e, s) for e, s in res if e not in used and Dm[u, e] < Nm[u]]
        STAT["n"] += 1
        if cand:
            _, e, s = min(cand); used.add(e); outs.append(s); ws.append(w10[r]); STAT["used"] += 1
    ws = np.array(ws, np.float32)
    if RENORM:
        ws = ws / ws.sum()
    pad = K - len(outs)
    outs += [outs[0]] * pad; ws = np.concatenate([ws, np.zeros(pad, np.float32)])
    base = type(sw).__mro__[1]
    y = (base.__call__(sw, x, mx.array(outs, mx.uint32).reshape(1, 1, K)) * mx.array(ws)[None, None, :, None].astype(x.dtype)).sum(axis=-2)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


def step_forward_bild(rt, ids, pos, score, tau):
    """SG.step_forward with BiLD-style fallback: each 1-row frugal step's next-token entropy decides whether the token is
    rolled back (MTPLX rollback_after_verify, verified_tokens=1) and recomputed with the whole top-10 (MODE exact).
    Counts re-runs and SSD loads (ext misses) per decode step. The prefill's last token stays frugal."""
    from mtplx import generation as G
    from mtplx.attention_context import attention_phase
    from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache_lazy
    p0 = int(pos[0])
    ps = G.restore_or_prefill_prompt_state(rt, [int(x) for x in ids[:p0 + 1]], base_hidden_variant=None,
                                           mtp_history_policy="cycle", capture_hidden=False)
    cache, logits, out = ps.trunk_cache, ps.logits, []
    for t in pos:
        if t > p0:
            m0 = ext.stats()["misses"]
            snap = snapshot_untrimmable_cache_lazy(cache)
            x = mx.array([[int(ids[t])]])
            with attention_phase("ar_decode"):
                logits = rt.forward_ar(x, cache=cache)[:, -1, :]
            lp = logits.astype(mx.float32); lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            ent = float(-(mx.exp(lp) * lp).sum())
            if ent > tau:
                rollback_after_verify(cache, snap, verified_tokens=1)
                MODE["exact"] = True
                try:
                    with attention_phase("ar_decode"):
                        logits = rt.forward_ar(x, cache=cache)[:, -1, :]
                    mx.eval(logits)
                finally:
                    MODE["exact"] = False
                STAT["rerun"] = STAT.get("rerun", 0) + 1
            STAT["steps"] = STAT.get("steps", 0) + 1
            STAT["loads"] = STAT.get("loads", 0) + ext.stats()["misses"] - m0
        r = score(np.array([t]), logits)
        mx.eval(r)
        out.append({k: np.array(v) for k, v in r.items()})
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


qwen4_exp.SparseMoeBlock.__call__ = block
if LATE is not None:   # MTPLX DecoderLayer.__call__ + the half-layer-late writes (NewIdeas' tf_defer125.py dh semantics)
    assert SYNCGATE, "lAtB needs the CPU residency path (residency before the call)"
    from mtplx.models.qwen4_exp import _hyper_residual_write
    PENDH = {}

    def layer_call(self, hidden, *, input_ids, ssm_mask, cache):
        if "ple" in self:
            hidden = hidden + self.ple(hidden, input_ids, cache)
        mixed, hyper, inject = self.attn_hyper_connection(hidden)
        if self.is_linear:
            block_out = self.linear_attn(mixed, ssm_mask, cache)
        else:
            block_out = self.self_attn(mixed, cache)
        L = self._bt_L
        for lo, inj in PENDH.pop(L, []):                  # layer L-1's late ranks, before this layer's MLP read
            hyper = _hyper_residual_write(hyper, lo, inj)
        mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))
        block_out = self.mlp(mixed)
        hidden = _hyper_residual_write(hyper, block_out, inject)
        lo = STASH.pop(L, None)
        if lo is not None:
            PENDH.setdefault(L + 1, []).append((lo, inject))
        return hidden
    _TM = rt.model.language_model.model if hasattr(rt.model, "language_model") else rt.model.model
    for _i, _l in enumerate(_TM.layers):
        _l._bt_L = _i
    qwen4_exp.DecoderLayer.__call__ = layer_call
items = SG.tf_items()
head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
seqs = [i for i, it in enumerate(items) if it["set"] in SETS][::EVERY]
out = {s: [] for s in SETS}
t0 = time.time()
for i in seqs:
    it = items[i]
    if BILD_TAU is not None:
        r = step_forward_bild(rt, it["ids"], it["pos"], lambda pos, lg: head(i, pos, lg), BILD_TAU)
    else:
        def sc(pos, lg, i=i):
            if ADA is not None:
                lp = lg.astype(mx.float32); lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
                LAST["ent"] = float(-(mx.exp(lp) * lp).sum())
            return head(i, pos, lg)
        LAST["ent"] = 99.0
        r = SG.step_forward(rt, it["ids"], it["pos"], sc)
    st.gate.check()
    out[it["set"]].append(r)
cat = {s: np.concatenate([r["kld"] for r in v]) for s, v in out.items() if v}
agr = {s: np.concatenate([r["agree"] for r in v]) for s, v in out.items() if v}
allk = np.concatenate(list(cat.values()))
line = (f"{VAR:6s} C={C} nw={NW} kc={KC} renorm={int(RENORM)}: KLD {allk.mean():.4f} over {len(allk)} positions "
        f"({len(seqs)} seqs, every {EVERY}) | " + " ".join(f"{s} {cat[s].mean():.4f} (top-1 {agr[s].mean():.3f})" for s in cat)
        + (f" | substituted {STAT['used'] / STAT['n']:.2f} of misses" if STAT["n"] else "")
        + (f" | mean waited ranks {STAT['w'] / STAT['wn']:.2f}" if STAT["wn"] else "")
        + (f" | per token: {48 * STAT['full'] / STAT['calls']:.1f} full + {48 * STAT['low'] / STAT['calls']:.1f} MSB-{TAILBITS} reads" if STAT["calls"] else "")
        + (f" | BiLD tau {BILD_TAU}: re-ran {STAT.get('rerun', 0) / max(1, STAT.get('steps', 0)):.1%} of steps, "
           f"{STAT.get('loads', 0) / max(1, STAT.get('steps', 0)):.1f} expert loads per step" if BILD_TAU is not None else "")
        + (f" | measured {STAT['ld'] / max(1, STAT.get('steps') or STAT['c1'] / 48):.1f} expert reads per step" if STAT.get("c1") else "")
        + (f" | late ranks {LATE[0]}-{LATE[1] - 1}: {NLAY * STAT.get('late', 0) / max(1, STAT.get('blk', 1)):.1f} per step half a "
           f"layer late" if LATE is not None else "")
        + (f" | pre-routed layers {PRE[0]}-{PRE[1] - 1}: pre-picks in the true top-2 {STAT.get('pre_top2', 0) / max(1, STAT.get('pre_n', 1)):.1%}"
           f", in the top-10 {STAT.get('pre_top10', 0) / max(1, STAT.get('pre_n', 1)):.1%}; true rank 0 / rank 1 picked by band "
           + " ".join(f"L{8 * b}-{8 * b + 7} {c[1] / c[0]:.0%}/{c[2] / c[0]:.0%}" for b, c in sorted(STAT.get("pre_band", {}).items()))
           if PRE is not None else "")
        + f" | {time.time() - t0:.0f}s")
print(line, flush=True)
os.makedirs(os.path.expanduser("~/Breakthrough/data/tfvar"), exist_ok=True)
RT = "" if abs(frac - 0.10) < 1e-9 else f"_r{round(100 * frac):02d}"   # other residencies get their own file
np.savez(os.path.expanduser(f"~/Breakthrough/data/tfvar/{VAR}{RT}_e{EVERY}.npz"),   # per position, for paired comparisons
         **{f"kld_{s}": cat[s] for s in cat}, **{f"agree_{s}": agr[s] for s in agr})
open(os.path.expanduser("~/Breakthrough/data/tfvar/results.txt"), "a").write(line + "\n")
