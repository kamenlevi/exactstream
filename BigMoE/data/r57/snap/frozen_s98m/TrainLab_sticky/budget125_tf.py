"""TrainLab (10-22): real 125B KLD of the per-pass union budget at R-row verify passes, on the Eval suite's exact
reference engine (read-only import of ~/EvalSuite + ~/BigMoE), teacher-forced on the suite's own-output sets and scored
against its cached reference (engine.var_head). Each sequence: the suite's exact prompt prefill, then the completion in
R-row decode_verify forwards = MTP verify passes whose drafts are all right. Variants:
  exact   natural routing in R-row passes (the R-row floor vs the cached reference)
  bU      per-pass union budget U: the U experts with the largest summed w^2 over the pass's rows (w = each row's
          top-10 gate weights normalised over its top-10) form the pass set; every row takes its top-10 within it. Done
          by a gate wrapper (router logits outside the set -> -inf), so the block's softmax / top-k / renormalisation
          stay the suite's code. Published as batch-aware routing (XShare ICML'26; OEA 2511.02237).
  oK      OEA's own rule (2511.02237): the pass set is the union of every row's top-K; rows fill to 10 within it
TL_MTP=1 (NewIdeas 21:10): load the MTP head too and record, per position, the drafter's depth-1 top-1 log-prob for the
same prediction (_pos.npz key <arm>__mtp_lp1), from the arm's own hidden states, as spec125's _MTPHead would draft it.
With TL_MTP=1 the exact arm also keeps the target's and the drafter's top-16 ids / log-probs per position (exact__tgt_top_ids
/ _lp, exact__mtp_top_ids / _lp; NewIdeas' speculative-sampling acceptance check).
Arms: exact (the loaded variant's routing and plan), bU / oK (budgets), pPLAN (B2: tier plan PLAN for this arm, e.g.
p4433333333 = v3, p4422222222 = ranks 2-9 at MSB-2), cC_rR (B2 cold-only: LRU residency at C, cold experts at ranks >= R
at MSB-2) and 3b<C_m>t<C_2> (MoonShots' 3B: every rank MSB-3, a base pool of C_m; cold experts at ranks 4-9 at 2 bits
through a 2-bit cache of C_2; SSD bytes against v3's 147 + 46 pools), and B10 v2's dense layer arms (dense_layers.py)
<k><L>[.<L2>...][p<PLAN>]: layers L... computed densely on verify passes from source k (d = v3's own codes, ranks 0-1
4-bit and 2-9 MSB-3; x = exact 4-bit; t = Fit12's trellis experts, base + residual for ranks 0-1, base for 2-9), the
other layers at plan PLAN (e.g. t24p4444444444: layer 24 trellis alone) or the loaded variant's. S5 / S9 (NewIdeas
Sun 09:45): pPLAN[h], a PLAN of K < 10 digits = top-K verify rows (each row's top K of the router, renormalized over the
K, as Fit12's TK1 hook; the prompt prefill and the MTP drafter keep top-10 at the plan padded with its last digit), h =
the trunk's HC mixers and PLE key / value in 8-bit (hc8_switch.py: the pinned hc8.install, switched per arm), e.g.
p4433333333h = v3 + HC8, p44333333 = v3 + top-8, p44333333h = both. R-26 (NewIdeas Tue 20:1x): qTTT = DYNK at tau
TTT / 1000 (dynk.py; MoonShots' R-25): each verify row drops its rank-8 / rank-9 expert when that expert's
top-10-renormalized weight < tau (the rest renormalized), inside the suite's block on v3's own sorted scores (same ids,
order and tiers), trunk layers only; the prompt prefill and the MTP drafter keep top-10; the drop counts and dropped
gate mass land in the json (_dynk). TL_ONCE=ARM[,ARM]: those arms run on the first sequence only (smokes, e.g. q000 =
the DYNK block with nothing dropped, which must equal v3 bit for bit). TL_LOCAL=1 also keeps,
per position, lkld = KLD(the run's first arm || this arm) over the full vocabulary, from both arms' logits in the same
pass: with the first arm p4444444444 it isolates one layer's damage, free of the R-row floor's and other layers' cross
terms (_pos.npz key <arm>__lkld; sticky/b10_extrap.py). Quiet-window safe: waits while ~/.local/share/quiet/PAUSE exists
before every forward; a sequence whose spin gate timed out is dropped for all arms (counter reset) and the run goes on.
usage: [TL_VARIANT=plan4433333333] [TL_MTP=1] budget125_tf.py R VAR[,VAR...] SET[,SET...] EVERY
       e.g. budget125_tf.py 4 exact,b14,b16,b18,o4,o5 tf_ifeval,tf_multi,tf_code,tf_gsm 4
-> data/budget125_tf.txt (a line per variant and set), data/budget125_tf_R<R>[_<TL_VARIANT>].json, and per position
   (_pos.npz next to it: <arm>__kld / agree / lp1 = the arm's own top-1 log-prob / margin = top-1 - top-2 log-prob /
   ent / seq / pos / part (index into 'parts')) for NewIdeas' uncertainty-gated budget bet."""
import functools, json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import env  # noqa: F401,E402  (process environment first: before mlx / mtplx / bigmoe)
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from evalsuite import stages as SG, engine as EN, model as M  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_layers as DL  # noqa: E402
import dynk as DK  # noqa: E402

OUT = os.path.expanduser("~/TrainLab/data/budget125_tf.txt")
BUD = {"U": None, "k0": None, "nat": [], "used": [], "tau": None, "dyn": [], "verify": False}
CHUNK = int(os.environ.get("TL_CHUNK", "0"))   # R-31: verify passes' expert computation in <= CHUNK-row calls (every arm)
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")     # quiet windows (RULES.md 04:00): stop at a pass boundary
# B2 cold-only arms "cC_rR" (NewIdeas Sun 05:00): an LRU replay of each layer's residency at C experts (seeded by the
# prompt's rows in order, then one update per verify pass with the pass's union); a cold expert that every row of the
# pass uses at rank >= R is read and computed as MSB-2, everything else at v3's tiers (4-bit ranks 0-1, else 3-bit)
from collections import OrderedDict
BASE_B, SIDE_B = 2162688, 770048                              # records3: MSB-3 base part, LSB sidecar part
COLD = {"on": False, "C": 128, "r2": 4, "wt": None, "lru": {}, "cur": {}, "cold": {}, "two": {}, "acc": None,
        "mode": "single", "C2": 32, "lru2": {}, "lv3": {}, "lsd": {}, "pbits": [4, 4, 3, 3, 3, 3, 3, 3, 3, 3]}
Q2_B = 1540096                                                # MoonShots' 2-bit code part (budget24_q1540096.txt)


def _lru_use(lru, e, cap):
    """touch e in an LRU of capacity cap; True if it was resident"""
    if e in lru:
        lru.move_to_end(e)
        return True
    lru[e] = None
    if len(lru) > cap:
        lru.popitem(last=False)
    return False


def _observe_3b(L, ids, base):
    q2, lv3, lsd = (COLD[k].setdefault(L, OrderedDict()) for k in ("lru2", "lv3", "lsd"))
    if ids.shape[0] > 8:                                      # prompt prefill: full experts into base (and v3's pools)
        for row in ids:
            for j, e in enumerate(row):
                e = int(e)
                _lru_use(base, e, COLD["C"]); _lru_use(lv3, e, 147)
                if j < 2:
                    _lru_use(lsd, e, 46)
        COLD["cur"][L] = None
        return
    best = {}
    for row in ids:
        for j, e in enumerate(row):
            best[int(e)] = min(best.get(int(e), 99), j)
    cold = {e for e in best if e not in base}
    two = {e for e in cold if best[e] >= COLD["r2"]}
    COLD["cur"][L], COLD["cold"][L], COLD["two"][L] = ids, cold, two
    a = COLD["acc"]
    order = sorted(best, key=best.get)                        # rank order; a pass's resident experts stay pinned:
    for pool, cap, need in ((base, COLD["C"], lambda e: e not in two), (q2, COLD["C2"], lambda e: e in two),
                            (lv3, 147, lambda e: True), (lsd, 46, lambda e: best[e] < 2)):
        hit = [e for e in order if need(e) and e in pool]     # touch the hits first, then the misses
        miss = [e for e in order if need(e) and e not in pool]
        for e in hit + miss:
            _lru_use(pool, e, cap)
        if pool is base:
            a["b2_bytes"] += BASE_B * len(miss); a["cold_reads"] += len(miss)
        elif pool is q2:
            a["b2_bytes"] += Q2_B * len(miss); a["two_reads"] += len(miss); a["cold_reads"] += len(miss)
        else:
            a["v3_bytes"] += (BASE_B if pool is lv3 else SIDE_B) * len(miss)


def cold_observe(L, ids, w=None):
    """ids: np [rows, k] rank-ordered expert ids of one call at layer L; w: their top-k weights (arms cC_wT only)"""
    lru = COLD["lru"].setdefault(L, OrderedDict())
    C = COLD["C"]
    if COLD["mode"] == "3b":                                  # MoonShots' 3B: base pool C_m (MSB-3, no sidecar) + a
        return _observe_3b(L, ids, lru)                       # 2-bit cache C_2; v3's pools (147 + 46) as the reference
    if ids.shape[0] > 8:                                      # prompt prefill: replay its rows in order, no accounting
        for row in ids:
            for e in row:
                e = int(e)
                if e in lru:
                    lru.move_to_end(e)
                else:
                    lru[e] = None
                    if len(lru) > C:
                        lru.popitem(last=False)
        COLD["cur"][L] = None
        return
    best = {}
    for row in ids:
        for j, e in enumerate(row):
            best[int(e)] = min(best.get(int(e), 99), j)
    cold = {e for e in best if e not in lru}
    if COLD["wt"] is not None:                                # HOBBIT's importance rule: 2-bit if the expert's
        wmax = {}                                             # largest top-k weight over the pass's rows < T
        for row, wr in zip(ids, w):
            for e, x in zip(row, wr):
                wmax[int(e)] = max(wmax.get(int(e), 0.0), float(x))
        two = {e for e in cold if wmax[e] < COLD["wt"]}
    else:
        two = {e for e in cold if best[e] >= COLD["r2"]}
    COLD["cur"][L], COLD["cold"][L], COLD["two"][L] = ids, cold, two
    a = COLD["acc"]
    for e in cold:                                            # one read per cold expert per pass
        v3 = BASE_B + (SIDE_B if best[e] < 2 else 0)
        a["v3_bytes"] += v3
        a["b2_bytes"] += Q2_B if e in two else v3              # a 2-bit read = records4's planar P2 part (MoonShots)
        a["cold_reads"] += 1
        a["two_reads"] += e in two
    for e in sorted(best, key=lambda e: (e not in lru, best[e])):  # residents first (pinned during the pass)
        _lru_use(lru, e, C)


def pause_wait():
    while os.path.exists(PAUSE):
        time.sleep(0.2)


def peak_rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30      # bytes on macOS


class BudgetGate(nn.Module):
    """one layer's router; with BUD['U'] set and >= 2 rows, logits outside the pass's union budget become -inf"""

    def __init__(self, g, k, L=None):
        super().__init__()
        self.g, self.k, self.L = g, k, L

    def __call__(self, x):
        lg = self.g(x)
        if COLD["on"]:                                            # B2 cold-only: residency replay for this call
            lf_ = lg.astype(mx.float32).reshape(-1, lg.shape[-1])
            o_ = mx.argsort(-lf_, axis=-1)[:, :self.k]               # rank-ordered ids + their top-k weights
            w_ = mx.softmax(mx.take_along_axis(lf_, o_, axis=-1), axis=-1) if COLD["wt"] is not None else None
            cold_observe(self.L, np.array(o_), None if w_ is None else np.array(w_))
        U, k0 = BUD["U"], BUD["k0"]
        if (U is None and k0 is None) or lg.shape[-2] < 2:
            return lg
        E = lg.shape[-1]
        lf = lg.astype(mx.float32).reshape(-1, E)
        top = mx.argpartition(-lf, kth=self.k - 1, axis=-1)[:, :self.k]
        BUD["nat"].append((mx.zeros((E,), mx.float32).at[top.reshape(-1)].add(1.0) > 0).sum())
        if k0 is not None:                                           # OEA: the union of the rows' top-k0
            keep = mx.zeros((E,), mx.float32).at[mx.argpartition(-lf, kth=k0 - 1, axis=-1)[:, :k0].reshape(-1)].add(1.0) > 0
            n = keep.sum()                                           # (< 10 only for near-identical rows: then the
            keep = mx.where(n >= self.k, keep, mx.ones_like(keep))  # pass runs unrestricted)
        else:
            w = mx.softmax(mx.take_along_axis(lf, top, axis=-1), axis=-1)
            c = mx.zeros((E,), mx.float32).at[top.reshape(-1)].add((w * w).reshape(-1))
            keep = mx.zeros((E,), mx.float32).at[mx.argpartition(-c, kth=U - 1)[:U]].add(1.0) > 0
        out = mx.where(keep, lg, mx.array(-mx.inf, lg.dtype))
        sel = mx.argpartition(-out.astype(mx.float32).reshape(-1, E), kth=self.k - 1, axis=-1)[:, :self.k]
        BUD["used"].append((mx.zeros((E,), mx.float32).at[sel.reshape(-1)].add(1.0) > 0).sum())
        return out


def main():
    R, VARS, SETS, EVERY = int(sys.argv[1]), sys.argv[2].split(","), sys.argv[3].split(","), int(sys.argv[4])
    limit = float(os.environ.get("TL_LIMIT", "1e9")) - 120
    from mtplx import generation as G
    from mtplx.attention_context import attention_phase
    t_start = time.time()
    v = M.variant(os.environ.get("TL_VARIANT", "ref"))           # e.g. plan4433333333 = v3's tiers (Kamen's combined
    print(f"variant {v}", flush=True)                            # check for fast modes: budget + tiers together)
    MTP = os.environ.get("TL_MTP") == "1"                         # NewIdeas 21:10: the drafter's own confidence
    LOCAL = os.environ.get("TL_LOCAL") == "1"                     # B10 v2: KLD against the run's first arm (lkld)
    if MTP:                                                       # M.load(ref / plan) with the MTP head loaded too
        from bigmoe.loader import load_streamed
        from evalsuite import env as EV
        rt, st = load_streamed(EV.REF_MODEL, 128, mtp=True)
        for L, sw in enumerate(l.mlp.switch_mlp for l in M._text(rt.model if hasattr(rt, "model") else rt).model.layers):
            sw._bm["E_full"] = st.hdr[st.prefix_fmt.format(L=L) + ".gate_proj.weight"][3][0]
        M.install(rt.model if hasattr(rt, "model") else rt)
        M.configure(v)
    else:
        rt, st = M.load(v)
    model = rt.model if hasattr(rt, "model") else rt
    layers = M._text(model).model.layers
    for L_, l in enumerate(layers):
        l.mlp.gate = BudgetGate(l.mlp.gate, l.mlp.top_k, L_)
    if any(a[0] in "qu" for a in VARS) or CHUNK:         # R-26 DYNK / R-31 UNIQ arms: the drop inside the suite's block, on v3's own
        blk = type(layers[0].mlp)                        # sorted scores (dynk.block_dynk), trunk layers only (the MTP
        suite_block = blk.__call__                       # drafter's block keeps top-10); other calls = the suite's block
        assert (suite_block.__name__, suite_block.__module__) == ("block", "evalsuite.model"), suite_block
        TRUNK = {id(l.mlp) for l in layers}

        def dynk_or_suite(self, x):                      # TL_CHUNK: every arm's verify passes go through the copy too
            if id(self) not in TRUNK or not BUD["verify"] or (BUD["tau"] is None and not (CHUNK and x.shape[-2] > CHUNK)):
                return suite_block(self, x)
            assert getattr(self.switch_mlp, "_bm", None) is not None and M.STATE.sorted, "DYNK needs the sorted block"
            return DK.block_dynk(self, x, 0.0 if BUD["tau"] is None else BUD["tau"], M.STATE, BUD["dyn"], chunk=CHUNK)
        blk.__call__ = dynk_or_suite
        print(f"R-26 DYNK: the suite's block wrapped on {len(TRUNK)} trunk layers", flush=True)
    ONCE = set(filter(None, os.environ.get("TL_ONCE", "").split(",")))     # arms run on the first sequence only (smokes)
    MAXPOS = int(os.environ.get("TL_MAXPOS", "0"))       # R-31 (b) profile: positions per sequence (0 = all)
    PROF = os.environ.get("TL_PROF") == "1"              # ... per pass: time, the suite's tier calls, RSS, MLX memory;
    import subprocess                                    # stops at swap + 1 GB (NewIdeas 21:0x)
    swap = lambda: float(subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode().split("used = ")[1].split("M")[0])
    SWAP0 = swap() if PROF else 0.0
    DL.install_block(M)                                  # B10 v2: dense layer arms d<L> / x<L> / t<L> (dense_layers.py)
    assert {l.mlp.top_k for l in layers} == {10}
    HC = None
    if any(a.startswith("p") and a.endswith("h") for a in VARS):     # S5: HC8 arms pPLANh, switched per arm
        import hc8_switch as HC
        n_, mb0, mb1 = HC.build(model)
        print(f"hc8: {n_} modules, {mb0:.0f} MB bf16 -> {mb1:.0f} MB 8-bit (switched per arm)", flush=True)
    DSRC = {}                                            # (kind, L) -> source, built on first use

    def dsrc(kind, L):
        if (kind, L) not in DSRC:
            t0 = time.time()
            if kind == "t":                              # Fit12's decoder (TL_B10_DECODER), every expert decoded once
                DSRC[kind, L] = DL.TrSrc(L, DL.load_decoder(os.environ.get("TL_B10_DECODER", "~/Fit12/scripts/fit_b10.py")),
                                         workers=int(os.environ.get("TL_B10_WORKERS", "4")))
                print(f"  int8 step error vs the decoded weights (relative): "
                      + ", ".join(f"{lv} {e:.1e}" for lv, e in DSRC[kind, L].err.items()), flush=True)
            else:                                        # d / x share layer L's codes
                base = DSRC.get(("d", L)) or DSRC.get(("x", L))
                s = DL.V3Src.__new__(DL.V3Src)
                if base is not None:
                    s.__dict__.update(base.__dict__)
                else:
                    DL.V3Src.__init__(s, st, L, M)
                s.all4 = kind == "x"
                DSRC[kind, L] = s
            print(f"dense source {kind}{L} ready in {time.time() - t0:.0f}s, peak RSS {peak_rss():.1f} GB", flush=True)
        return DSRC[kind, L]

    for var in VARS:                                     # every dense source before the first sequence
        if var[0] in "dxt" and var[1:2].isdigit():
            for L in var[1:].partition("p")[0].split("."):
                dsrc(var[0], int(L))
    LREF = {}                                            # TL_LOCAL: the first arm's log-softmax rows, this sequence

    def local_kld(var, t, logits):
        """TL_LOCAL=1: per position KLD(first arm of VARS || arm) over the full vocabulary, same sequence and pass"""
        lf = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
        la = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
        if var == VARS[0]:
            LREF[t] = la
            mx.eval(la)
            return mx.zeros((la.shape[0],), mx.float32)
        lr = LREF[t]
        return (mx.exp(lr) * (lr - la)).sum(axis=-1)
    SEGS0 = M.STATE.segs                               # the loaded variant's plan; arms "pPLAN" set their own
    if M.STATE.segs or any(a.startswith("p") for a in VARS):   # plan variants: the suite tiers multi-row calls only on its
        sw0 = next(l.mlp.switch_mlp for l in layers   # full-E prefill scratch; a verify pass on the pool slots (BigMoE's
                   if getattr(l.mlp.switch_mlp, "_bm", None) is not None)    # gated path) asserts. Split such calls
        base = type(sw0).__mro__[1]                    # into single rows: each takes the suite's decode tier branch on
        inner = base.__call__                          # the same slots (routing stays per pass, computed in the block)

        def split_rows(self, x, indices, *a, **kw):
            bm = getattr(self, "_bm", None)
            if (COLD["on"] and bm is not None and M.STATE.tiers_prefill and indices.ndim >= 2
                    and self.up_proj.weight.shape[0] != bm.get("E_full", -1) and COLD["cur"].get(bm["L"]) is not None):
                ids, two, segs0 = COLD["cur"][bm["L"]], COLD["two"][bm["L"]], M.STATE.segs
                outs = []
                for r in range(indices.shape[-2]):                 # per row: rank 0-1 4-bit, cold 2-bit, else 3-bit
                    pb = COLD["pbits"]
                    bits = [2 if int(ids[r][j]) in two else pb[j] for j in range(indices.shape[-1])]
                    parts, perm = [], []
                    for b in (4, 3, 2):
                        pos_ = [j for j, bj in enumerate(bits) if bj == b]
                        if not pos_:
                            continue
                        M.STATE.segs = [(0, len(pos_), b)]
                        parts.append(inner(self, x[..., r:r + 1, :], indices[..., r:r + 1, :][..., mx.array(pos_)], *a, **kw))
                        perm += pos_
                    y = mx.concatenate(parts, axis=-2)
                    inv = mx.array(np.argsort(np.array(perm)).astype(np.int32))
                    outs.append(mx.take(y, inv, axis=-2))
                M.STATE.segs = segs0
                return mx.concatenate(outs, axis=-3)
            if (bm is not None and M.STATE.tiers_prefill and indices.ndim >= 2 and indices.size != indices.shape[-1]
                    and self.up_proj.weight.shape[0] != bm.get("E_full", -1)):
                return mx.concatenate([inner(self, x[..., r:r + 1, :], indices[..., r:r + 1, :], *a, **kw)
                                       for r in range(indices.shape[-2])], axis=-3)
            return inner(self, x, indices, *a, **kw)
        base.__call__ = split_rows
        print("plan variant: multi-row verify passes on pool slots run row by row through the decode tier path",
              flush=True)
    items = SG.tf_items()
    head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
    by = {s: [i for i, it in enumerate(items) if it["set"] == s][::EVERY] for s in SETS}
    seqs = [x for grp in zip(*[by[s] + [None] * (max(map(len, by.values())) - len(by[s])) for s in SETS])
            for x in grp if x is not None]                    # the parts interleaved: a stop keeps all of them
    acc = {var: {} for var in VARS}
    uni = {var: [[], []] for var in VARS}                     # per full R-row pass: (used, natural) mean over layers
    vt = os.environ.get("TL_VARIANT", "ref")
    tag = os.environ.get("TL_TAG", "")                            # e.g. b2 (NewIdeas' confidence-gated precision bet)
    jpath = os.path.expanduser(f"~/TrainLab/data/budget125_tf_R{R}{'' if vt == 'ref' else '_' + vt}{'_mtp' if MTP else ''}"
                               f"{'_' + tag if tag else ''}.json")
    res, done = {var: {} for var in VARS}, 0
    FIELDS = ("kld", "agree", "lp1", "margin", "ent", "seq", "pos", "part") + (
        ("mtp_lp1", "mtp_hit", "tgt_top_ids", "tgt_top_lp", "mtp_top_ids", "mtp_top_lp") if MTP else ()) + (
        ("cold_frac",) if any(a.startswith("c") or a.startswith("3b") for a in VARS) else ()) + (   # top-16: exact arm only
        ("lkld",) if LOCAL else ())
    BYTES = {}                                                    # B2 cold-only arms: SSD bytes per verify position
    DYNS = {}                                                     # R-26 DYNK arms: drop counts over verify row-layers
    PP = {var: {f: [] for f in FIELDS} for var in VARS}                       # for the uncertainty-gated budget bet
    ppath = jpath.replace(".json", "_pos.npz")

    def rec(var, r, positions, i, part, mtp_lp1=None, mtp_hit=None, mtp_top=None, cold_frac=None, lkld=None):
        if "cold_frac" in PP[var]:
            PP[var]["cold_frac"].append(np.full(len(positions), np.nan, np.float32) if cold_frac is None
                                        else np.asarray(cold_frac, np.float32))
        if mtp_lp1 is None:
            mtp_lp1, mtp_hit = np.full(len(positions), np.nan, np.float32), np.full(len(positions), 255, np.uint8)
        if MTP and var == "exact":                                # NewIdeas 21:20 (350 board A1, speculative-sampling
            P_ = PP[var]                                          # acceptance 1 - TV): target and drafter top-16
            P_["tgt_top_ids"].append(np.array(r["ids"])[:, :16].astype(np.int32))
            P_["tgt_top_lp"].append(np.array(r["lp"])[:, :16].astype(np.float16))
            if mtp_top is None:
                mtp_top = (np.full((len(positions), 16), -1, np.int32), np.full((len(positions), 16), np.nan, np.float16))
            P_["mtp_top_ids"].append(mtp_top[0].astype(np.int32)); P_["mtp_top_lp"].append(mtp_top[1].astype(np.float16))
        lp = np.array(r["lp"]).astype(np.float32)                              # the arm's own top-K log-probs, sorted
        P = PP[var]
        P["kld"].append(np.array(r["kld"], np.float32)); P["agree"].append(np.array(r["agree"], np.uint8))
        P["lp1"].append(lp[:, 0]); P["margin"].append(lp[:, 0] - lp[:, 1]); P["ent"].append(np.array(r["ent"], np.float32))
        P["seq"].append(np.full(len(positions), i, np.int32)); P["pos"].append(np.asarray(positions, np.int32))
        P["part"].append(np.full(len(positions), SETS.index(part), np.int8))
        if LOCAL:                                                 # KLD(first arm || arm), full vocabulary
            P["lkld"].append(np.array(lkld, np.float32))
        if MTP:                                                   # the drafter's depth-1 top-1 log-prob for the same
            P["mtp_lp1"].append(np.asarray(mtp_lp1, np.float32))   # prediction (NaN where not available) and whether
            P["mtp_hit"].append(np.asarray(mtp_hit, np.uint8))      # its top-1 is the true next token (255 = n/a)
    print(f"{len(seqs)} sequences ({ {s: len(by[s]) for s in SETS} }), arms {VARS}, R {R}; load {time.time() - t_start:.0f}s",
          flush=True)
    for n_done, i in enumerate(seqs):
        if time.time() - t_start > limit:
            print(f"time limit: stopped after {n_done} sequences", flush=True)
            break
        it = items[i]
        ids, pos = it["ids"], it["pos"]
        p0 = int(pos[0])
        t_seq = time.time()
        snap = {var: ({f: len(PP[var][f]) for f in FIELDS}, len(acc[var].get(it["set"], [])), len(uni[var][0]))
                for var in VARS}
        snapb = {var: dict(BYTES[var]) for var in BYTES}
        gate_bad = False
        LREF.clear()
        for var in VARS:
            if var in ONCE and n_done > 0:                             # TL_ONCE: a smoke arm, first sequence only
                continue
            U = int(var[1:]) if var.startswith("b") else None
            k0 = int(var[1:]) if var.startswith("o") else None
            tau_ = int(var[1:]) / 1000 if var.startswith("q") else None    # R-26 DYNK qTTT: tau = TTT / 1000, verify
            if var.startswith("u"):                                         # R-31 UNIQ uTTT[n]: check-level mass rule,
                tau_ = functools.partial(DK.uniq_scores, theta=int(var[1:].rstrip("n")) / 1000, r0=4,   # theta TTT / 1000,
                                         renorm=not var.endswith("n"))     # ranks >= 4; n = not renormalized
            pv = var[1:].rstrip("h") if var.startswith("p") else None      # B2: precision arms pPLAN; S9: K < 10
            K_ = len(pv) if pv else 10                                      # digits = top-K verify rows; S5: h = HC8
            assert K_ <= 10, var
            P10 = M.plan_segments(pv + pv[-1] * (10 - K_)) if pv else SEGS0    # the prompt and the drafter: top-10
            M.STATE.segs = P10
            if HC is not None:
                HC.use(pv is not None and var.endswith("h"))
            if var.startswith("c") or var.startswith("3b"):            # B2 cold-only arm cC_rR / MoonShots' 3B arm
                if var.startswith("3b"):                               # 3b<C_m>t<C_2>: plan 3333333333, cold ranks
                    C_, T_ = var[2:].split("t")                        # 4-9 at 2 bits, a 2-bit cache of C_2
                    COLD.update(mode="3b", C=int(C_), C2=int(T_), r2=4, wt=None, pbits=[3] * 10)
                    plan_ = "3333333333"
                elif "w" in var:                                       # cC_wT: 2-bit if cold and its largest
                    C_, T_ = var[1:].split("w")                        # top-k weight in the pass < T / 1000
                    COLD.update(mode="single", C=int(C_), r2=99, wt=int(T_) / 1000, pbits=[4, 4] + [3] * 8)
                    plan_ = "4433333333"
                else:
                    C_, R_ = var[1:].split("r")
                    COLD.update(mode="single", C=int(C_), r2=int(R_), wt=None, pbits=[4, 4] + [3] * 8)
                    plan_ = "4433333333"
                COLD.update(on=True, lru={}, lru2={}, lv3={}, lsd={}, cur={}, cold={}, two={},
                            acc=BYTES.setdefault(var, dict(v3_bytes=0.0, b2_bytes=0.0, cold_reads=0, two_reads=0, rows=0)))
                M.STATE.segs = M.plan_segments(plan_)
            else:
                COLD["on"] = False
            if var[0] in "dxt" and var[1:2].isdigit():                 # B10 v2: layers L[.L2...] dense from a source
                Ls, _, plan_ = var[1:].partition("p")                  # (d: v3's own weights, x: exact 4-bit, t:
                DL.DENSE.update(on=True, src={int(L): dsrc(var[0], int(L)) for L in Ls.split(".")})   # Fit12's trellis);
                if plan_:                                              # the other layers at plan PLAN (suffix pPLAN)
                    M.STATE.segs = M.plan_segments(plan_)              # or the loaded variant's
                assert M.STATE.segs, "dense arms need a tier plan (TL_VARIANT=plan... or a pPLAN suffix)"
            else:
                DL.DENSE["on"] = False
            M.STATE.tiers_decode = bool(M.STATE.segs)
            M.STATE.memo.clear()
            BUD["U"] = BUD["k0"] = BUD["tau"] = None                     # exact prompt
            BUD["verify"] = False
            M.STATE.tiers_prefill = False                                # deployed v3: exact 4-bit prompt prefill,
            M.STATE.pass_id += 1                                         # tiers on the verify passes (below)
            pause_wait()
            ps = G.restore_or_prefill_prompt_state(rt, [int(x) for x in ids[:p0 + 1]], base_hidden_variant=None,
                                                   mtp_history_policy="cycle", capture_hidden=MTP)
            if MTP:                                               # the drafter's state after the prompt: its committed
                mcache = ps.committed_mtp_cache if ps.committed_mtp_cache is not None else rt.make_mtp_cache()
                hprev = ps.hidden[:, -1:, :]                          # history and the last prompt hidden h_p0
            cache = ps.trunk_cache
            r = head(i, np.array([p0]), ps.logits)
            lk = local_kld(var, p0, ps.logits) if LOCAL else None
            mx.eval(r, *([lk] if LOCAL else []))
            kl, ag = [np.array(r["kld"])], [np.array(r["agree"])]
            rec(var, r, [p0], i, it["set"], lkld=lk)
            BUD["U"], BUD["k0"], BUD["tau"] = U, k0, tau_                # rows only, at the loaded variant's tiers
            BUD["verify"] = True
            M.STATE.tiers_prefill = bool(M.STATE.segs)                   # plan variants: R-row passes tiered
            if K_ < 10:                                                  # S9: verify rows keep their top K
                M.STATE.segs = M.plan_segments(pv)
                for l in layers:
                    l.mlp.top_k = K_
            t, end = p0 + 1, int(pos[-1]) + 1
            if MAXPOS:
                end = min(end, p0 + 1 + MAXPOS)
            while t < end:
                n = min(R, end - t)
                M.STATE.pass_id += 1                                     # invalidates the tiered-scratch memo (plan variants)
                pause_wait()
                tp_, s0_ = time.time(), dict(M.STATE.stats)
                with attention_phase("decode_verify"):
                    out = rt.forward_ar(mx.array([[int(x) for x in ids[t:t + n]]]), cache=cache, return_hidden=MTP)
                lg, hid = (out[0], out[1]) if MTP else (out, None)
                r = head(i, np.arange(t, t + n), lg[0])
                lk = local_kld(var, t, lg[0]) if LOCAL else None
                dl = None
                if MTP:                                           # spec125._MTPHead: the depth-1 draft of token p+1 is
                    H_ = mx.concatenate([hprev, hid[:, :-1, :]], axis=1)  # the head on the committed pair (h_(p-1),
                    if K_ < 10:                                   # S9: the drafter's block keeps top-10 (its graph
                        M.STATE.segs = P10                        # takes the plan when it is built, right here)
                    mlg = rt.draft_mtp(H_, mx.array([[int(x) for x in ids[t:t + n]]]), mtp_cache=mcache)   # token p);
                    if K_ < 10:
                        M.STATE.segs = M.plan_segments(pv)
                    mlg = mlg[0] if isinstance(mlg, (tuple, list)) else mlg       # teacher-forced, every pair commits
                    mf = mlg[0].astype(mx.float32)
                    dl = (mf - mx.logsumexp(mf, axis=-1, keepdims=True)).max(axis=-1)
                    dh = mx.argmax(mf, axis=-1) == mx.array([int(x) for x in ids[t + 1:t + n + 1]])
                    hprev = hid[:, -1:, :]
                    mt = None
                    if var == "exact":
                        lpm = mf - mx.logsumexp(mf, axis=-1, keepdims=True)
                        ti = mx.argpartition(-lpm, kth=15, axis=-1)[:, :16]
                        tv = mx.take_along_axis(lpm, ti, axis=-1)
                        o_ = mx.argsort(-tv, axis=-1)
                        mt = (mx.take_along_axis(ti, o_, axis=-1), mx.take_along_axis(tv, o_, axis=-1))
                    mx.eval(dl, dh, *(mt or ()))
                mx.eval(r, *BUD["nat"], *BUD["used"], *BUD["dyn"], *([lk] if LOCAL else []))
                if PROF:
                    sw_ = swap()
                    rss_ = int(os.popen(f"ps -o rss= -p {os.getpid()}").read().strip() or 0) / 2 ** 20
                    print(f"PROF {var} R {R} pass at {t}: {n} rows {time.time() - tp_:.2f}s ({(time.time() - tp_) / n:.3f} s/pos); "
                          + ", ".join(f"{k} +{M.STATE.stats[k] - s0_.get(k, 0)}" for k in M.STATE.stats)
                          + f"; RSS {rss_:.1f} GB (peak {peak_rss():.1f}), MLX active "
                          f"{getattr(mx, 'get_active_memory', lambda: 0)() / 2 ** 30:.1f} GB peak "
                          f"{getattr(mx, 'get_peak_memory', lambda: 0)() / 2 ** 30:.1f} GB; swap {sw_ / 1024:.1f} GB "
                          f"({(sw_ - SWAP0) / 1024:+.2f})", flush=True)
                    if sw_ > SWAP0 + 1024:
                        print("PROF: swap grew > 1 GB: stopping", flush=True)
                        os._exit(3)
                if BUD["dyn"]:                                     # R-26: rows, dropping rank 9, dropping 8 + 9, mass
                    DYNS[var] = DYNS.get(var, 0) + np.sum([np.array(a) for a in BUD["dyn"]], axis=0)
                    BUD["dyn"].clear()
                kl.append(np.array(r["kld"])); ag.append(np.array(r["agree"]))
                cf = None
                if COLD["on"]:                                         # per row: share of its rank >= 2 experts cold
                    COLD["acc"]["rows"] += n
                    fr = [[sum(int(e) in COLD["cold"][L_] for e in COLD["cur"][L_][j][2:]) / (len(COLD["cur"][L_][j]) - 2)
                           for L_ in COLD["cur"] if COLD["cur"][L_] is not None] for j in range(n)]
                    cf = np.array([np.mean(f) if f else np.nan for f in fr], np.float32)
                rec(var, r, np.arange(t, t + n), i, it["set"], None if dl is None else np.array(dl),
                    None if dl is None else np.array(dh).astype(np.uint8),
                    None if (dl is None or mt is None) else (np.array(mt[0]), np.array(mt[1])), cf, lk)
                if n == R and BUD["nat"]:
                    uni[var][0].append(float(np.mean([float(a) for a in BUD["used"]])))
                    uni[var][1].append(float(np.mean([float(a) for a in BUD["nat"]])))
                BUD["nat"].clear(); BUD["used"].clear()
                t += n
            BUD["U"] = BUD["k0"] = BUD["tau"] = None
            BUD["verify"] = False
            for l in layers:
                l.mlp.top_k = 10
            try:
                st.gate.check()
            except RuntimeError as e:                                  # a timed-out spin gate: this sequence is
                gate_bad = True                                        # invalid for every arm; drop it, continue
                np.frombuffer(memoryview(st.gate.status), np.int32)[0] = 0
                print(f"  {it['key']} arm {var}: {e} -> sequence dropped", flush=True)
            acc[var].setdefault(it["set"], []).append((np.concatenate(kl), np.concatenate(ag)))
        if gate_bad:                                               # roll every arm back to the sequence's start
            for var in VARS:
                lens, na, nu = snap[var]
                for f in FIELDS:
                    del PP[var][f][lens[f]:]
                if it["set"] in acc[var]:
                    del acc[var][it["set"]][na:]
                del uni[var][0][nu:]; del uni[var][1][nu:]
                if var in snapb:
                    BYTES[var].update(snapb[var])
            print(f"[{n_done + 1}/{len(seqs)}] {it['key']}: DROPPED (spin gate timeout)", flush=True)
            continue
        print(f"[{n_done + 1}/{len(seqs)}] {it['key']}: {len(pos)} positions x {len(VARS)} arms in "
              f"{time.time() - t_seq:.0f}s", flush=True)
        res = {}
        for var in VARS:
            res[var] = {}
            for s, lst in acc[var].items():
                k = np.concatenate([a for a, _ in lst]); g = np.concatenate([b for _, b in lst])
                res[var][s] = dict(kld=float(k.mean()), p999=float(np.quantile(k, 0.999)), agree=float(g.mean()),
                                   n=int(k.size), seqs=len(lst))
            if var in BYTES and BYTES[var]["rows"]:                     # B2: SSD bytes per verify position
                b = BYTES[var]; nr = b["rows"]
                res[var]["_bytes"] = dict(v3_MB_per_tok=b["v3_bytes"] / nr / 1e6, b2_MB_per_tok=b["b2_bytes"] / nr / 1e6,
                                          saved=1 - b["b2_bytes"] / max(b["v3_bytes"], 1.0), cold_reads_per_tok=b["cold_reads"] / nr,
                                          two_reads_per_tok=b["two_reads"] / nr)
            if uni[var][0]:
                res[var]["_union"] = dict(used=float(np.mean(uni[var][0])), natural=float(np.mean(uni[var][1])))
            if var in DYNS and var.startswith("q"):                     # R-26: shares of verify row-layers
                d_ = DYNS[var]
                res[var]["_dynk"] = dict(row_layers=float(d_[0]), drop_any=float(d_[1] / d_[0]), drop_both=float(d_[2] / d_[0]),
                                         experts_dropped_per_row=float((d_[1] + d_[2]) / d_[0]), mass=float(d_[3] / d_[0]))
            if var in DYNS and var.startswith("u"):                     # R-31: rows with a drop, dropped uses, mass
                d_ = DYNS[var]
                res[var]["_uniq"] = dict(row_layers=float(d_[0]), rows_with_drop=float(d_[1] / d_[0]),
                                         dropped_per_row=float(d_[2] / d_[0]), mass=float(d_[3] / d_[0]))
        json.dump(res, open(jpath, "w"))
        np.savez(ppath, parts=np.array(SETS), **{f"{var}__{f}": np.concatenate(PP[var][f]) for var in VARS for f in FIELDS
                                               if PP[var][f]})
        done += 1
    lines = [f"{time.strftime('%m-%d %H:%M')} budget125_tf R={R} variant {vt}: {done} sequences, "
             f"{time.time() - t_start:.0f}s, peak RSS {peak_rss():.1f} GB"]
    for var in VARS:
        u = res[var].get("_union")
        lines.append(f"  {var:6s} " + " | ".join(f"{s} KLD {res[var][s]['kld']:.4f} top-1 {res[var][s]['agree']:.3f} "
                                                 f"(n {res[var][s]['n']})" for s in SETS if s in res[var])
                     + (f" | distinct experts per pass/layer {u['used']:.1f} of natural {u['natural']:.1f} "
                        f"({u['used'] / u['natural']:.2f}x)" if u else "")
                     + (" | SSD per token v3 {v3_MB_per_tok:.1f} MB -> {arm} {b2_MB_per_tok:.1f} MB ({saved:.1%} saved), "
                        "cold reads {cold_reads_per_tok:.1f} / tok of which 2-bit {two_reads_per_tok:.1f}".format(arm=var, **res[var]["_bytes"])
                        if "_bytes" in res[var] else "")
                     + (" | DYNK over {row_layers:.0f} verify row-layers: rank 9 dropped {drop_any:.1%}, ranks 8 + 9 "
                        "{drop_both:.1%}, {experts_dropped_per_row:.2f} experts and {mass:.2%} of the top-10 gate mass "
                        "dropped per row-layer".format(**res[var]["_dynk"]) if "_dynk" in res[var] else "")
                     + (" | UNIQ over {row_layers:.0f} verify row-layers: {rows_with_drop:.1%} of rows drop, {dropped_per_row:.2f} "
                        "experts and {mass:.2%} of the top-10 gate mass dropped per row-layer".format(**res[var]["_uniq"])
                        if "_uniq" in res[var] else ""))
    print("\n".join(lines), flush=True)
    open(OUT, "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
