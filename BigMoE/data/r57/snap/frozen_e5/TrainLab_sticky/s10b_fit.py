"""S10b long-lead expert predictor: offline fit on BigMoE's S10b trace (NewIdeas Sun 25 Oct 15:27, QUEUE.md row S10b;
b31 Mon 08:22; b32 = b31 plus residency, Mon 10:14).
CPU only (numpy, 4 BLAS threads). Predicts layer T = L + d's routed experts (lead d = 1, 2, 3) from layer L's MoE input
(the router's input) at the trace's 8 predictor layers L, to drive prefetch only (decode stays exact).
Models per (L, d), in NewIdeas' order; each starts from M0's scores (step 0 = the plain router):
  M0  plain router: router_T applied to x_L as is (the engine's look-ahead; = BigMoE s10b_check.py's baseline).
  M1  linear on top of router_T: a * router_T(x_L) + z_L dW + b   (z = x_L standardized on the training rows).
  M1a M1 + an inter-layer affinity table (published: expert affinity across layers): + the sum of A[e] over the 10
      experts routed at L itself (known as soon as x_L is).
  M2  small MLP: a * router_T(x_L) + relu(z W1 + c) W2 + b (b31: only if the best linear model misses 65 % on
      validation; b32: always tried).
  Loss: softmax cross-entropy against T's 10 routed ids (1/10 each); AdamW (batch 256), lr in {3e-4, 1e-3, 3e-3} x
  decoupled weight decay wd in {0.1, 10}, early stopping by epoch (<= 10). No other sweeps.
Split by PROMPT, never by row. Selection (lead 2 only, pooled over the 8 layers): train on prompts 0-2 of each domain,
validate on prompt 3 -> (model, wd, epochs); M2 replaces the best linear model only if it is >= 1 point better there.
Judge (primary): refit on prompts 0-3, score prompt 4 of each domain (held out, never used in selection).
Secondary: leave-one-prompt-index-out over 0..4 (5 folds, each trained on the other four prompts of every domain), pooled.
Join (BigMoE's rule): per (pass, layer) the FIRST call in file order; MTPLX's commit re-runs layer 1 over the kept rows of
a partly accepted verify under the same seq (fewer rows, all resident, not in miss.bin), which those later calls are.
Scores (s10b_check.py's definitions; a row's misses = its routed ids at T that miss.bin flags not resident in that pass),
summed over rows, layers and domains:
  b31: recall@16 of the misses, top-16 over all 512 experts per row.
  b32 (res.bin: pool T's residency at base layer L's prefetch decision, T = L+1..L+3): DEPLOYABLE recall@k, the row's
  top-k among the experts NOT resident at the decision (the engine skips those), k = 2 / 4 / 8 / 12 / 16; and the read
  budget per pass and target layer (the union of the pass's rows' picks = what the SSD reads): recall of the pass's
  misses, unique reads, reads never routed at T in the pass, precision, for the top-k among non-resident (k = 1..16)
  and the top-K over all 512 with resident skipped (K = 4..64; all12 on M0 = the shipped engine's lead-1 policy). The
  catch at READS = 4.8 unique reads per pass-layer is linear in reads between the two budgets around it. Selection
  uses that catch (lead 2, top-k among non-resident).
Lead 2 decides; leads 1 and 3 beside it; recall@16 of all routed ids for context.
Pre-registered (REPORT.md 'S10b'): b31: held-out lead-2 recall@16 of misses < 55 % -> KILL; >= 65 % -> engine A/B.
b32 (NewIdeas 13:46, the bar in reads): held-out lead-2 per-pass catch at 4.8 reads per pass-layer >= 50.4 % (the shipped
lead-1 policy's catch at that budget) -> pass (b33 labels, then an engine A/B design); < 45 % -> park S10b; between ->
b33 labels decide. (Superseded, NewIdeas 09:46: per-row deployable recall@16 >= 65 % A/B, < 55 % stop.)
EXPLORATORY (not pre-registered; S10B_TARGET=miss): the same models, grid, split and judge, but trained on the row's
MISSED ids at T only (1/|misses| each; rows without misses at T leave the loss), i.e. the objective the score counts.
S10c-1 (NewIdeas 10-26 14:08; REPORT 'S10c-1'): the model FROZEN, no refit on the scored trace. S10B_FROZEN=M1a,3e-4,10,5
skips the selection and takes that recipe (b32's pick); S10B_TRAIN=DIR trains it there (b32: S10b's splits and seeds,
so S10b's models, whose lines on DIR's own held-out prompts are printed beside) and scores TRACE_DIR (b33: its own x,
residency and labels, no join); S10B_LABEL=ld counts late + demand misses only (b33's miss.bin entries are
e | miss << 16 | kind << 17, kind 0 demand, 1 on time, 2 late, 3 queued: drop kind 1, as s10b_check.py). Verdict:
held-out lead-2 per-pass catch at 4.8 reads per pass-layer in the shipped rule's family (top-K over all 512, resident
skipped), M1a minus the router: PASS >= +10 points, KILL < +5, held-out and leave-one-out both; anything else between.
S10c (2) export (S10B_EXPORT=DIR, linear models only; BigMoE's format): the lead-2 judge models (training trace's
prompts 0-3, the ones scored on held-out prompt 4) folded into one map on x for the engine, s = x Wf + bf + the sum of
A[e] over the experts routed at L (Wf = a W_T + dW / sd, bf = b - (mu / sd) dW): DIR/m1a_8pairs.npz (L{L}_Wf f16,
L{L}_bf f32, L{L}_A f16), m1a_8pairs.json (layout, formula; the folded f16 copies scored beside the f32 models) and
m1a_8pairs_check.json (top-12 per row for the first 3 calls of the first predictor layer in p04_code4).
All-layer export (NewIdeas 20:4x, after b34): S10B_LEADS=2 fits lead 2 only, on every trace layer whose target exists
(L + 2 < NL), and S10B_EXPORT_NAME names the files (default m1a_8pairs).
usage: [S10B_TARGET=miss] [S10B_LABEL=ld] [S10B_LEADS=2] [S10B_FROZEN=KIND,LR,WD,EPOCHS [S10B_TRAIN=DIR] [S10B_EXPORT=DIR
  [S10B_EXPORT_NAME=STEM]]]
  s10b_fit.py TRACE_DIR [TAG]  -> data/s10b_fit[_TAG].txt and .json"""
import json, os, sys, time
from types import SimpleNamespace
for _k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import numpy as np

D = os.path.expanduser(sys.argv[1])
TAG = "_" + sys.argv[2] if len(sys.argv) > 2 else ""
OUT = os.path.expanduser(os.environ.get("TL_DATA", "~/TrainLab/data"))
EPOCHS, BATCH, HID = int(os.environ.get("S10B_EPOCHS", 10)), 256, 512
GRID = [(lr, wd) for lr in (3e-4, 1e-3, 3e-3) for wd in (0.1, 10.0)]
LEADS, TOP = tuple(int(x) for x in os.environ.get("S10B_LEADS", "1,2,3").split(",")), 16    # all-layer export: S10B_LEADS=2
assert 2 in LEADS, "lead 2 decides (selection, verdicts, export)"
KS = (2, 4, 8, 12, 16)                                   # per row: the top-k among non-resident (deployable recall@k)
KF, KA = (1, 2, 3, 4, 5, 6, 8, 12, 16), (4, 6, 8, 12, 16, 24, 32, 48, 64)
POLS = tuple(f"k{k}" for k in KF) + tuple(f"all{k}" for k in KA)   # read policies per pass: top-k among non-resident,
SHOWN = ("k2", "k4", "k8", "k12", "k16", "all12")        # top-K over all 512 resident skipped; s10b_check's sweep
READS = 4.8      # the shipped lead-1 policy's unique reads per pass-layer (BigMoE check2: 50.4 % of the misses held-out)
C_DM, C_PL = 4, 5 + len(KS)            # count() columns: deployable misses (then hits per k); pass-layers (then misses,
NC = C_PL + 2 + 3 * len(POLS)          # then caught / reads / unused per policy)
TARGET = os.environ.get("S10B_TARGET", "routed")
assert TARGET in ("routed", "miss"), TARGET
LD = os.environ.get("S10B_LABEL", "") == "ld"            # S10c: late + demand misses only (drop kind 1, on time)
FROZEN = os.environ.get("S10B_FROZEN")                   # S10c: "KIND,LR,WD,EPOCHS", no selection
TRAIN_D = os.path.expanduser(os.environ.get("S10B_TRAIN", "")) or None   # S10c: train there, score TRACE_DIR
assert FROZEN or not TRAIN_D, "S10B_TRAIN needs S10B_FROZEN: no selection on the scored trace"
EXPORT = os.path.expanduser(os.environ.get("S10B_EXPORT", "")) or None   # S10c (2): folded lead-2 weights for the engine
assert FROZEN or not EXPORT, "S10B_EXPORT needs S10B_FROZEN"
meta = json.load(open(os.path.join(D, "meta.json")))
NL, LP, E, K = meta["n_layers"], meta["layers_pred"], meta["E"], meta["K"]
LP_ALL, LP = LP, [L for L in LP if L + max(LEADS) < NL]   # an all-layer trace: only layers whose targets exist
EXPN = os.environ.get("S10B_EXPORT_NAME", "m1a_8pairs")   # the export's file stem (the all-layer one: m1a_46pairs)
PR = meta["prompts"]
DOMS = list(dict.fromkeys(q["domain"] for q in PR))
DOM = np.array([DOMS.index(q["domain"]) for q in PR])
IDX = np.array([int(q["idx"]) for q in PR])
T0, LOG = time.time(), []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


W = {int(k[1:]): v.astype(np.float32).T.copy() for k, v in np.load(os.path.join(D, "routers.npz")).items()}  # [D, E]


def load(D, head):
    """one trace in BigMoE's layout -> its rows at the predictor layers; `head` starts its summary line"""
    mt = json.load(open(os.path.join(D, "meta.json")))
    assert (mt["n_layers"], mt["layers_pred"], mt["E"], mt["K"]) == (NL, LP_ALL, E, K) and [(q["name"], q["domain"], q["idx"])
           for q in mt["prompts"]] == [(q["name"], q["domain"], q["idx"]) for q in PR], f"{D}: not the same layout / prompts"
    # misses: (pass, trunk layer) -> the experts served there that were not resident (first record wins, as s10b_check);
    # b33's kinds: 0 demand, 1 on time (the prefetch was ready when T's service started), 2 late, 3 queued; ld drops kind 1
    raw = np.fromfile(os.path.join(D, "miss.bin"), np.int32)
    MISS, dup, kinds, i = {}, 0, np.zeros(4, np.int64), 0
    while i < len(raw):
        seq, L, n = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
        v = raw[i + 3:i + 3 + n]
        if L < NL:
            if (seq, L) in MISS:
                dup += 1
            else:
                mk = ((v >> 16) & 1) == 1
                kinds += np.bincount((v[mk] >> 17) & 3, minlength=4)
                if LD:
                    mk &= ((v >> 17) & 3) != 1
                MISS[(seq, L)] = np.sort(v[mk] & 0xFFFF)
        i += 3 + n

    # residency (b32's res.bin): (pass, L, T) -> offset of pool T's words at base layer L's prefetch decision, predictor
    # layers only (first record wins, as s10b_check); bit e % 32 of word e // 32 = expert e resident; T = NL is the MTP pool
    RESB, rr, RW = {}, None, (E + 31) // 32
    if os.path.exists(os.path.join(D, "res.bin")):
        rr, i, RW = np.fromfile(os.path.join(D, "res.bin"), "<i4"), 0, mt["res"]["W"]
        while i < len(rr):
            seq, L, nT = int(rr[i]), int(rr[i + 1]), int(rr[i + 2])
            i += 3
            for _ in range(nT):
                if L in LP:
                    RESB.setdefault((seq, L, int(rr[i])), i + 1)
                i += 1 + RW

    # rows: every verify pass (a seq with all trunk layers, equal rows), at each predictor layer L
    X = {L: [] for L in LP}
    IDL = {L: [] for L in LP}
    PN = {L: [] for L in LP}
    SEQ = {L: [] for L in LP}
    IDT = {(L, d): [] for L in LP for d in LEADS}
    MS = {(L, d): [] for L in LP for d in LEADS}
    OK = {(L, d): [] for L in LP for d in LEADS}
    RB = {(L, d): [] for L in LP for d in LEADS}       # pool T's residency at L's decision, per row (the record's bytes)
    RK = {(L, d): [] for L in LP for d in LEADS}       # ... and whether the pass has that record
    cnt = dict(passes=0, rerun_calls=0, mtp_only=0, partial=0, unequal_rows=0, no_x=0, no_miss_record=0, no_res=0)
    owner = {}
    for pn, q in enumerate(PR):
        p = dict(np.load(os.path.join(D, q["name"] + ".npz")))
        by = {}
        for j, (s, L) in enumerate(zip(p["seq"].tolist(), p["L"].tolist())):
            c = by.setdefault(s, {})
            if L in c:                     # BigMoE's join rule: the FIRST call per (pass, layer); MTPLX's commit re-runs
                cnt["rerun_calls"] += 1    # layer 1 over the kept rows of a partly accepted verify under the same seq
            else:
                c[L] = j
        for s, calls in by.items():
            assert owner.setdefault(s, pn) == pn, f"pass {s} appears in two prompts: miss.bin keys would be ambiguous"
            if not all(L in calls for L in range(NL)):
                cnt["mtp_only" if set(calls) == {NL} else "partial"] += 1
                continue
            R = {int(p["rows"][calls[L]]) for L in range(NL)}
            if len(R) != 1:
                cnt["unequal_rows"] += 1
                continue
            R = R.pop()
            cnt["passes"] += 1

            def ids(L):
                o = int(p["row_off"][calls[L]])
                return p["ids"][o:o + R].astype(np.int64)
            for L in LP:
                jo = int(p["x_off"][calls[L]])
                if jo < 0:
                    cnt["no_x"] += 1
                    continue
                X[L].append(p["x"][jo:jo + R])
                IDL[L].append(ids(L))
                PN[L].append(np.full(R, pn, np.int32))
                SEQ[L].append(np.full(R, s, np.int64))
                for d in LEADS:
                    it, m = ids(L + d), MISS.get((s, L + d))
                    cnt["no_miss_record"] += m is None
                    IDT[L, d].append(it)
                    MS[L, d].append(np.isin(it, m) if m is not None else np.zeros(it.shape, bool))
                    OK[L, d].append(np.full(R, m is not None))
                    ro = RESB.get((s, L, L + d))
                    cnt["no_res"] += rr is not None and ro is None
                    RB[L, d].append(np.repeat(rr[ro:ro + RW].view(np.uint8)[None], R, 0) if ro is not None
                                    else np.zeros((R, 4 * RW), np.uint8))
                    RK[L, d].append(np.full(R, ro is not None))
    cat = lambda a: {k: np.concatenate(v) for k, v in a.items()}
    X, IDL, PN, SEQ, IDT, MS, OK, RB, RK = cat(X), cat(IDL), cat(PN), cat(SEQ), cat(IDT), cat(MS), cat(OK), cat(RB), cat(RK)
    nrows = len(PN[LP[0]])
    say(f"{head} {D} (targets: {TARGET} ids{'' if TARGET == 'routed' else ', EXPLORATORY'}): {len(PR)} prompts ({', '.join(f'{d} {int((DOM == k).sum())}' for k, d in enumerate(DOMS))}), "
        f"{cnt['passes']} verify passes, {nrows} rows per predictor layer, layers {LP}; miss.bin {len(MISS)} trunk records "
        f"({dup} duplicates; misses by kind: demand {kinds[0]}, on time {kinds[1]}, late {kinds[2]}, queued {kinds[3]}"
        f"{', scored: late + demand (S10B_LABEL=ld)' if LD else ''}); skipped: {cnt['rerun_calls']} re-run calls (first call per pass and layer kept), "
        f"{cnt['mtp_only']} MTP-only seqs, {cnt['partial']} partial, {cnt['unequal_rows']} "
        f"unequal-row passes, {cnt['no_x']} without x; (row-layer, lead) without a miss record: {cnt['no_miss_record']}"
        + (f"; res.bin: {len(RESB)} (pass, L -> T) records at the predictor layers, scored (pass, L -> T) without one: "
           f"{cnt['no_res']}" if rr is not None else "; no res.bin: no deployable score"))
    assert all(len(PN[L]) == nrows for L in LP) and nrows > 0
    return SimpleNamespace(D=D, X=X, IDL=IDL, PN=PN, SEQ=SEQ, IDT=IDT, MS=MS, OK=OK, RB=RB, RK=RK, RESB=RESB, cnt=cnt,
                           nrows=nrows, dom=DOM[PN[LP[0]]], idx=IDX[PN[LP[0]]])


SC = load(D, "S10b fit on")                                                   # the scored trace (TRACE_DIR)
TRS = load(TRAIN_D, "S10c: the frozen recipe is trained on") if TRAIN_D else SC   # the training trace
if TRS is not SC:
    w2 = np.load(os.path.join(TRAIN_D, "routers.npz"))
    assert sorted(int(k[1:]) for k in w2.files) == sorted(W) and all(
        np.array_equal(w2[k].astype(np.float32).T, W[int(k[1:])]) for k in w2.files), "the two traces' routers differ"
RESB, nrows = SC.RESB, SC.nrows


def topk_mask(s):
    m = np.zeros(s.shape, bool)
    np.put_along_axis(m, np.argpartition(-s, TOP - 1, axis=1)[:, :TOP], True, axis=1)
    return m


class Model:
    def __init__(self, kind, dm, rng, lr):
        self.lr = lr
        P = {"a": np.ones(1, np.float32), "b": np.zeros(E, np.float32)}
        if kind in ("M1", "M1a"):
            P["dW"] = np.zeros((dm, E), np.float32)
        if kind == "M1a":
            P["A"] = np.zeros((E, E), np.float32)
        if kind == "M2":
            P["W1"] = (rng.standard_normal((dm, HID)) * np.sqrt(2 / dm)).astype(np.float32)
            P["c1"], P["W2"] = np.zeros(HID, np.float32), np.zeros((HID, E), np.float32)
        self.P, self.t = P, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def fwd(self, z, r, oh):
        P, h = self.P, None
        s = P["a"] * r + P["b"]
        if "dW" in P:
            s += z @ P["dW"]
        if "A" in P:
            s += oh @ P["A"]
        if "W1" in P:
            h = np.maximum(z @ P["W1"] + P["c1"], 0)
            s += h @ P["W2"]
        return s, h

    def step(self, z, r, oh, h, G, wd):
        P, g = self.P, {"a": np.array([(G * r).sum()], np.float32), "b": G.sum(0)}
        if "dW" in P:
            g["dW"] = z.T @ G
        if "A" in P:
            g["A"] = oh.T @ G
        if "W1" in P:
            g["W2"] = h.T @ G
            gh = (G @ P["W2"].T) * (h > 0)
            g["W1"], g["c1"] = z.T @ gh, gh.sum(0)
        self.t += 1
        for k, gk in g.items():
            m, v = self.m[k], self.v[k]
            m *= 0.9
            m += 0.1 * gk
            v *= 0.999
            v += 0.001 * gk * gk
            if k in ("dW", "A", "W1", "W2"):
                P[k] *= 1 - self.lr * wd
            P[k] -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


class Data:
    """one (L, d) problem: standardized x (stats from the training rows), router_T(x_L), multi-hot ids at L, and pool
    T's residency at L's prefetch decision (b32)"""
    def __init__(self, L, d, tr, S=None, ref=None):
        S = S or SC
        x = bf16(S.X[L])
        if ref is None:
            mu, sd = x[tr].mean(0), x[tr].std(0)
            self.mu, self.sd = mu, np.maximum(sd, 1e-3 * np.median(sd) + 1e-12)
        else:                       # S10c: another trace's rows under the training trace's statistics (model frozen)
            self.mu, self.sd = ref.mu, ref.sd
        self.z = (x - self.mu) / self.sd
        self.x, self.T = x, L + d
        self.r = x @ W[L + d]
        self.oh = np.zeros((len(x), E), np.float32)
        np.put_along_axis(self.oh, S.IDL[L], 1.0, axis=1)
        self.it, self.ms, self.ok = S.IDT[L, d], S.MS[L, d], S.OK[L, d]
        self.rb = np.unpackbits(S.RB[L, d], axis=1, bitorder="little")[:, :E].astype(bool)
        self.rk, self.sq, self.dom = S.RK[L, d], S.SEQ[L], S.dom
        if TARGET == "routed":
            self.tw = np.full(self.it.shape, 1.0 / K, np.float32)          # target weight of each routed id
        else:
            nm = self.ms.sum(1, keepdims=True)
            self.tw = np.where(self.ms, 1.0 / np.maximum(nm, 1), 0).astype(np.float32)

    def scores(self, mdl, ix):
        if mdl is None:
            return self.r[ix]
        ch = np.array_split(ix, max(1, len(ix) // 2048))
        if isinstance(mdl, Folded):
            return np.concatenate([self.x[c] @ mdl.Wf + mdl.bf + self.oh[c] @ mdl.A for c in ch])
        return np.concatenate([mdl.fwd(self.z[c], self.r[c], self.oh[c])[0] for c in ch])

    def fold(self, mdl):
        """S10c (2): a linear model as one map on x (the engine's form): s = x Wf + bf + sum of A over L's routed experts"""
        P = mdl.P
        assert "W1" not in P, "only linear models fold"
        dW = P.get("dW", np.zeros((self.x.shape[1], E), np.float32))
        return {"Wf": (P["a"][0] * W[self.T] + dW / self.sd[:, None]).astype(np.float16),
                "bf": (P["b"] - (self.mu / self.sd) @ dW).astype(np.float32),
                "A": P.get("A", np.zeros((E, E), np.float32)).astype(np.float16)}

    def count(self, mdl, ix):
        """per domain, over rows ix that have a miss record: [0-3] top-16 over all 512 (b31's score): hits on misses,
        misses, hits on routed, routed. DEPLOYABLE (rows whose pass has a residency record): [C_DM] misses, then the
        hits of the row's top-k among the experts not resident at the decision, k in KS. Read budget per pass and target
        layer (the union over the pass's rows, s10b_check's sweep): [C_PL] pass-layers, [C_PL + 1] the pass's misses,
        then per policy in POLS: misses caught, unique reads, reads never routed at T in the pass."""
        ix = ix[self.ok[ix]]
        s, it, ms = self.scores(mdl, ix), self.it[ix], self.ms[ix]
        V = np.zeros((len(ix), NC))
        inn = np.take_along_axis(topk_mask(s), it, axis=1)
        V[:, :4] = np.stack([(inn & ms).sum(1), ms.sum(1), inn.sum(1), np.full(len(ix), K)], 1)
        jx = np.flatnonzero(self.rk[ix])
        if len(jx):
            s, it, ms, rb, sq = s[jx], it[jx], ms[jx], self.rb[ix[jx]], self.sq[ix[jx]]
            mm, rt = np.zeros(rb.shape, bool), np.zeros(rb.shape, bool)       # the row's misses / routed ids at T
            np.put_along_axis(mm, it, ms, axis=1)
            np.put_along_axis(rt, it, True, axis=1)
            sd = np.where(rb, -np.inf, s)                                      # the engine skips resident experts
            pk = np.argpartition(-sd, max(KF) - 1, axis=1)[:, :max(KF)]
            pk = np.take_along_axis(pk, np.argsort(-np.take_along_axis(sd, pk, axis=1), axis=1, kind="stable"), axis=1)
            pa = np.argpartition(-s, min(max(KA), E) - 1, axis=1)[:, :min(max(KA), E)]   # over all 512 (the engine's rule)
            pa = np.take_along_axis(pa, np.argsort(-np.take_along_axis(s, pa, axis=1), axis=1, kind="stable"), axis=1)
            hit = np.take_along_axis(mm & ~rb, pk, axis=1)
            V[jx, C_DM] = ms.sum(1)
            V[jx, C_DM + 1:C_PL] = np.stack([hit[:, :k].sum(1) for k in KS], 1)
            st = np.flatnonzero(np.r_[True, sq[1:] != sq[:-1]])               # each pass's first row (contiguous)
            M, RT = np.logical_or.reduceat(mm, st, axis=0), np.logical_or.reduceat(rt, st, axis=0)
            V[jx[st], C_PL], V[jx[st], C_PL + 1] = 1, M.sum(1)
            for j, pol in enumerate(POLS):
                pm = np.zeros(rb.shape, bool)
                np.put_along_axis(pm, (pk if pol[0] == "k" else pa)[:, :int(pol.lstrip("kal"))], True, axis=1)
                U = np.logical_or.reduceat(pm & ~rb, st, axis=0)
                V[jx[st], C_PL + 2 + 3 * j:C_PL + 5 + 3 * j] = np.stack([(U & M).sum(1), U.sum(1), (U & ~RT).sum(1)], 1)
        c = np.zeros((len(DOMS), NC))
        np.add.at(c, self.dom[ix], V)
        return c

    def fit(self, kind, lr, wd, tr, epochs, rng, val=None):
        mdl, hist = Model(kind, self.z.shape[1], rng, lr), []
        tr = tr[self.tw[tr].sum(1) > 0]
        for _ in range(epochs):
            perm = rng.permutation(tr)
            for b0 in range(0, len(perm), BATCH):
                ix = perm[b0:b0 + BATCH]
                z, r, oh = self.z[ix], self.r[ix], self.oh[ix]
                s, h = mdl.fwd(z, r, oh)
                s -= s.max(1, keepdims=True)
                pr = np.exp(s)
                pr /= pr.sum(1, keepdims=True)
                np.put_along_axis(pr, self.it[ix], np.take_along_axis(pr, self.it[ix], axis=1) - self.tw[ix], axis=1)
                mdl.step(z, r, oh, h, pr / len(ix), wd)
            if val is not None:
                hist.append(self.count(mdl, val))
        return mdl, hist


class Folded:
    """an exported (folded, f16) model, scored in f32 like an engine that accumulates in f32"""
    def __init__(self, f):
        self.Wf, self.bf, self.A = (f[k].astype(np.float32) for k in ("Wf", "bf", "A"))


rec = lambda c: c[..., 0].sum() / max(c[..., 1].sum(), 1)                          # top-16 over all 512 (b31's score)
dep = lambda c, k=TOP: c[..., C_DM + 1 + KS.index(k)].sum() / max(c[..., C_DM].sum(), 1)   # DEPLOYABLE recall@k


def bud(c, pol):
    """read budget per pass and target layer: recall of the pass's misses, unique reads, reads never routed, precision"""
    j = C_PL + 2 + 3 * POLS.index(pol)
    n, m, ca, rd, un = (c[..., x].sum() for x in (C_PL, C_PL + 1, j, j + 1, j + 2))
    return ca / max(m, 1), rd / max(n, 1), un / max(n, 1), ca / max(rd, 1)


def at_reads(c, fam, target=READS):
    """(recall of the pass's misses at `target` unique reads per pass-layer, lower, upper policy): linear in reads between
    the family's two budgets around it (NewIdeas 13:46: the router's ~34 % at lead 2 is this of k2 / k4); None outside"""
    pts = sorted((bud(c, p)[1], bud(c, p)[0], p) for p in POLS if p.startswith(fam))
    for (r0, y0, p0), (r1, y1, p1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return y0 + (target - r0) / max(r1 - r0, 1e-9) * (y1 - y0), p0, p1
    return None


fmt = lambda a: f"{100 * a[0]:.1f}" if a else "n/a"


def summ(c):
    o = {"all16": float(rec(c))}
    if RESB:
        o["dep"] = {k: float(dep(c, k)) for k in KS}
        o["dep16_domains"] = {dm: float(dep(c[..., i, :])) for i, dm in enumerate(DOMS)}
        o["budget"] = {pol: dict(zip(("recall", "reads", "unused", "precision"), map(float, bud(c, pol)))) for pol in POLS}
        o["at_reads"] = {fam: (lambda a: a and {"recall": float(a[0]), "between": a[1:]})(at_reads(c, fam))
                         for fam in ("k", "all")}
        o["at_reads_domains"] = {dm: fmt(at_reads(c[..., i:i + 1, :], "k")) for i, dm in enumerate(DOMS)}
    return o


# the selection score: b32, the decision metric (per-pass catch at READS reads, top-k among non-resident); b31 recall@16
SCORE, SNAME = ((lambda c: (at_reads(c, "k") or (0.0,))[0]), f"per-pass catch at {READS} reads") if RESB else \
    (rec, "recall@16")
RES = {"counts": SC.cnt, "rows_per_layer": nrows, "layers": LP, "domains": DOMS, "res_records": len(RESB),
       "label": "late + demand" if LD else "all misses"}
if TRS is not SC:
    RES["train"] = {"trace": TRS.D, "counts": TRS.cnt, "rows_per_layer": TRS.nrows}
rowsof = lambda sel, S=None: np.flatnonzero(np.isin((S or SC).idx, sel))

# M0 on every prompt and on the held-out prompts (must equal s10b_check.py's printout)
c0, h0 = {}, {}
for d in LEADS:
    for L in LP:
        dat = Data(L, d, np.arange(nrows))
        c0[d] = c0.get(d, 0) + dat.count(None, np.arange(nrows))
        h0[d] = h0.get(d, 0) + dat.count(None, rowsof([4]))
say("M0 plain router, all prompts (= s10b_check): " + "; ".join(f"lead {d} recall@16 of misses {100 * rec(c0[d]):.1f} % "
    f"(of all routed {100 * c0[d][:, 2].sum() / c0[d][:, 3].sum():.1f} %)" for d in LEADS))
if RESB:
    for d in LEADS:
        say(f"M0 DEPLOYABLE lead {d} (= s10b_check; per row, top-k among the experts not resident at the decision; all "
            f"prompts | held-out): " + ", ".join(f"k{k} {100 * dep(c0[d], k):.1f} | {100 * dep(h0[d], k):.1f}" for k in KS)
            + "; k16 by domain: " + ", ".join(f"{dm} {100 * dep(c0[d][i]):.1f}" for i, dm in enumerate(DOMS)))
    say("M0 BUDGET (= s10b_check; per pass and target layer: recall of the pass's misses, all prompts | held-out; unique "
        "non-resident reads (of them never routed at T in the pass); precision = caught / reads; all12 = top-12 over all "
        "512, resident skipped = the shipped engine's lead-1 policy):")
    for d in LEADS:
        for pol in SHOWN:
            b, bh = bud(c0[d], pol), bud(h0[d], pol)
            say(f"  lead {d} {pol:>5}: recall {100 * b[0]:5.1f} % | {100 * bh[0]:5.1f} %, reads {b[1]:5.1f} ({b[2]:5.1f} "
                f"unused) vs misses {c0[d][:, C_PL + 1].sum() / max(c0[d][:, C_PL].sum(), 1):4.1f} per pass-layer, "
                f"precision {100 * b[3]:4.1f} %")
    for d in LEADS:
        say(f"M0 AT {READS} READS lead {d} (per pass-layer, linear in reads between the two budgets around it; all prompts "
            f"| held-out): top-k among non-resident {fmt(at_reads(c0[d], 'k'))} | {fmt(at_reads(h0[d], 'k'))} %, top-K over "
            f"all 512 (resident skipped) {fmt(at_reads(c0[d], 'all'))} | {fmt(at_reads(h0[d], 'all'))} %")
RES["m0_all"], RES["m0_heldout"] = {d: summ(c0[d]) for d in LEADS}, {d: summ(h0[d]) for d in LEADS}

if FROZEN:                              # S10c: the recipe is fixed (b32's pick): no selection on any trace
    KIND, LR, WD, EP = (f(x) for f, x in zip((str, float, float, int), FROZEN.split(",")))
    assert KIND in ("M1", "M1a", "M2") and EP > 0, FROZEN
    say(f"frozen recipe (S10B_FROZEN, no selection): {KIND} lr {LR:.0e} wd {WD} {EP} epochs, trained on {TRS.D}"
        + (" (not the scored trace: no refit there)" if TRS is not SC else ""))
    RES["pick"] = {"model": KIND, "lr": LR, "wd": WD, "epochs": EP, "frozen": True, "trained_on": TRS.D}
else:
    # selection: lead 2, train prompts 0-2, validate on prompt 3, pooled over the predictor layers
    tr, va = rowsof([0, 1, 2]), rowsof([3])
    sel = {}

    def select(kinds):
        for L in LP:
            dat = Data(L, 2, tr)
            sel.setdefault(("M0", 0.0, 0.0), np.zeros((1, len(DOMS), NC)))[0] += dat.count(None, va)
            for kind in kinds:
                for lr, wd in GRID:
                    _, hist = dat.fit(kind, lr, wd, tr, EPOCHS, np.random.default_rng(L), val=va)
                    sel.setdefault((kind, lr, wd), np.zeros((EPOCHS, len(DOMS), NC)))
                    sel[kind, lr, wd] += np.array(hist)
            say(f"  selection: layer {L} done ({time.time() - T0:.0f} s)")

    def best(kinds):
        c = [(SCORE(h[e]), k, lr, wd, e + 1) for (k, lr, wd), h in sel.items() if k in kinds for e in range(len(h))]
        return max(c, key=lambda t: t[0])

    show = lambda h: " ".join(f"{100 * SCORE(h[e]):.1f}" + (f" ({100 * dep(h[e]):.1f})" if RESB else "") for e in range(len(h)))
    select(("M1", "M1a"))
    say(f"selection (lead 2, validation = prompt 3 of each domain, pooled over layers): {SNAME} of misses by epoch"
        + (" (per-row deployable recall@16 in brackets)" if RESB else ""))
    for (k, lr, wd), h in sel.items():
        say(f"  {k:3s} lr {lr:.0e} wd {wd:4.1f}: " + show(h))
    pick = best(("M1", "M1a"))
    if RESB or pick[0] < 0.65:              # b32: M2 always tried (the plain router alone sits near 65 % there)
        select(("M2",))
        for lr, wd in GRID:
            say(f"  M2  lr {lr:.0e} wd {wd:4.1f}: " + show(sel["M2", lr, wd]))
        p2 = best(("M2",))
        if p2[0] >= pick[0] + 0.01:
            pick = p2
    v0 = SCORE(sel["M0", 0.0, 0.0][0])
    say(f"picked {pick[1]} lr {pick[2]:.0e} wd {pick[3]} {pick[4]} epochs: validation lead-2 {SNAME} of misses "
        f"{100 * pick[0]:.1f} % (M0 {100 * v0:.1f} %)")
    RES["selection"] = {f"{k} lr {lr} wd {wd}": [float(SCORE(x)) for x in h] for (k, lr, wd), h in sel.items()}
    RES["selection_score"] = SNAME
    RES["pick"] = {"model": pick[1], "lr": pick[2], "wd": pick[3], "epochs": pick[4], "val_score": float(pick[0]),
                   "val_m0": float(v0)}
    KIND, LR, WD, EP = pick[1:]


EXP, EXPA = {"judge": {}}, []


def judge(train_idx, test_idx, judge_fold=False):
    """{lead: {"m0": [L x domains x NC], "fit": [...]}} for the picked model trained on the training trace's train_idx
    prompts, counted on the scored trace's test_idx prompts; S10c (another training trace) adds "m0_tr" / "fit_tr", the
    same models on the training trace's own test_idx prompts (they must reproduce S10b there)"""
    cross = TRS is not SC
    tr_, te_, te2 = rowsof(train_idx, TRS), rowsof(test_idx), rowsof(test_idx, TRS)
    out = {d: {k: [] for k in ("m0", "fit") + (("m0_tr", "fit_tr") if cross else ())} for d in LEADS}
    for L in LP:
        for d in LEADS:
            dat = Data(L, d, tr_, TRS)
            mdl, _ = dat.fit(KIND, LR, WD, tr_, EP, np.random.default_rng(100 + L))
            sc = Data(L, d, None, SC, ref=dat) if cross else dat
            out[d]["m0"].append(sc.count(None, te_))
            out[d]["fit"].append(sc.count(mdl, te_))
            if EXPORT and judge_fold and d == 2:                          # the judge models, folded to f16
                EXP["judge"][L] = f = dat.fold(mdl)
                out[d].setdefault("fold", []).append(sc.count(Folded(f), te_))
                t32, t16 = (np.sort(np.argpartition(-sc.scores(m, te_), TOP - 1, axis=1)[:, :TOP], axis=1)
                            for m in (mdl, Folded(f)))
                EXPA.append((t32 == t16).all(1))
            if cross:
                out[d]["m0_tr"].append(dat.count(None, te2))
                out[d]["fit_tr"].append(dat.count(mdl, te2))
    return {d: {k: np.array(v) for k, v in o.items()} for d, o in out.items()}


def lines(name, J):
    say(name)
    for d in LEADS:
        m0, ft = J[d]["m0"], J[d]["fit"]
        dom = " | ".join(f"{dm} {100 * rec(ft[:, k]):.1f} (M0 {100 * rec(m0[:, k]):.1f})" for k, dm in enumerate(DOMS))
        say(f"  lead {d}: {KIND} recall@16 of misses {100 * rec(ft):.1f} % vs M0 {100 * rec(m0):.1f} % "
            f"({int(ft[..., 1].sum())} misses) | {dom} | of all routed {100 * ft[..., 2].sum() / ft[..., 3].sum():.1f} % "
            f"(M0 {100 * m0[..., 2].sum() / m0[..., 3].sum():.1f} %)")
    say("  lead 2 by layer: " + " | ".join(f"L{L} {100 * rec(J[2]['fit'][i]):.1f} (M0 {100 * rec(J[2]['m0'][i]):.1f})"
                                          for i, L in enumerate(LP)))
    if not RESB:
        return
    for d in LEADS:
        m0, ft = J[d]["m0"], J[d]["fit"]
        say(f"  DEPLOYABLE lead {d}: {KIND} recall@16 of misses {100 * dep(ft):.1f} % vs M0 {100 * dep(m0):.1f} % "
            f"({int(ft[..., C_DM].sum())} misses) | "
            + " | ".join(f"{dm} {100 * dep(ft[:, k]):.1f} (M0 {100 * dep(m0[:, k]):.1f})" for k, dm in enumerate(DOMS))
            + " | per row at k " + ", ".join(f"{k}: {100 * dep(ft, k):.1f} (M0 {100 * dep(m0, k):.1f})" for k in KS[:-1]))
    say("  DEPLOYABLE lead 2 by layer: " + " | ".join(f"L{L} {100 * dep(J[2]['fit'][i]):.1f} (M0 {100 * dep(J[2]['m0'][i]):.1f})"
                                                    for i, L in enumerate(LP)))
    say(f"  read budget per pass and target layer (the union over the pass's rows), {KIND} vs M0:")
    for d in LEADS:
        for pol in SHOWN:
            f_, m_ = bud(J[d]["fit"], pol), bud(J[d]["m0"], pol)
            say(f"    lead {d} {pol:>5}: recall {100 * f_[0]:5.1f} % (M0 {100 * m_[0]:5.1f}), reads {f_[1]:5.1f} (M0 "
                f"{m_[1]:5.1f}), unused {f_[2]:5.1f} (M0 {m_[2]:5.1f}), precision {100 * f_[3]:4.1f} % (M0 {100 * m_[3]:4.1f})")
        fa, ma, fA, mA = (at_reads(J[d][w], fam) for fam in ("k", "all") for w in ("fit", "m0"))
        say(f"    lead {d} AT {READS} READS: top-k among non-resident {KIND} {fmt(fa)} % (M0 {fmt(ma)}) [{KIND} "
            f"{'-'.join(fa[1:]) if fa else 'n/a'}] | top-K over all 512, resident skipped: {KIND} {fmt(fA)} % (M0 {fmt(mA)}) "
            f"[{KIND} {'-'.join(fA[1:]) if fA else 'n/a'}] | by domain (top-k): "
            + ", ".join(f"{dm} {fmt(at_reads(J[d]['fit'][:, i:i + 1], 'k'))} (M0 {fmt(at_reads(J[d]['m0'][:, i:i + 1], 'k'))})"
                        for i, dm in enumerate(DOMS)))


J = judge([0, 1, 2, 3], [4], judge_fold=True)
lines("JUDGE (primary): trained on prompts 0-3, held-out prompt 4 of each domain", J)
F = [judge([j for j in range(5) if j != f], [f]) for f in range(5)]
JL = {d: {k: sum(f[d][k] for f in F) for k in F[0][d]} for d in LEADS}
lines("leave-one-prompt-index-out (secondary, 5 folds pooled)", JL)
RES["judge"] = {d: {k: summ(v) for k, v in J[d].items()} for d in LEADS}
RES["lopo"] = {d: {k: summ(v) for k, v in JL[d].items()} for d in LEADS}
if TRS is not SC and RESB:
    say(f"the frozen models on their training trace's own prompts ({TRS.D}; they must reproduce S10b there), lead 2 at "
        f"{READS} reads: " + "; ".join(
            f"{nm}: top-K over all 512 (resident skipped) {fmt(at_reads(Jx[2]['fit_tr'], 'all'))} % (M0 "
            f"{fmt(at_reads(Jx[2]['m0_tr'], 'all'))}), top-k among non-resident {fmt(at_reads(Jx[2]['fit_tr'], 'k'))} % "
            f"(M0 {fmt(at_reads(Jx[2]['m0_tr'], 'k'))})" for nm, Jx in (("held-out", J), ("leave-one-out", JL))))
if EXPORT:                              # S10c (2), BigMoE's format: the judge models (prompts 0-3), folded
    os.makedirs(EXPORT, exist_ok=True)
    np.savez(os.path.join(EXPORT, f"{EXPN}.npz"),
             **{f"L{L}_{k}": v for L, f in EXP["judge"].items() for k, v in f.items()})
    # BigMoE's engine check: per row, the top-12 experts by s for the first 3 calls of layer LP[0] in the scored trace's
    # held-out prompt 4 of the first domain (b33: p04_code4), in the npz's call order
    q4 = next(q for q in PR if q["idx"] == 4)
    p, f, L0, chk = np.load(os.path.join(SC.D, q4["name"] + ".npz")), Folded(EXP["judge"][LP[0]]), LP[0], []
    for j in np.flatnonzero(p["L"] == L0)[:3]:
        R, xo, ro = int(p["rows"][j]), int(p["x_off"][j]), int(p["row_off"][j])
        assert xo >= 0, "a predictor-layer call without x"
        il = p["ids"][ro:ro + R].astype(np.int64)
        oh = np.zeros((R, E), np.float32)
        np.put_along_axis(oh, il, 1.0, axis=1)
        s_ = bf16(p["x"][xo:xo + R]) @ f.Wf + f.bf + oh @ f.A
        top = np.argsort(-s_, axis=1, kind="stable")[:, :12]
        chk.append({"call": int(j), "seq": int(p["seq"][j]), "L": L0, "T": L0 + 2, "rows": R, "ids_L": il.tolist(),
                    "top12": top.tolist(), "scores": np.take_along_axis(s_, top, axis=1).round(5).tolist()})
    json.dump({"prompt": q4["name"], "trace": SC.D, "weights": f"{EXPN}.npz",
               "computed": "f32: x = the trace's bf16 x, Wf and A as stored (f16 -> f32), bf f32; s = x @ Wf + bf + "
                           "A[ids_L].sum(0); top-12 over all experts by s, descending (no residency skip)",
               "calls": chk}, open(os.path.join(EXPORT, f"{EXPN}_check.json"), "w"), indent=1)
    fa, ff = at_reads(J[2]["fold"], "all"), at_reads(J[2]["fit"], "all")
    ag = float(np.concatenate(EXPA).mean())
    json.dump({"model": KIND, "recipe": {"lr": LR, "wd": WD, "epochs": EP}, "trained_on": TRS.D,
               "training_prompts": [q["name"] for q in PR if q["idx"] in (0, 1, 2, 3)], "scored_on": SC.D,
               "layers": {str(L): L + 2 for L in LP}, "hidden": int(SC.X[LP[0]].shape[1]), "experts": E,
               "keys": "L{L}_Wf [hidden, experts] f16, L{L}_bf [experts] f32, L{L}_A [experts, experts] f16",
               "formula": "s_T = x_L @ Wf + bf + A[ids_L].sum(0); x_L = layer L's MoE input (the router's input, raw: "
                          "the fit's standardization is folded into Wf and bf) per row; ids_L = the experts routed at L "
                          "for that row; s_T = scores of layer T = L + 2's experts; the A/B pick rule: top-K over all "
                          "experts by s_T, resident ones skipped",
               "heldout_shipped_rule_at_4.8_reads": {"folded_f16": fa and float(fa[0]), "f32": ff and float(ff[0]),
                                                     "router": float(at_reads(J[2]["m0"], "all")[0])},
               "rows_top16_identical_to_f32": ag}, open(os.path.join(EXPORT, f"{EXPN}.json"), "w"), indent=1)
    say(f"EXPORT (S10c (2), {EXPORT}/{EXPN}.npz + _check.json + .json): lead-2 {KIND} judge models (prompts 0-3) "
        f"folded to f16 (s = x Wf + bf + sum of A over L's routed experts), {len(LP)} pairs L -> L+2; on the held-out "
        f"prompts, shipped rule at {READS} reads: folded {fmt(fa)} % (f32 {fmt(ff)}, router "
        f"{fmt(at_reads(J[2]['m0'], 'all'))}); top-k among non-resident folded {fmt(at_reads(J[2]['fold'], 'k'))} % (f32 "
        f"{fmt(at_reads(J[2]['fit'], 'k'))}); rows whose top-{TOP} set is identical to the f32 model's {100 * ag:.2f} %; "
        f"check: {q4['name']} layer {L0}, calls {[c['call'] for c in chk]} ({sum(c['rows'] for c in chk)} rows)")
    RES["export"] = {"dir": EXPORT, "heldout_folded": fa and float(fa[0]), "heldout_f32": ff and float(ff[0]),
                     "rows_top16_identical": ag}
LBL = "pre-registered" if TARGET == "routed" else "EXPLORATORY, not pre-registered: cannot change the verdict alone"
if FROZEN and LD and RESB:              # S10c-1 (REPORT 'S10c-1'): late + demand misses, the shipped rule's family
    if TARGET == "routed" and TRS is SC:
        LBL = "not the pre-registered design: the model was refit on the scored trace"
    A = {nm: (at_reads(Jx[2]["fit"], "all"), at_reads(Jx[2]["m0"], "all")) for nm, Jx in (("held-out", J),
                                                                                         ("leave-one-out", JL))}
    dif = {nm: float(a[0] - b[0]) if a and b else None for nm, (a, b) in A.items()}
    verdict = ("no number (the sweep does not bracket the budget)" if None in dif.values() else
               "PASS (>= +10 points, held-out and leave-one-out): (2) the engine A/B, BigMoE designs it"
               if min(dif.values()) >= 0.10 else
               "KILL (< +5 points, held-out and leave-one-out: its edge was on-time prefetches; S10b / S10c closed)"
               if max(dif.values()) < 0.05 else
               "between (+5 to +10 points, or held-out and leave-one-out on different sides): (2) only if BigMoE prices "
               "the integration as small (NewIdeas' call)")
    verdict = verdict if TARGET == "routed" else f"no verdict (on the pre-registered lines it would read: {verdict})"
    pts = {nm: "n/a" if v is None else f"{100 * v:+.1f}" for nm, v in dif.items()}
    say(f"VERDICT S10c-1 ({LBL}; NewIdeas 14:08): late + demand misses, lead 2, per-pass catch at {READS} reads per "
        f"pass-layer, the shipped rule's family (top-K over all 512, resident skipped): "
        + "; ".join(f"{nm} {KIND} {fmt(a)} % vs router {fmt(b)} % ({pts[nm]} points)" for nm, (a, b) in A.items())
        + f" -> {verdict}; top-k among non-resident beside it: held-out {fmt(at_reads(J[2]['fit'], 'k'))} % (router "
        f"{fmt(at_reads(J[2]['m0'], 'k'))}), leave-one-out {fmt(at_reads(JL[2]['fit'], 'k'))} % (router "
        f"{fmt(at_reads(JL[2]['m0'], 'k'))}); {time.time() - T0:.0f} s")
    RES["s10c"] = {nm: {"fit": a and float(a[0]), "m0": b and float(b[0]), "diff": dif[nm]} for nm, (a, b) in A.items()}
elif RESB:
    a2 = at_reads(J[2]["fit"], "k")
    verdict = ("no number (the sweep does not bracket the budget)" if a2 is None else
               "PASS (>= 50.4 %, the shipped lead-1 catch at that budget): b33 labels, then an engine A/B design"
               if a2[0] >= 0.504 else "park S10b (< 45 %)" if a2[0] < 0.45 else
               "between 45 and 50.4 %: b33 labels (late + demand misses) decide")
    verdict = verdict if TARGET == "routed" else f"no verdict (on the pre-registered lines it would read: {verdict})"
    say(f"VERDICT ({LBL}; NewIdeas 13:46, the bar in reads): held-out lead-2 per-pass catch of the misses at {READS} reads "
        f"per pass-layer {fmt(a2)} % ({KIND}, top-k among non-resident, {'-'.join(a2[1:]) if a2 else 'n/a'}) vs M0 "
        f"{fmt(at_reads(J[2]['m0'], 'k'))} % -> {verdict}; the shipped rule's family (top-K over all 512, resident "
        f"skipped) {fmt(at_reads(J[2]['fit'], 'all'))} % (M0 {fmt(at_reads(J[2]['m0'], 'all'))}); leave-one-out "
        f"{fmt(at_reads(JL[2]['fit'], 'k'))} % (M0 {fmt(at_reads(JL[2]['m0'], 'k'))}); per-row deployable recall@16 "
        f"{100 * dep(J[2]['fit']):.1f} % vs M0 {100 * dep(J[2]['m0']):.1f} % (the 09:46 lines, superseded); "
        f"{time.time() - T0:.0f} s")
else:
    r2 = rec(J[2]["fit"])
    verdict = ("KILL (< 55 %)" if r2 < 0.55 else "ENGINE A/B (>= 65 %)" if r2 >= 0.65 else
               "alive, below the A/B line (55-65 %): NewIdeas decides the next step")
    say(f"VERDICT ({LBL}): held-out lead-2 recall@16 of misses {100 * r2:.1f} % ({KIND}) vs M0 "
        f"{100 * rec(J[2]['m0']):.1f} % -> {verdict}; leave-one-out {100 * rec(JL[2]['fit']):.1f} % (M0 "
        f"{100 * rec(JL[2]['m0']):.1f} %); {time.time() - T0:.0f} s")
RES["verdict"] = verdict
open(os.path.join(OUT, f"s10b_fit{TAG}.txt"), "w").write("\n".join(LOG) + "\n")
json.dump(RES, open(os.path.join(OUT, f"s10b_fit{TAG}.json"), "w"), indent=1)
