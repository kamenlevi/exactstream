"""SIB, sibling-state drafting in the sandbox (PLAN900 bet 9b; FAR's idea 1, ~/FableGod/far/IDEAS_1115_info.md
section 1; NewIdeas 12:2x). Pre-registered in results/LEVERS_A.md, section SIB, before any run.

build  For each window: the head's teacher-forced chain table (depth 31) and its decode walk (core.kept_walk's checks).
       For each check that missed at depth k < 31, the head's FREE-RUNNING draft from that check's anchor s (its own
       guesses fed back, as a real chain drafts) gives the rejected continuation: the wrong guess and the drafter's own
       tokens after it, L = min(8, 31 - k, rows left) tokens from position s' + 1 (s' = s + k + 1, the next check start).
       The 2B then runs, per post-miss start s', one full causal forward (no cache: the capture's own path) of two rows:
       the true tokens up to s' + the continuation, and the true tokens up to s' + L. Their pre-norm residuals at rows
       s'+1 .. s'+L are the sibling rows and the recomputed true rows (true_r, a harness control). (Amendment 13:1x:
       the first version branched a KV / Gated-DeltaNet cache, which sat 2 % off the stored states.)
       -> data/sib/SET/KEY.npz, and a per-set summary (the walk's kept/32 and the free-running agreement: VOID checks 1-2)
score  For each head and set: the teacher-forced chain at every anchor, depth 0-8, in four modes:
         own (the carried stand-in), true (the stored true W, evalx's oracle), true_r (the recomputed true rows),
         sib (the sibling rows for d <= L, the carried stand-in past L),
       read at the post-miss starts only: walk-strict P(right at d | right at 0..d-1) and marginal accuracy per band
       1-2 / 3-6 / 7-8 / 1-8 / 3-8, G = (sib - own) / (true - own) with a bootstrap CI over windows, the direction error of
       each state source against the true W, and the target's margin at the misses.
verdict  Applies the pre-registered lines to a frozen-head score and a fine-tune score.
All numbers are LAB numbers (teacher-forced, the 2B's own captured path), not real decoding.
usage: sibstate.py build --sets he,mbpp[,pool] [--head s30] [--n N]
       sibstate.py score --heads s30[,sib1_ctl,sib1_sib] --sets he,mbpp [--n N] --out results/sib_X.json
       sibstate.py verdict --frozen results/sib_s30.json [--ft results/sib_ft.json] --out results/sib_verdict.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

BANDS = {"1-2": (1, 2), "3-6": (3, 6), "7-8": (7, 8), "1-8": (1, 8), "3-8": (3, 8)}
ARMS = ("own", "true", "true_r", "sib")
GATE_KEPT = {"mbpp": 4.421133525456292, "he": 6.430905077262693}    # results/gate_chain.json, s30 chain
NSIB, DCHAIN = 8, 31


def H(*p):
    return os.path.join(os.path.expanduser("~/Sandbox98"), *p)


def set_files(st, n=0):
    from sb import core as C
    if st == "pool":
        fs = [os.path.join(C.DATA, "mc", k + ".npz") for k in json.load(open(os.path.join(C.DATA, "split.json")))["pool"]]
    else:
        fs = C.files(st)
    return fs[:n] if n else fs


def walk(rt, r0, Dp=DCHAIN):
    """core.kept_walk with the checks themselves: [(row p, run k)], k capped at Dp (kept = k + 1)"""
    sub = rt[r0:]; out = []; p = 0
    Dp = min(Dp, rt.shape[1])
    while p < sub.shape[0]:
        k = 0
        while k < Dp and sub[p, k] == 1:
            k += 1
        out.append((r0 + p, k)); p += k + 1
    return out


# ---------------------------------------------------------------- build
def deeper_tok(C, mx, D, h, tok, anc, j, K1, V1, ck, cv):
    """core.deeper with explicit input tokens tok [n] (the drafter's own guesses) instead of ids[anc + j]"""
    x = C.fuse(D, h, C.embed(tok)[None])
    ks, vs = [], []
    for li, L in enumerate(C.layers_of(D)):
        q, gate, k, v = C.qkv(L, L.input_layernorm(x), pos=anc + j - 1)
        ks.append(k); vs.append(v)
        o = C.ttt_attn(L.scale, q, K1[li], V1[li], [c[li] for c in ck] + [k], [c[li] for c in cv] + [v], anc)
        blk = L.o_proj(o.transpose(0, 2, 1, 3).reshape(1, h.shape[1], -1) * mx.sigmoid(gate))
        x = C.block(L, x, blk)
    ck.append(mx.stack(ks)); cv.append(mx.stack(vs))
    return x


def freerun(C, mx, D, W, ida, a0, anc_np, Dmax):
    """the head's own free-running draft from anchors anc_np: depth 1 reads (W[s], ids[s+1]); depth j >= 2 reads its
    carried output and its own depth-(j-1) guess. -> G [n, Dmax] (G[:, d] is the guess for position s + d + 2)"""
    h_all, K1, V1 = C.depth1(D, W, ida, a0)
    anc = mx.array(np.asarray(anc_np, np.int32))
    h = h_all[:, anc - a0]
    G = np.zeros((len(anc_np), Dmax), np.int32)
    g = mx.argmax(C.lm_logits(D.norm(h[0]).astype(mx.bfloat16)), -1)
    mx.eval(g); G[:, 0] = np.array(g)
    ck, cv = [], []
    for j in range(2, Dmax + 1):
        h = deeper_tok(C, mx, D, h, g, anc, j, K1, V1, ck, cv)
        g = mx.argmax(C.lm_logits(D.norm(h[0]).astype(mx.bfloat16)), -1)
        mx.eval(g, h, *ck[-1:], *cv[-1:])
        G[:, j - 1] = np.array(g)
    return G


def res_fwd(C, mx, toks, cache):
    """the 2B over toks [n] through cache (the text model's own forward, cut before its final norm) -> res [n, d]"""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    t = C.tm()
    x = t.embed_tokens(mx.array(np.asarray(toks, np.int32))[None])
    fa, ssm = create_attention_mask(x, cache[t.fa_idx]), create_ssm_mask(x, cache[t.ssm_idx])
    for layer, c in zip(t.layers, cache):
        x = layer(x, mask=ssm if layer.is_linear else fa, cache=c)
    return x[0]


def res_full(C, mx, tokmat):
    """the 2B over token rows tokmat [B, T] (one length) in one full causal forward with no cache, as core.target_states
    runs the capture -> pre-norm residuals [B, T, d]"""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    t = C.tm()
    x = t.embed_tokens(mx.array(np.asarray(tokmat, np.int32)))
    fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
    for layer in t.layers:
        x = layer(x, mask=ssm if layer.is_linear else fa, cache=None)
    return x


def snap(cache):
    from mlx_lm.models.cache import ArraysCache
    return [list(c.cache) if isinstance(c, ArraysCache) else c.offset for c in cache]


def restore(cache, s):
    from mlx_lm.models.cache import ArraysCache
    for c, v in zip(cache, s):
        if isinstance(c, ArraysCache):
            c.cache = list(v)
        else:
            c.trim(c.offset - v)


def build_window(C, mx, D, f):
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"]); S = len(ids)
    a0 = max(0, gen - 2); r0 = max(0, gen - 1 - a0)
    rt, _, _ = C.chain_table(D, f, Dp=DCHAIN)
    checks = walk(rt, r0)
    kept = [k + 1 for _, k in checks]
    assert kept == C.kept_walk(rt, r0), "walk() differs from core.kept_walk"
    lp = np.asarray(z["lp"], np.float32)
    st, prev, kk, LL, marg = [], [], [], [], []
    for (p, k), (p2, _) in zip(checks[:-1], checks[1:]):
        if k >= DCHAIN:                                    # a full 32-token check: no rejected rows
            continue
        s, s2 = a0 + p, a0 + p2
        assert s2 == s + k + 1 and rt[p, k] == 0
        L = min(NSIB, DCHAIN - k, S - 3 - s2)
        if L <= 0:
            continue
        st.append(s2); prev.append(s); kk.append(k); LL.append(L)
        marg.append(float(lp[s2, 0] - lp[s2, 1]))           # the target's margin at the missed row (row s2 predicts s2 + 1)
    n = len(st)
    out = {"starts": np.array(st, np.int32), "prev": np.array(prev, np.int32), "k": np.array(kk, np.int32),
           "L": np.array(LL, np.int32), "margin": np.array(marg, np.float32), "kept": np.array(kept, np.int32),
           "a0": a0, "gen": gen, "S": S}
    if n == 0:
        out.update(cont=np.zeros((0, NSIB), np.int32), sib=np.zeros((0, NSIB, 2048), np.float16),
                   tru=np.zeros((0, NSIB, 2048), np.float16), agree=np.zeros(0, bool), tru_err=np.zeros(0, np.float32),
                   pre_bit=np.zeros(0, bool), tru_bit=np.float32(np.nan))
        return out
    # the free-running drafts of the checks that missed
    W = C.load_W(z)
    Dmax = int(min(DCHAIN, max(k + L for k, L in zip(kk, LL))))
    G = freerun(C, mx, D, W, mx.array(ids), a0, prev, Dmax)
    agree = np.zeros(n, bool); cont = np.full((n, NSIB), -1, np.int32)
    for i, (s, k, L) in enumerate(zip(prev, kk, LL)):
        agree[i] = bool((G[i, :k] == ids[s + 2:s + k + 2]).all() and G[i, k] != ids[s + k + 2])
        cont[i, :L] = G[i, k:k + L]
    # the 2B (amendment 2, Mon 14:3x): per post-miss start ONE forward shaped exactly like the capture's (core.target_states:
    # batch 1, the whole length S, no cache) over the true tokens with rows s'+1 .. s'+L replaced by the rejected
    # continuation; the sibling states are its rows s'+1 .. s'+L. The 13:1x version ran shorter forwards (prefix length,
    # two rows), which pick other matmul kernels on short prefixes and sat 1.2 % off the stored states (smoke 14:24); the
    # first version branched a KV / Gated-DeltaNet cache (2 % off). Scratch check 14:27: the capture-shaped forward gives
    # the stored states bit for bit, and a replaced window leaves every earlier row bit for bit unchanged (causal).
    # true_r = one capture-shaped forward of the true tokens per window (expected: bitwise = the stored rows); pre_bit =
    # each start's own rows <= s' bitwise equal to the stored ones (the replaced window touched nothing before it).
    d = int(z["res"].shape[1])
    sib = np.zeros((n, NSIB, d), np.float16); tru = np.zeros((n, NSIB, d), np.float16)
    ref16 = np.asarray(z["res"], np.float16)
    def full(tok):
        r = C.target_states(tok, logits=False)[0]; mx.eval(r)
        return np.array(r.astype(mx.float16))
    t_true = full(ids)
    tru_bit = float((t_true == ref16).mean())
    pre_bit = np.zeros(n, bool); rels = []
    ref_all = ref16.astype(np.float32)
    for i, (s2, L) in enumerate(zip(st, LL)):
        nid = ids.copy(); nid[s2 + 1:s2 + 1 + L] = cont[i, :L]
        r = full(nid)
        sib[i, :L] = r[s2 + 1:s2 + 1 + L]; tru[i, :L] = t_true[s2 + 1:s2 + 1 + L]
        pre_bit[i] = bool((r[:s2 + 1] == ref16[:s2 + 1]).all())
        ref = ref_all[s2 + 1:s2 + 1 + L]; got = tru[i, :L].astype(np.float32)
        rels.append(np.sqrt(((got - ref) ** 2).mean(1)) / np.maximum(np.sqrt((ref ** 2).mean(1)), 1e-6))
    rel = np.concatenate(rels)
    out.update(cont=cont, sib=sib, tru=tru, agree=agree, tru_err=rel.astype(np.float32), pre_bit=pre_bit,
               tru_bit=np.float32(tru_bit))
    mx.clear_cache()
    return out


def cmd_build(a):
    import mlx.core as mx
    from sb import core as C, resume as R
    C.limit_memory(float(os.environ.get("SB_MEM_GB", "5")))
    R.arm_sigterm()
    C.ARGMAX_ROWS = 256                                    # 2048-row logits chunks put the smoke at 7.76 GB
    C.load_target()                                        # the trunk stays: the sibling rows need the 2B
    D = C.load_drafter(a.head)
    t0 = time.time()
    for st in a.sets.split(","):
        od = H("data", "sib", st + a.tag); os.makedirs(od, exist_ok=True)
        fs = set_files(st, a.n)
        kept, agree, nst, nmiss_lt2, nL8, rel, pbit, tbit = [], [], 0, 0, 0, [], [], []
        for wi, f in enumerate(fs):
            key = os.path.basename(f)[:-4]; p = os.path.join(od, key + ".npz")
            if not os.path.exists(p):
                C.wait_pause()
                if R.STOP["f"]:
                    print(f"[sib] SIGTERM: stopping before {key}; finished windows are kept", flush=True); sys.exit(75)
                o = build_window(C, mx, D, f)
                np.savez(p + ".tmp.npz", **o); os.replace(p + ".tmp.npz", p)
            o = np.load(p)
            kept += list(o["kept"]); agree += list(o["agree"]); nst += len(o["starts"])
            nmiss_lt2 += int((o["margin"] < 2.0).sum()); nL8 += int((o["L"] < NSIB).sum())
            if len(o["tru_err"]):
                rel.append(o["tru_err"])
            pbit += list(o["pre_bit"]) if "pre_bit" in o.files else []
            if "tru_bit" in o.files and np.isfinite(o["tru_bit"]):
                tbit.append(float(o["tru_bit"]))
            if (wi + 1) % 10 == 0 or wi == len(fs) - 1:
                print(f"[sib] build {st} {wi + 1}/{len(fs)}: {nst} post-miss starts, {time.time() - t0:.0f}s, "
                      f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
        rel = np.concatenate(rel) if rel else np.zeros(1)
        summ = {"set": st, "head": a.head, "windows": len(fs), "checks": len(kept), "kept_per_32": float(np.mean(kept)),
                "gate_kept": GATE_KEPT.get(st) if a.head == "s30" else None, "starts": nst,
                "agree": float(np.mean(agree)) if agree else None, "margin_lt2": nmiss_lt2 / max(nst, 1),
                "L_lt8": nL8 / max(nst, 1), "true_path_rel_err": {"mean": float(rel.mean()), "p99": float(np.quantile(rel, 0.99)),
                                                                  "max": float(rel.max())},
                "prefix_bitwise": float(np.mean(pbit)) if pbit else None,
                "true_bitwise": float(np.mean(tbit)) if tbit else None}
        if summ["gate_kept"] is not None and not a.n:
            summ["void1_ok"] = abs(summ["kept_per_32"] - summ["gate_kept"]) <= 0.01
        summ["void2_ok"] = summ["agree"] is None or summ["agree"] >= 0.99
        json.dump(summ, open(os.path.join(od, "_summary.json"), "w"), indent=1)
        print(f"[sib] build {st} DONE: {json.dumps(summ)}", flush=True)


# ---------------------------------------------------------------- score
def rep_rows(C, mx, o, S, field):
    """[S, 8, d] fp32 replacement rows (post-norm) at the sibling slots and the [S, 8] mask"""
    st, L = o["starts"], o["L"]
    n = len(st); d = o[field].shape[-1]
    idx_s = np.repeat(st, NSIB); idx_d = np.tile(np.arange(NSIB), n)
    vi = np.flatnonzero(idx_d < np.repeat(L, NSIB))
    msk = np.zeros((S, NSIB), bool); msk[idx_s[vi], idx_d[vi]] = True
    rep = mx.zeros((S, NSIB, d), mx.float32)
    if len(vi):
        post = C.final_norm(mx.array(o[field].reshape(-1, d)[vi])).astype(mx.float32)
        rep[mx.array(idx_s[vi]), mx.array(idx_d[vi])] = post
    return rep, mx.array(msk), msk


def make_fn(mx, rep, msk, rec=None):
    def fn(j, h, anc):
        if j < 2 or j > NSIB + 1:
            return h
        m = msk[anc, j - 2]
        if rec is not None:                                # own: record the carried rows at the sibling slots
            rec.append((j, np.array(anc), np.array(m), h[0]))
            return h
        return mx.where(m[:, None], rep[anc, j - 2], h[0])[None]
    return fn


def dir_err(x, t):
    x = x / np.maximum(np.sqrt((x ** 2).mean(-1, keepdims=True)), 1e-6)
    t = t / np.maximum(np.sqrt((t ** 2).mean(-1, keepdims=True)), 1e-6)
    return np.sqrt(((x - t) ** 2).mean(-1))


def counts(cell):
    """cell [n, 9] int8 (1 right, 0 wrong, -1 no target) -> [5, 9]: walk-strict den / num, marginal den / num, and
    kept within depths 0-8 (row 4, col 0: sum of runs capped at 9; col 1: cells)"""
    valid, right = cell >= 0, cell == 1
    alive = np.ones_like(right); alive[:, 1:] = np.cumprod(right[:, :-1], 1).astype(bool)
    c = np.zeros((5, cell.shape[1]), np.int64)
    c[0] = (alive & valid).sum(0); c[1] = (alive & right).sum(0)
    c[2] = valid.sum(0); c[3] = right.sum(0)
    run = np.where(right.all(1), cell.shape[1], np.argmin(right, 1))
    c[4, 0] = int(np.minimum(run, cell.shape[1]).sum()); c[4, 1] = cell.shape[0]
    return c


def band_vals(c):
    """c [arms, 5, 9] summed counts -> {kind: {band: {arm: value}}}"""
    out = {"strict": {}, "marginal": {}}
    for kind, (di, ni) in (("strict", (0, 1)), ("marginal", (2, 3))):
        for b, (lo, hi) in BANDS.items():
            out[kind][b] = {arm: float(c[ai, ni, lo:hi + 1].sum() / max(c[ai, di, lo:hi + 1].sum(), 1))
                            for ai, arm in enumerate(ARMS)}
    out["depth0"] = float(c[0, 1, 0] / max(c[0, 0, 0], 1))
    out["kept9"] = {arm: float(c[ai, 4, 0] / max(c[ai, 4, 1], 1)) for ai, arm in enumerate(ARMS)}
    return out


def gap(v, ceil="true"):
    den = v[ceil] - v["own"]
    return (v["sib"] - v["own"]) / den if abs(den) > 1e-9 else None


def score_set(C, mx, D, st, n, a):
    fs = set_files(st, n)
    od = H("data", "sib", st + a.tag)
    per, de = [], {"own": np.zeros((2, NSIB)), "sib": np.zeros((2, NSIB)), "true_r": np.zeros((2, NSIB))}
    marg, nL8, nst = [], 0, 0
    for f in fs:
        C.wait_pause()
        key = os.path.basename(f)[:-4]
        o = np.load(os.path.join(od, key + ".npz"))
        if len(o["starts"]) == 0:
            continue
        z = np.load(f)
        S = len(z["ids"]); a0 = int(o["a0"])
        W = C.load_W(z).astype(mx.float32)
        Wn = np.array(W[0])
        rep_s, msk, msk_np = rep_rows(C, mx, o, S, "sib")
        rep_t, _, _ = rep_rows(C, mx, o, S, "tru")
        rec = []
        tabs = {"own": C.chain_table(D, f, Dp=NSIB + 1, fn=make_fn(mx, None, msk, rec))[0],
                "true": C.chain_table(D, f, Dp=NSIB + 1, R=W)[0],
                "true_r": C.chain_table(D, f, Dp=NSIB + 1, fn=make_fn(mx, rep_t, msk))[0],
                "sib": C.chain_table(D, f, Dp=NSIB + 1, fn=make_fn(mx, rep_s, msk))[0]}
        rows = o["starts"] - a0
        per.append(np.stack([counts(tabs[arm][rows]) for arm in ARMS]))
        # direction error against the true W at the sibling slots (W row = anchor + j - 1)
        for j, anc, m, h in rec:
            sel = np.flatnonzero(m)
            if len(sel) == 0:
                continue
            tw = Wn[anc[sel] + j - 1]
            de["own"][0, j - 2] += dir_err(np.array(h[mx.array(sel)]), tw).sum(); de["own"][1, j - 2] += len(sel)
        for nm, rep in (("sib", rep_s), ("true_r", rep_t)):
            for slot in range(NSIB):
                ss = np.flatnonzero(msk_np[:, slot])
                if len(ss) == 0:
                    continue
                x = np.array(rep[mx.array(ss), slot]); tw = Wn[ss + slot + 1]
                de[nm][0, slot] += dir_err(x, tw).sum(); de[nm][1, slot] += len(ss)
        marg.append(o["margin"]); nL8 += int((o["L"] < NSIB).sum()); nst += len(o["starts"])
        mx.clear_cache()
    per = np.stack(per)                                     # [windows, arms, 5, 9]
    v = band_vals(per.sum(0))
    rng = np.random.default_rng(1116)
    boot = {kind: {b: [] for b in BANDS} for kind in ("strict", "marginal")}
    diff = {b: [] for b in BANDS}
    for _ in range(2000):
        bv = band_vals(per[rng.integers(0, len(per), len(per))].sum(0))
        for kind in boot:
            for b in BANDS:
                g = gap(bv[kind][b])
                if g is not None:
                    boot[kind][b].append(g)
        for b in BANDS:
            diff[b].append(bv["strict"][b]["sib"] - bv["strict"][b]["own"])
    res = {"windows": len(per), "starts": nst, "L_lt8": nL8 / max(nst, 1),
           "margin_lt2": float((np.concatenate(marg) < 2.0).mean()), "margin_median": float(np.median(np.concatenate(marg))),
           "depth0": v["depth0"], "kept9": v["kept9"], "strict": v["strict"], "marginal": v["marginal"],
           "G": {kind: {b: gap(v[kind][b]) for b in BANDS} for kind in ("strict", "marginal")},
           "G_true_r": {kind: {b: gap(v[kind][b], "true_r") for b in BANDS} for kind in ("strict", "marginal")},
           "G_ci95": {kind: {b: [float(np.percentile(boot[kind][b], 2.5)), float(np.percentile(boot[kind][b], 97.5))]
                             for b in BANDS} for kind in boot},
           "sib_minus_own_ci95": {b: [float(np.percentile(diff[b], 2.5)), float(np.percentile(diff[b], 97.5))] for b in BANDS},
           "dir_err": {nm: [float(de[nm][0, s] / max(de[nm][1, s], 1)) for s in range(NSIB)] for nm in de},
           "per_window": per.tolist()}
    res["void3_ok"] = abs(res["strict"]["1-8"]["true_r"] - res["strict"]["1-8"]["true"]) <= 0.003
    return res


def show(tag, r):
    print(f"[sib] {tag}: {r['windows']} windows, {r['starts']} post-miss starts (L < 8: {100 * r['L_lt8']:.1f} %; "
          f"target margin < 2 nats at the misses: {100 * r['margin_lt2']:.1f} %, median {r['margin_median']:.2f}); "
          f"depth 0 {100 * r['depth0']:.2f}", flush=True)
    for kind in ("strict", "marginal"):
        for b in BANDS:
            x = r[kind][b]; g = r["G"][kind][b]; ci = r["G_ci95"][kind][b]
            print(f"[sib]   {kind:8s} {b:4s}: own {100 * x['own']:.2f}  true {100 * x['true']:.2f}  "
                  f"true_r {100 * x['true_r']:.2f}  sib {100 * x['sib']:.2f}  G {g if g is None else round(g, 3)} "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)
    print(f"[sib]   kept within 0-8: " + "  ".join(f"{k} {v:.3f}" for k, v in r["kept9"].items())
          + f"; dir err d1-8 own {np.round(r['dir_err']['own'], 3).tolist()} sib {np.round(r['dir_err']['sib'], 3).tolist()} "
          f"true_r {np.round(r['dir_err']['true_r'], 3).tolist()}; VOID-3 ok {r['void3_ok']}", flush=True)


def cmd_score(a):
    import mlx.core as mx
    from sb import core as C
    C.limit_memory(float(os.environ.get("SB_MEM_GB", "5")))
    C.ARGMAX_ROWS = 256
    C.load_target(); C.drop_trunk()
    t0 = time.time()
    out = {"args": vars(a), "res": {}}
    for head in a.heads.split(","):
        D = C.load_drafter(head)
        for st in a.sets.split(","):
            r = score_set(C, mx, D, st, a.n, a)
            out["res"][f"{head}|{st}"] = r
            show(f"{head}|{st}", r)
            print(f"[sib] {head}|{st} done, {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    out["wall_s"] = time.time() - t0; out["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    print(f"[sib] DONE score -> {a.out}", flush=True)


# ---------------------------------------------------------------- verdict
def cmd_verdict(a):
    fr = json.load(open(a.frozen))["res"]
    ft = json.load(open(a.ft))["res"] if a.ft else None
    bs = {st: json.load(open(H("data", "sib", st, "_summary.json"))) for st in ("he", "mbpp")}
    void = []
    for st in ("he", "mbpp"):
        if not bs[st].get("void1_ok", False):
            void.append(f"VOID-1 {st}: walk kept/32 {bs[st]['kept_per_32']:.4f} vs gate {bs[st]['gate_kept']:.4f}")
        if not bs[st].get("void2_ok", False):
            void.append(f"VOID-2 {st}: free-running agreement {bs[st]['agree']:.4f} < 0.99")
        if not fr[f"s30|{st}"]["void3_ok"]:
            x = fr[f"s30|{st}"]["strict"]["1-8"]
            void.append(f"VOID-3 {st}: true_r {100 * x['true_r']:.2f} vs true {100 * x['true']:.2f}")
    G = {st: fr[f"s30|{st}"]["G"]["strict"] for st in ("he", "mbpp")}
    gft = None
    if ft:
        gft = {}
        for st in ("he", "mbpp"):
            c, s = ft[f"sib1_ctl|{st}"]["strict"], ft[f"sib1_sib|{st}"]["strict"]
            gft[st] = {b: ((s[b]["sib"] - c[b]["own"]) / (c[b]["true"] - c[b]["own"])
                           if abs(c[b]["true"] - c[b]["own"]) > 1e-9 else None) for b in BANDS}
    def passes(g):
        return g is not None and g["he"]["1-8"] is not None and g["mbpp"]["1-8"] is not None \
            and g["he"]["1-8"] >= 0.50 and g["mbpp"]["1-8"] >= 0.25
    def below(g, b):                                       # G < 0.25 on HE over band b (no fine-tune counts as below)
        return g is None or g["he"][b] is None or g["he"][b] < 0.25
    if void:
        verdict, why = "VOID", "; ".join(void)
    elif gft is None:                                      # the main read-out (frozen s30): only a PASS is final here
        side = ("KILL side" if below(G, "1-8") or below(G, "3-8") else "BETWEEN side")
        verdict = "PASS" if passes(G) else f"NOT PASS on the frozen head ({side}; the fine-tune decides the rest)"
        why = (f"frozen G HE 1-8 {G['he']['1-8']:.3f} (3-8 {G['he']['3-8']:.3f}) / MBPP 1-8 {G['mbpp']['1-8']:.3f} "
               f"(3-8 {G['mbpp']['3-8']:.3f})")
    elif passes(G) or passes(gft):
        verdict = "PASS"
        why = f"frozen G HE {G['he']['1-8']:.3f} / MBPP {G['mbpp']['1-8']:.3f}" + (
            f"; fine-tune G_FT HE {gft['he']['1-8']:.3f} / MBPP {gft['mbpp']['1-8']:.3f}" if gft else "")
    elif (below(G, "1-8") and below(gft, "1-8")) or (below(G, "3-8") and below(gft, "3-8")):
        verdict = "KILL"
        why = (f"frozen G HE 1-8 {G['he']['1-8']:.3f}, 3-8 {G['he']['3-8']:.3f}" +
               (f"; fine-tune G_FT HE 1-8 {gft['he']['1-8']:.3f}, 3-8 {gft['he']['3-8']:.3f}" if gft else "; no fine-tune"))
    else:
        verdict = "BETWEEN"
        why = (f"frozen G HE {G['he']['1-8']:.3f} / MBPP {G['mbpp']['1-8']:.3f}" +
               (f"; fine-tune G_FT HE {gft['he']['1-8']:.3f} / MBPP {gft['mbpp']['1-8']:.3f}" if gft else ""))
    out = {"verdict": verdict, "why": why, "G_frozen": G, "G_ft": gft, "build": bs}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[sib] VERDICT {verdict}: {why}", flush=True)


def cmd_smokecheck(a):
    """the smoke's harness gate (exit 0 = go on to the full run): free-running agreement >= 0.99, the recomputed true
    rows within 0.5 % (mean relative RMS) of the stored residuals, true_r within 1 pt of true (walk-strict 1-8) per set,
    and (amendment 2) >= 99.9 % of the starts' rows before the replaced window bitwise equal to the stored ones"""
    sc = json.load(open(a.out))["res"]
    bad = []
    for st in a.sets.split(","):
        b = json.load(open(H("data", "sib", st + a.tag, "_summary.json")))
        r = sc[f"s30|{st}"]["strict"]["1-8"]
        if b["agree"] is None or b["agree"] < 0.99:
            bad.append(f"{st}: free-running agreement {b['agree']}")
        if b["true_path_rel_err"]["mean"] > 0.005:          # full forwards (amendment 13:1x): ~0 expected
            bad.append(f"{st}: recomputed true rows rel err {b['true_path_rel_err']['mean']:.4f}")
        if abs(r["true_r"] - r["true"]) > 0.01:
            bad.append(f"{st}: true_r {100 * r['true_r']:.2f} vs true {100 * r['true']:.2f}")
        if b.get("prefix_bitwise") is None or b["prefix_bitwise"] < 0.999:
            bad.append(f"{st}: rows before the window bitwise on {b.get('prefix_bitwise')} of the starts")
        print(f"[sib] smoke {st}: agree {b['agree']}, true-path rel err {b['true_path_rel_err']}, true bitwise "
              f"{b.get('true_bitwise')}, prefix bitwise {b.get('prefix_bitwise')}, walk-strict 1-8 "
              f"true {100 * r['true']:.2f} / true_r {100 * r['true_r']:.2f}", flush=True)
    print(f"[sib] SMOKECHECK {'FAIL: ' + '; '.join(bad) if bad else 'OK'}", flush=True)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "score", "verdict", "smokecheck"))
    ap.add_argument("--sets", default="he,mbpp"); ap.add_argument("--head", default="s30")
    ap.add_argument("--heads", default="s30"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=""); ap.add_argument("--frozen", default=""); ap.add_argument("--ft", default="")
    ap.add_argument("--tag", default="", help="data/sib/SET + TAG (the smoke: _smoke)")
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score, "verdict": cmd_verdict, "smokecheck": cmd_smokecheck}[a.cmd](a)


if __name__ == "__main__":
    main()
