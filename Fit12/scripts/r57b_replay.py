"""Fit12 R-57b-R LOOKAHEAD-RESERVED replay (Fri 30 Oct 2026; NewIdeas 19:3x; PLAN.md 'R-57b-R', written before any data).
CPU only (numpy, <= 4 threads), no GPU, no lock. Data: an s10b-format capture of the 45 GB live REF2b loop (BigMoE's cap45):
per prompt pNN_*.npz (seq, L, rows, row_off, ids [rows, 10] score-descending, x_off, x = the MoE input as bf16 bits),
routers.npz (w{T} [512, 2560] f16), miss.bin (int32 records [seq, L, n, (e | miss << 16) x n]), meta.json.
(1) Recall of layer L+1's routing from layer L's MoE input, L = 0..46: raw = router_{L+1}(x_L); corrected = A_L raw + b_L
    (512 x 512 + bias, ridge; lambda by leave-one-prompt-out inside the training half), 2-fold by prompt. Per row top-10
    recall and the check's union recall, per pair and mean. First a router check: top10(router_L(x_L)) == the capture's ids.
(2) Cache replay per layer: LRU at C mirroring the loader's ensure() / make_room() (a miss evicts the oldest resident not
    needed now; then the needed experts are touched in row order). Baseline demand misses per (check, layer) are compared
    with miss.bin at the capture's C. Then N reserved slots per layer: at each check, L+1's top-N non-resident forecast
    experts (ranked by their best row log-prob) are read into a FIFO reserve; a demand miss found there is saved (promoted
    as the miss would be, so the main cache stays the baseline's); reserve entries dropped unused are wasted reads.
    In-time completion of the reads is not modelled (an upper bound).
Usage: [BigMoE venv] r57b_replay.py CAPDIR [C_CAPTURE]   -> /Users/kamen/Fit12/data/r57b/r57b_<capdir name>.txt / .json
       r57b_replay.py --selftest                         (synthetic data: LRU mirror vs a literal loader transcription)"""
import os
for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "4"                                 # RULES: CPU-only jobs use <= 4 cores
import glob, json, sys, time
from collections import OrderedDict
import numpy as np

OUT = "/Users/kamen/Fit12/data/r57b"
RESET = os.environ.get("R57B_RESET") == "1"              # 1: fresh pools per prompt (default: carried over, one process)
NL, K, E = 48, 10, 512
CS, NS = (192, 224, 256, 288, 320), (8, 16, 32)   # (23:4x) C 256+ answer S45b-2's curve without the memory
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)        # ridge, x mean diagonal of the Gram matrix
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def bf16(u):                                             # bf16 bits (uint16) -> float32
    return (u.astype(np.uint32) << 16).view(np.float32)


def top(a, k=K):                                         # top-k columns per row (unordered)
    return np.argpartition(-a, k - 1, axis=-1)[..., :k]


def need_of(ids):                                        # the loader's need: unique ids in row-major order
    seen, need = set(), []
    for e in ids.reshape(-1).tolist():
        if e not in seen:
            seen.add(e); need.append(e)
    return need


# ---------------------------------------------------------------- data
def load_capture(cap):
    """-> prompts: list of dicts {name, checks: [{seq, ids: [48 x (rows, 10)], x: [48 x (rows, 2560) f32 or None]}],
    pre: [(L, ids)] calls before the first check (prefill, if traced)}; W: {T: [512, 2560] f32}; miss: {(seq, L): (n, n_miss)}"""
    prompts = []
    files = sorted(set(glob.glob(os.path.join(cap, "p[0-9][0-9]_*.npz")) + glob.glob(os.path.join(cap, "*_gate_*.npz"))))
    first_seq = {p: int(np.load(p)["seq"].min()) for p in files}
    for p in sorted(files, key=first_seq.get):          # the process order: the gate's pass counter runs across prompts
        d = np.load(p)
        seq, L, rows, ro, ids, xo, x = (d[k] for k in ("seq", "L", "rows", "row_off", "ids", "x_off", "x"))
        first, order = {}, []
        for i in range(len(seq)):                        # each (seq, layer)'s FIRST call (commit re-runs are not served)
            key = (int(seq[i]), int(L[i]))
            if key not in first and int(L[i]) < NL:
                first[key] = i; order.append(key)
        by = {}
        for s, l in order:
            by.setdefault(s, {})[l] = first[(s, l)]
        checks, pre = [], []
        for s in sorted(by):
            ls = by[s]
            full = len(ls) == NL and len({int(rows[i]) for i in ls.values()}) == 1 and int(rows[ls[0]]) <= 16
            if not full:
                if not checks:
                    pre += [(l, ids[ro[i]:ro[i] + rows[i]]) for l, i in sorted(ls.items())]
                continue
            c = {"seq": s, "ids": [ids[ro[ls[l]]:ro[ls[l]] + rows[ls[l]]].astype(np.int64) for l in range(NL)], "x": []}
            for l in range(NL):
                i = ls[l]
                c["x"].append(bf16(x[xo[i]:xo[i] + rows[i]]) if xo[i] >= 0 else None)
            checks.append(c)
        prompts.append({"name": os.path.basename(p)[:-4], "checks": checks, "pre": pre,
                        "res0": d["res0"] if "res0" in d.files else None})   # cap45: slot per expert at the loop start
    W = {}
    rp = os.path.join(cap, "routers.npz")
    if os.path.exists(rp):
        r = np.load(rp)
        W = {int(k[1:]): r[k].astype(np.float32) for k in r.files if k.startswith("w")}
    miss = {}
    mp = os.path.join(cap, "miss.bin")
    if os.path.exists(mp):
        a = np.fromfile(mp, dtype=np.int32); i = 0
        while i + 3 <= len(a):
            s, l, n = int(a[i]), int(a[i + 1]), int(a[i + 2]); v = a[i + 3:i + 3 + n]; i += 3 + n
            if (s, l) not in miss:
                miss[(s, l)] = (n, int(((v >> 16) & 1).sum()))
    return prompts, W, miss


# ---------------------------------------------------------------- cache
def serve(od, need, C):
    """The loader's ensure() without prefetch: returns the misses; od = OrderedDict, last = most recent."""
    ns = set(need)
    miss = [e for e in need if e not in od]
    n_ev = len(od) + len(miss) - C
    if n_ev > 0:
        vic = []
        for e in od:                                     # oldest first, skipping experts needed now
            if e not in ns:
                vic.append(e)
                if len(vic) == n_ev:
                    break
        for v in vic:
            del od[v]
    for e in need:                                       # touch in row order: the last needed ends most recent
        if e in od:
            od.move_to_end(e)
        else:
            od[e] = True
    return miss


def serve_literal(lru, need, C):
    """A line-by-line transcription of bigmoe_ext.cpp ensure() + make_room() (no prefetch, no LFU), list front = most
    recent; for the self-test only."""
    misses = []
    for e in need:
        if e in lru:
            continue
        misses.append(e)
        while len(lru) >= C and lru:                     # make_room: evict from the tail, skipping needed
            j = len(lru) - 1
            while j >= 0 and lru[j] in need:
                j -= 1
            if j < 0:
                break
            del lru[j]
        lru.insert(0, e)
    for e in need:
        if e in lru:
            lru.remove(e); lru.insert(0, e)
    return misses


def ranked(score, n_max):                                # experts by best row log-prob, descending
    ls = score - score.max(-1, keepdims=True)
    ls = ls - np.log(np.exp(ls).sum(-1, keepdims=True))
    best = ls.max(0)
    return np.argsort(-best)[:n_max]


def replay(prompts, C, fore=None, N=0):
    """fore[(prompt index, check index, T)] -> [rows, 512] forecast logits for layer T (T = 1..47). Returns totals and
    per (seq, L) baseline misses."""
    tot = {"need": 0, "miss": 0, "saved": 0, "reads_pf": 0, "wasted": 0, "checks": 0, "checks16": 0}
    per = {}
    for pi, P in enumerate(prompts):
        if pi == 0 or RESET:                             # one process runs the prompts in file order: pools carry over
            main = [OrderedDict() for _ in range(NL)]
            res = [OrderedDict() for _ in range(NL)]     # reserve: expert -> used flag
        if P["res0"] is not None:                        # cap45: the resident set at the loop start (PF1 adopt_full).
            for l in range(NL):                          # Its LRU order is not exported: residents kept from the last
                r0 = P["res0"][l]                        # prompt keep their order, the prefill's adoptions go to the
                od = OrderedDict((e, True) for e in main[l] if r0[e] >= 0)   # front in slot order (an approximation)
                for e in np.argsort(r0, kind="stable").tolist():
                    if r0[e] >= 0 and e not in od:
                        od[e] = True
                while len(od) > C:
                    od.popitem(last=False)
                main[l] = od
        for l, ids in P["pre"]:                          # traced prefill calls warm the pools (wider than C: the last C)
            serve(main[l], need_of(ids)[-C:], C)
        for ci, c in enumerate(P["checks"]):
            tot["checks"] += 1; tot["checks16"] += c["ids"][0].shape[0] == 16
            for l in range(NL):
                need = need_of(c["ids"][l])
                if N and fore is not None and l + 1 < NL:     # while L computes: L+1's reads into its reserve
                    T = l + 1
                    f = fore.get((pi, ci, T))
                    if f is not None:
                        R, m = res[T], main[T]
                        picks = [int(e) for e in ranked(f, 4 * N) if int(e) not in m][:N]
                        for e in picks:
                            if e in R:
                                R.move_to_end(e)
                            else:
                                R[e] = False; tot["reads_pf"] += 1
                        while len(R) > N:
                            e, used = R.popitem(last=False)
                            tot["wasted"] += 0 if used else 1
                miss = serve(main[l], need, C)
                R = res[l]
                sv = [e for e in miss if e in R]
                for e in sv:                             # promoted into the main cache (as the miss would be)
                    del R[e]
                tot["need"] += len(need); tot["miss"] += len(miss); tot["saved"] += len(sv)
                per[(c["seq"], l)] = (len(need), len(miss))
        if pi == len(prompts) - 1 or RESET:
            for R in res:
                tot["wasted"] += sum(1 for u in R.values() if not u)
    return tot, per


# ---------------------------------------------------------------- forecasts
def pairs_data(prompts, W, T):
    """rows of (prompt, check) for target layer T: raw = router_T(x_{T-1}), true logits = router_T(x_T), true ids."""
    out = []
    for pi, P in enumerate(prompts):
        for ci, c in enumerate(P["checks"]):
            xs, xt = c["x"][T - 1], c["x"][T]
            if xs is None or xt is None:
                continue
            out.append((pi, ci, xs @ W[T].T, xt @ W[T].T, c["ids"][T]))
    return out


def recall(f, ids):
    t = top(f)
    r = np.mean([len(set(a.tolist()) & set(b.tolist())) / K for a, b in zip(t, ids)])
    uf, ut = set(t.reshape(-1).tolist()), set(ids.reshape(-1).tolist())
    return r, len(uf & ut) / len(ut), len(uf), len(ut)


def ridge(X, Y, lam):
    """the correction as a residual on the raw logits: Y - X ~ X D + b, ridge on D (lambda -> inf: raw + mean residual)"""
    X1 = np.hstack([X, np.ones((len(X), 1), X.dtype)]).astype(np.float64)
    G = X1.T @ X1
    reg = lam * np.mean(np.diag(G)[:-1]) * np.eye(G.shape[0]); reg[-1, -1] = 0
    return np.linalg.solve(G + reg, X1.T @ (Y - X).astype(np.float64)).astype(np.float32)


def apply(B, X):
    return X + X @ B[:-1] + B[-1]


def fit_fold(rows, train_p):
    """ridge with lambda by leave-one-prompt-out inside the training prompts (mean top-10 recall)"""
    tr = [r for r in rows if r[0] in train_p]
    best, bl = -1.0, LAMS[0]
    if len(train_p) >= 2:
        for lam in LAMS:
            sc = []
            for hp in train_p:
                a = [r for r in tr if r[0] != hp]; b = [r for r in tr if r[0] == hp]
                if not a or not b:
                    continue
                B = ridge(np.vstack([r[2] for r in a]), np.vstack([r[3] for r in a]), lam)
                sc += [recall(apply(B, r[2]), r[4])[0] for r in b]
            if sc and np.mean(sc) > best:
                best, bl = float(np.mean(sc)), lam
    B = ridge(np.vstack([r[2] for r in tr]), np.vstack([r[3] for r in tr]), bl)
    return B, bl


def main(cap, c_cap):
    t0 = time.time()
    prompts, W, miss = load_capture(cap)
    name = os.path.basename(os.path.normpath(cap))
    nchk = sum(len(P["checks"]) for P in prompts)
    rows16 = [c["ids"][0].shape[0] for P in prompts for c in P["checks"]]
    xl = sorted({l for P in prompts for c in P["checks"] for l in range(NL) if c["x"][l] is not None})
    say(f"R-57b-R replay of {cap}: {len(prompts)} prompts, {nchk} checks (rows median {np.median(rows16):.0f}), "
        f"prefill calls traced {sum(len(P['pre']) for P in prompts)}, x at {len(xl)} layers, routers {len(W)}, "
        f"miss.bin records {len(miss)}; loaded in {time.time() - t0:.0f} s")
    res = {"cap": cap, "C_capture": c_cap}
    # ---- router check
    if W and xl:
        ag = []
        for T in sorted(W):
            for P in prompts:
                for c in P["checks"]:
                    if c["x"][T] is not None:
                        ag.append(recall(c["x"][T] @ W[T].T, c["ids"][T])[0])
        say(f"  router check: top10(router_L(x_L)) vs the capture's ids, mean set agreement {np.mean(ag):.4f} over "
            f"{len(ag)} (check, layer) pairs (must be >= 0.99)")
        res["router_check"] = float(np.mean(ag))
    # ---- baseline replay vs miss.bin
    if c_cap and miss:
        _, per = replay(prompts, c_cap)
        both = [(per[k][1], miss[k][1]) for k in per if k in miss]
        eq = sum(a == b for a, b in both)
        tr, tm = sum(a for a, _ in both), sum(b for _, b in both)
        say(f"  baseline vs miss.bin at C {c_cap}: {len(both)} (check, layer) pairs matched; equal miss counts in "
            f"{eq} ({100 * eq / max(1, len(both)):.1f} %); misses replay {tr} vs loader {tm} ({100 * (tr / max(1, tm) - 1):+.2f} %)")
        res["miss_check"] = {"pairs": len(both), "equal": eq, "replay": tr, "loader": tm}
    # ---- recall (1)
    fore_raw, fore_cor, fore_orc, rec = {}, {}, {}, []
    for T in range(1, NL):                               # the raw forecast needs x only at the source layer; the oracle
        if T in W:                                       # (layer T's own scores) is the reserve's ceiling at each N
            for pi, P in enumerate(prompts):
                for ci, c in enumerate(P["checks"]):
                    if c["x"][T - 1] is not None:
                        fore_raw[(pi, ci, T)] = c["x"][T - 1] @ W[T].T
                    if c["x"][T] is not None:
                        fore_orc[(pi, ci, T)] = c["x"][T] @ W[T].T
    if W and len(prompts) >= 2:
        half = [set(range(0, len(prompts), 2)), set(range(1, len(prompts), 2))]
        for T in range(1, NL):
            rows = pairs_data(prompts, W, T) if T in W else []
            if not rows:
                continue
            rr = [recall(r[2], r[4]) for r in rows]
            cr, lams = [], []
            for f in (0, 1):
                B, lam = fit_fold(rows, half[f])
                lams.append(lam)
                for r in rows:
                    if r[0] in half[1 - f]:
                        fc = apply(B, r[2]); fore_cor[(r[0], r[1], T)] = fc
                        cr.append(recall(fc, r[4]))
            rec.append((T, np.mean([a[0] for a in rr]), np.mean([a[1] for a in rr]), np.mean([a[0] for a in cr]),
                        np.mean([a[1] for a in cr]), np.mean([a[2] for a in cr]), np.mean([a[3] for a in cr]), lams))
        say("  (1) recall of layer T's routing from layer T-1's MoE input (per row top-10 | union of the check):")
        for T, a, au, b, bu, nf, nt, lams in rec:
            say(f"    T {T:2d}: raw {a:.3f} | {au:.3f}; corrected {b:.3f} | {bu:.3f} (union sizes forecast {nf:.0f} vs "
                f"true {nt:.0f}; lambda {lams[0]:g} / {lams[1]:g})")
        mr, mc = float(np.mean([r[1] for r in rec])), float(np.mean([r[3] for r in rec]))
        mru, mcu = float(np.mean([r[2] for r in rec])), float(np.mean([r[4] for r in rec]))
        say(f"    mean over {len(rec)} pairs: raw {mr:.3f} | union {mru:.3f}; corrected {mc:.3f} | union {mcu:.3f}")
        res["recall"] = {"raw": mr, "raw_union": mru, "cor": mc, "cor_union": mcu,
                         "per_T": [[float(v) if not isinstance(v, list) else v for v in r] for r in rec]}
    # ---- replay (2)
    say("  (2) cache replay (per 16-row check = every pass's misses incl. commit replays / the 16-row passes, as S45 "
        "counts; saved / wasted = reserve hits / reserve reads dropped unused):")
    res["replay"] = {}
    for C in CS:
        b, _ = replay(prompts, C)
        n = max(1, b["checks16"])
        say(f"    C {C}: baseline {b['miss'] / n:.1f} misses per check ({b['need'] / n:.0f} needed; {b['checks']} passes, "
            f"{b['checks16']} of them 16-row)")
        res["replay"][C] = {"base": b}
        for lab, fo in (("raw", fore_raw), ("corrected", fore_cor), ("oracle", fore_orc)):
            if not fo:
                continue
            for N in NS:
                t, _ = replay(prompts, C, fo, N)
                assert t["miss"] == b["miss"] and t["need"] == b["need"], "the main cache must stay the baseline's"
                say(f"      {lab:9s} N {N:2d}: saved {t['saved'] / n:6.1f} per check ({100 * t['saved'] / max(1, b['miss']):5.1f} % "
                    f"of misses), prefetch reads {t['reads_pf'] / n:6.1f}, wasted {t['wasted'] / n:6.1f}; demand misses "
                    f"{(b['miss'] - t['saved']) / n:.1f}")
                res["replay"][C][f"{lab}_N{N}"] = t
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"r57b_{name}.json"), "w"), default=str)
    open(os.path.join(OUT, f"r57b_{name}.txt"), "w").write("\n".join(lines) + "\n")
    say(f"  done in {time.time() - t0:.0f} s -> {OUT}/r57b_{name}.txt")


# ---------------------------------------------------------------- export (BigMoE's R-57b build, Sat 31 Oct 01:2x)
def export(cap, out):
    """One map per target layer T = 1..47, fit on ALL prompts (lambda by leave-one-prompt-out), in BigMoE's form:
    corrected_T = raw_T @ W{T} + b{T}, raw_T = router_T(x_{T-1}) in f32 on the raw LOGITS (not log-probs); W = I + D of the
    residual ridge, f16; b f32. Checks the f16 rounding: recall with the exported maps vs the f32 fit, in sample."""
    t0 = time.time()
    prompts, W, _ = load_capture(cap)
    allp = {int(v) for v in os.environ["R57B_TRAIN"].split(",")} if os.environ.get("R57B_TRAIN") else set(range(len(prompts)))   # prompt indices in process order
    d, lam_t, chk = {}, {}, []
    for T in range(1, NL):
        rows = pairs_data(prompts, W, T) if T in W else []
        if not rows:
            continue
        B, lam = fit_fold(rows, allp)
        Wt = (np.eye(E, dtype=np.float32) + B[:-1]).astype(np.float16)
        bt = B[-1].astype(np.float32)
        d[f"W{T}"], d[f"b{T}"], lam_t[T] = Wt, bt, lam
        r32 = np.mean([recall(apply(B, r[2]), r[4])[0] for r in rows])
        r16 = np.mean([recall(r[2] @ Wt.astype(np.float32) + bt, r[4])[0] for r in rows])
        chk.append((T, r32, r16))
    d["spec"] = np.array("corrected_T = raw_T @ W{T} + b{T}; raw_T = router_T(x_{T-1}) f32 logits (cap45 x, bf16 -> f32); "
                         "rank = max over rows of log_softmax(corrected_T); fit on all of cap45 (6 prompts), ridge residual")
    d["lambda"] = np.array([lam_t.get(T, np.nan) for T in range(NL)], np.float32)
    np.savez(out, **d)
    say(f"export -> {out}: {len(lam_t)} layers in {time.time() - t0:.0f} s; in-sample recall f32 {np.mean([c[1] for c in chk]):.4f} "
        f"vs exported f16 {np.mean([c[2] for c in chk]):.4f} (max |diff| per layer {max(abs(c[1] - c[2]) for c in chk):.4f})")


# ---------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(57)
    ok = True
    # (a) the LRU mirror vs the literal loader transcription on random need streams
    for C in (24, 40):
        od, lru = OrderedDict(), []
        for step in range(3000):
            pool = rng.integers(0, 120 if step % 3 else 60)
            need = need_of(rng.choice(np.arange(pool + 30), size=(rng.integers(1, 5), 4)))
            need = need[:C]
            m1, m2 = serve(od, need, C), serve_literal(lru, need, C)
            if m1 != m2 or list(reversed(list(od))) != lru:
                ok = False; print(f"selftest: LRU mismatch at C {C} step {step}"); break
    print(f"selftest (a) LRU mirror == literal ensure()/make_room(): {ok}")
    # (b) a synthetic capture through the whole pipeline (hidden size 64 instead of 2560)
    D = 64
    d = os.path.join(OUT, "selftest_cap"); os.makedirs(d, exist_ok=True)
    W = {T: rng.standard_normal((E, D)).astype(np.float16) for T in range(NL)}
    M = [np.linalg.qr(rng.standard_normal((D, D)))[0].astype(np.float32) for _ in range(NL)]   # learnable layer maps
    np.savez(os.path.join(d, "routers.npz"), **{f"w{T}": W[T] for T in range(1, NL)})
    miss_rec = []
    for p in range(4):
        seq, Ls, rows, ids, xo, xs = [], [], [], [], [], []
        n_x, s = 0, 0
        for chk in range(24):
            for r in (16, int(rng.integers(1, 17))):     # a 16-row check, then its commit replay (k+1 rows)
                x = rng.standard_normal((r, D)).astype(np.float32)
                for l in range(NL):
                    x = (0.95 * x @ M[l] + 0.3 * rng.standard_normal((r, D))).astype(np.float32)
                    xb = (x.view(np.uint32) >> 16).astype(np.uint16)
                    lg = bf16(xb) @ W[l].astype(np.float32).T
                    seq.append(1000 * p + s); Ls.append(l); rows.append(r)
                    ids.append(np.argsort(-lg, -1)[:, :K].astype(np.int16)); xo.append(n_x); xs.append(xb); n_x += r
                s += 1
        rws = np.array(rows, np.int16)
        np.savez(os.path.join(d, f"p{p:02d}_syn{p}.npz"), seq=np.array(seq, np.int32), L=np.array(Ls, np.int16),
                 rows=rws, row_off=np.cumsum(rws, dtype=np.int64) - rws, ids=np.concatenate(ids),
                 x_off=np.array(xo, np.int64), x=np.concatenate(xs))
    prompts, _, _ = load_capture(d)
    lrus = [[] for _ in range(NL)]                        # a miss.bin from the literal loader at C 192, one process,
    for pi, P in enumerate(prompts):                      # with a synthetic res0 per prompt (prefill adoptions)
        if RESET:
            lrus = [[] for _ in range(NL)]
        r0 = -np.ones((NL, E), np.int32)
        for l in range(NL):
            keep = [e for e in lrus[l] if rng.random() < 0.9]
            new = [int(e) for e in rng.permutation(E)[:6] if e not in lrus[l]]
            for j, e in enumerate(rng.permutation(keep + new).tolist()):
                r0[l, e] = j
            kept = [e for e in lrus[l] if r0[l, e] >= 0]
            adopt = [e for e in np.argsort(r0[l], kind="stable").tolist() if r0[l, e] >= 0 and e not in kept]
            lrus[l] = list(reversed(adopt)) + kept
        P["res0"] = r0
        f = glob.glob(os.path.join(d, f"p{pi:02d}_*.npz"))[0]
        z = dict(np.load(f)); z["res0"] = r0; np.savez(f, **z)
        for c in P["checks"]:
            for l in range(NL):
                need = need_of(c["ids"][l]); m = set(serve_literal(lrus[l], need, 192))
                miss_rec += [c["seq"], l, len(need)] + [e | ((e in m) << 16) for e in need]
    np.array(miss_rec, np.int32).tofile(os.path.join(d, "miss.bin"))
    del lines[:]
    main(d, 192)
    r = json.load(open(os.path.join(OUT, "r57b_selftest_cap.json")))
    ok_b = r["router_check"] > 0.99 and r["miss_check"]["equal"] == r["miss_check"]["pairs"] > 0
    ok_c = r["recall"]["cor"] > r["recall"]["raw"] + 0.1          # the layer maps are linear: the correction must help
    print(f"selftest (b) synthetic capture: router check {r['router_check']:.4f}, miss.bin equal "
          f"{r['miss_check']['equal']} / {r['miss_check']['pairs']}: {ok_b}; recall raw {r['recall']['raw']:.3f} -> "
          f"corrected {r['recall']['cor']:.3f}: {ok_c}")
    ok_b = ok_b and ok_c
    print("SELFTEST", "PASS" if ok and ok_b else "FAIL")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--export"]:
        export(sys.argv[2], sys.argv[3])
    elif sys.argv[1:2] == ["--selftest"]:
        selftest()
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
