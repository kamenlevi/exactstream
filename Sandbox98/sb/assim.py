"""ASSIM screen (THINK_1116c drift wall; Fable C's ASSIMILATED DRAFTING and Fable B's MEANPULL / ERRMEM, both
IDEAS_1116c_drift.md): the frozen s30 head's carried stand-in corrected online, per answer window, from what the
32-row checks of the EARLIER blocks of the same answer give for free: the target's true state at every row they accepted.

The decode trajectory is the SIB build's (data/sib/SET/*.npz 'kept': the s30 chain's teacher-forced 32-row checks from
the answer's first anchor; block b starts at anchor s_b and accepts k_b = kept_b - 1 drafted tokens). Block b's check
gives the true W at s_b + d for d = 1 .. k_b + 1 (the bonus input token's row and the k_b accepted rows), and the head's
stand-in at depth d of block b (the carried state that stands in for W[s_b + d]) was computed on the same tokens there,
so r_{b,d} = unit(W[s_b + d]) - unit(stand-in_{b,d}) is an exact, runtime-available measurement of the drift.
All vectors live in unit-RMS space (the head's input norm makes the scale irrelevant); a corrected stand-in keeps the
carried row's RMS. Depth d = chain step j - 1; depth 0 (the true anchor state) is never touched.

Arm parameters (presets below; custom: NAME=src:open/pool:depth/ema:0.3/beta:0.5/K:0.2/dmin:3):
  K, dmin   MEANPULL: u <- unit((1 - K) u + K m) at depths >= dmin, m = unit(the EMA of the true W rows 0..s_b,
            half-life --mean_hl rows): the window's slow component (Fable B (a))
  src       none | open: residuals of the UNCORRECTED chain (the score's own stand-ins; Fable C's screen 1 as written)
            | closed: residuals of this arm's own pre-correction stand-ins at the same depth, i.e. what the drafter
            itself would compare after each check (Fable B (b)); runtime-exact up to one approximation: a residual's
            validity uses the uncorrected trajectory's k_b
  pool      pool: one vector over all accepted depths (open only) | depth: one vector per depth d
  ema       mean: the running mean over every earlier residual row | a float: the bias-corrected EMA over blocks
  beta      the gain on the residual estimate e: u <- unit(u + beta e) once the window has a history
Rows scored: --rows sib = the SIB main read-out's cells (post-miss starts, depths 0-8; its own / true counts must be
reproduced exactly: VOID otherwise) | --rows blocks = every block anchor of the trajectory, depths 0-30.
Report: walk-strict and marginal per band, G = (arm - own) / (true - own) with a bootstrap over windows (2000, rng 1116),
kept per check at the trajectory's anchors (blocks), and the direction error of the stand-in the head read, per depth.
usage: sb/assim.py --rows sib --out results/assim_sib.json (default arms: every preset but noop)
       [--sets he,mbpp] [--n N] [--tag _smoke] [--smoke (adds the no-op arm, which must equal own bit for bit)]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

PRESETS = {      # NewIdeas 18:5x picks (Fable C's arm as written, its EMA variant; Fable B's MEANPULL K grid + ERRMEM)
    "assim": dict(src="open", pool="pool", ema="mean", beta=1.0, blocks="sib", kplus=0),         # C: previous starts, d <= k
    "assim_ema4": dict(src="open", pool="pool", ema=float(1 - 0.5 ** 0.25), beta=1.0, blocks="sib", kplus=0),  # hl 4 starts
    "assim_all": dict(src="open", pool="pool", ema="mean", beta=1.0, blocks="all", kplus=1),     # every check, d <= k + 1
    "errmem": dict(src="closed", pool="depth", ema=0.5, beta=1.0),
    "mp20": dict(K=0.2, dmin=3), "mp35": dict(K=0.35, dmin=3), "mp50": dict(K=0.5, dmin=3),
    "mp35_errmem": dict(K=0.35, dmin=3, src="closed", pool="depth", ema=0.5, beta=1.0),
    "noop": dict(),                                                                    # smoke: must equal own
}
BASE = dict(src="none", pool="pool", ema="mean", beta=1.0, K=0.0, dmin=1, blocks="all", kplus=1)
# residual rows: blocks 'all' = every earlier check of the answer, 'sib' = only the earlier post-miss starts (Fable C as
# written); kplus 1 = depths 1 .. k + 1 (the bonus input's row and the k accepted rows), 0 = depths 1 .. k (Fable C)
BANDS = {"sib": {"1-2": (1, 2), "3-6": (3, 6), "7-8": (7, 8), "1-8": (1, 8), "3-8": (3, 8)},
         "blocks": {"1-2": (1, 2), "3-6": (3, 6), "7-14": (7, 14), "15-30": (15, 30), "3-14": (3, 14), "3-30": (3, 30),
                    "1-30": (1, 30)}}


def parse_arms(spec):
    arms = {}
    for it in spec.split(","):
        it = it.strip()
        if not it:
            continue
        name, _, cfg = it.partition("=")
        a = dict(BASE)
        if cfg:
            for kv in cfg.split("/"):
                k, _, v = kv.partition(":")
                a[k] = v if k in ("src", "pool", "blocks") or (k == "ema" and v == "mean") else float(v)
            a["dmin"] = int(a["dmin"]); a["kplus"] = int(a["kplus"])
        else:
            a.update(PRESETS[name])
        if a["src"] == "closed" and a["pool"] != "depth":
            sys.exit(f"{name}: a closed-loop residual can only pool per depth (deeper depths are not computed yet)")
        arms[name] = a
    return arms


def unit(x):
    return x / np.maximum(np.sqrt((x * x).mean(-1, keepdims=True)), 1e-6)


def causal(R, valid, ema):
    """R [nb, dim] residual sums per block, valid [nb] counts (rows) -> E [nb, dim], has [nb]: the estimate each block
    sees, from the blocks before it only (mean over every earlier row, or the bias-corrected EMA over block means)"""
    nb, dim = R.shape
    E = np.zeros((nb, dim), np.float32); has = np.zeros(nb, bool)
    if ema == "mean":
        cs = np.cumsum(R, 0); cn = np.cumsum(valid)
        E[1:] = cs[:-1] / np.maximum(cn[:-1], 1)[:, None]; has[1:] = cn[:-1] > 0
        return E, has
    a = float(ema); e = np.zeros(dim); w = 0.0
    for b in range(nb):
        if w > 0:
            E[b] = e / w; has[b] = True
        if valid[b] > 0:
            e = (1 - a) * e + a * (R[b] / valid[b]); w = (1 - a) * w + a
    return E, has


def dirr(u, t):
    return np.sqrt(((unit(u) - t) ** 2).mean(-1))


def run_window(C, mx, SB, D, f, st, a, arms, DP):
    key = os.path.basename(f)[:-4]
    o = np.load(SB.H("data", "sib", st + a.tag, key + ".npz"))
    z = np.load(f)
    S = len(z["ids"]); gen = int(z["gen"]); a0 = int(o["a0"])
    assert a0 == max(0, gen - 2) and int(o["S"]) == S
    r0 = max(0, gen - 1 - a0)
    pos, kb, p = [], [], r0
    for kp in o["kept"]:
        pos.append(a0 + p); kb.append(int(kp) - 1); p += int(kp)
    pos, kb = np.array(pos, np.int64), np.array(kb, np.int64)
    nb = len(pos)
    if a.rows == "sib":
        assert set(o["starts"].tolist()) <= set(pos.tolist()), "SIB starts are not block anchors"
        scored = np.isin(pos, o["starts"])
        srows = o["starts"] - a0
    else:
        scored = np.ones(nb, bool); srows = pos - a0
    W = C.load_W(z).astype(mx.float32); Wn = np.array(W[0])
    dim = Wn.shape[1]
    # unit true rows at every block's depths, their validity, and the residual validity (accepted rows only)
    dd = np.arange(DP)
    tpos = pos[:, None] + dd[None]                                        # [nb, DP]
    tval = tpos <= S - 1
    tW = np.zeros((nb, DP, dim), np.float32)
    tW[tval] = unit(Wn[tpos[tval]])
    is_start = np.isin(pos, o["starts"])

    def rmask(arm):                                    # which (block, depth) residual rows this arm may use
        m_ = tval & (dd[None] >= 1) & (dd[None] <= (kb + arm["kplus"])[:, None])
        return m_ & is_start[:, None] if arm["blocks"] == "sib" else m_
    # MEANPULL's slow component (Fable B: the window's EMA of the true W over accepted rows): the EMA of the true W
    # rows from the answer's first anchor a0 up to each block's anchor, half-life a.mean_hl rows
    al = 1 - 0.5 ** (1.0 / a.mean_hl)
    m = np.zeros(dim); w = 0.0; mrow = np.zeros((nb, dim), np.float32); bi = 0
    order = np.argsort(pos)
    for q in range(a0, int(pos.max()) + 1 if nb else a0):
        m = (1 - al) * m + al * Wn[q]; w = (1 - al) * w + al
        while bi < nb and pos[order[bi]] == q:
            mrow[order[bi]] = unit(m / w); bi += 1
    rows = pos - a0
    out = {}

    def hook(name, arm, rec=None, oc=None, och=None):
        de = np.zeros((2, DP)); state = {"chk": False}; rv = rmask(arm)

        def fn(j, h, anc):
            if j < 2:
                return h
            d = j - 1
            if d >= DP:
                return h
            n = h.shape[1]
            sel = rows < n
            if not sel.any():
                return h
            rr = rows[sel]
            ri = mx.array(rr.astype(np.int32))
            if not state["chk"]:
                assert (np.array(anc)[rr] == pos[sel]).all(), "row / anchor mismatch"
                state["chk"] = True
            hb = np.array(h[0][ri].astype(mx.float32))
            nrm = np.maximum(np.sqrt((hb * hb).mean(-1, keepdims=True)), 1e-6)
            u = hb / nrm
            if rec is not None:
                rec[sel, d] = u; recd[sel, d] = True
            changed = False
            if arm["K"] > 0 and d >= arm["dmin"]:
                u = unit((1 - arm["K"]) * u + arm["K"] * mrow[sel]); changed = True
            corr = has = None
            if arm["src"] == "open":
                corr, has = oc[sel, d], och[sel, d]
            elif arm["src"] == "closed":
                v = rv[sel, d]
                R = np.where(v[:, None], tW[sel, d] - u, 0.0)
                corr, has = causal(R, v.astype(np.int64), arm["ema"])
            if has is not None and has.any():
                u = u.copy(); u[has] = unit(u[has] + arm["beta"] * corr[has]); changed = True
            sc = scored[sel] & tval[sel, d]
            if sc.any():
                de[0, d] += dirr(u[sc], tW[sel, d][sc]).sum(); de[1, d] += int(sc.sum())
            if not changed:
                return h
            hh = h[0]
            hh[ri] = mx.array((u * nrm).astype(np.float32)).astype(hh.dtype)
            return hh[None]
        return fn, de

    # own (uncorrected; records its stand-ins at the block anchors) and true
    rec = np.zeros((nb, DP, dim), np.float32); recd = np.zeros((nb, DP), bool)
    fown, de_own = hook("own", BASE, rec=rec)
    rt_own = C.chain_table(D, f, Dp=DP, fn=fown)[0]
    if a.rows == "blocks":
        assert [k + 1 for _, k in SB.walk(rt_own, r0)] == o["kept"].tolist(), "trajectory differs from the build's"
    rt_true = C.chain_table(D, f, Dp=DP, R=W)[0]
    out["own"] = (rt_own, de_own); out["true"] = (rt_true, np.zeros((2, DP)))
    # the open-loop estimates (from the uncorrected stand-ins): pooled over accepted depths, or per depth; only rows
    # the chain actually computed (an anchor leaves the batch near the answer's end)
    resall = tW - rec                                                    # [nb, DP, dim]
    pre = {}
    for name, arm in arms.items():
        oc = och = None
        if arm["src"] == "open":
            k = (arm["pool"], str(arm["ema"]), arm["blocks"], arm["kplus"])
            if k not in pre:
                rv0 = rmask(arm) & recd
                res0 = np.where(rv0[..., None], resall, 0.0)
                if arm["pool"] == "pool":
                    E, has = causal(res0.sum(1), rv0.sum(1), arm["ema"])
                    pre[k] = (np.repeat(E[:, None], DP, 1), np.repeat(has[:, None], DP, 1))
                else:
                    Ed, hd = np.zeros_like(res0), np.zeros(rv0.shape, bool)
                    for d in range(1, DP):
                        Ed[:, d], hd[:, d] = causal(res0[:, d], rv0[:, d].astype(np.int64), arm["ema"])
                    pre[k] = (Ed, hd)
            oc, och = pre[k]
        fa, de_a = hook(name, arm, oc=oc, och=och)
        out[name] = (C.chain_table(D, f, Dp=DP, fn=fa)[0], de_a)
    mx.clear_cache()
    cnt = {nm: SB.counts(rt[srows][:, :DP]) for nm, (rt, _) in out.items()}
    des = {nm: de for nm, (_, de) in out.items()}
    info = {"blocks": nb, "scored": int(scored.sum()), "resid_rows": int((rmask(BASE) & recd).sum())}
    return key, cnt, des, info


def band_vals(c, bands):
    """c [5, DP] summed counts -> strict / marginal per band, depth 0, kept (mean run + 1 within the read depths)"""
    v = {"strict": {}, "marginal": {}}
    for b, (lo, hi) in bands.items():
        v["strict"][b] = float(c[1, lo:hi + 1].sum() / max(c[0, lo:hi + 1].sum(), 1))
        v["marginal"][b] = float(c[3, lo:hi + 1].sum() / max(c[2, lo:hi + 1].sum(), 1))
    v["depth0"] = float(c[1, 0] / max(c[0, 0], 1)); v["kept"] = float(c[4, 0] / max(c[4, 1], 1) + 1)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", choices=("sib", "blocks"), default="sib")
    ap.add_argument("--arms", default="assim,assim_ema4,assim_all,errmem,mp20,mp35,mp50,mp35_errmem")
    ap.add_argument("--sets", default="he,mbpp"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--tag", default=""); ap.add_argument("--mean_hl", type=float, default=64.0)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx
    from sb import core as C, sibstate as SB
    C.limit_memory(float(os.environ.get("SB_MEM_GB", "3")))
    C.ARGMAX_ROWS = 256
    C.load_target(); C.drop_trunk()
    D = C.load_drafter("s30")
    arms = parse_arms(a.arms + (",noop" if a.smoke else ""))
    DP = SB.NSIB + 1 if a.rows == "sib" else SB.DCHAIN
    bands = BANDS[a.rows]
    names = ["own", "true"] + list(arms)
    ref = json.load(open(SB.H("results", "sib_s30.json")))["res"] if a.rows == "sib" and not a.tag else None
    t0 = time.time(); res = {"args": vars(a), "arms": arms, "sets": {}}
    for st in a.sets.split(","):
        fs = SB.set_files(st, a.n)
        per, dsum, keys, infos = [], {nm: np.zeros((2, DP)) for nm in names}, [], []
        for f in fs:
            C.wait_pause()
            if a.rows == "sib" and len(np.load(SB.H("data", "sib", st + a.tag, os.path.basename(f)))["starts"]) == 0:
                continue                                                 # as the SIB score: no post-miss start, no cells
            key, cnt, des, info = run_window(C, mx, SB, D, f, st, a, arms, DP)
            keys.append(key); infos.append(info)
            per.append(np.stack([cnt[nm] for nm in names]))
            for nm in names:
                dsum[nm] += des[nm]
            print(f"[assim] {st} {key}: {info['blocks']} blocks, {info['scored']} scored, {info['resid_rows']} residual rows"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
        per = np.stack(per)                                              # [windows, arms, 5, DP]
        tot = per.sum(0)
        v = {nm: band_vals(tot[i], bands) for i, nm in enumerate(names)}
        voids = {}
        if ref is not None:                                              # VOID 1: own / true = the SIB main read-out's cells
            pw = np.array(ref[f"s30|{st}"]["per_window"])[:len(per)]
            voids["own_true_match_sib_s30"] = bool((pw[:, 0] == per[:, 0]).all() and (pw[:, 1] == per[:, 1]).all())
        if a.smoke:
            voids["noop_equals_own"] = bool((per[:, names.index("noop")] == per[:, 0]).all())
        rng = np.random.default_rng(1116)
        idx = rng.integers(0, len(per), (a.boot, len(per)))
        G, Gci, Dci = {}, {}, {}
        bt = [band_vals(per[ix].sum(0)[i], bands) for ix in idx for i in (0, 1)] if a.boot else []
        for i, nm in enumerate(names[2:], 2):
            G[nm], Gci[nm], Dci[nm] = {}, {}, {}
            arm_b = [band_vals(per[ix, i].sum(0), bands) for ix in idx] if a.boot else []
            for b in bands:
                den = v["true"]["strict"][b] - v["own"]["strict"][b]
                G[nm][b] = (v[nm]["strict"][b] - v["own"]["strict"][b]) / den if abs(den) > 1e-9 else None
                if a.boot:
                    gs, ds = [], []
                    for t in range(a.boot):
                        o_, t_, x_ = bt[2 * t]["strict"][b], bt[2 * t + 1]["strict"][b], arm_b[t]["strict"][b]
                        ds.append(x_ - o_)
                        if abs(t_ - o_) > 1e-9:
                            gs.append((x_ - o_) / (t_ - o_))
                    Gci[nm][b] = [float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))] if gs else None
                    Dci[nm][b] = [float(100 * np.percentile(ds, 2.5)), float(100 * np.percentile(ds, 97.5))]
        de = {nm: [float(dsum[nm][0, d] / dsum[nm][1, d]) if dsum[nm][1, d] else None for d in range(DP)] for nm in names}
        lo3 = 3; hi3 = 8 if a.rows == "sib" else 30
        de38 = {nm: float(dsum[nm][0, lo3:hi3 + 1].sum() / max(dsum[nm][1, lo3:hi3 + 1].sum(), 1)) for nm in names}
        res["sets"][st] = {"windows": len(per), "blocks": int(sum(x["blocks"] for x in infos)),
                           "scored": int(sum(x["scored"] for x in infos)),
                           "resid_rows": int(sum(x["resid_rows"] for x in infos)), "vals": v, "G": G, "G_ci95": Gci,
                           "delta_ci95_pt": Dci, "dir_err": de, f"dir_err_{lo3}-{hi3}": de38, "voids": voids,
                           "keys": keys, "per_window": per.tolist()}
        r = res["sets"][st]
        print(f"\n[assim] {st}: {r['windows']} windows, {r['blocks']} blocks, {r['scored']} scored rows, rows {a.rows};"
              f" voids {voids}")
        print("  walk-strict  " + "  ".join(f"{b:>6}" for b in bands) + f"   kept   dir_err {lo3}-{hi3}")
        for nm in names:
            line = "  ".join(f"{100 * v[nm]['strict'][b]:6.2f}" for b in bands)
            print(f"  {nm:>10}  {line}   {v[nm]['kept']:5.2f}  {de38[nm]:.3f}")
        for nm in names[2:]:
            gl = "  ".join(f"{b} {G[nm][b]:+.2f}" + (f" [{Gci[nm][b][0]:+.2f},{Gci[nm][b][1]:+.2f}]" if Gci.get(nm, {}).get(b) else "")
                           for b in bands if G[nm][b] is not None)
            print(f"  G {nm:>10}: {gl}")
    # the authors' lines (NewIdeas 18:5x), per arm, on HE: PASS G(3-8) >= 0.25 with the CI's lower end > 0 AND the
    # direction error at depths 3-8 down >= 15 %; KILL G(3-8) < 0.10 or the error not down; BETWEEN otherwise
    vb, lb = ("3-8", "3-8") if a.rows == "sib" else ("3-30", "3-30")
    hs = res["sets"].get("he")
    if hs is not None and a.rows == "sib":
        res["verdict"] = {}
        for nm in names[2:]:
            g = hs["G"][nm][vb]; ci = (hs["G_ci95"].get(nm) or {}).get(vb)
            e0, e1 = hs[f"dir_err_{lb}"]["own"], hs[f"dir_err_{lb}"][nm]
            fall = (e0 - e1) / e0 if e0 else 0.0
            if g is not None and g >= 0.25 and ci and ci[0] > 0 and fall >= 0.15:
                vd = "PASS"
            elif g is None or g < 0.10 or fall <= 0:
                vd = "KILL"
            else:
                vd = "BETWEEN"
            res["verdict"][nm] = {"verdict": vd, "G": g, "G_ci95": ci, "dir_err_fall": fall}
            print(f"VERDICT {nm}: {vd} -- HE G(3-8) {g:+.3f} CI {ci}, direction error 3-8 {e0:.3f} -> {e1:.3f}"
                  f" ({100 * fall:+.1f} % fall)")
    res["wall_s"] = time.time() - t0; res["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[assim] DONE {a.out} wall {res['wall_s']:.0f}s peak {res['mlx_peak_gb']:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
