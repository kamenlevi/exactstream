"""MTPSCREEN (MTP Head, Wed 11-04 15:4x; Leap's question on the orchestrator's 300 route; pre-registered in REPORT 'MTPSCREEN'):
does the MTP-assisted pick keep its kept per check when the MTP step's head is screened to a token subset? (Leap 30e92a6:
the 4-bit lm_head is ~1.7 of the ~2.2 ms step; a 32K screen 0.24 ms, ~2K rows 0.02 ms.)
Chain: d24_mtptop3.py's (d22_ttt's teacher-forced b5 chain over the 24 held-out HumanEval+ windows, data/d16w). At the TOP16
walk's HPD-0 anchors A (data/d02b_walk_kw_code.npz; MTP anchor s = A - 1, depth j predicts ids[A + j], DFlash2's depth j) the
MTP's top-1 token and p under four heads:
  full  the whole vocabulary (E1 COVERAGE's pick: code DFlash2 9.54 -> pick 11.70, oracle 12.73);
  a     Leap's FR-Spec top-32K (~/Leap/data/frspec_rank.npy[:32768]) + the session tokens (prompt + answer before the input);
  b     DFlash2's 16 candidates at (A, j) + the session tokens;   b0  the 16 candidates alone.
p = exp(top-1 logit - logsumexp over the head's set). The pick (E1's rule): at each depth the token of the more confident
drafter (MTP p vs DFlash2's teacher-forced selector max); kept = 1 + the leading run of right picks. Also: the pick with a
shifted MTP threshold (MTP wins when p > cf + delta; delta swept here, so optimistic), MTP alone, DFlash2 alone, the union.
Output: data/mtpscreen.txt (appended), data/mtpscreen_code.npz.   Usage: [BigMoE venv] python sticky/mtpscreen.py [--run b5]"""
import os, sys, time, argparse, resource
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/TrainLab_sticky"))
import d22_ttt as T                                          # its imports set MTPLX's env, then mlx
import numpy as np
import mlx.core as mx

D = 15
FR = np.unique(np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))[:32768].astype(np.int64))
HEADS = ("full", "a", "b", "b0")
DELTAS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def sub_top(lg_row, idx):
    """top-1 token and p over the subset idx (int array) of one row's logits"""
    v = mx.take(lg_row, mx.array(idx.astype(np.int32)))
    return mx.argmax(v), mx.exp(v.max() - mx.logsumexp(v))


def main():
    T.M.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b5"); ap.add_argument("--frac", type=float, default=0.03)
    a = ap.parse_args()
    rt, lm, head, P = T.load_head(a)
    Z = np.load(os.path.join(T.TD, "d02b_walk_kw_code.npz"))
    names = [str(x)[:-4] if str(x).endswith(".npz") else str(x) for x in Z["names"]]
    t0 = time.time()
    say(f"MTPSCREEN {time.strftime('%m-%d %H:%M')}: the MTP-assisted pick (head {a.run}, teacher-forced) with the MTP's head over "
        f"full / a (FR-Spec top-32K + session) / b (DFlash2's 16 candidates + session) / b0 (the 16 alone), at the TOP16 walk's "
        f"HPD-0 anchors ({len(names)} code windows)")
    REC = {}                                                 # (w, c) -> per depth j: [true, dtok, dcf, (tok, p) per head]
    for w, key in enumerate(names):
        x = T.window(key)
        ids = x["ids"]; S = len(ids)
        gen = T.gen_of(key); a0 = max(0, gen - 2)
        ck, cand, tsm = Z[f"{w}_checks"], Z[f"{w}_cand"], Z[f"{w}_tsm"].astype(np.float32)
        zi = Z[f"{w}_ids"]
        m_ = min(len(zi), S)
        assert int(Z[f"{w}_off"]) == 0 and (zi[:m_] == ids[:m_]).all(), key
        rows = {int(A) - 1 - a0: (c, int(A), int(m)) for c, (k, m, A) in enumerate(ck) if int(A) - 1 - a0 >= 0}

        def on_depth(j, h):
            n = min(h.shape[1], S - 1 - j - a0)
            sel = [(r, c, A) for r, (c, A, m) in rows.items() if r < n and A + j < S and j <= min(D, m)]
            if not sel:
                return
            lg = T.logits(lm, head, h[:, mx.array([r for r, _, _ in sel])])[0]     # only the anchors' rows
            full_t, full_p = mx.argmax(lg, axis=-1), mx.exp(lg.max(axis=-1) - mx.logsumexp(lg, axis=-1))
            outs = []
            for i, (r, c, A) in enumerate(sel):
                sess = np.unique(ids[:A + j].astype(np.int64))       # the prompt + answer before depth j's prediction
                cd = cand[c, j - 1].astype(np.int64)
                sets = {"a": np.union1d(FR, sess), "b": np.union1d(cd, sess), "b0": np.unique(cd)}
                res = {}
                for hname, idx in sets.items():
                    ti, p = sub_top(lg[i], idx)
                    res[hname] = (idx, ti, p)
                outs.append(res)
            mx.eval(full_t, full_p, [v for res in outs for (_, ti, p) in res.values() for v in (ti, p)])
            ft, fp = np.array(full_t), np.array(full_p)
            for i, (r, c, A) in enumerate(sel):
                o = np.argsort(-tsm[c, j - 1], kind="stable")
                rec = [int(ids[A + j]), int(cand[c, j - 1, o[0]]), float(tsm[c, j - 1, o[0]]), (int(ft[i]), float(fp[i]))]
                for hname in ("a", "b", "b0"):
                    idx, ti, p = outs[i][hname]
                    rec.append((int(idx[int(ti.item())]), float(p.item())))
                REC.setdefault((w, c), {})[j] = rec

        T.chain(lm, head, P, x["W"], mx.array(ids)[None], D, on_depth, a0)
        if w in (0, len(names) - 1):
            say(f"  window {w + 1}/{len(names)} {key}: {len(rows)} anchors, {time.time() - t0:.0f} s")

    def kept(rule):                                          # rule(rec) -> picked token or None (no opinion: stop)
        ks = []
        for (w, c), R in REC.items():
            run = 0
            for j in range(1, D + 1):
                if j not in R:
                    break
                if rule(R[j]) != R[j][0]:
                    break
                run += 1
            ks.append(run + 1)
        return float(np.mean(ks))

    hi = {h: 3 + i for i, h in enumerate(HEADS)}
    base = kept(lambda r: r[1])
    say(f"  {len(REC)} checks; DFlash2 alone {base:.2f} kept per check (the walk: 9.57; E1: 9.54); MTP alone (full) "
        f"{kept(lambda r: r[3][0]):.2f} (E1 10.13); union oracle {kept(lambda r: r[0] if r[0] in (r[1], r[3][0]) else -1):.2f} "
        f"(E1 12.73)")
    for h in HEADS:
        k0 = kept(lambda r, h=h: r[hi[h]][0] if r[hi[h]][1] > r[2] else r[1])
        sw = {d: kept(lambda r, h=h, d=d: r[hi[h]][0] if r[hi[h]][1] > r[2] + d else r[1]) for d in DELTAS}
        bd = max(sw, key=sw.get)
        un = kept(lambda r, h=h: r[0] if r[0] in (r[1], r[hi[h]][0]) else -1)
        say(f"    head {h:4s}: pick {k0:.2f} (x{k0 / base:.3f}); best threshold delta {bd:+.1f}: {sw[bd]:.2f} (x{sw[bd] / base:.3f}); "
            f"union oracle {un:.2f}; MTP top-1 = full's on {100 * np.mean([r[hi[h]][0] == r[3][0] for R in REC.values() for r in R.values()]):.1f} % of depths")
    say(f"  max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB, {time.time() - t0:.0f} s")
    keys = sorted(REC)
    arr = np.full((len(keys), D, 11), np.nan)                # per check, depth: true, dtok, dcf, then (tok, p) per head
    for i, k in enumerate(keys):
        for j, r in REC[k].items():
            arr[i, j - 1, :3] = r[:3]
            arr[i, j - 1, 3:] = [v for t in r[3:] for v in t]
    np.savez(os.path.join(T.TD, "mtpscreen_code.npz"), wc=np.array(keys, np.int32), rec=arr, heads=np.array(HEADS))
    open(os.path.join(T.TD, "mtpscreen.txt"), "a").write("\n".join(lines) + "\n\n")
    if os.environ.get("MTPS_TIME", "1") == "1":         # Leap's ask: the 1-row step, full head vs screened vs no head
        lines.clear()
        try:
            time_steps(rt, lm, T.window(names[0]))
        except Exception as e:                           # best effort: the replay above is already saved
            say(f"  step timing failed: {type(e).__name__}: {e}")
        open(os.path.join(T.TD, "mtpscreen.txt"), "a").write("\n".join(lines) + "\n\n")


def time_steps(rt, lm, x, n=400, reps=40, warm=8):
    """the engine's chain step (r57_screen.py MTPHead.draft_cut: mtp_forward at 1 row, argmax + softmax max, one sync) on a
    400-pair MTP cache, with lm.lm_head as loaded, as Leap's ScreenedHead (32K + 0 session rows: its .logits), and as a
    16-column stand-in (the layer + mixer alone)"""
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
    from screened_head import ScreenedHead
    model = rt.model if hasattr(rt.model, "mtp_forward") else lm
    W, ids = x["W"], x["ids"]
    n = min(n, W.shape[1] - 2)
    orig = lm.lm_head
    sh = ScreenedHead(orig, np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))[:32768], extra_cap=4096)
    out = {}
    for name, hd in (("full", orig), ("screened32K", sh.logits), ("stand-in", lambda z: z[..., :16])):
        lm.lm_head = hd
        mc = model.make_mtp_cache()
        model.mtp_update_cache(W[:, :n - 1], mx.array([ids[1:n].tolist()]), mtp_cache=mc)
        lg, h = model.mtp_forward(W[:, n - 1:n], mx.array([[int(ids[n])]]), mtp_cache=mc, return_hidden=True)
        mx.eval(lg, h)
        tok, ts = int(ids[n + 1]), []
        for i in range(warm + reps):
            t0 = time.perf_counter()
            lgx, h = model.mtp_forward(h, mx.array([[tok]]), mtp_cache=mc, return_hidden=True)
            lx = lgx[0, -1].astype(mx.float32)
            tk, p = mx.argmax(lx), mx.softmax(lx).max()
            mx.eval(tk, p)
            if i >= warm:
                ts.append(time.perf_counter() - t0)
        out[name] = 1e3 * float(np.median(ts))
    lm.lm_head = orig
    say(f"  MTP step at 1 row (mtp_forward + argmax / softmax max + one sync, {reps} steps on a {n}-pair cache, median): "
        f"full head {out['full']:.3f} ms, screened 32K {out['screened32K']:.3f} ms, 16-column stand-in (layer + mixer) "
        f"{out['stand-in']:.3f} ms")


if __name__ == "__main__":
    main()
