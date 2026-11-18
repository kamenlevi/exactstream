"""PK v2: the persistent level (pk_level.PK) vs ra_sf.draft_sf on real windows: tokens, parents, depth, pop order and
path_p bits per tree; ms per pop for both (MoonShots, Wed 18 Nov 2026).
usage: ~/BigMoE/.venv/bin/python -I -B pk_tree_test.py [--set heplus24] [--windows 2,3] [--anchors 6]"""
import argparse
import os
import sys
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98/draft"))
sys.path.insert(0, H("~/BigMoE/scripts"))
import mlx.core as mx  # noqa: E402
import ra_sf as RA  # noqa: E402
import r57_draft as R  # noqa: E402
import pk_level as PKL  # noqa: E402
SYNC_EACH = os.environ.get("PK_SYNC_EACH") == "1"
import dd_ceiling as CE  # noqa: E402
import dd_ttt as DD  # noqa: E402
import dd_tree as DT  # noqa: E402


def diag(pk, lm, view, root, sh, imap, n=15, k=8):
    """per pop: PK's hidden / logits / tail vs draft_sf's recorded forwards (the first difference wins)"""
    rec = []
    RA.SCREENED["on"] = True
    try:
        osf = RA.draft_sf(lm, view, root, n=n, k=k, imap=imap, rec=rec)
    finally:
        RA.SCREENED["on"] = False
    o = pk.L.defs
    qc = view.cache[0]
    Lc = int(qc.offset)
    kb, vb = qc.kv.keys, qc.kv.values
    cap_kv = int(kb.shape[2])
    im32 = imap.astype(mx.int32)
    meta0 = mx.array([0, 0, Lc, n, k, cap_kv * 256, int(sh.n_extra), 0], mx.int32)
    dep = pk.ksetup(inputs=[view.logits[0, -1].astype(mx.float32), view.h.reshape(-1), im32, pk.scr, meta0],
                    grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.int32])[0]
    mx.eval(dep)
    for t in range(n - 1):
        meta = mx.array([t, pk.base, Lc, n, k, cap_kv * 256, int(sh.n_extra), 0], mx.int32)
        dep = pk.kmain(inputs=[pk.dw, pk.ds, pk.db, pk.sm, pk.fs] + pk.ex + [sh.w, sh.s, sh.b, sh.xw, sh.xs, sh.xb] + pk.emb +
                       [kb.reshape(-1), vb.reshape(-1), im32, pk.scr, pk.sync, meta, dep], grid=(PKL.NTG * PKL.NT, 1, 1),
                       threadgroup=(PKL.NT, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.int32])[0]
        pk.base += PKL.NBAR
        mx.eval(dep); mx.synchronize()
        SB, SFv = pk.scr.view(mx.bfloat16), pk.scr.view(mx.float32)
        hpk = SB[o["HS"][1] + (t + 1) * 10240:o["HS"][1] + (t + 2) * 10240]
        lgpk = SFv[o["lg"][1]:o["lg"][1] + lgref_n(rec[t][0])]
        lg_s, h_s = rec[t][0].reshape(-1).astype(mx.float32), rec[t][1].reshape(-1)
        eh = bool(mx.array_equal(hpk, h_s).item()); el = bool(mx.array_equal(lgpk, lg_s).item())
        tis_r, tv_r, tp_r = RA.topk_ra(lg_s, k)
        lse_r = mx.logsumexp(lg_s)
        nd = int((lgpk != lg_s).sum().item()) if not el else 0
        if not (eh and el):
            stages(pk, lm, view, rec, t, Lc, kb, vb, n)
            bad = np.nonzero(np.array(lgpk != lg_s))[0][:6].tolist() if not el else []
            print(f"    diag t {t}: h {'=' if eh else 'DIFF'} lg {'=' if el else str(nd) + ' DIFF at ' + str(bad)} "
                  f"(pk {[float(lgpk[i].item()) for i in bad[:3]]} sf {[float(lg_s[i].item()) for i in bad[:3]]})", flush=True)
            return
    print("    diag: every pop's h / lg bitwise; the difference is in the tail", flush=True)


def stages(pk, lm, view, rec, t, Lc, kb, vb, n):
    """node t's forward_static stage by stage (the stock ops) vs PK's scratch (the last launch = node t)"""
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_math import attention_gate
    o = pk.L.defs
    S32 = np.array(pk.scr.view(mx.int32))
    SB = pk.scr.view(mx.bfloat16)
    pops = S32[o["pops"][1]:o["pops"][1] + 4 * n].reshape(n, 4)
    par, tok, d0 = int(pops[t, 0]), int(pops[t, 1]), int(pops[t, 2])
    nanc = S32[o["nanc"][1]:o["nanc"][1] + n * n].reshape(n, n)
    anc = [int(a) for a in nanc[t, :d0] if a >= 0]
    head = lm.mtp; L0 = head.layers[0]; att = L0.self_attn
    hp = view.h if par < 0 else rec[par][1]
    x = head._mtp_prepare_inputs_impl(hp, lm.model.embed_tokens(mx.array([[tok]])))
    mixed, hyper, inject = L0.attn_hyper_connection(x)
    qf, kf, vf = att.q_proj(mixed), att.k_proj(mixed), att.v_proj(mixed)
    q, gate = mx.split(qf.reshape(1, 1, 24, -1), 2, axis=-1); gate = gate.reshape(1, 1, -1)
    k, v = kf.reshape(1, 1, 2, -1), vf.reshape(1, 1, 2, -1)
    q, k = att.q_norm(q), att.k_norm(k)
    q, k = RA._rope_qk(att, q, k, mx.array([Lc + d0], mx.int32))
    q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
    Kc, Vc = kb[..., :Lc, :], vb[..., :Lc, :]
    k, v = k.astype(Kc.dtype), v.astype(Vc.dtype)
    if t:
        kd = SB[o["kd"][1]:o["kd"][1] + t * 512].reshape(t, 2, 256).transpose(1, 0, 2)[None]
        vd = SB[o["vd"][1]:o["vd"][1] + t * 512].reshape(t, 2, 256).transpose(1, 0, 2)[None]
        am = mx.array([j in anc for j in range(t)])
        mask = mx.concatenate([mx.ones((Lc,), mx.bool_), am, mx.ones((1,), mx.bool_)]).reshape(1, 1, 1, -1)
        Ka, Va = mx.concatenate([Kc, kd, k], axis=2), mx.concatenate([Vc, vd, v], axis=2)
    else:
        mask = None
        Ka, Va = mx.concatenate([Kc, k], axis=2), mx.concatenate([Vc, v], axis=2)
    out = Q._verify_sdpa(q, Ka, Va, scale=att.scale, mask=mask)
    gated = attention_gate(out.transpose(0, 2, 1, 3).reshape(1, 1, -1), gate)
    bo1 = Q._linear(att.o_proj, gated)
    mixed2, hyper2, inj2 = L0.mlp_hyper_connection(hyper, pending=(bo1, inject))
    bo2 = L0.mlp(mixed2)
    h = Q._hyper_residual_write(hyper2, bo2, inj2)
    print(f"    node t {t}: par {par} tok {tok} d0 {d0} anc {anc} Lc {Lc}", flush=True)
    for nm, ref, off, size in (("fuse x", x, "x1", 10240), ("mixed1", mixed, "mixed1", 2560), ("inj1", inject, "inj1", 4),
                               ("qf", qf, "qf", 12288), ("kf", kf, "kf", 512), ("vf", vf, "vf", 512),
                               ("gated", gated, "gated", 6144), ("bo1", bo1, "bo1", 2560), ("hw2", hyper2, "hw2", 10240),
                               ("mixed2", mixed2, "mixed2", 2560), ("inj2", inj2, "inj2", 4), ("bo2", bo2, "bo2", 2560)):
        mine = SB[o[off][1]:o[off][1] + size]
        r = ref.reshape(-1)
        eq = bool(mx.array_equal(mine, r).item())
        print(f"      {nm}: {'=' if eq else str(int((mine != r).sum().item())) + ' differ'}", flush=True)
        if not eq:
            break
    hk = SB[o["HS"][1] + (t + 1) * 10240:o["HS"][1] + (t + 2) * 10240]
    print(f"      h: {'=' if bool(mx.array_equal(hk, h.reshape(-1)).item()) else 'differs'}", flush=True)


def lgref_n(lg):
    return int(lg.reshape(-1).shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="heplus24")
    ap.add_argument("--windows", default="2,3")
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    rt, lm, head = DD.M.load(0.03)
    R.install(lm, "q8h32")
    pk = PKL.PK(lm, n=a.n, k=a.k)
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    TK = np.load(os.path.join(DT.OUT, f"topk_t30b_{a.set}.npz"))
    res = dict(trees=0, same=0, fallback=0, ms_sf=[], ms_pk=[])
    for fi in [int(x) for x in a.windows.split(",")]:
        z = np.load(FS[fi])
        ids = np.asarray(z["ids"], np.int32)
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        sel = (TK["depth"] == 0) & (TK["fi"] == fi)
        ancs = np.sort((TK["row"][sel] - 1).astype(np.int64))
        pick = ancs[::max(1, len(ancs) // a.anchors)][:a.anchors]
        R.begin_window(lm, ids[:int(z["gen"])])
        mtp = RA.Head(lm)
        done_upto = -1
        for s in [int(x) for x in pick]:
            R.observe(lm, ids[:s + 2])
            mtp.append(W[:, done_upto + 1:s + 1], ids[done_upto + 2:s + 2])
            done_upto = s
            mx.eval(mtp.logits, mtp.h)
            root = int(ids[s + 1])
            sh = R.ST["sh"]
            imap = RA.idmap(sh)
            rowv = R._screened_row(mtp.logits[0, -1], sh)
            view = R._Root(mtp.cache, rowv[None, None, :], mtp.h)
            RA.SCREENED["on"] = True
            try:
                for rep in range(2):
                    mx.synchronize(); t0 = time.perf_counter()
                    osf = RA.draft_sf(lm, view, root, n=a.n, k=a.k, imap=imap)
                    t1 = time.perf_counter()
                    opk = pk.draft(view, root, sh, imap, sync_each=SYNC_EACH)
                    t2 = time.perf_counter()
            finally:
                RA.SCREENED["on"] = False
            res["trees"] += 1
            if opk is None:
                res["fallback"] += 1
                print(f"  w{fi} s{s}: PK fell back", flush=True)
                continue
            same = all(np.array_equal(osf[k_], opk[k_]) for k_ in ("tokens", "parents", "depth", "pop", "path_p", "p"))
            res["same"] += int(same)
            if not same:
                diag(pk, lm, view, root, sh, imap, n=a.n, k=a.k)
            res["ms_sf"].append((t1 - t0) * 1e3 / a.n); res["ms_pk"].append((t2 - t1) * 1e3 / a.n)
            print(f"  w{fi} s{s} Lc {int(mtp.cache[0].offset)}: {'IDENTICAL' if same else 'DIFFERENT'} | ms per pop SF "
                  f"{res['ms_sf'][-1]:.3f} PK {res['ms_pk'][-1]:.3f}"
                  + ("" if same else "\n    fields differ: " + str([k_ for k_ in ("tokens", "parents", "depth", "pop", "path_p", "p")
                                                                   if not np.array_equal(osf[k_], opk[k_])])
                     + f"\n    SF pop {osf['pop'].tolist()} par {osf['parents'].tolist()}\n    PK pop {opk['pop'].tolist()} par {opk['parents'].tolist()}"
                     + f"\n    SF path_p {np.array(osf['path_p']).tolist()}\n    PK path_p {np.array(opk['path_p']).tolist()}"), flush=True)
    print(f"[pk_tree_test] trees {res['trees']}, identical {res['same']}, fallback {res['fallback']}; ms per pop median SF "
          f"{np.median(res['ms_sf']) if res['ms_sf'] else 0:.3f} PK {np.median(res['ms_pk']) if res['ms_pk'] else 0:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
