"""TrainLab COPY (MTP Head, Tue 17 Nov, S22 QATC2) of MoonShots' week98/p3_dd_tree.py (sha256 in data/s22/m3/SOURCES), changed only by: OUT = ~/TrainLab/data/s22/m3; --weights PATH (default dd_<run> as before); --tag T (appended to the dump / json names so runs with other weights do not collide); TOPK stays --run's (layout only: files, anchors, walk starts).
WEEK98 PILLAR 3, M3 (MoonShots, Mon 16 Nov 2026; LEVERS_A "PILLAR 3" M3; NewIdeas 08:1x: use BigMoE's dd_tree.py).
A COPY of ~/BigMoE/scripts/dd_tree.py's dump loop (BigMoE's / TrainLab's modules imported read-only; Forest and
tree_attn imported from dd_tree itself; outputs in ~/MoonShots/week98/p3/, never in BigMoE's folders) with hooks for
PILLAR 3's draft-only byte cuts (the C1-C8 ladder of p3_m012.py):
  - weights: t30b's fp32 trained Linears fake-quantized the way the engine would store them (bf16 -> affine g64 quantize
    -> dequantize): q / k / v / o, the shared expert and fc_hidden / fc_embedding at 4 bits, the router at 8 (C1 = the
    formats p3_m012's C1 reads); the head's bf16 hyper-connection mixes at 4; then q / o at 3 / 2 (C4 / C6), fc + the
    mixes at 2 (C7), k / v / the shared expert at 2 (C8);
  - routing: the routed experts' top-k at draft time (10 -> 4 / 2);
  - vocab: the screened head = FR-Spec's top-N ids (~/Leap/data/frspec_rank.npy[:N]) + the session's own tokens (the
    prompt + the committed answer up to the anchor, first-appearance order, cap 4096: Leap's BP-1 convention). Logits
    outside the set are masked to -inf, so the subset logits are the full head's at the same ids by construction. The
    screened KERNEL (Leap's ScreenedHead: gathered 4-bit rows) is checked against the full head bitwise on real states
    (each window's roots), and the share of nodes whose full-head top-1 falls outside the set (the pick change) is counted.
  - Q4 / H8: the 4-bit weights alone (full head, top-10) and the 8K head alone (fp32 weights), to split C1's cost.
  - Q8 (Mon 16 Nov 21:1x, NewIdeas' ask after MTP Head's S17): Q4 with every 4 -> 8 (all trained parts + the head's HC
    mixes at 8 bits, g64; router 8; full head, top-10): is post-hoc 8-bit free on the 125B?
  - H16 / H32 / Q8H32 (Mon 16 Nov 22:3x, NewIdeas): the screen at FR-Spec top-16K / top-32K + the session extras (as
    H8), fp32 weights; and Q8 with the 32K screen stacked.
  - B16H32 (Tue 17 Nov 09:5x, the orchestrator's r57 format decision: Q8 + 32K if Q8 holds its line, else bf16 + 32K):
    t30b's 11 trained tensors rounded to bf16 (fp32 -> bf16 -> fp32, no quantization), HC mixes as stored, the 32K
    screen + the session extras, top-10: the fallback config's price.
Gate: --cut none at --run t30b must reproduce BigMoE's tree_t30b_<set>.npz bitwise (every array): --gate compares.
Kept is TRUE-STATE (lab): teacher-forced, the 125B's true state at every anchor; score with NewIdeas' treek_final.py
(--dump-dir ~/MoonShots/week98/p3 --dump-fmt tree_t30b_{tag}_<cut>.npz --out <a MoonShots path>).
usage: ~/BigMoE/.venv/bin/python -I -B p3_dd_tree.py --set heplus24|mbppbank --cut none|Q4|Q8|H8|H16|H32|Q8H32|B16H32|C1..C8 [--part i/n]
       [--merge n] [--gate]     (MEM ~11-12 GB: the day's one big GPU job; share lock with MEM, after `peer ls`)"""
import argparse, json, os, sys, time
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/BigMoE/scripts"))                  # read-only imports (run with -I -B: no bytecode written there)
import dd_tree as DT                                        # noqa: E402  (Forest, tree_attn, NKS, AKS)

OUT = H("~/TrainLab/data/s22/m3")
REF = DT.OUT                                                # BigMoE's ceiling dir: TOPK + the reference dumps, read-only
RANK_F = H("~/Leap/data/frspec_rank.npy")
CUTS = {                                                    # qo / kv / se / fc / hc bits (router 8 always), top-k, head rows
    "none": None,
    "Q4": dict(qo=4, kv=4, se=4, fc=4, hc=4, top=10, nh=None),
    "Q8": dict(qo=8, kv=8, se=8, fc=8, hc=8, top=10, nh=None),
    "H16": dict(qo=None, kv=None, se=None, fc=None, hc=None, top=10, nh=16384),
    "H32": dict(qo=None, kv=None, se=None, fc=None, hc=None, top=10, nh=32768),
    "Q8H32": dict(qo=8, kv=8, se=8, fc=8, hc=8, top=10, nh=32768),
    "B16H32": dict(qo=None, kv=None, se=None, fc=None, hc=None, top=10, nh=32768, bf16=True),
    "H8": dict(qo=None, kv=None, se=None, fc=None, hc=None, top=10, nh=8192),
    "C1": dict(qo=4, kv=4, se=4, fc=4, hc=4, top=10, nh=8192),
    "C2": dict(qo=4, kv=4, se=4, fc=4, hc=4, top=4, nh=8192),
    "C3": dict(qo=4, kv=4, se=4, fc=4, hc=4, top=4, nh=4096),
    "C4": dict(qo=3, kv=4, se=4, fc=4, hc=4, top=4, nh=4096),
    "C5": dict(qo=3, kv=4, se=4, fc=4, hc=4, top=2, nh=4096),
    "C6": dict(qo=2, kv=4, se=4, fc=4, hc=4, top=2, nh=4096),
    "C7": dict(qo=2, kv=4, se=4, fc=2, hc=2, top=2, nh=4096),
    "C8": dict(qo=2, kv=2, se=2, fc=2, hc=2, top=2, nh=2048),
}
EXTRA_CAP = 4096


def fq(mx, w, bits, gs=64):
    """w as the engine would store it: bf16, affine-quantized at `bits` (group gs), dequantized back to w's dtype"""
    q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=gs, bits=bits)
    return mx.dequantize(q, s, b, group_size=gs, bits=bits).astype(w.dtype)


def apply_cut(mx, c, P, head):
    """fake-quantize the draft-only copy's weights per the cut (in place); returns the list of what changed"""
    done = []
    if c is None:
        return done
    if c.get("bf16"):                                       # B16H32: t30b's trained tensors rounded to bf16, no quantization
        names = ("q_proj", "o_proj", "k_proj", "v_proj", "se_gate", "se_up", "se_down", "se_gatew", "fc_hidden",
                 "fc_embedding", "router")
        for name in names:
            lin = getattr(P, name)
            assert lin.weight.dtype == mx.float32, f"P.{name} is not fp32"
            lin.weight = lin.weight.astype(mx.bfloat16).astype(mx.float32)
            done.append(f"P.{name} bf16")
        assert len(done) == 11
    for name, key in (("q_proj", "qo"), ("o_proj", "qo"), ("k_proj", "kv"), ("v_proj", "kv"), ("se_gate", "se"),
                      ("se_up", "se"), ("se_down", "se"), ("fc_hidden", "fc"), ("fc_embedding", "fc"), ("router", "rt")):
        bits = 8 if key == "rt" else c[key]
        if key == "rt" and c["qo"] is None:                 # H8: the weights stay fp32
            continue
        if bits:
            lin = getattr(P, name)
            lin.weight = fq(mx, lin.weight, bits)
            done.append(f"P.{name} {bits}-bit")
    if c["hc"]:
        L0 = head.layers[0]
        for gname, g in (("attn_hc", L0.attn_hyper_connection), ("mlp_hc", L0.mlp_hyper_connection),
                         ("mixer", head.hyper_connection_mixer)):
            for sub in ("input_mix_weight_down", "input_mix_weight_up"):
                lin = getattr(g, sub)
                assert not hasattr(lin, "scales"), f"{gname}.{sub} is not bf16"
                lin.weight = fq(mx, lin.weight, c["hc"])
                done.append(f"head.{gname}.{sub} {c['hc']}-bit")
    head.layers[0].mlp.top_k = int(c["top"])
    done.append(f"experts top-{c['top']}")
    mx.eval(P.parameters(), head.parameters())
    return done


def screen_masks(ids, anc, base_mask):
    """[B, V] bool: base ids + the session's own tokens through each anchor's committed position s + 1 (first-appearance
    order, at most EXTRA_CAP of them outside the base)"""
    V = base_mask.shape[0]
    first = {}
    for i, t in enumerate(ids.tolist()):
        if t not in first:
            first[t] = i
    order = sorted((i, t) for t, i in first.items() if not base_mask[t])[:EXTRA_CAP]   # extras in appearance order
    pos = np.array([i for i, _ in order], np.int64)
    tok = np.array([t for _, t in order], np.int64)
    M = np.broadcast_to(base_mask, (len(anc), V)).copy()
    for b, s in enumerate(anc):
        M[b, tok[pos <= s + 1]] = True
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["heplus24", "mbppbank"], required=True)
    ap.add_argument("--run", default="t30b"); ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--nodes", type=int, default=32); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cut", choices=list(CUTS), default="none")
    ap.add_argument("--part", default="", help="i/n: only windows [len*i/n, len*(i+1)/n)")
    ap.add_argument("--merge", type=int, default=0, help="n: concatenate the n parts in order")
    ap.add_argument("--weights", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--gate", action="store_true", help="compare this cut's dump with BigMoE's reference, array by array")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stem = f"tree_{a.run}_{a.set}_{a.cut}" + (f"_{a.tag}" if a.tag else "")
    if a.gate:
        mine, ref = np.load(os.path.join(OUT, f"{stem}.npz")), np.load(os.path.join(REF, f"tree_{a.run}_{a.set}.npz"))
        res = {k: bool(np.array_equal(mine[k], ref[k])) for k in sorted(set(ref.files) | set(mine.files))
               if k in ref.files and k in mine.files}
        miss = sorted(set(ref.files) ^ set(mine.files))
        ok = all(res.values()) and not miss
        print(f"[p3 gate] {a.set} {a.cut}: {'PASS (bitwise, every array)' if ok else 'FAIL'}; {json.dumps(res)}"
              + (f"; keys not in both: {miss}" if miss else ""), flush=True)
        return
    if a.merge:
        TK = np.load(os.path.join(REF, f"topk_{a.run}_{a.set}.npz"))
        Z = [dict(np.load(os.path.join(OUT, f"{stem}.part{i}of{a.merge}.npz"))) for i in range(a.merge)]
        a.nodes, a.k = int(Z[0]["nodes"]), int(Z[0]["k"])
        N_ = {k: np.concatenate([z[k] for z in Z]) for k in DT.NKS}
        A_ = {k: np.concatenate([z[k] for z in Z]) for k in DT.AKS}
        assert (np.diff(A_["a_fi"]) >= 0).all(), "parts out of order"
        X = {k: [x for z in Z for x in json.loads(str(z["extra"]))[k]] for k in ("screen_checks", "pick_out")}
        np.savez_compressed(os.path.join(OUT, f"{stem}.npz"), files=TK["files"], nodes=a.nodes, k=a.k, **N_, **A_)
        return report(a, N_, A_, TK, round(float(sum(z["wall_s"] for z in Z)), 1),
                      round(float(max(z["peak_gb"] for z in Z)), 2), X, json.loads(str(Z[0]["extra"]))["cut_done"])
    import mlx.core as mx
    import dd_ceiling as CE
    import dd_ttt as DD
    from dd_topk import topk_rows
    T, M = DD.T, DD.M
    t0 = time.time()
    rt_, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    P.load_weights(a.weights or os.path.join(DD.CKD, f"dd_{a.run}.safetensors")); mx.eval(P.parameters())
    c = CUTS[a.cut]
    cut_done = apply_cut(mx, c, P, head)
    print(f"[p3 tree] cut {a.cut}: {cut_done}", flush=True)
    base = None
    sh = None
    if c is not None and c["nh"]:
        RANK = np.load(RANK_F)
        base = np.zeros(lm.lm_head.weight.shape[0], bool)
        base[np.asarray(RANK[:c["nh"]], np.int64)] = True
        sys.path.insert(0, H("~/Leap/scripts"))
        from screened_head import ScreenedHead           # the kernel check only (read-only import)
        sh = ScreenedHead(lm.lm_head, RANK[:c["nh"]], extra_cap=EXTRA_CAP)
    NEGINF = mx.array(float("-inf"), mx.float32)
    emb, layer = lm.model.embed_tokens, head.layers[0]
    att = layer.self_attn
    FS = {"heplus24": CE.HEP, "mbppbank": CE.MBB}[a.set]
    TK = np.load(os.path.join(REF, f"topk_{a.run}_{a.set}.npz"))
    assert [str(x) for x in TK["files"]] == [os.path.basename(f) for f in FS], "TOPK's files differ"
    d0 = TK["depth"] == 0
    NK = ("fi", "an", "node", "parent", "depth", "token", "path_p", "y", "on_true", "top_ids", "top_p", "p_ynext", "ynext_rank")
    AKs = ("a_fi", "a_an", "a_walk_start", "root_top_ids", "root_top_p")
    N_, A_ = {k: [] for k in NK}, {k: [] for k in AKs}
    X = {"screen_checks": [], "pick_out": []}
    lo, hi = 0, len(FS)
    if a.part:
        i_, n_ = (int(x) for x in a.part.split("/")); lo, hi = len(FS) * i_ // n_, len(FS) * (i_ + 1) // n_
    for fi, f in enumerate(FS):
        if not lo <= fi < hi:
            continue
        while os.path.exists(T.PAUSE):
            time.sleep(1)
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32)
        S = len(ids); gen = int(z["gen"]); a0 = max(0, gen - 2)
        sel = d0 & (TK["fi"] == fi)
        anc = (TK["row"][sel] - 1).astype(np.int64)                  # the anchors of TOPK's depth-0 cells
        ws = TK["walk_start"][sel]
        B = len(anc)
        if B == 0:
            continue
        mask = mx.array(screen_masks(ids, anc, base)) if base is not None else None
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        idm = mx.array(ids)[None]
        n1 = S - 1                                                     # depth 1 = the true pairs, as dd_d22_ttt.chain
        mixed, hyper, inject = layer.attn_hyper_connection(T.fuse(head, P, W[:, :n1], emb(idm[:, 1:n1 + 1])).astype(mx.bfloat16))
        q, gate, K1, V1 = T.qkv(att, P, mixed, mx.arange(n1))
        o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K1, V1, scale=att.scale, mask="causal")
        n = n1 - a0
        blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
        h1 = T.finish(head, P, hyper[:, a0:], inject[:, a0:], blk)
        R = h1[:, mx.array(anc - a0)]                                  # [1,B,Wd] the roots' carried states
        lg = T.logits(lm, head, R)[0]
        if mask is not None:
            # the screened kernel vs the full head's subset logits on these real states (bitwise), and the pick change
            mixd = head.hyper_connection_mixer(R)
            sl = sh.logits(mixd)
            fl = mx.take(lm.lm_head(mixd).astype(mx.float32).reshape(-1, lg.shape[-1]), sh.ids, axis=1)
            X["screen_checks"].append(bool(mx.array_equal(sl, fl).item()))
            X["pick_out"].append([int((~np.array(mx.take_along_axis(mask, mx.argmax(lg, -1)[:, None], 1))).sum()), int(B)])
            lg = mx.where(mask, lg, NEGINF)
        rti, rtp, _, _, _ = topk_rows(mx, lg, ids[anc + 2], a.k)
        mx.eval(K1, V1, R); del mixed, hyper, inject, q, gate, o, blk, h1, lg
        A_["a_fi"].append(np.full(B, fi)); A_["a_an"].append(anc); A_["a_walk_start"].append(ws)
        A_["root_top_ids"].append(rti); A_["root_top_p"].append(rtp)
        F = DT.Forest(rti, rtp)
        HS, KS, VS = [R], [], []
        ontrue = np.zeros((B, a.nodes), bool)
        ar = np.arange(B)
        for t in range(a.nodes):
            par, tok, dep, pp, ent = F.pop()
            grp = [np.flatnonzero(par == q_) for q_ in np.unique(par)]          # the parents' carried states, by group
            cat = mx.concatenate([HS[int(par[g_[0]]) + 1][:, mx.array(g_)] for g_ in grp], axis=1)
            inv = np.empty(B, np.int64); inv[np.concatenate(grp)] = np.arange(B)
            HP = cat[:, mx.array(inv)]
            pos = anc + dep + 1                                                 # chain depth j = d + 2 -> position s + j - 1
            x = T.fuse(head, P, HP, emb(mx.array(tok)[None]))
            mixed, hyper, inject = layer.attn_hyper_connection(x.astype(mx.bfloat16))
            q, gate, k, v = T.qkv(att, P, mixed, mx.array(pos.astype(np.int32)))
            KS.append(k); VS.append(v)
            J = int(dep.max()) + 1
            aidx = np.empty((B, J), np.int32); am = np.zeros((B, J), bool)
            for b in range(B):
                e = ent[b]
                aidx[b, :len(e)] = [qq * B + b for qq in e]; aidx[b, len(e):] = t * B + b; am[b, :len(e)] = True
            ai = mx.array(aidx)
            kst, vst = mx.concatenate(KS, axis=2), mx.concatenate(VS, axis=2)
            o = DT.tree_attn(mx, att.scale, q, K1, V1, kst[:, :, ai], vst[:, :, ai], mx.array(am), mx.array(anc.astype(np.int32)))
            blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, B, -1) * mx.sigmoid(gate))
            h = T.finish(head, P, hyper, inject, blk)
            lg = T.logits(lm, head, h)[0]
            if mask is not None:
                X["pick_out"].append([int((~np.array(mx.take_along_axis(mask, mx.argmax(lg, -1)[:, None], 1))).sum()), int(B)])
                lg = mx.where(mask, lg, NEGINF)
            yp = anc + dep + 2                                                  # the node's position
            y = np.where(yp <= S - 1, ids[np.minimum(yp, S - 1)], -1)
            yn = np.where(yp + 1 <= S - 1, ids[np.minimum(yp + 1, S - 1)], 0)
            ti, tp, py, yr, _ = topk_rows(mx, lg, yn, a.k)
            mx.eval(h)
            HS.append(h)
            ptrue = np.where(par >= 0, ontrue[ar, np.maximum(par, 0)], True)
            ontrue[:, t] = ptrue & (tok == y)
            F.push(t, ti, tp)
            for k_, v_ in (("fi", np.full(B, fi)), ("an", anc), ("node", np.full(B, t)), ("parent", par), ("depth", dep),
                           ("token", tok), ("path_p", pp.astype(np.float32)), ("y", y), ("on_true", ontrue[:, t].copy()),
                           ("top_ids", ti), ("top_p", tp), ("p_ynext", py), ("ynext_rank", yr)):
                N_[k_].append(v_)
        del HS, KS, VS, K1, V1, R
        mx.clear_cache()
    N_ = {k: np.concatenate(v) for k, v in N_.items()}
    A_ = {k: np.concatenate(v) for k, v in A_.items()}
    extra = json.dumps(dict(X, cut_done=cut_done))
    if a.part:
        i_, n_ = a.part.split("/")
        np.savez_compressed(os.path.join(OUT, f"{stem}.part{i_}of{n_}.npz"), nodes=a.nodes, k=a.k, extra=extra,
                            wall_s=time.time() - t0, peak_gb=mx.get_peak_memory() / 1e9, **N_, **A_)
        print(f"[p3 tree] part {a.part}: windows {lo}..{hi - 1}, {len(A_['a_an'])} anchors, {time.time() - t0:.0f} s", flush=True)
        return
    np.savez_compressed(os.path.join(OUT, f"{stem}.npz"), files=TK["files"], nodes=a.nodes, k=a.k, **N_, **A_)
    report(a, N_, A_, TK, round(time.time() - t0, 1), round(mx.get_peak_memory() / 1e9, 2), X, cut_done)


def report(a, N_, A_, TK, wall, peak, X, cut_done):
    """dd_tree.report's checks and preview (same code path: it is called with a MoonShots json path), plus the cut's
    screened-head checks"""
    import contextlib, io
    buf = io.StringIO()
    old = DT.OUT
    DT.OUT = OUT                                            # dd_tree.report writes tree_<run>_<set>.json under DT.OUT
    run0 = a.run
    a.run = f"{run0}"                                       # its json name: tree_<run>_<set>.json -> renamed below
    try:
        with contextlib.redirect_stdout(buf):
            DT.report(a, N_, A_, TK, wall, peak)
    finally:
        DT.OUT = old
    src = os.path.join(OUT, f"tree_{run0}_{a.set}.json")
    Rj = json.load(open(src))
    os.remove(src)
    Rj["cut"] = a.cut
    Rj["cut_done"] = cut_done
    if X["screen_checks"]:
        Rj["screen_kernel_bitwise_windows"] = f"{sum(X['screen_checks'])} / {len(X['screen_checks'])}"
        po = np.array(X["pick_out"]) if X["pick_out"] else np.zeros((0, 2))
        Rj["full_top1_outside_screened_set"] = round(float(po[:, 0].sum() / max(po[:, 1].sum(), 1)), 6)
    json.dump(Rj, open(os.path.join(OUT, f"tree_{run0}_{a.set}_{a.cut}" + (f"_{a.tag}" if a.tag else "") + ".json"), "w"), indent=1)
    print("[p3 tree]", json.dumps(Rj), flush=True)


if __name__ == "__main__":
    main()
