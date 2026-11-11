"""WALL 1105 probe (Fable C, Thu 5 Nov 2026): real expert OUTPUTS on BigMoE's cap45 checks (176 real 16-row DFlash2 checks,
code + math, MoE inputs per layer), numpy on the CPU, no GPU. Experts are dequantized here from the 4-bit MLX safetensors
(affine, group 64, fp32: w = scale * q + bias); the router's dequant is checked against cap45's routers.npz and its top-10
against the recorded ids. One layer at a time; ~1 min per layer.
What it measures per layer (all errors are relative to the row's full routed output sum_r w_r E_r(x), weights = the
router's softmax renormalised over the recorded top-10; the shared expert is not included):
  A  drop-k curve (ranks >= k removed, no renorm), norms by rank, cosine between a row's own experts' outputs.
  B  thin tail tiers: ranks >= 2 computed with only m of the 640 hidden units (static = units chosen per expert on the even
     checks' uses, scored on the odd checks; dynamic = the m largest |h_j| * ||D_j|| per input, an upper bound).
  C  substitution in a live-like block (rows 0 .. kept): each tail singleton (rank >= 2, used by no other row of the block)
     replaced by a block member's output on the same row: the oracle-best member by cosine with an optimal scale, and the
     member with the highest router score; vs simply dropping the singletons.
  D  tail surrogates: the previous row's tail; a ridge map x -> tail (leave-one-prompt-out, dual form).
Usage (BigMoE venv): nice -n 15 python -I wall_probe.py [layers...] -> wall_probe.out (appends)."""
import json, os, struct, sys, time
import numpy as np

MD = os.path.expanduser("~/BigMoE/models/bare"); CAP = os.path.expanduser("~/BigMoE/data/cap45")
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
LAYERS = [int(a) for a in sys.argv[1:]] or [12, 24, 36]
MS = (80, 160, 320)
RULES = ("dyn", "gate", "sk64", "sk128")
OUTF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wall_probe2.out")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
_st = {}
DT = {"U32": "<u4", "BF16": "<u2", "F16": "<f2", "F32": "<f4"}


def say(s):
    print(s, flush=True); open(OUTF, "a").write(s + "\n")


def tensor(name):
    f = idx[name]
    if f not in _st:
        p = os.path.join(MD, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        _st[f] = (hdr, 8 + n, np.memmap(p, dtype=np.uint8, mode="r"))
    hdr, base, mm = _st[f]; h = hdr[name]; b, e = h["data_offsets"]; dt = np.dtype(DT[h["dtype"]])
    return np.frombuffer(mm, dtype=dt, count=(e - b) // dt.itemsize, offset=base + b).reshape(h["shape"])


def bf(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def deq(w, s, b, bits=4):
    n = 32 // bits; mask = np.uint32((1 << bits) - 1); w = np.ascontiguousarray(w)
    q = np.empty(w.shape + (n,), np.float32)
    for i in range(n): q[..., i] = (w >> np.uint32(bits * i)) & mask
    q = q.reshape(w.shape[:-1] + (-1, 64))
    return (q * bf(s)[..., None] + bf(b)[..., None]).reshape(w.shape[:-1] + (-1,))


def load_checks():
    P = []
    for name in ORDER:
        z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
        seq, L, rows, roff, ids, xoff = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"], z["x_off"]
        checks, order = {}, []
        for n, (s, l, r) in enumerate(zip(seq, L, rows)):
            s, l, r = int(s), int(l), int(r)
            c = checks.setdefault(s, {"rows": {}, "x": {}, "kept": None})
            if s not in order: order.append(s)
            if l in c["rows"]: c["kept"] = r
            else: c["rows"][l] = ids[roff[n]:roff[n] + r].astype(np.int64); c["x"][l] = (int(xoff[n]), r)
        cks = [checks[s] for s in order if len(checks[s]["rows"]) == 48 and checks[s]["kept"] is not None]
        P.append((name, cks, z["x"]))
    return P


P = load_checks()
say(f"==== wall_probe2 {time.strftime('%H:%M')} layers {LAYERS}; checks per prompt {[len(c) for _, c, _ in P]}; kept mean {np.mean([c['kept'] for _, cks, _ in P for c in cks]):.2f}")
for L in LAYERS:
    t0 = time.time(); pre = f"language_model.model.layers.{L}.mlp."
    X, IDS, CK, PJ, PR = [], [], [], [], []; kept = []; ci = 0
    for p, (name, cks, xs) in enumerate(P):
        for c in cks:
            o, r = c["x"][L]
            if r != 16: continue
            X.append(bf(xs[o:o + 16])); IDS.append(c["rows"][L]); CK += [ci] * 16; PJ += list(range(16)); PR += [p] * 16
            kept.append(c["kept"]); ci += 1
    X = np.concatenate(X).astype(np.float32); IDS = np.concatenate(IDS); CK = np.array(CK); PJ = np.array(PJ); PR = np.array(PR)
    kept = np.array(kept); N = len(X); NC = ci
    R = deq(tensor(pre + "gate.weight"), tensor(pre + "gate.scales"), tensor(pre + "gate.biases"), 8)        # [512, 2560]
    try:
        rz = np.load(os.path.join(CAP, "routers.npz"))[f"w{L}"].astype(np.float32)
        rd = float(np.abs(R - rz).max()); rr = float(np.abs(rz).mean())
    except Exception as ex: rd, rr = float("nan"), float("nan")
    lg = X @ R.T
    top = np.argsort(-lg, axis=1)[:, :10]
    match = np.mean([len(set(top[i]) & set(IDS[i])) / 10 for i in range(N)])
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    w10 = np.take_along_axis(pr, IDS, axis=1); mass10 = float(w10.sum(1).mean()); wn = w10 / w10.sum(1, keepdims=True)
    say(f"L{L}: rows {N} ({NC} checks); router dequant max|diff| vs routers.npz {rd:.2e} (mean |w| {rr:.3f}); my top-10 = recorded ids {100 * match:.1f} %; top-10 softmax mass {mass10:.3f}; weights by rank {np.round(wn.mean(0), 3).tolist()}")
    # blocks (live-like): rows 0 .. kept of each check; member rows for the substitution test on every 2nd check
    use = [[] for _ in range(512)]
    for i in range(N):
        for sl in range(10): use[IDS[i, sl]].append((i, sl))
    mem = [[] for _ in range(512)]; blocks = []
    for c in range(0, NC, 2):
        rows = np.nonzero((CK == c) & (PJ <= min(kept[c], 15)))[0]
        U = np.unique(IDS[rows]); blocks.append((c, rows, U))
        for e in U:
            for r in rows:
                if e not in IDS[r]: mem[e].append(r)
    nmem = sum(len(m) for m in mem); MemOut = np.zeros((nmem, 2560), np.float16); memidx = {}; mi = 0
    C = np.zeros((N, 10, 2560), np.float32)
    evalrow = (CK % 2 == 1)
    Rst = {m: np.zeros((N, 2560), np.float32) for m in MS}
    RR = {(ru, m): np.zeros((N, 2560), np.float32) for ru in RULES for m in MS}; Rall = {m: np.zeros((N, 2560), np.float32) for m in MS}
    en_tail = 0.0; en_st = {m: 0.0 for m in MS}; EN = {(ru, m): 0.0 for ru in RULES for m in MS}
    Sel = {m: np.zeros((N, 10, 640), bool) for m in MS}
    cal_rows = ~evalrow; ev_, evec = np.linalg.eigh((X[cal_rows].T @ X[cal_rows]).astype(np.float64)); Pk = evec[:, ::-1][:, :128].astype(np.float32); Z = X @ Pk
    xen = float((Z[evalrow] ** 2).sum() / (X[evalrow] ** 2).sum()); xen64 = float((Z[evalrow][:, :64] ** 2).sum() / (X[evalrow] ** 2).sum())
    sw = pre + "switch_mlp."
    T = {k: tensor(sw + k) for k in ("gate_proj.weight", "gate_proj.scales", "gate_proj.biases", "up_proj.weight", "up_proj.scales",
                                     "up_proj.biases", "down_proj.weight", "down_proj.scales", "down_proj.biases")}
    for e in range(512):
        nu = len(use[e]); nm = len(mem[e])
        if nu + nm == 0: continue
        G = deq(T["gate_proj.weight"][e], T["gate_proj.scales"][e], T["gate_proj.biases"][e])
        Uu = deq(T["up_proj.weight"][e], T["up_proj.scales"][e], T["up_proj.biases"][e])
        D = deq(T["down_proj.weight"][e], T["down_proj.scales"][e], T["down_proj.biases"][e])
        ur = np.array([a for a, _ in use[e]], np.int64); us = np.array([b for _, b in use[e]], np.int64)
        rows_all = np.concatenate([ur, np.array(mem[e], np.int64)])
        Xr = X[rows_all]; g = Xr @ G.T; u = Xr @ Uu.T; h = g / (1.0 + np.exp(-g)) * u; out = h @ D.T
        if nu: C[ur, us] = out[:nu]
        for k, r in enumerate(mem[e]):
            MemOut[mi] = out[nu + k]; memidx[(r, e)] = mi; mi += 1
        if nu:
            dn = np.linalg.norm(D, axis=0); un = np.linalg.norm(Uu, axis=1); hf = h[:nu]; gf = g[:nu]; of = out[:nu]
            a = np.abs(hf) * dn
            sc = {"dyn": a, "gate": np.abs(gf / (1.0 + np.exp(-gf))) * dn * un}
            z = Z[ur]; GP = G @ Pk; UP = Uu @ Pk
            for r_ in (64, 128):
                gt = z[:, :r_] @ GP[:, :r_].T; ut = z[:, :r_] @ UP[:, :r_].T
                sc[f"sk{r_}"] = np.abs(gt / (1.0 + np.exp(-gt)) * ut) * dn
            cal = ~evalrow[ur]; ev = evalrow[ur] & (us >= 2); eva = evalrow[ur]
            imp = (a[cal] ** 2).mean(0) if cal.any() else dn ** 2; order_s = np.argsort(-imp)
            for m in MS:
                kth = np.partition(a, 640 - m, axis=1)[:, 640 - m][:, None]; Sel[m][ur, us] = a >= kth
            if eva.any():
                wa = wn[ur[eva], us[eva]][:, None]
                for m in MS:
                    kth = np.partition(a[eva], 640 - m, axis=1)[:, 640 - m][:, None]
                    np.add.at(Rall[m], ur[eva], wa * (of[eva] - (hf[eva] * (a[eva] >= kth)) @ D.T))
            if ev.any():
                he = hf[ev]; oe = of[ev]; rw = ur[ev]; ww = wn[rw, us[ev]][:, None]
                en_tail += float((oe ** 2).sum())
                for m in MS:
                    msk = np.zeros(640, np.float32); msk[order_s[:m]] = 1.0
                    res = oe - (he * msk) @ D.T
                    np.add.at(Rst[m], rw, ww * res); en_st[m] += float((res ** 2).sum())
                    for ru in RULES:
                        se = sc[ru][ev]; kth = np.partition(se, 640 - m, axis=1)[:, 640 - m][:, None]
                        res = oe - (he * (se >= kth)) @ D.T
                        np.add.at(RR[(ru, m)], rw, ww * res); EN[(ru, m)] += float((res ** 2).sum())
    full = np.einsum("ns,nsd->nd", wn, C); fn = np.linalg.norm(full, axis=1) + 1e-9
    # A: drop-k, norms, orthogonality
    cn = np.linalg.norm(C, axis=2)
    dk = []
    for k in range(1, 10):
        dk.append(float(np.mean(np.linalg.norm(np.einsum("ns,nsd->nd", wn[:, k:], C[:, k:]), axis=1) / fn)))
    Cn = C / (cn[:, :, None] + 1e-9); G2 = np.einsum("nsd,ntd->nst", Cn, Cn)
    offd = G2[:, np.triu_indices(10, 1)[0], np.triu_indices(10, 1)[1]]
    say(f"L{L} A: drop-k error (no renorm) k=1..9: {np.round(dk, 3).tolist()}; ||E|| by rank {np.round(cn.mean(0), 2).tolist()}; "
        f"cos between a row's experts: mean {offd.mean():+.3f}, mean |cos| {np.abs(offd).mean():.3f}, p95 |cos| {np.percentile(np.abs(offd), 95):.3f}; "
        f"||full routed|| {fn.mean():.2f} vs sum_r w_r||E_r|| {float((wn * cn).sum(1).mean()):.2f}")
    # B: thin tails
    ev = evalrow
    def be(Rm): return float(np.mean(np.linalg.norm(Rm[ev], axis=1) / fn[ev]))
    say(f"L{L} B: thin tails (ranks >= 2 with m of 640 hidden units; odd checks; block error / energy kept); input PCA-128 holds {xen:.2f} of ||x||^2 (PCA-64 {xen64:.2f}); reference drop-k: k2 {dk[1]:.3f}, k5 {dk[4]:.3f}, k8 {dk[7]:.3f}")
    for m in MS:
        say(f"   m {m}: static {be(Rst[m]):.3f} / {1 - en_st[m] / max(en_tail, 1e-9):.2f}; " + "; ".join(f"{ru} {be(RR[(ru, m)]):.3f} / {1 - EN[(ru, m)] / max(en_tail, 1e-9):.2f}" for ru in RULES)
            + f"; ALL ranks thin (oracle) {be(Rall[m]):.3f}")
    # G: group-equivalents on live-like blocks: a group with a rank 0-1 use is full; a tail-only group costs |union of its rows' selected units| / 640
    ge = {m: [0.0, 0.0] for m in MS}; ng = 0; ntail = 0; nsing = 0
    for c, rows, U in blocks:
        uses = {}
        for r in rows:
            for sl in range(10): uses.setdefault(int(IDS[r, sl]), []).append((r, sl))
        for e, lst in uses.items():
            ng += 1
            if any(sl < 2 for _, sl in lst):
                for m in MS: ge[m][0] += 1.0; ge[m][1] += 1.0
                continue
            ntail += 1; nsing += (len(lst) == 1)
            for m in MS:
                u_ = np.zeros(640, bool)
                for r, sl in lst: u_ |= Sel[m][r, sl]
                f_ = u_.sum() / 640.0; ge[m][0] += f_; ge[m][1] += 1.0 / 3.0 + 2.0 / 3.0 * f_
    say(f"L{L} G: live-like blocks: groups per block-layer {ng / len(blocks):.1f}; tail-only groups {100 * ntail / ng:.0f} % (single-row {100 * nsing / ng:.0f} %); group-equivalents with thin tails (oracle units / gate computed in full): "
        + "; ".join(f"m {m}: x{ge[m][0] / ng:.2f} / x{ge[m][1] / ng:.2f}" for m in MS))
    sub = C[ev][:, 2:].reshape(-1, 2560); sub = sub[np.random.default_rng(0).choice(len(sub), min(4000, len(sub)), replace=False)]
    sv = np.linalg.svd(sub, compute_uv=False) ** 2; cs_ = np.cumsum(sv) / sv.sum()
    say(f"L{L} S: shared output subspace of tail expert outputs ({len(sub)} uses): energy in the top 64 / 256 / 640 / 1280 directions {cs_[63]:.2f} / {cs_[255]:.2f} / {cs_[639]:.2f} / {cs_[1279]:.2f}")
    # C: substitution
    st = dict(n=0, cos_or=[], cos_rt=[], drop=[], sub_or=[], sub_rt=[], sing=0, pairs=0, un0=0, un1=0, rows=0)
    for c, rows, U in blocks:
        cnt = {}
        for r in rows:
            for e in IDS[r]: cnt[e] = cnt.get(e, 0) + 1
        st["un0"] += len(U); st["un1"] += sum(1 for e in U if cnt[e] > 1 or any(e in IDS[r][:2] for r in rows)); st["rows"] += len(rows)
        for r in rows:
            members = [e for e in U if e not in IDS[r]]
            if not members: continue
            MU = np.stack([MemOut[memidx[(r, e)]] for e in members]).astype(np.float32); mun = np.linalg.norm(MU, axis=1) + 1e-9
            rtb = int(np.argmax(lg[r, members]))
            d_drop = np.zeros(2560, np.float32); d_or = np.zeros(2560, np.float32); d_rt = np.zeros(2560, np.float32); ns = 0
            for sl in range(2, 10):
                e = IDS[r, sl]
                if cnt[e] > 1: continue
                v = C[r, sl]; vn = np.linalg.norm(v) + 1e-9; cs = (MU @ v) / (mun * vn); j = int(np.argmax(cs))
                st["cos_or"].append(float(cs[j])); st["cos_rt"].append(float(cs[rtb])); ns += 1
                alpha = max(float(MU[j] @ v) / float(mun[j] ** 2), 0.0)
                d_drop += wn[r, sl] * v; d_or += wn[r, sl] * (v - alpha * MU[j]); d_rt += wn[r, sl] * (v - MU[rtb])
            st["sing"] += ns; st["pairs"] += 8
            if ns:
                st["drop"].append(np.linalg.norm(d_drop) / fn[r]); st["sub_or"].append(np.linalg.norm(d_or) / fn[r]); st["sub_rt"].append(np.linalg.norm(d_rt) / fn[r])
    say(f"L{L} C: substitution in live-like blocks ({len(blocks)} blocks, {st['rows'] / len(blocks):.1f} rows): tail singletons {100 * st['sing'] / max(st['pairs'], 1):.0f} % of tail pairs; "
        f"union {st['un0'] / len(blocks):.1f} -> {st['un1'] / len(blocks):.1f} per layer without them (x{st['un1'] / max(st['un0'], 1):.2f}); "
        f"cos(singleton output, best member output on the same row): oracle mean {np.mean(st['cos_or']):.3f}, p90 {np.percentile(st['cos_or'], 90):.3f}, max {np.max(st['cos_or']):.3f}; router-best member {np.mean(st['cos_rt']):+.3f}; "
        f"row block error: drop singletons {np.mean(st['drop']):.3f}, oracle substitution (optimal scale) {np.mean(st['sub_or']):.3f}, router-best substitution {np.mean(st['sub_rt']):.3f}")
    # D: tail surrogates
    tail = np.einsum("ns,nsd->nd", wn[:, 2:], C[:, 2:]); tn = np.linalg.norm(tail, axis=1) + 1e-9
    ok = (PJ >= 1) & (PJ < kept[CK]); i1 = np.nonzero(ok)[0]; i0 = i1 - 1
    pe = np.linalg.norm(tail[i1] - tail[i0], axis=1) / fn[i1]; pc = np.einsum("nd,nd->n", tail[i1], tail[i0]) / (tn[i1] * tn[i0])
    res_lin = []
    for pe_ in (0, 2):
        tr = PR != pe_; te = PR == pe_; Xc = X[tr]; Yc = tail[tr] - tail[tr].mean(0); K = Xc @ Xc.T; best = None
        for lam in (0.003, 0.03, 0.3):
            A = np.linalg.solve(K + lam * np.trace(K) / len(K) * np.eye(len(K), dtype=np.float32), Yc)
            pred = (X[te] @ Xc.T) @ A + tail[tr].mean(0)
            er = float(np.mean(np.linalg.norm(tail[te] - pred, axis=1) / fn[te])); r2 = 1 - float(((tail[te] - pred) ** 2).sum() / ((tail[te] - tail[tr].mean(0)) ** 2).sum())
            if best is None or er < best[0]: best = (er, r2, lam)
        res_lin.append(best)
    say(f"L{L} D: tail (ranks 2-9) norm / full {float(np.mean(tn / fn)):.3f}; previous kept row's tail as surrogate: error {pe.mean():.3f}, cos {pc.mean():+.3f}; "
        f"ridge x -> tail, leave-one-prompt-out (HumanEval_68, HumanEval_91): error {res_lin[0][0]:.3f} / {res_lin[1][0]:.3f}, R2 {res_lin[0][1]:+.2f} / {res_lin[1][1]:+.2f} (best of 3 lambdas, optimistic); reference k2 {dk[1]:.3f}, k5 {dk[4]:.3f}, k8 {dk[7]:.3f}  ({time.time() - t0:.0f} s)")
