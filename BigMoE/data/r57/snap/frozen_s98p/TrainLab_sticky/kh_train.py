> (a quote from the project owner, kept private)
distillation -- pre-registration'). CPU only (MLX on the CPU device). Predict the cheap pass's CERTIFICATE (p1 under top-2
>= 0.99) from what the drafter knows before the check, and score the gate on the measured top-2 flips.
Rows: ~/FableGod/floor_rows.json (10,316 rows of the decision map, 36 windows: 12 code / ifeval / multi; rows with a
drafter cf only). Labels: cert2 = exp(top-1 lp under topk2) >= 0.99 (~/FableGod/decision_map/out/topk2.json); flips =
floor_rows' flip['topk2'] (every row perturbed: conservative). Features, per row = draft j + 1 of the HPD-0 walk's check at
anchor A (row A + j; the decision map's gate source): the drafter's hidden at draft position j (2560), its pick's selector
p1 / p2 / entropy, the unary top-16 p1 / p2 / entropy, cf, log(1 - cf), the running cf product, depth j
(~/TrainLab/data/d02b_walk_kh_<dom>.npz, TL_WALK_KH=1 at the HPD-0 anchors; its dcf must equal HPD-0's).
Models: LOGIT-S (logistic on the 10 scalars), LOGIT-H (logistic on hidden + scalars), MLP-H (primary: 2 layers, 256
hidden, on hidden + scalars). CV: 6 folds of 2 code windows; each fold trains on every other row (all domains) and scores
its 2 held-out code windows. Lines on the pooled out-of-fold code scores: coverage at <= 0.1 % top-2 flips (KILL < 60 %,
PASS >= 60 % and the deployment check holds), coverage at 0 flips; deployment check = each fold's threshold set on its
training code rows (the lowest score with 0 flips there), applied to its held-out rows: pooled coverage and flips.
Baselines the same way: cf alone, the certificate itself (the ceiling). Output: data/kh_read.txt (appended)."""
import json, math, os, sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

mx.set_default_device(mx.cpu)
FG = os.path.expanduser("~/FableGod")
TD = os.path.expanduser("~/TrainLab/data")
WALK = {"code": "d02b_walk_kh_code.npz", "ifeval": "d02b_walk_kh_ifeval.npz", "multi": "d02b_walk_kh_multi.npz"}
HPD0 = {"code": "d02b_walk_hpd0_code.npz", "ifeval": "d02b_walk_hpd0_ifeval.npz", "multi": "d02b_walk_hpd0_multi.npz"}
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

# ---------------------------------------------------------------- rows, labels, features
R = [r for r in json.load(open(os.path.join(FG, "floor_rows.json"))) if r["cf"] is not None]
T2 = json.load(open(os.path.join(FG, "decision_map/out/topk2.json")))
Z, H0, POS = {}, {}, {}
for dom, wf in WALK.items():
    Z[dom], H0[dom] = np.load(os.path.join(TD, wf)), np.load(os.path.join(TD, HPD0[dom]))
    names = [str(x) for x in Z[dom]["names"]]
    assert names == [str(x) for x in H0[dom]["names"]], dom
    for i, nm in enumerate(names):
        ch = Z[dom][f"{i}_checks"]
        assert np.array_equal(ch, H0[dom][f"{i}_checks"]), (dom, nm)                      # the HPD-0 anchors
        assert np.nanmax(np.abs(Z[dom][f"{i}_dcf"] - H0[dom][f"{i}_dcf"])) < 1e-5, (dom, nm)   # the same cf (HPD-0)
        for c, (k, m, A) in enumerate(ch):
            for j in range(min(k + 1, 15)):                                                # rows A .. A + k (bonus row k)
                POS[(nm, int(A) + j)] = (dom, i, c, j)
X, S, y, flip, cfv, dom_, win_ = [], [], [], [], [], [], []
miss = 0
for r in R:
    key = (r["nm"], r["i"])
    if key not in POS:
        miss += 1; continue
    dom, i, c, j = POS[key]
    z = Z[dom]
    cf = float(z[f"{i}_dcf"][c, j])
    assert abs(cf - r["cf"]) < 1e-5, (key, cf, r["cf"])
    st = z[f"{i}_kst"][c, j]
    run = float(np.prod(z[f"{i}_dcf"][c, :j + 1]))
    X.append(z[f"{i}_khid"][c, j].astype(np.float32))
    S.append([cf, math.log(max(1 - cf, 1e-6)), run, j / 15.0, *[float(v) for v in st]])
    w = T2[r["nm"]]; jj = r["i"] - w["a"]
    y.append(1.0 if math.exp(w["lp"][jj][0]) >= 0.99 else 0.0)
    flip.append(bool(r["flip"].get("topk2"))); cfv.append(cf); dom_.append(r["dom"]); win_.append(r["nm"])
X, S, y, flip, cfv = np.stack(X), np.array(S, np.float32), np.array(y, np.float32), np.array(flip), np.array(cfv)
dom_, win_ = np.array(dom_), np.array(win_)
say(("[post-hoc: trained on code rows only] " if os.environ.get("KH_CODEONLY") else "") + f"K-HEAD read-out: {len(y)} rows with features ({miss} rows not found in the walk); code {int((dom_ == 'code').sum())}, "
    f"ifeval {int((dom_ == 'ifeval').sum())}, multi {int((dom_ == 'multi').sum())}; cert2 share code "
    f"{y[dom_ == 'code'].mean():.3f}; top-2 flips code {int(flip[dom_ == 'code'].sum())}")

# ---------------------------------------------------------------- operating points
def coverage(score, fl, bar):
    """the largest share of rows above one threshold with flips <= bar x selected (bar 0: none)"""
    o = np.argsort(-score, kind="stable")
    f = np.cumsum(fl[o]); n = np.arange(1, len(o) + 1)
    ok = f <= bar * n if bar > 0 else f == 0
    # thresholds only between distinct scores
    sc = score[o]; brk = np.r_[sc[1:] != sc[:-1], True]
    good = np.nonzero(ok & brk)[0]
    return (good[-1] + 1) / len(o) if len(good) else 0.0

def thr_zero(score, fl):
    """the lowest threshold with zero flips at or above it (training rows)"""
    o = np.argsort(-score, kind="stable"); sc = score[o]; f = np.cumsum(fl[o])
    brk = np.r_[sc[1:] != sc[:-1], True]
    good = np.nonzero((f == 0) & brk)[0]
    return sc[good[-1]] if len(good) else np.inf

# ---------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d, h), nn.Linear(h, 1)
    def __call__(self, x):
        return self.l2(nn.relu(self.l1(x)))[:, 0]

def fit(Xtr, ytr, kind, seed):
    mx.random.seed(seed)
    m = MLP(Xtr.shape[1], 256) if kind == "mlp" else nn.Linear(Xtr.shape[1], 1)
    opt = optim.Adam(learning_rate=1e-3)
    xa, ya = mx.array(Xtr), mx.array(ytr)
    wd = 1e-4 if kind == "mlp" else 1e-3
    def loss(mod, xb, yb):
        z = mod(xb) if kind == "mlp" else mod(xb)[:, 0]
        l2 = sum((p ** 2).sum() for _, p in nn.utils.tree_flatten(mod.trainable_parameters()))
        return nn.losses.binary_cross_entropy(z, yb, with_logits=True, reduction="mean") + wd * l2
    lg = nn.value_and_grad(m, loss)
    rng = np.random.default_rng(seed)
    n = len(ytr); bs = 256
    for ep in range(30 if kind == "mlp" else 60):
        for b in np.array_split(rng.permutation(n), max(1, n // bs)):
            bi = mx.array(b.astype(np.int32))
            l, g = lg(m, xa[bi], ya[bi]); opt.update(m, g); mx.eval(m.parameters(), opt.state)
    return m

def predict(m, Xte, kind):
    z = m(mx.array(Xte)) if kind == "mlp" else m(mx.array(Xte))[:, 0]
    return np.array(z)

code_w = sorted(set(win_[dom_ == "code"]))
assert len(code_w) == 12, len(code_w)
folds = [code_w[k::6] for k in range(6)]
feats = {"LOGIT-S": S, "LOGIT-H": np.concatenate([X, S], axis=1), "MLP-H": np.concatenate([X, S], axis=1)}
res = {}
for name, F in feats.items():
    kind = "mlp" if name.startswith("MLP") else "lin"
    oof = np.full(len(y), np.nan); dep_sel, dep_fl, dep_n = 0, 0, 0
    for fw in folds:
        te = np.isin(win_, fw); tr = ~te
        if os.environ.get("KH_CODEONLY"):                 # post-hoc (11-02): train on the other code windows only
            tr = tr & (dom_ == "code")
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        m = fit((F[tr] - mu) / sd, y[tr], kind, seed=11)
        s_te = predict(m, (F[te] - mu) / sd, kind); oof[te] = s_te
        trc = tr & (dom_ == "code")
        t = thr_zero(predict(m, (F[trc] - mu) / sd, kind), flip[trc])
        sel = s_te >= t
        dep_sel += int(sel.sum()); dep_fl += int(flip[te][sel].sum()); dep_n += int(te.sum())
    cm = dom_ == "code"
    res[name] = (coverage(oof[cm], flip[cm], 0.001), coverage(oof[cm], flip[cm], 0.0), dep_sel / dep_n, dep_fl, dep_sel)
    say(f"  {name:8s}: out-of-fold code coverage at <= 0.1 % flips {100 * res[name][0]:5.1f} %, at 0 flips "
        f"{100 * res[name][1]:5.1f} %; deployment (train-fold 0-flip threshold) {100 * res[name][2]:5.1f} % covered, "
        f"{dep_fl} flips in {dep_sel} gated rows ({100 * dep_fl / max(dep_sel, 1):.2f} %)")
cm = dom_ == "code"
say(f"  baselines (code): cf alone at <= 0.1 % flips {100 * coverage(cfv[cm], flip[cm], 0.001):.1f} %, at 0 flips "
    f"{100 * coverage(cfv[cm], flip[cm], 0.0):.1f} %; the certificate itself (ceiling) {100 * coverage(y[cm], flip[cm], 0.0):.1f} %")
c1, c0, dc, dfl, dsel = res["MLP-H"]
v = ("KILL" if c1 < 0.60 else "PASS" if dfl <= 0.001 * max(dsel, 1) else "BETWEEN")
say(f"  LINE (MLP-H, code, out-of-fold coverage at <= 0.1 % top-2 flips; KILL < 60 %, PASS >= 60 % with the deployment "
    f"check's flips <= 0.1 %): {100 * c1:.1f} % -> {v}")
open(os.path.join(TD, "kh_read.txt"), "a").write("\n".join(lines) + "\n\n")
