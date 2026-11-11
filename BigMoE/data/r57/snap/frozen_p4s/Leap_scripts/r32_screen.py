"""R-32 STOPPRED (NewIdeas Tue ~20:35; Leap, pre-registered Tue 2026-10-27 20:37 EET in ~/Leap/TABLE_350.md 'R-32'):
predict a check's stop row from the target's own layer-12 state, so rows predicted rejected can be dropped from the
layers after 12 (their experts never read). Kept rows stay exact (every layer is causal); a wrong early stop costs
tokens only. MoonShots (R-24, from data/r24_routes.npz): rejected-only experts are 28.2 % of a check's per-layer
union; a cut from layer 12 with a perfect stop row reaches 75 % of that (union -21 %).
Data: CA's resident 12-layer slice (layers 0-11, MTPLX's verify forward read-only) on real drafted checks: DFlash2
(L1-20, block 16) on the 24 held-out code samples (dfl2d_code, the sample's real features), acceptance from the
sample's true tokens (the real target's greedy output), then the real commit of keep = accepted + 1.
  layer 12 (the slice's head = the logit lens at layer 12): p(next draft token), top-1 p, entropy per row
  layer 0 (dfl2_trunc-style baseline): the drafter's p of its chosen token and its selector margin per position
  routing at layers 0-11 (top-10 per row): the union proxy for the cut layers 12-47
Predictors (stop at the first draft whose score < tau; thresholds and probes fit on samples 0-11, scored on 12-23):
lens p(draft) alone; a logistic probe on layer-12 + drafter features; the layer-0 baselines (drafter p alone, probe
on drafter features). Whole-check cut = 36 / 48 x the cut in the layers after 12.
Lines (NewIdeas): PASS: at layer 12, whole-check union cut >= 10 % at <= 2 % fewer accepted tokens AND >= 5 points
better than the layer-0 baseline. KILL: < 5 % at <= 2 % token loss, or no better than layer 0.
Also saved for MoonShots (NewIdeas ~20:40): per-row top-10 expert ids AND their softmax gate weights at layers
0-11, with accept lengths, in r24_routes.npz's layout: data/r32_routes.npz.
Usage: [BigMoE venv] r32_screen.py -> ~/Leap/data/r32_screen.txt, data/r32_dump.npz, data/r32_routes.npz"""
import glob, os, sys, time
sys.argv = [sys.argv[0], "r32"]                          # ca_anat's MODE: load only, no run
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
import ca_anat as CA
import numpy as np
import mlx.core as mx
import mlx.nn as nn

Q, CS, model, D = CA.Q, CA.CS, CA.model, CA.D
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party"))
import dflash_model_mlx as DF
class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales",
                  "biases")] + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales",
                  "biases")])
del w
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
BS, R = 16, 16

def select_conf(block, hidden):
    """CA's draft path; also the drafter's p of each chosen token and the selector's best - second score"""
    logits = draft.compute_logits(hidden).astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    sel = draft.candidate_selector
    cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
    unary = mx.take_along_axis(logits, cand, axis=-1)
    hp = sel.hidden_projection(hidden)
    pred, path, margin = block[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        top2 = mx.sort(sc, axis=-1)[:, -2:]
        margin.append(top2[:, 1] - top2[:, 0])
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    path = mx.stack(path, axis=1)
    pch = mx.exp(mx.take_along_axis(logits, path[..., None], axis=-1)[..., 0] - lse)
    return path, pch, mx.stack(margin, axis=1)

ROUTE, ROUTE_ON, ORIG_ARGP = [], [False], mx.argpartition
def argp(a, kth, axis=-1, **kw):
    out = ORIG_ARGP(a, kth=kth, axis=axis, **kw)
    if ROUTE_ON[0] and a.shape[-1] == 512 and kth == -10:   # a = the softmax gates (MTPLX's _verify_call)
        top = out[..., -10:]
        ROUTE.append((top, mx.take_along_axis(a, top, axis=-1)))
    return out
mx.argpartition = argp

# ---------------------------------------------------------------- the dump
REC = {k: [] for k in ("acc", "drafts", "lens_pd", "lens_pmax", "lens_ent", "dr_p", "dr_margin", "routes", "gates",
                       "sample")}
t_run = time.time()
paths = sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))
for si, zp in enumerate(paths):
    while os.path.exists(PAUSE):
        time.sleep(5)
    z = np.load(zp)
    f, sid, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
    rc = model.make_cache()
    pre = [int(t) for t in sid[:gen - off]]
    for a0 in range(0, len(pre), 1024):
        lg0 = model(mx.array([pre[a0:a0 + 1024]]), cache=rc)
        mx.eval(lg0, CA.states(rc))
    dc = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c_ in dc:
        c_.offset = start
    fed, a = start, gen
    while a - off + BS < len(sid):
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(sid[a - off])] + [MASK] * (BS - 1)])
        path, pch, marg = select_conf(blk, draft.hidden_states(blk, ctx, dc, 1))
        dr = np.array(path[0]).tolist()
        fed = a
        true = sid[a - off + 1:a - off + BS]
        k = 0
        while k < BS - 1 and dr[k] == true[k]:
            k += 1
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
        x = mx.array([[int(sid[a - off])] + dr[:BS - 1]])
        snap = CS.snapshot_untrimmable_cache_lazy(rc)
        ROUTE.clear(); ROUTE_ON[0] = True
        lg = CA.forward("A", x, rc)[0].astype(mx.float32)   # [16, V]: the logit lens at layer 12
        ROUTE_ON[0] = False
        lse = mx.logsumexp(lg, axis=-1)
        logp = lg - lse[:, None]
        nxt = mx.array(dr[:BS - 1])
        lens_pd = mx.exp(mx.take_along_axis(logp[:BS - 1], nxt[:, None], axis=-1)[:, 0])
        lens_pmax = mx.exp(mx.max(logp, axis=-1))
        lens_ent = -mx.sum(mx.exp(logp) * logp, axis=-1)
        rt = mx.stack([t_.reshape(R, 10) for t_, _ in ROUTE])
        gw = mx.stack([g_.reshape(R, 10) for _, g_ in ROUTE]).astype(mx.float32)
        mx.eval(lens_pd, lens_pmax, lens_ent, rt, gw, pch, marg)
        model.commit_verified_window(rc, snap.states, keep_tokens=k + 1, verified_tokens=R)
        mx.eval(CA.states(rc))
        for key, val in (("acc", k), ("drafts", dr[:BS - 1]), ("lens_pd", np.array(lens_pd)),
                         ("lens_pmax", np.array(lens_pmax)), ("lens_ent", np.array(lens_ent)),
                         ("dr_p", np.array(pch[0])), ("dr_margin", np.array(marg[0])),
                         ("routes", np.array(rt).astype(np.int16)), ("gates", np.array(gw).astype(np.float16)),
                         ("sample", si)):
            REC[key].append(val)
        a = a + k + 1
    del rc, dc
    mx.clear_cache()
mx.argpartition = ORIG_ARGP
A = {k: np.array(v) for k, v in REC.items()}
np.savez(os.path.join(D, "r32_dump.npz"), **A)
np.savez(os.path.join(D, "r32_routes.npz"), routes=A["routes"], gates=A["gates"], accepted=A["acc"],   # r24's layout
         layers=np.arange(A["routes"].shape[1]), sample=A["sample"], samples=np.array([os.path.basename(p) for p in paths]))
n = len(A["acc"])
say(f"R-32: {n} real drafted checks on {len(paths)} samples in {time.time() - t_run:.0f} s; accepted drafts "
    f"{A['acc'].mean():.2f} per check (tokens per check {A['acc'].mean() + 1:.2f}); data/r32_dump.npz")

# ---------------------------------------------------------------- the analysis (CPU)
S = A["acc"] + 1                                         # true stop row: the first rejected draft row (16 = none)
L = A["routes"].shape[1]
uall = np.array([[len(set(A["routes"][i, l].ravel())) for l in range(L)] for i in range(n)])
def ukeep(i, rows):
    return np.array([len(set(A["routes"][i, l, :rows].ravel())) for l in range(L)])
CUT_SHARE = 36 / 48

def evaluate(stop):
    """stop[i] = predicted first dropped row (1..16); returns (token loss, whole-check union cut, pairs cut)"""
    tok = np.minimum(stop, S)
    loss = 1 - tok.sum() / S.sum()
    uk = np.array([ukeep(i, int(stop[i])) for i in range(n)])
    ucut = 1 - uk.sum() / uall.sum()
    pcut = 1 - np.minimum(stop, 16).sum() / (16 * n)
    return loss, CUT_SHARE * ucut, CUT_SHARE * pcut

def stops(score, tau):
    """first draft row j (1..15) whose score[j - 1] < tau, else 16"""
    below = score < tau
    first = np.where(below.any(axis=1), below.argmax(axis=1) + 1, 16)
    return first

oracle = evaluate(S)
say(f"  oracle stop (perfect predictor): token loss {oracle[0]:.1%}, whole-check union cut {oracle[1]:.1%}, pairs "
    f"{oracle[2]:.1%} (MoonShots' R-24: ~21 % / ~26 %)")

samp = A["sample"]
fit, test = samp < len(paths) // 2, samp >= len(paths) // 2
def rows_labels(mask):
    """per-row training data: draft rows j = 1..S (accepted before the stop: 1; the stop row: 0)"""
    X, y, idx = [], [], []
    for i in np.where(mask)[0]:
        for j in range(1, min(int(S[i]), 15) + 1):
            X.append(j); y.append(1 if j < S[i] else 0); idx.append((i, j))
    return np.array(X), np.array(y), idx

def feats(i, j, kind):
    base = [A["dr_p"][i, j - 1], A["dr_margin"][i, j - 1], j / 15]
    if kind == "drafter":
        return base
    return base + [A["lens_pd"][i, j - 1], A["lens_pmax"][i, j - 1], A["lens_ent"][i, j - 1] / 10]

def fit_logreg(X, y, steps=3000, lr=0.5, l2=1e-3):
    X = np.hstack([X, np.ones((len(X), 1))])
    wv = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ wv))
        wv -= lr * (X.T @ (p - y) / len(y) + l2 * wv)
    return wv

def probe_scores(kind):
    _, y, idx = rows_labels(fit)
    X = np.array([feats(i, j, kind) for i, j in idx])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    wv = fit_logreg((X - mu) / sd, y)
    out = np.zeros((n, 15))
    for i in range(n):
        Xi = (np.array([feats(i, j, kind) for j in range(1, 16)]) - mu) / sd
        out[i] = 1 / (1 + np.exp(-(np.hstack([Xi, np.ones((15, 1))]) @ wv)))
    return out

PRED = {"layer 12: lens p(draft)": A["lens_pd"], "layer 12: probe (lens + drafter)": probe_scores("lens"),
        "layer 0: drafter p": A["dr_p"], "layer 0: probe (drafter)": probe_scores("drafter")}
TAUS = np.concatenate([[0.0], np.geomspace(1e-4, 0.999, 120)])
best = {}
for name, sc in PRED.items():
    pick = None
    for tau in TAUS:                                     # threshold chosen on the fit half: the largest cut at <= 2 % loss
        st = stops(sc, tau)
        tokf = np.minimum(st[fit], S[fit]); lossf = 1 - tokf.sum() / S[fit].sum()
        if lossf <= 0.02:
            pick = tau
    st = stops(sc, pick)
    sub = test
    tok = np.minimum(st[sub], S[sub]); loss = 1 - tok.sum() / S[sub].sum()
    uk = np.array([ukeep(i, int(st[i])) for i in np.where(sub)[0]])
    ucut = CUT_SHARE * (1 - uk.sum() / uall[sub].sum())
    pcut = CUT_SHARE * (1 - np.minimum(st[sub], 16).sum() / (16 * sub.sum()))
    best[name] = (ucut, loss, pick, pcut)
    say(f"  {name:34s}: tau {pick:.4g} (fit half, <= 2 % loss) -> held-out half: token loss {loss:.1%}, whole-check "
        f"union cut {ucut:.1%}, pairs cut {pcut:.1%}")
l12 = max(best[k][0] for k in best if k.startswith("layer 12"))
l0 = max(best[k][0] for k in best if k.startswith("layer 0"))
ok_loss = all(best[k][1] <= 0.025 for k in best)
v = "PASS" if l12 >= 0.10 and l12 - l0 >= 0.05 else "KILL" if l12 < 0.05 or l12 <= l0 else "BETWEEN"
say(f"R-32 verdict (PASS: layer-12 union cut >= 10 % at <= 2 % token loss AND >= 5 points over layer 0; KILL: < 5 % "
    f"or no better than layer 0): {v} (layer 12 {l12:.1%}, layer 0 {l0:.1%}; held-out losses within 2.5 %: {ok_loss})")
open(os.path.join(D, "r32_screen.txt"), "w").write("\n".join(lines) + "\n")
