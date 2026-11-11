"""D-22 MTPTTT (MTP Head, Fri 10-30; REPORT 'D-22'): train the 125B's native MTP head with the EAGLE-3 training-time
test inside its attention, so every depth is trained the way the chain runs at inference.
Chain semantics (SpecDecode spec125._MTPHead; MTPLX qwen4_exp.Qwen4ExpMTP): the MTP cache holds one entry per committed
pair (the trunk's pre-mixer widened stream at u, token u+1). For anchor s (the last committed pair), depth 1 runs the
pair (W[s], ids[s+1]) at position s and predicts ids[s+2]; depth j >= 2 runs (the head's own depth-(j-1) pre-mixer
output, token s+j) at position s+j-1, attends to the true pairs u <= s plus its own chain entries at s+1..s+j-1, and
predicts ids[s+j+1]. The 10-22 run (mtp125n_d4) trained depth j over a fresh cache instead and lost depths 2-4.
Forward = mtp125.py's plain-MLX head (dense attention, = QSA below ~2051 keys; frozen 4-bit experts through gather_qmm
with stop_gradient routing; fp32 copies of fc_hidden, fc_embedding, attention q/k/v/o, router, shared expert). Every
true pair gives keys / values; only anchors from a0 on (the answer) run the full layer.
Teacher-forced tokens: on the accepted prefix the chain's own tokens are the true ones, and acceptance stops at the
first miss, so the read-out counts the same accepted runs as a real chain.
Data: L1-20's windows (~/Leap/data/l120: ids, asst, the 125B's top-32 per row, split) + D-16a's widened stream
(data/d16w/<key>.npz wd, offset 0). Held out: data/d16w tf_code_HumanEval_* (24), tf_gsm_* (60).
Commands:
  gate                 untrained head vs MTPLX's real chain (D-17's data/d17 chains, 8 code + 8 gsm prompts): argmax
                       agreement per depth on positions where the real chain's earlier drafts were right
  train --run NAME     [--depth 8] [--lr 1e-5] [--wdecay 0.8] [--epochs 2]; resumable, PAUSE-aware, SIGTERM checkpoints
  eval --run NAME|stock [--sets code,gsm]: the chain from the true prefix at every position of the held-out answers,
                       walked like D-17 (tokens per pass, K 3 / 7 / 15), per-sample rows saved for paired CIs.
                       agent turns are longer than 2,051 keys, where MTPLX's QSA selects keys and this dense forward
                       does not: approximate there.
Usage: [BigMoE venv] python sticky/d22_ttt.py CMD ...   (outputs data/d22_*.txt / .npz, checkpoints ckpt/d22_*)"""
import argparse, glob, json, os, signal, sys, time
os.environ.setdefault("TL_PLAIN_MIN", "1")                 # MTPLX's NAX sorted gather has no VJP (mtp125.py)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/TrainLab_root"))
import mtp125 as M                                          # sets MTPLX's env (eager HC read for the VJP), imports mlx
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import mtplx.models.qwen4_exp as Q

H = os.path.expanduser
TD = H("~/TrainLab/data")
CK = H("~/TrainLab/ckpt")
L120 = H("~/Leap/data/l120")
PAUSE = H("~/.local/share/quiet/PAUSE")
NEG = -1e9


# ---------------------------------------------------------------- the head, one depth at a time
def qkv(att, P, x, pos):
    """q [1,H,n,hd], gate [1,n,H*hd], k / v [1,KV,n,hd] for rows x [1,n,D] at absolute positions pos [n]"""
    B, n, _ = x.shape
    xf = x.astype(mx.float32)
    q, gate = mx.split(P.q_proj(xf).reshape(B, n, att.n_heads, -1), 2, axis=-1)
    k = P.k_proj(xf).reshape(B, n, att.n_kv_heads, -1)
    v = P.v_proj(xf).reshape(B, n, att.n_kv_heads, -1)
    q, k = att.q_norm(q), att.k_norm(k)
    cos, sin = Q._rope_cos_sin(pos, att._inv_freq, att._rope_attention_scaling)
    q, k = Q._apply_partial_rope(q, cos, sin), Q._apply_partial_rope(k, cos, sin)
    return q.transpose(0, 2, 1, 3), gate.reshape(B, n, -1), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)


def ttt_attn(scale, q, K1, V1, ck, cv, a0):
    """Depth >= 2: query row i (anchor s = a0 + i) attends to the true pairs u <= s (K1 / V1 [1,KV,S1,hd]) and to its
    own chain entries (ck / cv: one [1,KV,n,hd] per depth 2..j, same rows). q [1,H,n,hd] -> [1,H,n,hd]"""
    _, Hh, n, hd = q.shape
    KV, S1 = K1.shape[1], K1.shape[2]
    r = Hh // KV
    qg = q.reshape(1, KV, r, n, hd)
    st = (qg @ K1[:, :, None].swapaxes(-1, -2)) * scale                         # [1,KV,r,n,S1]
    st = mx.where(mx.arange(S1)[None, :] <= (a0 + mx.arange(n))[:, None], st, NEG)
    sc = mx.stack([(qg * k[:, :, None]).sum(-1) for k in ck], -1) * scale     # [1,KV,r,n,J]
    m = mx.stop_gradient(mx.maximum(st.max(-1, keepdims=True), sc.max(-1, keepdims=True)))
    et, ec = mx.exp(st - m), mx.exp(sc - m)
    o = et @ V1[:, :, None]
    for i, v in enumerate(cv):
        o = o + ec[..., i:i + 1] * v[:, :, None]
    o = o / (et.sum(-1, keepdims=True) + ec.sum(-1, keepdims=True))
    return o.reshape(1, Hh, n, hd)


def fuse(head, P, Win, emb):
    B, n, Wd = Win.shape
    hn = head.pre_fc_norm_hidden(Win).reshape(B, n, head._hc, -1).astype(mx.float32)
    en = P.fc_embedding(head.pre_fc_norm_embedding(emb).astype(mx.float32))
    return (P.fc_hidden(hn) + en[:, :, None, :]).reshape(B, n, Wd)


def finish(head, P, hyper, inject, blk):
    """mtp125.head_train after the attention: HC + MoE -> the pre-mixer widened output"""
    layer = head.layers[0]
    mixed, hyper, inject = layer.mlp_hyper_connection(hyper, pending=(blk.astype(hyper.dtype), inject))
    blk = M.moe_train(layer.mlp, P, mixed).astype(hyper.dtype)
    return hyper + (blk[..., None, :] * inject[..., :, None]).reshape(*hyper.shape)


def chain(lm, head, P, W, ids, D, on_depth, a0=0):
    """W [1,S,Wd] (bf16) the trunk's widened stream, ids [1,S]. True pairs u = 0..S-2 give keys / values; anchors
    s = a0..S-j-1 run depth j (it needs token s+j); on_depth(j, h) gets h [1,n,Wd], row i = anchor a0 + i, which
    predicts ids[a0 + i + j + 1]."""
    S = W.shape[1]
    emb = lm.model.embed_tokens
    layer = head.layers[0]
    att = layer.self_attn
    n1 = S - 1
    mixed, hyper, inject = layer.attn_hyper_connection(fuse(head, P, W[:, :n1], emb(ids[:, 1:n1 + 1])).astype(mx.bfloat16))
    q, gate, K1, V1 = qkv(att, P, mixed, mx.arange(n1))
    o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K1, V1, scale=att.scale, mask="causal")   # bottom-right aligned
    n = n1 - a0
    blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
    h = finish(head, P, hyper[:, a0:], inject[:, a0:], blk)
    on_depth(1, h)
    ck, cv = [], []
    for j in range(2, D + 1):
        n = S - j - a0
        if n <= 4:
            break
        mixed, hyper, inject = layer.attn_hyper_connection(
            fuse(head, P, h[:, :n], emb(ids[:, a0 + j:a0 + j + n])).astype(mx.bfloat16))
        q, gate, k, v = qkv(att, P, mixed, a0 + mx.arange(n) + (j - 1))
        ck.append(k); cv.append(v)
        o = ttt_attn(att.scale, q, K1, V1, [c[:, :, :n] for c in ck], [c[:, :, :n] for c in cv], a0)
        blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate))
        h = finish(head, P, hyper, inject, blk)
        on_depth(j, h)


def logits(lm, head, h):
    return lm.lm_head(head.hyper_connection_mixer(h)).astype(mx.float32)


# ---------------------------------------------------------------- data
def window(key, folder=L120):
    w = np.load(os.path.join(TD, "d16w", key + ".npz"))
    assert int(w["off"]) == 0, key
    ids = np.asarray(w["ids"], np.int32)
    out = {"W": mx.array(w["wd"]).astype(mx.bfloat16)[None], "ids": ids}
    lf = os.path.join(folder, key + ".npz")
    if os.path.exists(lf):
        z = np.load(lf)
        n = min(len(ids), len(z["ids"]))
        assert (z["ids"][:n] == ids[:n]).all(), key
        out.update(tk=np.asarray(z["tk"], np.int32), lp=np.asarray(z["lp"], np.float32), asst=np.asarray(z["asst"], np.float32),
                   split=str(z["split"]))
    return out


_GEN = {}
def gen_of(key):
    """prompt length: the Eval suite's tf_gen records, Chat1000's agent turns"""
    if not _GEN:
        for l in open(H("~/EvalSuite/cache/ref/tf_gen.jsonl")):
            r = json.loads(l)
            _GEN[r["key"].replace("/", "_")] = len(r["prompt_ids"])
        for l in open(os.path.join(TD, "mgsm_items.jsonl")):
            r = json.loads(l)
            _GEN[r["key"]] = len(r["prompt_ids"])
        for f in glob.glob(H("~/Chat1000/data/agent125/*.npz")):
            _GEN["agent_" + os.path.basename(f)[:-4]] = int(np.load(f)["prompt"].shape[0])
    return _GEN[key]


# ---------------------------------------------------------------- loss
_CK = {}
def depth_kl(lm, head):
    if "f" not in _CK:
        def f(h, tk, lp, mask):
            tk, lp, mask = mx.stop_gradient(tk), mx.stop_gradient(lp), mx.stop_gradient(mask)
            lg = logits(lm, head, h)[0]
            lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
            return (M.kl_top(lq, lp) * mask).sum() / mx.maximum(mask.sum(), 1.0)
        _CK["f"] = mx.checkpoint(f)
    return _CK["f"]


def depth_kl_auf(lm, head):
    """depth_kl + the depth's argmax hits (detached) for AUF's alive mask"""
    if "fa" not in _CK:
        def f(h, tk, lp, mask, tgt):
            tk, lp, mask = mx.stop_gradient(tk), mx.stop_gradient(lp), mx.stop_gradient(mask)
            lg = logits(lm, head, h)[0]
            lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
            hit = mx.stop_gradient((mx.argmax(lg, -1) == tgt).astype(mx.float32))
            return (M.kl_top(lq, lp) * mask).sum() / mx.maximum(mask.sum(), 1.0), hit
        _CK["fa"] = mx.checkpoint(f)
    return _CK["fa"]


def first_asst(x):
    a = np.flatnonzero(x["asst"] > 0)
    return max(0, int(a[0]) - 2) if len(a) else 0


def ttt_loss(lm, head, P, x, D, wdecay, auf=False):
    """x = window(): the per-depth KL to the 125B's top-32 at row s + j (it predicts ids[s + j + 1]), on assistant
    targets; anchors from 2 before the first assistant token.
    auf (b5; Spec-AUF, arXiv 2607.01893, accept-until-fail): an anchor's loss stops at its chain's first predicted failure
    (depth j counts only while the head's argmax was right at every depth < j; the failing depth itself is kept), and
    depths are weighted by their kept share of depth 1's anchors, so the loss is a token-level mean over the kept support."""
    W, ids = x["W"], x["ids"]
    S, a0 = len(ids), first_asst(x)
    tk, lp, asst = mx.array(x["tk"]), mx.array(x["lp"]), mx.array(x["asst"])
    per, acc = [], {"tot": 0.0, "w": 0.0}
    alive, ida = [None], mx.array(ids)

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0, tk.shape[0] - j - a0)     # rows a0+i+j of tk / lp, ids[a0+i+j+1] exist
        if n <= 0:
            return
        m = asst[a0 + j + 1:a0 + j + 1 + n]
        w = wdecay ** (j - 1)
        if auf:
            if alive[0] is not None:
                m = m * alive[0][:n]
            l, hit = depth_kl_auf(lm, head)(h[:, :n], tk[a0 + j:a0 + j + n], lp[a0 + j:a0 + j + n], m,
                                            ida[a0 + j + 1:a0 + j + 1 + n])
            if j == 1:
                acc["m1"] = mx.maximum(m.sum(), 1.0)
            w = w * mx.stop_gradient(m.sum() / acc["m1"])
            alive[0] = hit if alive[0] is None else alive[0][:n] * hit
        else:
            l = depth_kl(lm, head)(h[:, :n], tk[a0 + j:a0 + j + n], lp[a0 + j:a0 + j + n], m)
        per.append(l); acc["tot"] = acc["tot"] + w * l; acc["w"] = acc["w"] + w
    chain(lm, head, P, W, mx.array(ids)[None], D, on_depth, a0)
    return acc["tot"] / mx.maximum(acc["w"], 1e-6) if auf else acc["tot"] / acc["w"], per


# ---------------------------------------------------------------- chain read-out
def chain_hits(lm, head, P, x, D, a0):
    """per depth j: (right, argmax) arrays over anchors s = a0..; right = the depth-j draft equals ids[s + j + 1]"""
    W, ids = x["W"], x["ids"]
    S = len(ids)
    out = []

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0)
        am = mx.argmax(logits(lm, head, h[:, :n])[0], -1)
        mx.eval(am)
        a = np.array(am)
        out.append((a == ids[a0 + j + 1:a0 + j + 1 + n], a))
    chain(lm, head, P, W, mx.array(ids)[None], D, on_depth, a0)
    return out


def runs(right, S, a0, K=15):
    """accepted chain length of every anchor s (consecutive right drafts from depth 1, capped at K), indexed by s"""
    k = np.zeros(S, np.int64)
    alive = np.ones(S, bool)
    alive[:a0] = False
    for j in range(min(K, len(right))):
        r = np.zeros(S, bool)
        r[a0:a0 + len(right[j][0])] = right[j][0]
        alive &= r
        k += alive
    return k


def walk(kk, gen, ids, end_ids, K):
    """D-17's walk over the answer (cut at its first end token): the pass at output position p uses anchor gen + p - 2"""
    out = ids[gen:]
    e = next((i for i, t in enumerate(out) if int(t) in end_ids), len(out) - 1)
    L, p, tok, passes = e + 1, 1, 0, 0
    while p < L:
        s = gen + p - 2
        k = min(int(kk[s]) if 0 <= s < len(kk) else 0, K, L - p)
        tok += k + 1; passes += 1; p += k + 1
    return tok, passes


# ---------------------------------------------------------------- commands
def load_head(a):
    rt, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    if a.run and a.run != "stock":
        P.load_weights(os.path.join(CK, f"d22_{a.run}.safetensors"))
        print(f"[d22] loaded ckpt d22_{a.run}", flush=True)
    mx.eval(P.parameters())
    return rt, lm, head, P


def end_tokens():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(H("~/BigMoE/models/bare/tokenizer.json"))
    return {tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}


def cmd_gate(a):
    rt, lm, head, P = load_head(a)
    agree = np.zeros((2, 15)); tot = np.zeros((2, 15))
    for f in sorted(glob.glob(os.path.join(TD, "d17", "tf_code_*.npz")) + glob.glob(os.path.join(TD, "d17", "tf_gsm_*.npz"))):
        key = os.path.basename(f)[:-4]
        if not os.path.exists(os.path.join(TD, "d16w", key + ".npz")):
            continue
        z = np.load(f)
        x = window(key)
        gen, out, mtp = int(z["prompt"].shape[0]), [int(t) for t in z["out"]], z["mtp"]
        assert [int(t) for t in x["ids"][:gen]] == [int(t) for t in z["prompt"]], key
        full = [int(t) for t in x["ids"][gen:]]                  # the suite's stored answer; D-17's decode may part
        n = min(len(out), len(full))                             # from it: compare over the common prefix only
        n = next((i for i in range(n) if full[i] != out[i]), n)
        a0 = max(0, gen - 2)
        right = chain_hits(lm, head, P, x, 15, a0)
        dom = 0 if "code" in key else 1
        for p in range(1, n):
            i = gen + p - 2 - a0                                     # row of anchor s = gen + p - 2
            for d in range(1, 16):
                if mtp[p][d - 1] < 0 or p + d - 1 >= n or d - 1 >= len(right) or i >= len(right[d - 1][1]):
                    break
                tot[dom, d - 1] += 1
                agree[dom, d - 1] += int(right[d - 1][1][i] == mtp[p][d - 1])
                if mtp[p][d - 1] != out[p + d - 1]:          # the real chain left the true path: its next inputs differ
                    break
        print(f"[d22] gate {key}: depth-1 agreement so far {agree[dom, 0] / max(tot[dom, 0], 1):.4f}", flush=True)
    lines = ["D-22 gate: my training-time-test chain (untrained head) vs MTPLX's real chain (D-17), argmax agreement per",
             "depth on positions where the real chain's earlier drafts were right (n per depth in brackets)"]
    for dom, nm in ((0, "code"), (1, "gsm")):
        lines.append(f"  {nm}: " + " ".join(f"{agree[dom, d] / max(tot[dom, d], 1):.3f}[{int(tot[dom, d])}]" for d in range(15)))
    a17 = agree[:, :7].sum() / max(tot[:, :7].sum(), 1)
    lines.append(f"  depth 1 {agree[:, 0].sum() / max(tot[:, 0].sum(), 1):.4f} (gate >= 0.98); depths 1-7 {a17:.4f} (gate >= 0.95)")
    print("\n".join(lines), flush=True)
    open(os.path.join(TD, "d22_gate.txt"), "w").write("\n".join(lines) + "\n")


def cmd_eval(a):
    rt, lm, head, P = load_head(a)
    end_ids = end_tokens()
    pat = {"code": "tf_code_HumanEval_*", "gsm": "tf_gsm_*", "ifeval": "tf_ifeval_*", "multi": "tf_multi_*", "agent": "agent_*",
           "mgsm": "mgsm_*"}
    res, KK, lines = {}, {}, [f"D-22 read-out, head '{a.run}': the chain from the true prefix at every position of the held-out "
                      f"answers, walked as D-17 (tokens per pass)"]
    for st in a.sets.split(","):
        rows = []
        for f in sorted(glob.glob(os.path.join(TD, "d16w", pat[st] + ".npz"))):
            while os.path.exists(PAUSE):
                time.sleep(1)
            key = os.path.basename(f)[:-4]
            x = window(key)
            gen = gen_of(key)
            a0 = max(0, gen - 2)
            right = chain_hits(lm, head, P, x, 15, a0)
            kk = runs(right, len(x["ids"]), a0)
            r = [walk(kk, gen, x["ids"], end_ids, K) for K in (3, 7, 15)]
            hits = [float(right[d][0].mean()) if d < len(right) else np.nan for d in range(3)]   # short answers: the chain stops early
            rows.append((key, *[v for t in r for v in t], *hits))
            KK[key] = kk
        res[st] = rows
        A = np.array([r[1:7] for r in rows], float)
        lines.append(f"  {st} ({len(rows)}): " + ", ".join(f"K{K} {A[:, 2 * i].sum() / A[:, 2 * i + 1].sum():.2f}"
                                                          for i, K in enumerate((3, 7, 15)))
                     + "; depth 1-3 hit (teacher-forced, every anchor) " + " ".join(f"{np.nanmean([r[7 + d] for r in rows]):.3f}" for d in range(3)))
        print(lines[-1], flush=True)
    np.savez(os.path.join(TD, f"d22_eval_{a.run}.npz"), **{st: np.array([r[1:] for r in rows], float) for st, rows in res.items()},
             **{st + "_keys": np.array([r[0] for r in rows]) for st, rows in res.items()})
    np.savez(os.path.join(TD, f"d22_runs_{a.run}.npz"), **KK)    # accepted run per anchor, for D-19-style comparisons
    open(os.path.join(TD, f"d22_eval_{a.run}.txt"), "w").write("\n".join(lines) + "\n")


def cmd_loss(a):
    """Paired screen: per-window, per-depth KL (forward only) of head --run on (1) the first --n windows of training
    run --of's first epoch, in its order (each was unseen when its step began), (2) the 10 validation windows."""
    rt, lm, head, P = load_head(a)
    keys = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(L120, "*.npz")))
    split = {k: str(np.load(os.path.join(L120, k + ".npz"))["split"]) for k in keys}
    tr = [k for k in keys if split[k] == "train"]
    seed = json.load(open(os.path.join(CK, f"d22_{a.of}.json")))["args"]["seed"]
    perm = [int(i) for i in np.random.default_rng(seed).permutation(len(tr))]
    order = [tr[perm[-1 - i]] for i in range(min(a.n, len(tr)))]             # cmd_train pops from the end
    out = {}
    for name, ks in (("order", order), ("val", [k for k in keys if split[k] == "val"])):
        rows = []
        for k in ks:
            while os.path.exists(PAUSE):
                time.sleep(1)
            _, per = ttt_loss(lm, head, P, window(k), a.depth, a.wdecay)
            mx.eval(per)
            rows.append([float(v) for v in per] + [float("nan")] * (a.depth - len(per)))
        out[name] = np.array(rows)
        print(f"[d22] loss '{a.run}' on {name} ({len(ks)}): per depth " + " ".join(f"{v:.4f}" for v in np.nanmean(out[name], 0)),
              flush=True)
    np.savez(os.path.join(TD, f"d22_loss_{a.run}.npz"), **out, order_keys=np.array(order))


def cmd_train(a):
    t0 = time.time()
    limit = float(os.environ.get("TL_LIMIT", "1e9")) - 120
    rt, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    os.makedirs(CK, exist_ok=True)
    ckw, ckj = os.path.join(CK, f"d22_{a.run}.safetensors"), os.path.join(CK, f"d22_{a.run}.json")
    keys = []                                                    # (key, folder): --data l120[,d22c]
    for nm in a.data.split(","):
        fd = {"l120": L120, "d22c": os.path.join(TD, "d22c")}[nm]
        for f in sorted(glob.glob(os.path.join(fd, "*.npz"))):
            k = os.path.basename(f)[:-4]
            if str(np.load(f)["split"]) == "train" and os.path.exists(os.path.join(TD, "d16w", k + ".npz")):
                keys.append((k, fd))
    state = {"step": 0, "epoch": 0, "order": [], "loss": [], "per": [], "args": vars(a), "status": "running"}
    if os.path.exists(ckj):
        state = json.load(open(ckj)); P.load_weights(ckw)
        print(f"[d22] resume at step {state['step']} (epoch {state['epoch']}); Adam moments restart", flush=True)
    P.freeze(); P.unfreeze()
    for nm in filter(None, a.freeze.split(",")):                 # e.g. --freeze router: keep the expert choice stock
        getattr(P, nm).freeze()
    steps = int(a.epochs * len(keys))
    s0 = state["step"]
    if a.sched == "cos":                                         # warm-up 20, then cosine to 10 % at the last step
        opt = optim.Adam(learning_rate=lambda s: a.lr * mx.minimum(1.0, (s + 1) / 20.0)
                         * (0.1 + 0.45 * (1 + mx.cos(np.pi * mx.minimum(1.0, s / max(1, steps - 1))))))
    else:
        opt = optim.Adam(learning_rate=lambda s: a.lr * mx.minimum(1.0, (s + 1) / 20.0))
    if state["step"]:                                            # a resume continues the schedule (moments restart)
        opt.state["step"] = mx.array(state["step"], mx.uint64)
    cur = {}
    lossf = nn.value_and_grad(P, lambda: ttt_loss(lm, head, P, cur["x"], a.depth, a.wdecay, a.auf)[0])
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    rng = np.random.default_rng(a.seed + state["epoch"])
    ntr = sum(v.size for _, v in tree_flatten(P.trainable_parameters()))
    print(f"[d22] train {a.run}: {len(keys)} windows, {steps} steps, depth {a.depth}, lr {a.lr}, wdecay {a.wdecay}, auf {a.auf}, "
          f"{ntr / 1e6:.1f}M trainable; loaded in {time.time() - t0:.0f}s", flush=True)

    def save(status):
        mx.save_safetensors(ckw + ".tmp.safetensors", dict(tree_flatten(P.parameters())))
        os.replace(ckw + ".tmp.safetensors", ckw)
        state["status"] = status
        json.dump(state, open(ckj + ".tmp", "w")); os.replace(ckj + ".tmp", ckj)
    last = time.time()
    while state["step"] < steps:
        while os.path.exists(PAUSE):
            time.sleep(1)
        if not state["order"]:
            state["order"] = [int(i) for i in rng.permutation(len(keys))]; state["epoch"] += 1
        cur["x"] = window(*keys[state["order"].pop()])
        t1 = time.time()
        loss, g = lossf()
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(P, g)
        mx.eval(P.parameters(), opt.state, loss)
        state["step"] += 1; state["loss"].append(float(loss))
        if state["step"] % 10 == 0 or state["step"] == s0 + 1:
            _, per = ttt_loss(lm, head, P, cur["x"], a.depth, a.wdecay)          # the full-support per-depth KL, as before
            per = [float(v) for v in per]; state["per"].append([state["step"]] + per)
            print(f"[d22] step {state['step']}/{steps} loss {float(loss):.4f} (mean of last 10 "
                  f"{np.mean(state['loss'][-10:]):.4f}); this window per depth " + " ".join(f"{v:.3f}" for v in per)
                  + f"; {time.time() - t1:.1f}s/step, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        if stop["now"] or time.time() - t0 > limit:
            save("stopped"); print(f"[d22] stopped at step {state['step']}: checkpoint saved; resume with the same command", flush=True)
            sys.exit(3)
        if time.time() - last > 300:
            save("running"); last = time.time()
    save("done")
    print(f"[d22] DONE {a.run}: {state['step']} steps in {time.time() - t0:.0f}s", flush=True)


def main():
    M.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--run", default="")
    ap.add_argument("--frac", type=float, default=0.03)       # trunk experts resident: the minimum (the trunk is unused)
    ap.add_argument("--depth", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wdecay", type=float, default=0.8); ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sets", default="code,gsm")
    ap.add_argument("--of", default="base"); ap.add_argument("--n", type=int, default=76)
    ap.add_argument("--freeze", default=""); ap.add_argument("--sched", default="const")
    ap.add_argument("--data", default="l120")
    ap.add_argument("--auf", action="store_true")                # b5: Spec-AUF's accept-until-fail loss support
    a = ap.parse_args()
    {"gate": cmd_gate, "train": cmd_train, "eval": cmd_eval, "loss": cmd_loss}[a.cmd](a)


if __name__ == "__main__":
    main()
