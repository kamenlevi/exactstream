"""WEEK98 CHAIN SIZE (orchestrator 01:3x, after Leap's KNEE probe 4be8cf9): how small can the 2B's chain head be and keep
its deep accuracy? Leap: weights <= ~8-12 MB per step stay in the GPU cache (1.1-3.2 TB/s, ~1 ms per 30 steps); 16-48 MB
read at ~500 GB/s (~3-5 ms per 30 steps). The head as it is: 60.8M params (34 MB at 4-bit) + the tied lm_head over
248,320 tokens (286 MB at 4-bit) per step.
Arms, all from s30o, trained the same steps in one process (train_multi's Arm: s30o's Adam, pool2, cosine re-anneal):
  ctl    the full head
  ffnK   the FFN cut to K neurons (6144 -> 3072 / 1536): neurons ranked by output energy E[a_i^2] ||down_i||^2 on the
         head's own chain inputs (4 pool2 files), down_proj refit by ridge least squares to the full FFN's output, the
         Adam moments sliced to the kept neurons
Eval-only, on each trained arm: 4-bit weights (group 64, post-training) and lm_head shortlists (the K most frequent
answer tokens of the training captures; the argmax is taken over the list). Plus the pruned+refit heads BEFORE training
(1/2, 1/4, 1/8; MBPP only): the damage and what the refit keeps.
Read-out: the teacher-forced chain on MBPP-50 / HE-30 (bands 0-0 and 23-30, kept/32), paired against the control at the
same precision and list. MB per chain step at 4-bit next to each arm. No checkpoints are written (in-process eval); on a
SIGTERM (the guard's yield) the arms + Adam are saved once and the rerun resumes (sb/resume.py).
LINE (orchestrator 01:3x, written before any run): a head of <= 48 MB per step (better <= 12 MB) that holds MBPP-50 23-30
within ~1 pt of the full head at matched steps -> the cache-resident chain is a live GOAL A form; a large drop -> block
drafting stays the route.
usage: chainsize.py [--steps 200] [--keeps 3072,1536] [--pre 3072,1536,768] [--lists 32768,16384,8192,4096]"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sb import core as C
from sb import resume as R
from sb.compare import paired
from sb.train import split
from sb.train_multi import Arm

V = 248320
GS = {"G": None, "n": 0}


class Rec(nn.Module):
    """wraps down_proj: accumulates the Gram matrix of its input (the FFN activations) on the chain's own rows"""
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def __call__(self, x):
        a = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        GS["G"] = a.T @ a if GS["G"] is None else GS["G"] + a.T @ a
        GS["n"] += a.shape[0]
        mx.eval(GS["G"])
        return self.lin(x)


def ffn_stats(D, files):
    lin = D.down_proj
    D.down_proj = Rec(lin)
    try:
        for f in files:
            C.chain_table(D, f)
    finally:
        D.down_proj = lin
    return GS["G"], GS["n"]


def prune(D, keep, G):
    """cut the FFN to the top-`keep` neurons by output energy; refit down_proj (ridge LS to the full output) -> S, errors"""
    dw = D.down_proj.weight                                  # [d, ffn]
    en = np.array(mx.diagonal(G) * (dw * dw).sum(0))
    S = np.sort(np.argsort(-en)[:keep]); Sm = mx.array(S)
    GSS = G[Sm][:, Sm]; GSf = G[Sm]
    lam = 1e-3 * float(mx.mean(mx.diagonal(GSS)))
    rhs = GSf @ dw.T                                         # [k, d]
    Wt = mx.linalg.solve(GSS + lam * mx.eye(keep), rhs, stream=mx.cpu)
    tot = float(mx.sum(dw * (dw @ G)))                       # E||Y||^2 (x n)

    def err(Wk):                                             # E||Y - A_S Wk||^2 / E||Y||^2, Wk [k, d]
        return (tot - 2 * float(mx.sum(Wk * rhs)) + float(mx.sum(Wk * (GSS @ Wk)))) / tot
    e_trunc, e_refit = err(dw[:, Sm].T), err(Wt)
    g = nn.Linear(D.d, keep, bias=False); g.weight = D.gate_proj.weight[Sm]
    u = nn.Linear(D.d, keep, bias=False); u.weight = D.up_proj.weight[Sm]
    dn = nn.Linear(keep, D.d, bias=False); dn.weight = Wt.T
    D.gate_proj, D.up_proj, D.down_proj = g, u, dn
    mx.eval(D.parameters())
    return S, e_trunc, e_refit


def shape_ffn(D, keep):                                      # the pruned shapes, for a resume (weights come from the state)
    D.gate_proj = nn.Linear(D.d, keep, bias=False); D.up_proj = nn.Linear(D.d, keep, bias=False)
    D.down_proj = nn.Linear(keep, D.d, bias=False)


def mb_per_step(D, K, bits=4):
    """bytes per chain step with the body at `bits` (group 64: bits/8 B per weight + an fp16 scale and bias per 64), norms
    fp16, and the lm_head over K tokens at 4-bit as the target stores it (0.5625 B per weight)"""
    b = 0.0
    for _, m in D.named_modules():
        if isinstance(m, (nn.Linear, nn.QuantizedLinear)):
            o, i = m.weight.shape
            i = i * 32 // m.bits if isinstance(m, nn.QuantizedLinear) else i
            b += o * i * (bits / 8 + 4 / 64)
        elif isinstance(m, nn.RMSNorm):
            b += m.weight.size * 2
    return (b + K * D.d * 0.5625) / 1e6


def shortlist_masks(lists):
    """the K most frequent next tokens on the training captures' answer rows (base + pool + mc2; no eval rows)"""
    cp = os.path.join(C.DATA, "shortlist_counts.npy")      # sb/shortlist.py: Magicoder python (19.6M tokens) + captures
    if os.path.exists(cp):
        cnt = np.load(cp)
    else:
        s = json.load(open(os.path.join(C.DATA, "split.json")))
        fs = [os.path.join(C.DATA, "mc", k + ".npz") for k in s["base"] + s["pool"]] + \
             [os.path.join(C.DATA, "mc2", k + ".npz") for k in s["mc2"]]
        cnt = np.zeros(V, np.int64)
        for f in fs:
            z = np.load(f); ids = np.asarray(z["ids"]); gen = int(z["gen"])
            np.add.at(cnt, ids[gen:], 1)
    order = np.argsort(-cnt, kind="stable")
    masks, lists_ = {"full": None}, {}
    for K in lists:
        mk = np.full(V, -1e9, np.float32); mk[order[:K]] = 0.0
        masks[f"L{K // 1024}K"] = mx.array(mk); lists_[f"L{K // 1024}K"] = set(order[:K].tolist())
    return masks, lists_, int(cnt.sum())


def chain_multi(D, f, masks, Dp=31):
    """C.chain_table with several argmax rules from one chain pass: masks {name: None (full vocab) | additive [V]}"""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"]); W = C.load_W(z)
    a0 = max(0, gen - 2); S = len(ids); n = S - 1 - a0
    rts = {k: np.full((n, Dp), -1, np.int8) for k in masks}
    C.REPL.update(R=None, K=1, fn=None)
    C.CUR["f"], C.CUR["train"], C.CUR["x"] = f, False, None
    hs = []

    def on_depth(j, h, anc):
        m = min(h.shape[1], S - 1 - j - a0)
        if m > 0:
            hs.append((j, m, D.norm(h[0, :m]).astype(mx.bfloat16)))
    C.chain(D, W, mx.array(ids), Dp, on_depth, a0)
    Hs = mx.concatenate([x for _, _, x in hs], 0)
    ams = {k: [] for k in masks}
    for c0 in range(0, Hs.shape[0], 2048):
        lg = C.lm_logits(Hs[c0:c0 + 2048])
        for k, mk in masks.items():
            ams[k].append(mx.argmax(lg if mk is None else lg + mk, -1))
        mx.eval([v[-1] for v in ams.values()])
    for k in masks:
        am = np.array(mx.concatenate(ams[k])); o = 0
        for j, m, _ in hs:
            a = am[o:o + m]; o += m
            rts[k][:m, j - 1] = (a == ids[a0 + j + 1:a0 + j + 1 + m]).astype(np.int8)
    return rts, gen, a0


def evaluate(D, sets, masks):
    """-> {set: {mask: summarize(...)}}"""
    out = {}
    for st, fs in sets.items():
        tabs = {k: [] for k in masks}
        for f in fs:
            C.wait_pause()
            rts, gen, a0 = chain_multi(D, f, masks)
            for k in masks:
                tabs[k].append((os.path.basename(f)[:-4], rts[k], gen, a0))
            mx.clear_cache()
        out[st] = {k: C.summarize(v) for k, v in tabs.items()}
    return out


def brief(s):
    r = s[0]
    return {"0-0": round(100 * r["0-0"], 2), "23-30": round(100 * r["23-30"], 2), "kept32": round(r["kept_per_32"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--rows", type=int, default=96); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--keeps", default="3072,1536"); ap.add_argument("--pre", default="3072,1536,768")
    ap.add_argument("--lists", default="32768,16384,8192,4096"); ap.add_argument("--nstat", type=int, default=4)
    ap.add_argument("--nm", type=int, default=50); ap.add_argument("--nh", type=int, default=30)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--mem", type=float, default=7.0)
    ap.add_argument("--precs", default="fp32,q8,q4")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.nm, a.nh, a.nstat, a.pre = 3, 2, 2, 1, "768"
    tag = "chainsize" + ("_smoke" if a.smoke else "")
    C.limit_memory(a.mem)
    t0 = time.time()
    C.load_target(); C.drop_trunk()
    sets = {"mbpp": C.files("mbpp")[:a.nm], "he": C.files("he")[:a.nh]}
    masks, lists_, ntok = shortlist_masks([int(k) for k in a.lists.split(",") if k])
    cover = {}
    for st, fs in sets.items():
        tot = 0; inl = {k: 0 for k in lists_}
        for f in fs:
            z = np.load(f); ids = np.asarray(z["ids"]); tgt = ids[int(z["gen"]):]
            tot += len(tgt)
            for k, L in lists_.items():
                inl[k] += int(sum(1 for t in tgt.tolist() if t in L))
        cover[st] = {k: round(100 * v / tot, 3) for k, v in inl.items()}
    print(f"[cs] shortlists from {ntok} training answer tokens; eval tokens covered (%): {json.dumps(cover)}", flush=True)
    sp = os.path.join(C.CKPT, tag + "_state")
    out = {"args": vars(a), "cover": cover, "pre": {}, "post": {}, "mb": {}, "prune": {}}
    meta = None
    if os.path.exists(sp + ".safetensors"):
        meta = json.loads(mx.load(sp + ".safetensors", return_metadata=True)[1]["meta"])
        out = meta["out"]
        print(f"[cs] resuming at step {meta['i']} (pre-training reads kept)", flush=True)
    keeps = [int(k) for k in a.keeps.split(",") if k]
    ta = argparse.Namespace(start="s30o", resume_opt="s30o", const=False, lr=a.lr, warmup=20, steps=a.steps,
                            depth=a.depth, wdecay=0.97)
    tr, _ = split("pool2")
    sl = C.pass_slices(tr, a.rows, 1115)
    a.steps = ta.steps = min(a.steps, len(sl))
    if meta is None:
        D0 = C.load_drafter("s30o")
        G, nrow = ffn_stats(D0, tr[:a.nstat])
        print(f"[cs] FFN Gram on {nrow} chain rows of {a.nstat} pool2 files, {time.time() - t0:.0f}s", flush=True)
        for k in [int(x) for x in a.pre.split(",") if x]:     # pruned + refit heads before any training (MBPP only)
            Dk = C.load_drafter("s30o")
            S, et, er = prune(Dk, k, G)
            ev = evaluate(Dk, {"mbpp": sets["mbpp"]}, {"full": None})
            out["pre"][f"ffn{k}"] = {"err_trunc": round(et, 4), "err_refit": round(er, 4), **brief(ev["mbpp"]["full"])}
            print(f"[cs] pre ffn{k}: output error trunc {et:.4f} refit {er:.4f}; MBPP-50 {json.dumps(out['pre'][f'ffn{k}'])}", flush=True)
            del Dk; mx.clear_cache()
        out["prune"]["rows"] = nrow
    arms = {"ctl": Arm("ctl", "-", ta)}
    for k in keeps:
        A_ = Arm(f"ffn{k}", "-", ta)
        if meta is None:
            S, _, _ = prune(A_.D, k, G)
            Sm = mx.array(S)
            st = A_.opt.state
            for nm_ in ("gate_proj", "up_proj"):
                for mv in ("m", "v"):
                    st[nm_]["weight"][mv] = st[nm_]["weight"][mv][Sm]
            for mv in ("m", "v"):
                st["down_proj"]["weight"][mv] = st["down_proj"]["weight"][mv][:, Sm]
        else:
            shape_ffn(A_.D, k)
        arms[f"ffn{k}"] = A_
    i0 = 0
    if meta is not None:
        R.load(sp, {k: v.D for k, v in arms.items()}, {k: v.opt for k, v in arms.items()})
        for k, v in arms.items():
            v.log = meta["logs"][k]
        i0 = meta["i"]
    else:
        G = None; GS["G"] = None; mx.clear_cache()
    R.arm_sigterm()
    t1 = time.time()
    print(f"[cs] arms {list(arms)}: {a.steps} steps x depth {a.depth} on pool2 from s30o (Adam continued), from step {i0}", flush=True)
    for i in range(i0, a.steps):
        C.wait_pause()
        f, b, r = sl[i]
        x = C.window_at(f, b, r, a.depth)
        for m in arms.values():
            m.step(x, ta)
        if (i + 1) % 20 == 0 or i == i0:
            print(f"[cs] step {i + 1}/{a.steps} " + " ".join(f"{k} {np.mean(m.log['loss'][-20:]):.4f}" for k, m in arms.items())
                  + f"; {(time.time() - t1) / (i + 1 - i0):.2f}s/step, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        if R.STOP["f"]:
            wrote = R.save(sp, {k: v.D for k, v in arms.items()}, {k: v.opt for k, v in arms.items()},
                           {"i": i + 1, "out": out, "logs": {k: v.log for k, v in arms.items()}})
            print(f"[cs] SIGTERM: state saved at step {i + 1} ({wrote / 1e9:.2f} GB), exiting to resume later", flush=True)
            sys.exit(75)
    R.finish(sp)
    print(f"[cs] trained in {time.time() - t1:.0f}s; evaluating", flush=True)
    for k, m in arms.items():
        for bits in (4, 8):
            out["mb"][f"{k}|q{bits}"] = {lk: round(mb_per_step(m.D, V if lk == "full" else int(lk[1:-1]) * 1024, bits), 1)
                                        for lk in masks}
        for prec in a.precs.split(","):
            D = m.D
            if prec != "fp32":
                D = copy.deepcopy(m.D)
                nn.quantize(D, group_size=64, bits=int(prec[1:]), class_predicate=lambda p, mod: isinstance(mod, nn.Linear))
            out["post"][f"{k}|{prec}"] = evaluate(D, sets, masks)
            r = out["post"][f"{k}|{prec}"]
            print(f"[cs] {k}|{prec}: " + " | ".join(f"{st} {lk} 0-0 {100 * r[st][lk][0]['0-0']:.2f} 23-30 {100 * r[st][lk][0]['23-30']:.2f} "
                  f"kept {r[st][lk][0]['kept_per_32']:.2f}" for st in sets for lk in ("full", "L16K", "L8K")) + f"; MB/step q4 {out['mb'][k + '|q4']}", flush=True)
    rng = np.random.default_rng(1114)
    res = {"cover": out["cover"], "pre": out["pre"], "mb": out["mb"], "arms": {}}
    for key, ev in out["post"].items():
        res["arms"][key] = {}
        for st in sets:
            for lk in masks:
                s_, per = ev[st][lk]
                c_, pc = out["post"]["ctl|fp32"][st]["full"]
                row = {"0-0": round(100 * s_["0-0"], 2), "23-30": round(100 * s_["23-30"], 2), "kept32": round(s_["kept_per_32"], 3)}
                for b in ("23-30", "0-0"):
                    d = paired(pc, per, b, rng)
                    row[f"d{b}"] = round(100 * d["diff"], 2); row[f"ci{b}"] = [round(100 * x, 2) for x in d["ci95"]]
                    row[f"p{b}"] = round(d["p_sign"], 4)
                res["arms"][key][f"{st}|{lk}"] = row
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
    print(f"[cs] DONE in {res['wall_s']:.0f}s -> results/{tag}.json", flush=True)


if __name__ == "__main__":
    main()
