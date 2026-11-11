"""F2a step 4 (MTP Head; GPU ~10 GB, ~3 min): the new chat head vs b5 on held-out chat, by the pre-registered line
(REPORT 'F2A night: pre-registration'). Teacher-forced chain (D-24's top_ranks), depths 1-6, kept per check with the
shipped cut (t 0.3, K6), tok/s [P] with chat's fitted check cost (CHAT150: 47.0 + 3.25 x rows + 0.43 + 2.59 x steps ms):
  heldout  D-24's held-out answers (30 IFEval + 24 multi, data/d16w; not in b5's or F2a's training)
  onpolicy the 125B's own greedy IFEval answers (data/f2a/eval, captured in the night's phase 1)
Usage: [BigMoE venv] python3 -I sticky/f2a_eval.py [run=f2a] -> data/f2a/f2a_eval.txt"""
import os, sys, glob, json
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/TrainLab_sticky"))
import d24_mtptop3 as D24
T = D24.T
M = T.M
import numpy as np
import mlx.core as mx

RUN = sys.argv[1] if len(sys.argv) > 1 else "f2a"
CK = os.path.expanduser("~/TrainLab/ckpt")
EV = os.path.expanduser("~/TrainLab/data/f2a/eval")
OUT = os.path.expanduser(f"~/TrainLab/data/f2a/{RUN}_eval.txt")   # f2a -> f2a_eval.txt (the pre-registered verdict)
D24.D = 6
L_ = []


def say(s):
    print(s, flush=True); L_.append(s)


def load_w(path):
    z = np.load(path)
    W = mx.array((z["wd"].astype(np.uint32) << 16).view(np.float32)).astype(mx.bfloat16)[None]
    return {"W": W, "ids": np.asarray(z["ids"], np.int32), "asst": np.asarray(z["asst"], np.float32)}


def kept_cut(p1, rk, gen, n_ids, t=0.3, K=6):
    a0 = max(0, gen - 2); Lw = n_ids - gen; pos = 1; T_ = C_ = R_ = S_ = 0
    while pos < Lw:
        s = gen + pos - 2 - a0
        if 0 <= s < len(rk) and (rk[s] >= 0).any():
            r_, q_ = rk[s], p1[s]; L = int((r_ >= 0).sum())
            miss = np.where(r_[:L] != 0)[0]; kstar = int(miss[0]) + 1 if len(miss) else L + 1
            m = int((np.cumprod(np.nan_to_num(q_[:min(K, L)], nan=0.0)) >= t).sum())
            k, rows, st = min(kstar, m + 1), m + 1, min(m + 1, min(K, L))
        else:
            k, rows, st = 1, 1, 0
        k = max(1, min(k, Lw - pos)); T_ += k; C_ += 1; R_ += rows; S_ += st; pos += k
    return T_, C_, R_, S_


M.limit_memory()
rt, lm, head = M.load(0.03)
heads = {}
names = ["b5"] + (["f2a"] if RUN != "f2a" and os.path.exists(os.path.join(CK, "d22_f2a.safetensors")) else []) + [RUN]
for nm in names:                                                  # stage 2 (amendment 2): b5, f2a, f2a2
    P = M.Trainable(head); P.load_weights(os.path.join(CK, f"d22_{nm}.safetensors")); mx.eval(P.parameters()); heads[nm] = P
sets = {"heldout ifeval": [("d16w", os.path.basename(f)[:-4]) for f in sorted(glob.glob(os.path.join(T.TD, "d16w", D24.PAT["ifeval"] + ".npz")))],
        "heldout multi": [("d16w", os.path.basename(f)[:-4]) for f in sorted(glob.glob(os.path.join(T.TD, "d16w", D24.PAT["multi"] + ".npz")))],
        "onpolicy ifeval": [("eval", f) for f in sorted(glob.glob(os.path.join(EV, "e_*.npz")))]}
res = {}
for st, items in sets.items():
    if not items:                                                 # e.g. the on-policy windows were not captured
        say(f"{st:16s}: no windows"); continue
    for nm, P in heads.items():
        hit = np.zeros(6); tot = np.zeros(6); TT = CC = RR = SS = 0
        for kind, k in items:
            if kind == "d16w":
                x = T.window(k); gen = T.gen_of(k)
            else:
                x = load_w(k); gen = int(np.argmax(x["asst"] > 0))
            a0 = max(0, gen - 2)
            r = D24.top_ranks(lm, head, P, x, a0)
            n = len(x["ids"]) - 1 - a0
            p1 = np.full((n, 6), np.nan, np.float32); rk = np.full((n, 6), -1, np.int8)
            for j, (p, kk) in enumerate(r[:6]):
                p1[:len(p), j] = p; rk[:len(kk), j] = kk
                v = kk >= 0; hit[j] += (kk[v] == 0).sum(); tot[j] += v.sum()
            a, b, c, d = kept_cut(p1, rk, gen, len(x["ids"]))
            TT += a; CC += b; RR += c; SS += d
        ms = 47.0 * CC + 3.25 * RR + 0.43 * CC + 2.59 * SS
        res[(st, nm)] = (TT / CC, TT / ms * 1e3)
        say(f"{st:16s} {nm:4s}: depth hits " + " ".join(f"{h / max(t, 1):.3f}" for h, t in zip(hit, tot)) +
            f" | kept/check {TT / CC:.3f} (rows {RR / CC:.2f}) | tok/s [P] {TT / ms * 1e3:.1f} | n {len(items)}")
ho = [(res[(s, RUN)][0] / res[(s, "b5")][0]) for s in ("heldout ifeval", "heldout multi")]
pooled = float(np.sqrt(ho[0] * ho[1]))
onp = res[("onpolicy ifeval", RUN)][0] / res[("onpolicy ifeval", "b5")][0] if ("onpolicy ifeval", "b5") in res else float("nan")
v = "PASS" if pooled >= 1.08 and onp >= 1.05 else ("KILL" if pooled < 1.03 else "BETWEEN")
say(f"Verdict F2a ({RUN} vs b5; pre-registered): {v}: held-out kept x{pooled:.3f} (geo of IFEval x{ho[0]:.3f}, multi x{ho[1]:.3f}), "
    f"on-policy IFEval kept x{onp:.3f} (PASS >= x1.08 AND on-policy >= x1.05; KILL < x1.03)")
if "f2a" in heads and RUN != "f2a":
    hs = [res[(s, RUN)][0] / res[(s, "f2a")][0] for s in ("heldout ifeval", "heldout multi")]
    g2 = float(np.sqrt(hs[0] * hs[1]))
    o2 = res[("onpolicy ifeval", RUN)][0] / res[("onpolicy ifeval", "f2a")][0] if ("onpolicy ifeval", "f2a") in res else float("nan")
    rd = "more data pays" if g2 >= 1.02 else ("flat" if g2 < 1.005 else "small")
    say(f"Data scaling ({RUN} = f2a + 2 epochs on ring 2, vs f2a; amendment 2): {rd}: held-out kept x{g2:.3f} "
        f"(IFEval x{hs[0]:.3f}, multi x{hs[1]:.3f}), on-policy x{o2:.3f} (more data pays >= x1.02; flat < x1.005)")
open(OUT, "w").write("\n".join(L_) + "\n")
