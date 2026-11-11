"""D1 variants (MTP Head, Sun 11-08 00:xx; CHAT150 strike): does a cheaper draft step keep the chat head's hits?
Teacher-forced chain (D-24's top_ranks / d22_ttt.chain, head b5) on the held-out IFEval (30) + multi (24) answers, depths
1-6, for draft-step variants that change the drafts (chat outputs never change: chat verifies exactly):
  out4      as shipped: the 4-bit 248K output layer
  out3/out2 the output layer re-quantized to 3 / 2 bits (MTPLX's draft output layer idea)
  p4        the head's trained parts (fc, attention, router, shared expert; bf16 in the engine) at 4 bits, output 4-bit
  p4out3    both
Saves per set and variant: depth-1..6 hit, kept per check with the shipped cut (t 0.3) -> data/d1_variants.txt.
Shared GPU, peak ~9-11 GB (as D-24)."""
import os, sys, glob, time, copy
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/TrainLab_sticky"))
import d24_mtptop3 as D24                                   # imports d22_ttt as T (MTPLX env), mlx
T = D24.T
import numpy as np
import mlx.core as mx
import mlx.nn as nn

D24.D = 6
OUT = os.path.expanduser("~/TrainLab/data/d1_variants.txt")
L_ = []


def say(s):
    print(s, flush=True); L_.append(s)


class A:
    run = "b5"; frac = 0.03


T.M.limit_memory()
rt, lm, head, P = T.load_head(A)
q = lm.lm_head
W4 = (q.weight, q.scales, q.biases)
wd = mx.dequantize(q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits)
heads = {4: lm.lm_head}
for b in (3, 2):
    w, s, bb = mx.quantize(wd, group_size=64, bits=b)
    ql = nn.QuantizedLinear(wd.shape[1], wd.shape[0], bias=False, group_size=64, bits=b)
    ql.weight, ql.scales, ql.biases = w, s, bb
    mx.eval(ql.parameters())
    heads[b] = ql
del wd
P4 = copy.deepcopy(P)
nn.quantize(P4, group_size=64, bits=4, class_predicate=lambda p_, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
mx.eval(P4.parameters())
VAR = [("out4", P, 4), ("out3", P, 3), ("out2", P, 2), ("p4", P4, 4), ("p4out3", P4, 3)]


def kept_cut(p1, rk, gen, n_ids, t=0.3, K=6):
    a0 = max(0, gen - 2); Lw = n_ids - gen; pos = 1; T_ = C_ = R_ = 0
    while pos < Lw:
        s = gen + pos - 2 - a0
        if 0 <= s < len(rk) and (rk[s] >= 0).any():
            r_, q_ = rk[s], p1[s]; L = int((r_ >= 0).sum())
            miss = np.where(r_[:L] != 0)[0]; kstar = int(miss[0]) + 1 if len(miss) else L + 1
            m = int((np.cumprod(np.nan_to_num(q_[:min(K, L)], nan=0.0)) >= t).sum())
            k, rows = min(kstar, m + 1), m + 1
        else:
            k, rows = 1, 1
        k = max(1, min(k, Lw - pos)); T_ += k; C_ += 1; R_ += rows; pos += k
    return T_, C_, R_


say("D1 variants: head b5 teacher-forced chain, depths 1-6; hits vs the answer text (see CHAT150's caveat)")
t0 = time.time()
for st in ("ifeval", "multi"):
    files = sorted(glob.glob(os.path.join(T.TD, "d16w", D24.PAT[st] + ".npz")))
    for nm, PP, bits in VAR:
        lm.lm_head = heads[bits]
        hit = np.zeros(6); tot = np.zeros(6); TT = CC = RR = 0
        for f in files:
            key = os.path.basename(f)[:-4]
            x = T.window(key); gen = T.gen_of(key); a0 = max(0, gen - 2)
            r = D24.top_ranks(lm, head, PP, x, a0)
            n = len(x["ids"]) - 1 - a0
            p1 = np.full((n, 6), np.nan, np.float32); rk = np.full((n, 6), -1, np.int8)
            for j, (p, k) in enumerate(r[:6]):
                p1[:len(p), j] = p; rk[:len(k), j] = k
                v = k >= 0; hit[j] += (k[v] == 0).sum(); tot[j] += v.sum()
            a, b_, c = kept_cut(p1, rk, gen, len(x["ids"]))
            TT += a; CC += b_; RR += c
        lm.lm_head = heads[4]
        say(f"{st:6s} {nm:7s}: depth hits " + " ".join(f"{h / t:.3f}" for h, t in zip(hit, tot)) +
            f" | kept/check (cut 0.3, K6) {TT / CC:.3f}, rows {RR / CC:.2f} | {time.time() - t0:.0f} s")
say(f"peak MLX memory {mx.get_peak_memory() / 2 ** 30:.1f} GB")
open(OUT, "w").write("\n".join(L_) + "\n")
