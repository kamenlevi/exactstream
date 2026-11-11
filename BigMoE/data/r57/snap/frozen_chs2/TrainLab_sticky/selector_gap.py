"""SELGAP (MTP Head, Tue 11-03 23:2x; MoonShots' WALL2_KEPT 'Already known' DCore line, a 10-minute check): how much
DFlash2 L1-20's first-order candidate selector adds over plain DFlash (each position's own argmax, no selector).
DCore (arXiv 2610.24993) puts a trained causal tail (DSpark / Domino) at ~+19 % kept over plain DFlash; TRIED SD-ARtail is
KILL on paper because DFlash2 already has the selector. MoonShots' line: selector gap < +10 % -> a trained tail deserves
a screen; >= +10 % -> the selector already holds most of a tail's gain (the KILL stands).
Data: the HPD-0 k-head walks (data/d02b_walk_kh_<dom>.npz: per check the drafter's normed hidden khid [15, 2560] f16 at
every draft position, its own chain dtok, the teacher-forced selector picks ttok). Unary = the 4-bit lm_head on khid
(compute_logits = lm_head(hidden)), argmax. Both variants teacher-forced at HPD-0's anchors (exact up to the first miss);
kept = accepted + 1 per 16-row check, no cf cut. Validation: ttok and dtok must sit in the recomputed exact top-16.
Output: data/selector_gap.txt (appended)."""
import os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
V = 248320
TD = os.path.expanduser("~/TrainLab/data")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)


w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())


def first_miss(dr, tr):
    k = 0
    while k < len(tr) and tr[k] >= 0 and dr[k] == tr[k]:
        k += 1
    return k


say(f"SELGAP {time.strftime('%m-%d %H:%M')}: DFlash2 L1-20 selector on vs off (unary argmax), HPD-0 anchors, teacher-forced")
for dom in ("code", "ifeval", "multi"):
    z = np.load(os.path.join(TD, f"d02b_walk_kh_{dom}.npz"))
    n = len(z["names"])
    on, off_, chk, bad16, badk = [], [], 0, 0, 0
    for i in range(n):
        ids, o = z[f"{i}_ids"], int(z[f"{i}_off"])
        ck, dt, tt, kh = z[f"{i}_checks"], z[f"{i}_dtok"], z[f"{i}_ttok"], z[f"{i}_khid"]
        top1 = np.zeros(dt.shape, np.int64); in16 = np.zeros(dt.shape, bool); din16 = np.zeros(dt.shape, bool)
        flat = kh.reshape(-1, 2560)
        for s in range(0, flat.shape[0], 1024):
            lg = tgt.lm_head(mx.array(flat[s:s + 1024]).astype(mx.bfloat16))
            t16 = mx.argpartition(lg, -16, axis=-1)[:, -16:]
            a1 = mx.argmax(lg, axis=-1)
            mx.eval(t16, a1)
            t16, a1 = np.array(t16), np.array(a1)
            r = np.arange(s, s + a1.shape[0])
            top1.reshape(-1)[r] = a1
            in16.reshape(-1)[r] = (t16 == tt.reshape(-1)[r][:, None]).any(1)
            din16.reshape(-1)[r] = (t16 == dt.reshape(-1)[r][:, None]).any(1)
        for c, (k_rec, a) in enumerate(zip(ck[:, 0], ck[:, 2])):
            true = np.full(15, -1, np.int64)
            seg = ids[a - o + 1:a - o + 16]
            true[:len(seg)] = seg
            kon, koff = first_miss(tt[c], true), first_miss(top1[c], true)
            badk += int(first_miss(dt[c], true) != k_rec)
            mm = int((true >= 0).sum())
            bad16 += int((~in16[c, :mm]).sum() + (~din16[c, :mm]).sum())
            if mm == 15:
                on.append(kon); off_.append(koff); chk += 1
    on, off_ = np.array(on), np.array(off_)
    kon, koff = on.mean() + 1, off_.mean() + 1
    say(f"  {dom:6s}: {chk} full checks; kept per 16-row check selector ON {kon:.3f} / OFF (unary argmax) {koff:.3f} -> "
        f"selector gap {100 * (kon / koff - 1):+.1f} %; P(depth >= d right) ON/OFF at d=1,3,8,15: "
        + " ".join(f"{(on >= d).mean():.3f}/{(off_ >= d).mean():.3f}" for d in (1, 3, 8, 15))
        + f"; checks where ON > OFF {(on > off_).mean():.3f}, OFF > ON {(off_ > on).mean():.3f}"
        + f"; validation: own-chain k mismatches {badk}, picks outside the recomputed top-16 {bad16}")
open(os.path.join(TD, "selector_gap.txt"), "a").write("\n".join(lines) + "\n\n")
