#!/usr/bin/env python3
"""EARLY DEATH stage 2, piece 1: the 125B's own LENS on a widened state (NewIdeas, Mon 16 Nov 21:2x; for Wed's probe,
research/week98/EARLYDEATH_PREREG.md, P-lens / P-tuned).

The 125B's head (mtplx models/qwen4_exp.py, TextModel.__call__): the last layer's WIDENED stream h [*, 4 x 2560] ->
model.hyper_connection_mixer (GatedResidual, use_combine=False; its reference path, lines ~1290-1300):
    normed = GroupedRMSNorm(h) (rms per 2560 group, eps 1e-6) * hc_norm.weight
    mix    = sigmoid(up(silu(down(normed) / 4)))  -> [*, 4, 2560]
    mixed  = mean_g(mix * normed_g)               -> [*, 2560]
-> lm_head (4-bit affine, group 64) -> logits. The lens applies the same head to a layer-L widened state.
Weights: ~/BigMoE/models/bare/model-00016-of-00017.safetensors (lazy mx.load, CPU device; read-only).

SELF-CHECK (run now, before the capture exists): the lens on the bank's FINAL widened stream wd must reproduce the
125B's stored top-32 (tk / lp) on the answer rows: top-1 agreement >= 99 % and median |lp diff| over the shared top-32
< 0.05 nats (fp16 storage). Both readings of hc_norm's weight are tried (raw, and +1 as sanitize does for raw-HF
checkpoints); exactly one must pass.
Usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B ed_lens.py [n_files]
"""
import glob
import json
import os
import sys

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
H = os.path.expanduser("~")
WF = f"{H}/BigMoE/models/bare/model-00016-of-00017.safetensors"
PFX = "language_model."
HC, HID, EPS, GS, BITS = 4, 2560, 1e-6, 64, 4


class Lens:
    def __init__(self, shift=False):
        w = mx.load(WF)
        g = lambda k: w[PFX + k]
        self.hcw = np.array(g("model.hyper_connection_mixer.hc_norm.weight").astype(mx.float32)) + (1.0 if shift else 0.0)
        self.down = np.array(g("model.hyper_connection_mixer.input_mix_weight_down.weight").astype(mx.float32))   # [320, 10240]
        self.up = np.array(g("model.hyper_connection_mixer.input_mix_weight_up.weight").astype(mx.float32))       # [10240, 320]
        self.qw, self.qs, self.qb = g("lm_head.weight"), g("lm_head.scales"), g("lm_head.biases")
        self.V = int(self.qs.shape[0])
        assert self.down.shape == (320, HC * HID) and self.up.shape == (HC * HID, 320), (self.down.shape, self.up.shape)

    def mixed(self, h, chunk=1024):
        """h [n, 10240] -> the mixer's output [n, 2560] (float32), in row chunks (memory-light)"""
        if len(h) > chunk:
            return np.concatenate([self.mixed(h[a:a + chunk], chunk) for a in range(0, len(h), chunk)])
        h = np.asarray(h, np.float32)
        n = h.shape[0]
        g_ = h.reshape(n, HC, HID)
        normed = (g_ / np.sqrt((g_ * g_).mean(-1, keepdims=True) + EPS)).reshape(n, HC * HID) * self.hcw
        z = normed @ self.down.T / HC
        z = z * (0.5 * (1.0 + np.tanh(0.5 * z)))         # silu, overflow-free sigmoid form
        u_ = z @ self.up.T
        mix = 0.5 * (1.0 + np.tanh(0.5 * u_))            # sigmoid
        return (mix.reshape(n, HC, HID) * normed.reshape(n, HC, HID)).mean(1)

    def rows(self, ids):
        """dequantized lm_head rows for token ids -> [len(ids), 2560] float32"""
        ix = mx.array(np.asarray(ids, np.int32))
        W = mx.dequantize(mx.take(self.qw, ix, 0), mx.take(self.qs, ix, 0), mx.take(self.qb, ix, 0), group_size=GS, bits=BITS)
        return np.array(W.astype(mx.float32))

    def logits_for(self, m, ids):
        """m [n, 2560], ids [n, k] -> logits [n, k] for each row's own token ids"""
        u, inv = np.unique(ids, return_inverse=True)
        R = self.rows(u)
        return np.einsum("nd,nkd->nk", m, R[inv.reshape(ids.shape)])

    def topk_full(self, m, k=32, chunk=16384):
        """full-vocab top-k ids, their log-softmax, over vocab chunks (memory-light)"""
        n = m.shape[0]
        best_v = np.full((n, k), -np.inf, np.float64); best_i = np.zeros((n, k), np.int64)
        lse = np.full(n, -np.inf)
        for a in range(0, self.V, chunk):
            ids = np.arange(a, min(a + chunk, self.V))
            L = m @ self.rows(ids).T
            mxv = L.max(1); lse = np.logaddexp(lse, mxv + np.log(np.exp(L - mxv[:, None]).sum(1)))
            allv = np.concatenate([best_v, L], 1); alli = np.concatenate([best_i, np.broadcast_to(ids, L.shape)], 1)
            sel = np.argpartition(-allv, k - 1, 1)[:, :k]
            best_v = np.take_along_axis(allv, sel, 1); best_i = np.take_along_axis(alli, sel, 1)
        o = np.argsort(-best_v, 1)
        return np.take_along_axis(best_i, o, 1), np.take_along_axis(best_v, o, 1) - lse[:, None]


def self_check(nf=2, rows_per_file=96):
    files = sorted(glob.glob(f"{H}/datasets/deepdraft/bank/tf_code_HumanEval_*.npz"))[:nf]
    out = {}
    for shift in (False, True):
        L = Lens(shift)
        agree = tot = 0; diffs = []
        for f in files:
            z = np.load(f)
            gen = int(z["gen"]); wd = z["wd"].astype(np.float32); tk = z["tk"]; lp = z["lp"].astype(np.float32)
            r = np.arange(max(0, gen - 1), min(len(wd), max(0, gen - 1) + rows_per_file))
            ti, tl = L.topk_full(L.mixed(wd[r]))
            agree += int((ti[:, 0] == tk[r, 0]).sum()); tot += len(r)
            for j, row in enumerate(r):                  # |lp diff| on the shared top-32 ids
                mine = dict(zip(ti[j].tolist(), tl[j].tolist()))
                diffs += [abs(mine[t] - v) for t, v in zip(tk[row].tolist(), lp[row].tolist()) if t in mine]
        out["shift" if shift else "raw"] = {"rows": tot, "top1_agree": agree / tot, "median_lp_diff": float(np.median(diffs)),
                                             "p95_lp_diff": float(np.percentile(diffs, 95)), "shared": len(diffs)}
        print(("shift +1" if shift else "raw"), out["shift" if shift else "raw"], flush=True)
    ok = [k for k, v in out.items() if v["top1_agree"] >= 0.99 and v["median_lp_diff"] < 0.05]
    out["verdict"] = f"PASS ({ok[0]})" if len(ok) == 1 else f"FAIL ({ok or 'none'} pass)"
    print("SELF-CHECK:", out["verdict"], flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, "ed_lens_selfcheck.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    self_check(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
