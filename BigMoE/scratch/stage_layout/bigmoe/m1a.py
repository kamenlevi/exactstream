"""S10c step 2 (NewIdeas Tue ~20:00, rapid-screen size): TrainLab's frozen M1a lead-2 forecaster in the engine.
BIGMOE_M1A=npz (keys L{L}_Wf [2560, 512], L{L}_bf [512], L{L}_A [512, 512], T = L + 2): at each source layer L the
forecaster gets a d = 2 list from M1a, s = x @ Wf + bf + sum of A[e] over L's routed ids (TrainLab's folded form, x =
layer L's MoE input, the tensor the router forecast already uses), and T's lead-1 router list (forecast at T - 1) is
dropped: M1a's picks REPLACE it, same rule (top-B per row, resident skipped by the ext) and same queue (PF_QD). Every
other layer keeps the shipped lead-1 router list. Prediction only: the decode stays exact."""
import os
import numpy as np
import mlx.core as mx


class M1a:
    m1a = True                                       # stream.py passes L's routed ids to forecasters that set this

    def __init__(self, z, L):
        self.L = L
        self.Wf = mx.array(z[f"L{L}_Wf"].astype(np.float16))      # stored f16 (46 pairs: ~143 MB), used in f32:
        self.bf = mx.array(z[f"L{L}_bf"].astype(np.float32))      # x is bf16 and M1a was fit in float
        self.A = mx.array(z[f"L{L}_A"].astype(np.float16))
        self.bf16 = self.bf.astype(mx.float16)
        mx.eval(self.Wf, self.bf, self.A, self.bf16)

    def __call__(self, x, ids):
        if F16:                                      # b39: f16 matmul (f32 accumulation inside), no per-call upcast
            return _score16(x, ids.reshape(x.shape[0], -1), self.Wf, self.A, self.bf16)
        s = x.astype(mx.float32) @ self.Wf.astype(mx.float32) + self.bf
        return s + self.A[ids.reshape(x.shape[0], -1)].astype(mx.float32).sum(axis=-2)


# BIGMOE_M1A_F16=1 (b39): b38's f32 path upcast each layer's Wf every pass (~8 MB of traffic per layer + ~7 kernels),
# which ate the gate-wait gain. This path scores in f16 in one compiled function (a few top-12 picks may differ from
# the f32 scores on near-ties; prediction only, the decode stays exact).
F16 = os.environ.get("BIGMOE_M1A_F16") == "1"
_score16 = mx.compile(lambda x, ids, Wf, A, bf: (x.astype(mx.float16) @ Wf) + A[ids].sum(axis=-2) + bf)


SKIP = (None, None, 0)                                 # an empty d = 1 list: keeps the d = 2 list positional


def attach(layers, nl, B, path):
    """rewire the look-ahead lists (switch_mlp._bm["next"]) for M1a's source layers; returns the (L, T) pairs.
    Any set of pairs, chains included (all 46: every layer's lead-1 list goes, each L forecasts L + 2 only)."""
    z = np.load(path)
    K = int(os.environ.get("BIGMOE_M1A_K", B))                    # top-K per row for M1a's lists (b34: 12 and 16)
    src = sorted(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_Wf"))
    tgt = {L + 2 for L in src}
    assert max(tgt) < nl, src
    bms = [layers[L].mlp.switch_mlp._bm for L in range(nl)]
    for L in range(nl - 1):                                      # shipped: at most the router list for L + 1
        nx = bms[L].get("next")
        assert not nx or (len(nx) == 1 and nx[0][0] is layers[L + 1].mlp.gate), f"layer {L}: not the shipped list"
    for L in range(nl - 1):
        if L + 1 in tgt and bms[L].get("next"):
            bms[L]["next"] = [SKIP]                              # T's lead-1 router list goes
    for L in src:
        bms[L]["next"] = [(bms[L].get("next") or [SKIP])[0], (M1a(z, L), None, K)]
    for L in range(nl - 1):
        if bms[L].get("next") == [SKIP]:
            bms[L].pop("next")                                   # nothing left to forecast from L
    return [(L, L + 2) for L in src]
