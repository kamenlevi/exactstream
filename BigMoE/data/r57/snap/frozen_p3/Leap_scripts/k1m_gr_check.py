"""K1M-GR (Leap, Sun 2026-11-08 00:5x): ref2 candidate's _gr with REF2_HC8_K1M on == off (and == the unfolded hc8r path) on
a GatedResidual with 8-bit g64 mixers at the 125B's shapes, rows 1-4, with and without the pending write: mixed,
written and inject bitwise. usage: [BigMoE venv] k1m_gr_check.py [candidate.py] -> stdout"""
import importlib.util, os, sys
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Leap/data/hc8f/ref2_cand2.py")
spec = importlib.util.spec_from_file_location("ref2_k1m_cand", path)
C = importlib.util.module_from_spec(spec)
sys.modules["ref2_k1m_cand"] = C
spec.loader.exec_module(C)
D, HC, LR = 2560, 4, 320
HCD = HC * D
BF = mx.bfloat16


class FakeGR(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_mix_weight_down = nn.QuantizedLinear(HCD, LR, bias=False, group_size=64, bits=8)
        self.input_mix_weight_up = nn.QuantizedLinear(LR, HCD, bias=False, group_size=64, bits=8)
        for q in (self.input_mix_weight_down, self.input_mix_weight_up):
            q.scales = (q.scales * 0.05).astype(BF)
            q.biases = (q.biases * 0.05).astype(BF)
        self.block_inject_weight = nn.Linear(HCD, HC, bias=False)
        self.block_inject_weight.weight = (self.block_inject_weight.weight * 0.2).astype(BF)
        self.hc_norm = nn.RMSNorm(HCD, eps=1e-6)
        self.hc_norm.weight = mx.array(np.random.default_rng(3).uniform(0.5, 1.5, (HCD,)).astype(np.float32)).astype(BF)
        self.hc_count = HC
        self.hidden_size = D


C._ON[0] = True
C._ORIG.setdefault("gr", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the fallback ran")))
C.CFG["hc8_fast"] = True
C.CFG["hc8_v"] = 4
g = FakeGR()
mx.eval(g.parameters())
assert C._hc8_params(g) is not None, "the fake module's mixers are not HC8's 8-bit form"
rng = np.random.default_rng(11)
bad = tot = 0
modes = {"unfolded": dict(hc8_fold=False, hc8_k1m=False, hc8_g=0), "fold": dict(hc8_fold=True, hc8_k1m=False, hc8_g=0),
         "fold+k1m": dict(hc8_fold=True, hc8_k1m=True, hc8_g=0), "k1m": dict(hc8_fold=False, hc8_k1m=True, hc8_g=0)}
if "hc8_g" in C.CFG:
    modes["gemv"] = dict(hc8_fold=False, hc8_k1m=False, hc8_g=4)
gerr = 0.0
gwbad = 0
for rows in (1, 2, 3, 4):
    for pend in (False, True):
        hi = mx.array(rng.standard_normal((1, rows, HCD)).astype(np.float32)).astype(BF)
        pe = None
        if pend:
            pe = (mx.array(rng.standard_normal((1, rows, D)).astype(np.float32)).astype(BF),
                  mx.array(rng.uniform(0.0, 2.0, (1, rows, HC)).astype(np.float32)).astype(BF))
        outs = {}
        for name, fl in modes.items():
            C.CFG.update(fl)
            c0 = dict(C.COUNTS)
            outs[name] = C._gr(g, hi, pe)
            mx.eval(outs[name])
            if name in ("fold+k1m", "k1m"):
                assert C.COUNTS["hc8_k1m"] == c0["hc8_k1m"] + 1, "K1M did not engage"
            if name == "gemv":
                assert C.COUNTS["hc8_g"] == c0["hc8_g"] + 1, "HC8-G did not engage"
                for a, b in ((outs["unfolded"][0], outs[name][0]), (outs["unfolded"][2], outs[name][2])):
                    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
                    gerr = max(gerr, float(mx.max(mx.abs(a32 - b32)).item() / max(1e-30, mx.max(mx.abs(a32)).item())))
                gwbad += not bool(mx.array_equal(outs["unfolded"][1], outs[name][1]))
                assert outs[name][0].shape == outs["unfolded"][0].shape and outs[name][1].shape == hi.shape
        for name in ("fold", "fold+k1m", "k1m"):
            ok = all(bool(mx.array_equal(a, b)) for a, b in zip(outs["unfolded"], outs[name]))
            tot += 1
            bad += not ok
            if not ok:
                print(f"  rows {rows} pend {pend}: {name} != unfolded", flush=True)
print(f"K1M-GR: {tot - bad} of {tot} cases bitwise (_gr fold / fold+k1m / k1m vs the unfolded hc8r path; rows 1-4, pending "
      f"off / on); COUNTS hc8_k1m {C.COUNTS['hc8_k1m']}, hc8_fold {C.COUNTS['hc8_fold']}", flush=True)
if "gemv" in modes:
    print(f"HC8-G via _gr (FAST): max rel error of mixed / inject vs the unfolded path {gerr:.2e}; written bitwise "
          f"{'yes' if gwbad == 0 else f'NO ({gwbad})'}; COUNTS hc8_g {C.COUNTS['hc8_g']}", flush=True)
sys.exit(1 if bad else 0)
