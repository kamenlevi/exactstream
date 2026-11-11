"""gated_norm_exact (gdn_exact.py) vs MTPLX's SigmoidRMSNormGated eager path, bitwise: rows of 48 heads x S (1..8)."""
import os, sys
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_third_party/ms_gdn"))   # read-only use
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from mtplx.models.qwen4_exp import SigmoidRMSNormGated
from gdn_exact import gated_norm_exact
ok = True
for seed, (sh, sw) in enumerate(((1.0, 0.5), (4.0, 1.5), (0.05, 0.2))):
    mx.random.seed(seed)
    m = SigmoidRMSNormGated(128, eps=1e-6)
    m.weight = (mx.random.normal((128,)) * sw + 1).astype(mx.bfloat16)
    for S in (1, 2, 4, 8, 12, 16):
        h = (mx.random.normal((1, S, 48, 128)) * sh).astype(mx.bfloat16)
        g = (mx.random.normal((1, S, 48, 128)) * 2).astype(mx.bfloat16)
        ref = m(h, g)
        out = gated_norm_exact(h, g, m.weight, m.eps)
        n = int((ref.reshape(-1).view(mx.uint16) != out.reshape(-1).view(mx.uint16)).sum().item())
        ok &= n == 0 and ref.dtype == out.dtype
        print(f"scale h {sh} w {sw} S={S}: mismatches {n} of {h.size} (ref dtype {ref.dtype})", flush=True)
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
