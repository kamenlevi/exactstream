#!/usr/bin/env python3
"""Export the MTP head's input path for bank_proxy.py's r2_head (Sat 14 Nov 2026): the head's pre_fc_norm_hidden weight
and eps (from the 125B's MTP module, as M.load builds it) and t30b's fc_hidden -> ~/datasets/deepdraft/ckpt/head_in.npz.
A check: numpy's fused input equals the head's own for one captured row block.
usage: [BigMoE venv] export_head_in.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dd_ttt as DD                                                  # noqa: E402
import numpy as np                                                   # noqa: E402
import mlx.core as mx                                                # noqa: E402

T, M = DD.T, DD.M
rt, lm, head = M.load(0.03)
P = M.Trainable(head)
P.load_weights(os.path.join(DD.CKD, "dd_t30b.safetensors"))
nm = head.pre_fc_norm_hidden
g = np.array(nm.weight.astype(mx.float32)); eps = float(getattr(nm, "eps", 1e-6))
F = np.array(P.fc_hidden.weight.astype(mx.float32))
print(f"[head_in] pre_fc_norm_hidden weight {g.shape}, eps {eps}; fc_hidden {F.shape}; head._hc {head._hc}", flush=True)
z = np.load(os.path.join(DD.CAP, sorted(os.listdir(DD.CAP))[0]))
W = np.asarray(z["wd"][:64]).astype(np.float32)
ref = np.array(P.fc_hidden(head.pre_fc_norm_hidden(mx.array(W).astype(mx.bfloat16)).reshape(64, head._hc, -1).astype(mx.float32)).reshape(64, -1))
def numpy_in(W, grouped):
    X = W.reshape(len(W), head._hc, -1) if grouped else W
    Xn = X / np.sqrt((X ** 2).mean(-1, keepdims=True) + eps)
    Xn = Xn.reshape(len(W), -1) * (np.tile(g, head._hc) if g.shape[0] * head._hc == W.shape[1] else g)
    return (Xn.reshape(len(W), head._hc, -1) @ F.T).reshape(len(W), -1)
res = {gr: float(np.linalg.norm(numpy_in(W, gr) - ref) / np.linalg.norm(ref)) for gr in (False, True)}
grouped = min(res, key=res.get); rel = res[grouped]
print(f"[head_in] norm class {type(nm).__name__}; numpy vs the head's own fused hidden input on 64 captured rows: rel diff "
      f"{res[False]:.2e} (one RMS over 10240) / {res[True]:.2e} (per HC stream) -> grouped {grouped}", flush=True)
np.savez(os.path.join(DD.CKD, "head_in.npz"), norm_w=g, fc_hidden=F, eps=eps, hc=head._hc, grouped=grouped, check_rel=rel)
print("[head_in] saved", flush=True)
