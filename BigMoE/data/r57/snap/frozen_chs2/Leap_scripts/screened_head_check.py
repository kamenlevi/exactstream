"""MTP-STEP-1 check (Leap, Wed 2026-11-04 15:3x): screened_head.ScreenedHead on the 125B's real 4-bit lm_head, base =
FR-Spec top-32K (~/Leap/data/frspec_rank.npy[:32768]):
 (1) the subset's logits == the full head's logits at those ids, bitwise (1 row); the extra rows (session tokens outside
     the base) likewise; the top-1 == the full head's top-1 whenever that token is in base + extra; p >= the full p;
 (2) ms per call at 1 row (64-call dependent chain), full head vs screened (no extra / 2,048 extra rows).
Usage: [BigMoE venv] screened_head_check.py -> ~/Leap/data/mtp_step0.txt (appends; GPU, MEM ~1.5 GB; share lock)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
import mlx.core as mx
import mlx.nn as nn
from screened_head import ScreenedHead

OUT = os.path.expanduser("~/Leap/data/mtp_step0.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
pre = "language_model.lm_head"
sh = mx.load(f"{D}/{wmap[pre + '.weight']}")
head = nn.QuantizedLinear(2560, sh[pre + ".weight"].shape[0], bias=False, group_size=64, bits=4)
head.update({"weight": sh[pre + ".weight"], "scales": sh[pre + ".scales"], "biases": sh[pre + ".biases"]})
mx.eval(head.parameters())
V = head["weight"].shape[0]
rank = np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))
base = rank[:32768]
sh_ = ScreenedHead(head, base, extra_cap=4096)
extra = rank[40000:42048]                                     # 2,048 tokens outside the base: a stand-in session set
sh_.add(extra)
say(f"\nMTP-STEP-1 {time.strftime('%a %H:%M:%S')}: ScreenedHead base {len(base)} + extra {sh_.n_extra} rows (cap 4096)")
mx.random.seed(3)
ok_l = ok_t = ok_p = True
n_in = 0
for i in range(32):
    h = (mx.random.normal((1, 2560)) * (0.5 + 0.1 * i)).astype(mx.bfloat16)
    full = head(h).astype(mx.float32)[0]
    lg = sh_.logits(h)[0]
    tok, p = sh_(h)
    mx.eval(full, lg, tok, p)
    F = np.array(full)
    sub = np.concatenate([np.sort(base), np.array(sh_.extra_list)])
    ok_l &= bool(np.array_equal(np.array(lg)[:len(sub)], F[sub]))
    t_full = int(np.argmax(F))
    if t_full in set(sub.tolist()):
        n_in += 1
        ok_t &= int(np.array(tok)[0]) == t_full
    pf = float(np.exp(F[t_full] - (F.max() + np.log(np.exp(F - F.max()).sum()))))
    ok_p &= float(np.array(p)[0]) >= pf - 1e-6 or int(np.array(tok)[0]) != t_full
say(f"(1) subset logits == full at those ids (bitwise): {ok_l}; top-1 == full's when it is in the subset ({n_in}/32): {ok_t}; "
    f"p >= full p: {ok_p}")
sh0 = ScreenedHead(head, base, extra_cap=4096)
ARMS = {"full": lambda x: mx.argmax(head(x), axis=-1), "screened": lambda x: sh0(x)[0], "screened+2k": lambda x: sh_(x)[0]}
x0 = (mx.random.normal((1, 2560)) * 0.5).astype(mx.bfloat16)
mx.eval(x0)
def chain(f):
    x = x0
    for _ in range(64):
        x = mx.depends(x0, f(x))
    return x
for f in ARMS.values():
    mx.eval(chain(f))
T = {k: [] for k in ARMS}
for r in range(18):
    for k in (list(ARMS) if r % 2 == 0 else list(ARMS)[::-1]):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(chain(ARMS[k])); mx.synchronize()
        T[k].append(1e3 * (time.perf_counter() - t0) / 64)
say("(2) ms per head call (1 row, 64-call chain, median of 15): " + ", ".join(f"{k} {np.median(v[3:]):.3f}" for k, v in T.items()))
