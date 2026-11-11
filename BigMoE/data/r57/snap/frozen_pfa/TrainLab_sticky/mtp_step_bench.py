"""D1 (CHAT150 strike, NewIdeas 00:4x): the chat draft step's anatomy (MTP Head, Sun 11-08, GPU ~3-6 GB, quiet).
The shipped chat drafter = r57_screen_fast.py MTPHead (D-23c's mtp15e): per drafted row one model.mtp_forward on the head
(trained parts of D-22 b5 in bf16 via mtp125.install), an fp32 softmax over the 248,320-token vocab, then mx.eval + .item()
before the next step. Timed here on the engine's shapes (random widened stream, 512 context rows; timing only):
  seq     the shipped loop: K steps, each evaluated and read back (host sync per step)
  nosync  the same K steps as one lazy graph (argmax -> next input on the GPU), one eval at the end; drafts compared to seq
  head    the 4-bit 248K output layer alone (hyper-connection mixer + lm_head) on one row
  core    the MTP layer without the output layer (emit_logits=False)
  dlh<b>  MTPLX's draft output layer re-quantized to b bits (draft_lm_head.py), one row
Usage: [BigMoE venv] python3 -I sticky/mtp_step_bench.py [K=4] [reps=40]  -> data/mtp_step_bench.txt"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/TrainLab_root"))
import mtp125 as M
import mlx.core as mx
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
OUT = os.path.expanduser("~/TrainLab/data/mtp_step_bench.txt")
L_ = []


def say(s):
    print(s, flush=True); L_.append(s)


M.limit_memory()
rt, lm, head = M.load(0.03)
TH = M.Trainable(head)
TH.load_weights(os.path.expanduser("~/TrainLab/ckpt/d22_b5.safetensors"))
M.install(head, TH, dtype=mx.bfloat16)
del TH
if os.environ.get("D1_P4") == "1":                        # D1 p4: the installed trained parts (bf16 Linear) at 4 bits
    import mlx.nn as nn
    nq = []
    nn.quantize(head, group_size=64, bits=4, class_predicate=lambda p, m: (
        isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear) and m.weight.dtype == mx.bfloat16
        and m.weight.shape[-1] % 64 == 0 and not nq.append(p)))
    mx.eval(head.parameters())
    print(f"D1_P4: {len(nq)} trained modules at 4 bits: {nq}", flush=True)
    OUT = OUT.replace(".txt", "_p4.txt")
model = rt.model if hasattr(rt.model, "mtp_forward") else lm
assert hasattr(model, "mtp_forward"), "no mtp_forward"
W = int(getattr(head, "_hc", 1)) * 2560
say(f"D1 draft-step anatomy: model {type(model).__name__}, widened width {W}, K {K}, reps {REPS}, "
    f"lm_head {type(lm.lm_head).__name__} bits {getattr(lm.lm_head, 'bits', '?')}")
mx.random.seed(0)
ctx = 512
hid = (mx.random.normal((1, ctx + 1, W)) * 0.5).astype(mx.bfloat16)
ids = mx.random.randint(1000, 30000, (1, ctx + 1))
cache = model.make_mtp_cache()
model.mtp_update_cache(hid[:, :-1], ids[:, 1:ctx + 1][:, :ctx], mtp_cache=cache)
mx.eval([c.state for c in cache])
lg0, h0 = model.mtp_forward(hid[:, -1:], ids[:, -1:], mtp_cache=cache, return_hidden=True)
mx.eval(lg0, h0)
base_off = cache[0].offset


def trim():
    n = cache[0].offset - base_off
    if n > 0:
        cache[0].trim(n)


def seq():
    l0 = lg0[0, -1].astype(mx.float32)
    tk, p = mx.argmax(l0), mx.softmax(l0).max()
    mx.eval(tk, p)
    toks, h = [int(tk.item())], h0
    for _ in range(K - 1):
        lgx, h = model.mtp_forward(h, mx.array([[toks[-1]]]), mtp_cache=cache, return_hidden=True)
        lx = lgx[0, -1].astype(mx.float32)
        tk, p = mx.argmax(lx), mx.softmax(lx).max()
        mx.eval(tk, p)
        toks.append(int(tk.item()))
    trim()
    return toks


def nosync():
    l0 = lg0[0, -1].astype(mx.float32)
    tks, ps = [mx.argmax(l0)], [mx.softmax(l0).max()]
    h = h0
    for _ in range(K - 1):
        lgx, h = model.mtp_forward(h, tks[-1].reshape(1, 1), mtp_cache=cache, return_hidden=True)
        lx = lgx[0, -1].astype(mx.float32)
        tks.append(mx.argmax(lx)); ps.append(mx.softmax(lx).max())
    t = mx.stack(tks); pp = mx.stack(ps)
    mx.eval(t, pp)
    out = t.tolist()
    trim()
    return out


def timeit(fn, reps=REPS):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


a, b = seq(), nosync()
say(f"drafts seq {a} | nosync {b} | equal {a == b}")
ms, p90 = timeit(seq)
say(f"seq    : {ms:6.2f} ms per chain of {K - 1} forwards ({ms / (K - 1):.2f} per drafted row; p90 {p90:.2f})")
ms2, p90 = timeit(nosync)
say(f"nosync : {ms2:6.2f} ms per chain ({ms2 / (K - 1):.2f} per row; p90 {p90:.2f}); x{ms / ms2:.2f} vs seq")
hrow = h0[:, -1:]


def head_only():
    x = lm.lm_head(head.hyper_connection_mixer(hrow)) if hasattr(head, "hyper_connection_mixer") else lm.lm_head(hrow)
    mx.eval(x)


def core_only():
    _, hh = model.language_model._mtp_core(hid[:, -1:], ids[:, -1:], mtp_cache=cache, emit_logits=False) \
        if hasattr(model, "language_model") and hasattr(model.language_model, "_mtp_core") else \
        model._mtp_core(hid[:, -1:], ids[:, -1:], mtp_cache=cache, emit_logits=False)
    mx.eval(hh); trim()


for nm, fn in (("head (4-bit 248K output layer)", head_only), ("core (MTP layer, no output layer)", core_only)):
    try:
        ms3, p90 = timeit(fn)
        say(f"{nm:34s}: {ms3:6.2f} ms (p90 {p90:.2f})")
    except Exception as e:
        say(f"{nm}: n/a ({type(e).__name__}: {str(e)[:120]})")
try:
    import mtplx.draft_lm_head as DLH
    q = lm.lm_head
    for bits in (() if os.environ.get("D1_SKIPOUT") == "1" else (3, 2)):
        wq = mx.dequantize(q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits)
        w2, s2, b2 = mx.quantize(wq, group_size=64, bits=bits)
        del wq
        mx.eval(w2, s2, b2)
        xin = head.hyper_connection_mixer(hrow) if hasattr(head, "hyper_connection_mixer") else hrow

        def f(w2=w2, s2=s2, b2=b2, bits=bits):
            mx.eval(mx.quantized_matmul(xin, w2, s2, b2, transpose=True, group_size=64, bits=bits))
        ms4, p90 = timeit(f)
        say(f"output layer re-quantized to {bits} bits        : {ms4:6.2f} ms (p90 {p90:.2f})")
        del w2, s2, b2
except Exception as e:
    say(f"draft output layer bits: n/a ({type(e).__name__}: {str(e)[:120]})")
def upd(k):
    def f():
        model.mtp_update_cache(hid[:, :k], ids[:, 1:k + 1], mtp_cache=cache)
        mx.eval([c.state for c in cache]); trim()
    return f


for k in (2, 4, 8):
    ms5, p90 = timeit(upd(k))
    say(f"cache update over {k} accepted rows        : {ms5:6.2f} ms (p90 {p90:.2f})")


def cycle():
    """engine-like draft of one chat check: append 3 accepted rows (update 2 + forward the last), then 2 synced steps"""
    model.mtp_update_cache(hid[:, :2], ids[:, 1:3], mtp_cache=cache)
    lg, h = model.mtp_forward(hid[:, 2:3], ids[:, 3:4], mtp_cache=cache, return_hidden=True)
    l0 = lg[0, -1].astype(mx.float32); tk, p = mx.argmax(l0), mx.softmax(l0).max(); mx.eval(tk, p); t = int(tk.item())
    for _ in range(2):
        lg, h = model.mtp_forward(h, mx.array([[t]]), mtp_cache=cache, return_hidden=True)
        lx = lg[0, -1].astype(mx.float32); tk, p = mx.argmax(lx), mx.softmax(lx).max(); mx.eval(tk, p); t = int(tk.item())
    trim()


ms6, p90 = timeit(cycle)
say(f"engine-like chat draft (append 3 + 2 steps): {ms6:6.2f} ms (p90 {p90:.2f})")
say(f"peak MLX memory {mx.get_peak_memory() / 2 ** 30:.1f} GB")
open(OUT, "w").write("\n".join(L_) + "\n")
