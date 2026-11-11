"""F1-R8 check (Leap, Tue 2026-11-03 22:5x; TABLE_350 'F1-R8', pre-registered 22:54): ref2's CFG["fatt_r8"] (fs4's R 8 form for
padded verify calls with <= 8 real rows) on F1's layer harness (layer 3's real weights, prefill of random hidden states):
 (1) bits, T0 2K and 10K, with fatt_verify on: an 8-row check's y with fatt_r8 on == off; 8 one-row AR steps' y on ==
     off; a 16-row check (fs4 R 16) == the AR steps with fatt_r8 on (SD == AR);
 (2) counts: every <= 8-row call took R 8, the 16-row check did not;
 (3) F1R8_SPEED=1: ms per 8-row layer call at 2K and 10K, fatt_r8 off / on (A-B-A-B, B6's chained method; a quiet lock).
Usage: [BigMoE venv] [F1R8_SPEED=1] f1r8_check.py -> ~/Leap/data/f1r8_check.txt (GPU ~3 GB)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
LEAP = os.path.expanduser("~/Leap")
sys.path.insert(0, f"{LEAP}/scripts")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind

OUT = f"{LEAP}/data/f1r8_check.txt"
SPEED = os.environ.get("F1R8_SPEED") == "1"
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def build():
    D = os.path.expanduser("~/BigMoE/models/bare")
    wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(f"{D}/config.json"))
    qc = cfg["quantization"]
    args = Q.TextArgs.from_dict(cfg["text_config"])
    pre = "language_model.model.layers.3.self_attn."
    keys = [k_ for k_ in wmap if k_.startswith(pre)]
    w = {}
    for f in sorted({wmap[k_] for k_ in keys}):
        sh = mx.load(f"{D}/{f}")
        w.update({k_[len(pre):]: sh[k_] for k_ in keys if wmap[k_] == f})
    att = Q.Attention(args)
    def qpred(p, m_):
        if not (isinstance(m_, nn.Linear) and f"{p}.scales" in w):
            return False
        c_ = qc.get(pre + p, {})
        return {"group_size": c_.get("group_size", qc["group_size"]), "bits": c_.get("bits", qc["bits"])}
    nn.quantize(att, group_size=qc["group_size"], bits=qc["bits"], class_predicate=qpred)
    att.load_weights(list(w.items()), strict=False)
    att.eval()
    mx.eval(att.parameters())
    return att, args


def cache_at(att, args, T0):
    mx.random.seed(T0)
    c = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
    for a0 in range(0, T0, 1024):
        n_ = min(1024, T0 - a0)
        y = att(mx.random.normal((1, n_, 2560)).astype(mx.bfloat16), c)
        mx.eval(y, [z for z in (c.kv.keys, c.kv.values, c.raw_keys, c.pooled) if z is not None])
    return c


def call(att, x, c):
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
        y = att(x, c)
    mx.eval(y)
    return y


def xrows(T0, R):
    mx.random.seed(T0 + 7 * R)
    x = mx.random.normal((1, R, 2560)).astype(mx.bfloat16)
    mx.eval(x)
    return x


bits = lambda a: np.array(a.view(mx.uint16)).astype(np.uint16)
ref2.apply()
ref2.CFG["fatt_verify"] = True
att, args = build()
say(f"F1-R8 check {time.strftime('%a %H:%M')} ({os.environ.get('F1R8_COND', 'share lock')}){' speed' if SPEED else ''}")
if not SPEED:
    ok = True
    for T0 in (2000, 10000):
        c = cache_at(att, args, T0)
        x8, x16 = xrows(T0, 8), xrows(T0, 16)
        res = {}
        for r8 in (False, True):
            ref2.CFG["fatt_r8"] = r8
            ref2.COUNTS["fatt_r8"] = ref2.COUNTS["fatt_verify"] = 0
            y8 = call(att, x8, c); c.trim(8)                                   # an 8-row check
            ar = [call(att, x16[:, r:r + 1], c) for r in range(16)]; c.trim(16)   # 16 one-row AR steps
            y16 = call(att, x16, c); c.trim(16)                               # the 16-row check over the same rows
            res[r8] = (y8, ar, y16, dict(r8=ref2.COUNTS["fatt_r8"], fs4=ref2.COUNTS["fatt_verify"]))
        e8 = bool(np.array_equal(bits(res[False][0]), bits(res[True][0])))
        ear = sum(bool(np.array_equal(bits(a), bits(b))) for a, b in zip(res[False][1], res[True][1]))
        sdar = sum(bool(np.array_equal(bits(res[True][2][:, r]), bits(res[True][1][r][:, 0]))) for r in range(16))
        cnt_ok = res[True][3] == {"r8": 17, "fs4": 18} and res[False][3] == {"r8": 0, "fs4": 18}
        ok &= e8 and ear == 16 and sdar == 16 and cnt_ok
        say(f"T0 {T0}: 8-row check r8 on == off {e8}; AR steps on == off {ear}/16; SD (16-row) == AR (r8 on) {sdar}/16; "
            f"counts off {res[False][3]}, on {res[True][3]} -> {'OK' if cnt_ok else 'WRONG'}")
        mx.clear_cache()
    say(f"F1-R8 (1)+(2): {'PASS' if ok else 'FAIL'}")
else:
    ms = {}
    for T0 in (2000, 10000):
        c = cache_at(att, args, T0)
        x8 = xrows(T0, 8)
        t = {False: [], True: []}
        for r8 in (False, True, False, True):
            ref2.CFG["fatt_r8"] = r8
            def go():
                ys = []
                for _ in range(8):
                    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
                        ys.append(att(x8, c))
                    c.trim(8)
                return ys
            for _ in range(2):
                mx.eval(go())
            ts = []
            for _ in range(20):
                mx.synchronize(); t0 = time.perf_counter(); mx.eval(go()); ts.append(time.perf_counter() - t0)
            t[r8].append(1e3 * float(np.median(ts)) / 8)
        ms[T0] = {k: float(np.mean(v)) for k, v in t.items()}
        say(f"(3) T0 {T0}: ms per 8-row layer call, r8 off {ms[T0][False]:.3f} {t[False]}, on {ms[T0][True]:.3f} {t[True]}; "
            f"{ms[T0][False] / ms[T0][True]:.2f}x")
        mx.clear_cache()
    f10 = ms[10000][False] / ms[10000][True]
    say(f"F1-R8 (3): {'PASS' if f10 >= 1.15 else 'KILL' if f10 < 1.05 else 'BETWEEN'} ({f10:.2f}x at 10K)")
ref2.CFG["fatt_r8"] = False
ref2.CFG["fatt_verify"] = False
