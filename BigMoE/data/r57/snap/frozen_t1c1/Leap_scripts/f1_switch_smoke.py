"""F1-SWITCH smoke (Leap, Tue 2026-11-03 22:3x; TABLE_350 'F1-SWITCH', pre-registered 22:25): ref2's CFG["fatt_verify"]
(fs4 in the QSA verify calls) on layer 3's real weights, B6's layer harness (1,024-row prefill chunks of random hidden
states), T0 2K and 10K:
 (2) flag off == the pre-switch ref2 (data/ref2_before_f1.py), BITWISE: a 16-row and an 8-row verify call's y, each
     version in its own subprocess (ref2.apply() patches MTPLX globally);
 (3) the switch takes effect: COUNTS per arm; y(on) vs y(off) mean |diff| / RMS;
 (4) SD == AR at the layer level, bitwise: a check of R real rows (16, 8) at T0..T0+R-1 vs REF2's AR (each row alone,
     padded to 16 by ref2's _att, the cache grown one token at a time); flag on (F1) and off (REF2-T's own, the control);
 (5) ms per 16-row layer call at 10K, flag on vs off (B6: fs4 1.141, ref2t 1.936).
Usage: [BigMoE venv] f1_switch_smoke.py -> ~/Leap/data/f1_switch_smoke.txt (GPU, ~3 GB; share lock)"""
import os, sys, json, time, subprocess
import numpy as np
import mlx.core as mx
import mlx.nn as nn
LEAP = os.path.expanduser("~/Leap")
sys.path.insert(0, f"{LEAP}/scripts")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
OUT = f"{LEAP}/data/f1_switch_smoke.txt"
DUMP = len(sys.argv) > 1 and sys.argv[1] == "--dump"
if DUMP and sys.argv[2] == "old":                       # the pre-switch ref2 under the name ref2
    import importlib.util
    spec = importlib.util.spec_from_file_location("ref2", f"{LEAP}/data/ref2_before_f1.py")
    ref2 = importlib.util.module_from_spec(spec); sys.modules["ref2"] = ref2; spec.loader.exec_module(ref2)
else:
    import ref2
from mtplx.models import qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind


def say(s):
    print(s, flush=True)
    if not DUMP:
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

if DUMP:   # (2): y of a 16-row and an 8-row verify call per T0, flag off (the old ref2 has no flag)
    which, path = sys.argv[2], sys.argv[3]
    if "fatt_verify" in ref2.CFG:
        ref2.CFG["fatt_verify"] = False
    att, args = build()
    res = {}
    for T0 in (2000, 10000):
        c = cache_at(att, args, T0)
        for R in (16, 8):
            res[f"{T0}_{R}"] = bits(call(att, xrows(T0, R), c)); c.trim(R)
    np.savez(path, **res)
    print(json.dumps({k: int(v) for k, v in ref2.COUNTS.items() if k in ("fatt_verify", "verify_old", "verify_old_eligible", "kb_pad", "sdpa_split")}))
    sys.exit(0)

say(f"F1-SWITCH smoke {time.strftime('%a %H:%M')} ({os.environ.get('F1_COND', 'share lock')})")
if len(sys.argv) > 1 and sys.argv[1] == "--speed":   # (5) alone (a quiet re-run): ms per 16-row layer call at 10K
    att, args = build()
    c = cache_at(att, args, 10000)
    x = xrows(10000, 16)
    ms = {}
    for flag in (False, True, False, True):          # A-B-A-B, the medians of each arm's two passes
        ref2.CFG["fatt_verify"] = flag
        def go():
            ys_ = []
            for _ in range(8):
                with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
                    ys_.append(att(x, c))
                c.trim(16)
            return ys_
        for _ in range(2):
            mx.eval(go())
        ts = []
        for _ in range(20):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(go()); ts.append(time.perf_counter() - t0)
        ms.setdefault(flag, []).append(1e3 * float(np.median(ts)) / 8)
    ref2.CFG["fatt_verify"] = False
    on, off = float(np.mean(ms[True])), float(np.mean(ms[False]))
    say(f"(5) T0 10000, quiet: ms per 16-row layer call, flag off {off:.3f} {ms[False]} (B6 ref2t 1.936), "
        f"on {on:.3f} {ms[True]} (B6 fs4 1.141); {off / on:.2f}x; (5) {'PASS' if abs(on / 1.141 - 1) <= 0.15 else 'FAIL'}")
    sys.exit(0)
verdict = {}
# (2) flag off == the pre-switch ref2, bitwise, in two subprocesses
py = sys.executable
tmp = {w: f"{LEAP}/data/f1_dump_{w}.npz" for w in ("old", "new")}
cnt = {}
for w in ("old", "new"):
    r = subprocess.run([py, __file__, "--dump", w, tmp[w]], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cnt[w] = r.stdout.strip().splitlines()[-1]
A, B = np.load(tmp["old"]), np.load(tmp["new"])
same = {k: bool(np.array_equal(A[k], B[k])) for k in A.files}
verdict["(2) flag off == pre-switch, bitwise"] = all(same.values())
say(f"(2) flag off vs data/ref2_before_f1.py, y bitwise per T0_R: {same}; counts old {cnt['old']}, new {cnt['new']}")
for w in tmp.values():
    os.remove(w)

att, args = build()
ok3, ok4 = True, True
ms = {}
for T0 in (2000, 10000):
    c = cache_at(att, args, T0)
    for R in (16, 8):
        x = xrows(T0, R)
        ys = {}
        for flag in (False, True):
            ref2.CFG["fatt_verify"] = flag
            for k_ in ("fatt_verify", "verify_old", "verify_old_eligible"):
                ref2.COUNTS[k_] = 0
            # (3) the check, then (4) the AR on the same cache
            y_sd = call(att, x, c); c.trim(R)
            n_sd = {k_: ref2.COUNTS[k_] for k_ in ("fatt_verify", "verify_old", "verify_old_eligible")}
            ar = [call(att, x[:, r:r + 1], c) for r in range(R)]
            c.trim(R)
            n_all = {k_: ref2.COUNTS[k_] for k_ in ("fatt_verify", "verify_old", "verify_old_eligible")}
            eq = [bool(np.array_equal(bits(y_sd[:, r]), bits(ar[r][:, 0]))) for r in range(R)]
            took = (n_all["fatt_verify"] == R + 1 and n_all["verify_old_eligible"] == 0) if flag else n_all["fatt_verify"] == 0
            ok3 &= took
            if flag:
                ok4 &= all(eq)
            ys[flag] = y_sd
            say(f"T0 {T0} R {R:2d} flag {'on ' if flag else 'off'}: counts check {n_sd}, check + {R} AR steps {n_all} "
                f"-> switch {'OK' if took else 'WRONG'}; SD == AR bitwise {sum(eq)}/{R}"
                + ("" if all(eq) else f" (first differing row {eq.index(False)})"))
        ro, rf = (np.array(ys[f].astype(mx.float32)) for f in (False, True))
        say(f"  y(on) vs y(off): mean |diff| / RMS {np.abs(rf - ro).mean() / np.sqrt((ro ** 2).mean()):.2e}")
    if T0 == 10000:   # (5) ms per 16-row layer call, B6's method (8 chained with trim, median of 20)
        x = xrows(T0, 16)
        for flag in (False, True):
            ref2.CFG["fatt_verify"] = flag
            def go():
                ys_ = []
                for _ in range(8):
                    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
                        ys_.append(att(x, c))
                    c.trim(16)
                return ys_
            for _ in range(2):
                mx.eval(go())
            ts = []
            for _ in range(20):
                mx.synchronize(); t0 = time.perf_counter(); mx.eval(go()); ts.append(time.perf_counter() - t0)
            ms[flag] = 1e3 * float(np.median(ts)) / 8
        say(f"(5) T0 10000: ms per 16-row layer call, flag off {ms[False]:.3f} (B6 ref2t 1.936), on {ms[True]:.3f} "
            f"(B6 fs4 1.141); {ms[False] / ms[True]:.2f}x")
    mx.clear_cache()
ref2.CFG["fatt_verify"] = False
verdict["(3) switch takes effect (counts)"] = ok3
verdict["(4) SD == AR bitwise, flag on"] = ok4
verdict["(5) speed within 15 % of B6's fs4"] = abs(ms[True] / 1.141 - 1) <= 0.15
r1 = ref2.self_test()
verdict["(1) self_test"] = all(r1.values())
say(f"F1-SWITCH smoke: {'PASS' if all(verdict.values()) else 'FAIL'} {verdict}")
