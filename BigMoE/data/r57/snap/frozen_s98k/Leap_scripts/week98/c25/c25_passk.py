"""c25_passk.py (Leap, Mon 16 Nov 2026 14:3x; CHECK25): per-class GPU time of the lossy 32-row check, ported from
Breakthrough's CHECK55 engine pass K (~/Breakthrough/scripts/m1/check55/c55_engine.py, read-only source; Leap's copy is
adapted to the c25 arms and knobs). Each op class is flushed into its own command buffers (the original mx.async_eval at
every class boundary) and tagged through libmtlprof (MP.tag), so a class's GPU ms = the GPU time of the command buffers that
carry its tag. The flushes change the scheduling, so the class times are scaled by (pass T GPU busy / pass K class sum) per
arm, as CHECK55 did. The wrappers go on the MTPLX classes and functions AFTER the FAST stack (ref2 / v32 / f32one / hc4i /
int8d) is installed, so they time whatever those installed. Bits: the pass-K logits vs the timing pass's (must agree).
run(E, G, P, S0, gids, gen, key, LG) from c25_engine.window() on the first window (C25_PASSK=1); E is c25_engine."""
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

CLASSES = ["other", "ple", "hc_read", "hc_write", "gdn_proj", "gdn_core", "att_proj", "att_core", "moe_router",
           "moe_experts", "moe_shared", "moe_tail", "head_mixer", "head_lm", "read", "commit"]
TI = {c: i for i, c in enumerate(CLASSES)}
K = dict(on=False, cur=0, layer=-1, lastu=0.0, acc={}, saved=[], ae0=None, MP=None, Q=None, vocab=0)


def _collect(objs, out):
    for o in objs:
        if isinstance(o, mx.array):
            out.append(o)
        elif isinstance(o, (tuple, list)):
            _collect(o, out)
        elif isinstance(o, dict):
            _collect(list(o.values()), out)
        elif o is not None and not isinstance(o, nn.Module) and hasattr(o, "state"):
            out.extend(K["Q"]._midloop_state_arrays([o]))
    return out


def _switch(tag, objs):
    MP = K["MP"]
    u = time.thread_time()
    a = K["acc"].setdefault((K["layer"], K["cur"]), [0.0, 0.0, 0, 0])
    a[0] += u - K["lastu"]
    arrs = _collect(objs, [])
    if arrs:
        c0 = MP.counts()
        u1 = time.thread_time()
        K["ae0"](*arrs)
        a[1] += time.thread_time() - u1
        a[2] += MP.counts()["disp"] - c0["disp"]
        a[3] += 1
    K["cur"] = tag
    MP.tag(tag)
    K["lastu"] = time.thread_time()


def _leaf(name, extra=None, after=None):
    def deco(f):
        def w(*a, **k):
            if not K["on"]:
                return f(*a, **k)
            nm = name(a) if callable(name) else name
            if nm is None:
                return f(*a, **k)
            prev = K["cur"]
            _switch(TI[nm], a)
            out = f(*a, **k)
            _switch(after(prev) if after else prev, (out,) + (tuple(extra(a, k)) if extra else ()))
            return out
        return w
    return deco


def _put(obj, attr, w):
    had = attr in getattr(obj, "__dict__", {}) if not isinstance(obj, dict) else True
    K["saved"].append((obj, attr, obj[attr] if isinstance(obj, dict) else getattr(obj, attr), had))
    if isinstance(obj, dict):
        obj[attr] = w
    else:
        setattr(obj, attr, w)


def install(G):
    Q = K["Q"]
    model = G["model"]
    inner = model.language_model.model
    mixer = inner.hyper_connection_mixer
    lidx = {id(l): i for i, l in enumerate(inner.layers)}
    qmm0 = mx.quantized_matmul

    def qmm_w(*a, **k):
        if not K["on"]:
            return qmm0(*a, **k)
        N, c = int(a[1].shape[0]), CLASSES[K["cur"]]
        nm = "head_lm" if N == K["vocab"] else "gdn_proj" if c == "gdn_core" else "att_proj" if c == "att_core" else \
            "moe_shared" if c in ("moe_router", "moe_tail") and N == 1 else None
        if nm is None:
            return qmm0(*a, **k)
        prev = K["cur"]
        _switch(TI[nm], a)
        out = qmm0(*a, **k)
        _switch(prev, (out,))
        return out
    _put(mx, "quantized_matmul", qmm_w)
    _put(Q.PLELayer, "__call__", _leaf("ple")(Q.PLELayer.__call__))
    _put(Q.GatedResidual, "__call__", _leaf(lambda a: "head_mixer" if a[0] is mixer else "hc_read")(Q.GatedResidual.__call__))
    _put(Q, "_hyper_residual_write", _leaf("hc_write")(Q._hyper_residual_write))
    _put(Q.GatedDeltaNet, "__call__", _leaf("gdn_core", extra=lambda a, k: tuple(a[3:4]) + ((k["cache"],) if "cache" in k else ()))(
        Q.GatedDeltaNet.__call__))
    _put(Q.Attention, "__call__", _leaf("att_core", extra=lambda a, k: tuple(a[2:3]) + ((k["cache"],) if "cache" in k else ()))(
        Q.Attention.__call__))
    _put(Q.SparseMoeBlock, "__call__", _leaf("moe_router")(Q.SparseMoeBlock.__call__))
    sw_cls = type(inner.layers[0].mlp.switch_mlp)
    _put(sw_cls, "__call__", _leaf("moe_experts", after=lambda p: TI["moe_tail"] if CLASSES[p] in ("moe_router", "moe_experts")
                                   else p)(sw_cls.__call__))
    try:
        import glue_fused
        _put(glue_fused, "shared_cat", _leaf(lambda a: "moe_shared")(glue_fused.shared_cat))
    except ImportError:
        pass
    if "shared_cat" in G:
        _put(G, "shared_cat", _leaf(lambda a: "moe_shared")(G["shared_cat"]))
    dec0 = Q.DecoderLayer.__call__

    def dec_w(self, *a, **k):
        if not K["on"]:
            return dec0(self, *a, **k)
        prevL = K["layer"]
        _switch(K["cur"], a + tuple(k.values()))
        K["layer"] = lidx.get(id(self), -1)
        out = dec0(self, *a, **k)
        _switch(K["cur"], (out,))
        K["layer"] = prevL
        return out
    _put(Q.DecoderLayer, "__call__", dec_w)


def uninstall():
    K["on"] = False
    for obj, attr, f, had in reversed(K["saved"]):
        if isinstance(obj, dict):
            obj[attr] = f
        elif had or not isinstance(obj, type):
            setattr(obj, attr, f)
        else:
            try:
                delattr(obj, attr)
            except AttributeError:
                setattr(obj, attr, f)
    K["saved"].clear()
    K["MP"].tag(63)


def run(E, G, P, S0, gids, gen, key, LG, arms=("F16", "F32s", "F32o", "F32c6i"), rounds=None):
    """per arm: 1 + rounds checks with every class flushed and tagged (the first discarded); per class GPU ms (the tagged
    command buffers' GPU time), dispatches, CPU; bits vs the timing pass's first round"""
    MP = sys.modules.get("mtlprof")
    assert MP is not None, "pass K needs libmtlprof (C25_PROF=1)"
    K.update(MP=MP, Q=sys.modules["mtplx.models.qwen4_exp"], vocab=int(E.S["vocab"]),
             ae0=E.S.get("ae0") or mx.async_eval)
    rounds = int(os.environ.get("C25_KROUNDS", "3")) if rounds is None else rounds
    arms = [a for a in arms if a in E.ARMS]
    install(G)
    out = {}
    try:
        for a in arms:
            n = 16 if a.startswith("F16") else 32
            rows = []
            for r in range(1 + rounds):
                K["acc"] = {}
                g0 = MP.tag_counts()
                i0 = MP.counts()["ncb"]
                E.knobs(G, a, n)
                try:
                    K["on"] = True
                    K["cur"], K["layer"] = TI["other"], -1
                    MP.tag(K["cur"])
                    K["lastu"] = time.thread_time()
                    sn = G["CS"].snapshot_untrimmable_cache_lazy(P)
                    m0 = MP.now()
                    with G["ref2"].verify_scope(G["model"]):
                        lg = G["model"](mx.array([gids[gen:gen + n]]), cache=P)
                    _switch(TI["read"], (lg,))
                    lg = lg[0].astype(mx.float32)
                    mx.eval(lg)
                    _switch(TI["commit"], ())
                    assert G["model"].commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
                    mx.eval(G["states"](P))
                    mx.synchronize()
                    _switch(TI["other"], ())
                    m1 = MP.now()
                finally:
                    K["on"] = False
                    E.unknob(G)
                i1 = MP.counts()["ncb"]
                g1 = MP.tag_counts()
                E.back(G, P, S0, n)
                if r == 0:
                    continue
                cb = MP.cbs(i0, i1)
                gpu = np.zeros(len(CLASSES))
                for s_, e_, t_, rd in zip(cb["start"], cb["end"], cb["tag"], cb["ready"]):
                    if rd and 0 <= t_ < len(CLASSES):
                        gpu[t_] += max(0.0, e_ - s_)
                cpu = np.zeros(len(CLASSES))
                for (L, t), v in K["acc"].items():
                    cpu[t] += v[0]
                rows.append(dict(gpu=gpu.tolist(), cpu=cpu.tolist(), disp=(g1 - g0)[:len(CLASSES)].tolist(),
                                 busy=MP.busy(cb, m0, m1), wall=m1 - m0, ncb=int(i1 - i0),
                                 bits=E.cmp(lg, LG[a]) if a in LG else None))
            out[a] = rows
    finally:
        uninstall()
    return out


def report(E, G, PK):
    """per arm: class GPU ms (scaled to the timing pass's GPU busy), dispatches, CPU; mean over the rounds and windows"""
    W = E._walls()
    say = lambda s: E._say(G, s)
    for a, rows in PK.items():
        if not rows:
            continue
        gpu = np.mean([r["gpu"] for r in rows], axis=0) * 1e3
        cpu = np.mean([r["cpu"] for r in rows], axis=0) * 1e3
        disp = np.mean([r["disp"] for r in rows], axis=0)
        bT = [v[a]["prof"]["busy"] for v in W.values() if a in v and v[a].get("prof")]
        bz = 1e3 * float(np.median(bT)) if bT else float("nan")
        sc = bz / max(gpu.sum(), 1e-9)
        say(f"  PASS K {a}: class GPU sum {gpu.sum():.2f} ms (pass K wall {1e3 * np.mean([r['wall'] for r in rows]):.2f}, "
            f"{np.mean([r['ncb'] for r in rows]):.0f} cbs) vs pass T GPU busy {bz:.2f} -> scale {sc:.3f}; bits vs pass T "
            f"{[r['bits'] for r in rows]}")
        order = np.argsort(-gpu)
        say("    " + "; ".join(f"{CLASSES[i]} {gpu[i] * sc:.2f} ms ({disp[i]:.0f} disp, CPU {cpu[i]:.2f})"
                               for i in order if gpu[i] > 0 or disp[i] > 0))
