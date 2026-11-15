"""c55_engine.py (Breakthrough, Sun 15 Nov 2026; CHECK55 step 0 on the full engine; booked by the orchestrator for ~04:15
Mon after EvalSuite's close-call base). Called from r57_c55.py, my copy of BigMoE's r57_screen_fast.py (read-only source
d148532; only its G32T time-mode block is replaced and its output dir moved to ~/Breakthrough/data/m1/check55/engine).
Env as BigMoE's G32T runner (exact-45 loop, C 160, MAXK 320, build_g32, v32 + V32_MOE1, SB_FIRST + PLE_LATE at 5-16 rows).

Per G32T window (window()):
  cold32  one 32-row check first, on fresh tokens (the window's first check after its prefill: cold PLE rows, expert misses);
  same    2 warm-up + R57_G32TIME rounds of the arms below, rotated, each on a rollback to the prefill (G32T's same tokens):
          A16; C32; C32sb (ref2 seam_hi 32: SB_FIRST + PLE_LATE at 17-32 rows, Leap 66d02e6); C32one (Leap's v32one, seam_hi
          32); A16cpu / C32cpu (+2 ms of Python thread CPU per check); C32h1 (the LM head as one 32-row MLX call; not exact).
  passK   on the first window only, guarded: each op class flushed into its own command buffers (A16, C32), bits checked.
Then finish(): fresh 16-row checks on C55_FRESH bank windows (each: prefill, a cold check, a warm re-check of the same
tokens), and the report + JSON. Instrument: libmtlprof (mtlprof.m), thread CPU, BigMoE's stats() (misses, SSD), a PLE
stage timer (wall + CPU of MTPLX's n-gram stage)."""
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mtlprof as MP

REC = []
S = dict(ok=False, head1=False, spin=0.0, ple_w=0.0, ple_u=0.0, ple_n=0, enc=0.0, bits={}, passk={}, notes=[])
ARMS = ["A16", "C32", "C32sb", "C32one", "A16cpu", "C32cpu", "C32h1"]
TAGS = {a: 32 + i for i, a in enumerate(ARMS)}
SPIN_MS = float(os.environ.get("C55_SPIN_MS", "2.0"))


def _say(G, s):
    G["say"](s)
    S["notes"].append(s)


def setup(G):
    if S["ok"]:
        return
    model, ref2, v32, Q = G["model"], G["ref2"], G["v32"], G["Q"]
    import v32one
    v32one.install(v32, ref2)
    v32one.ON[0] = False
    S["v32one"] = v32one
    S["vocab"] = int(model.language_model.lm_head.weight.shape[0])
    S["nl"] = len(model.language_model.model.layers)
    qmm0 = mx.quantized_matmul

    def qmm_ko(*a, **k):                                 # C32h1: the LM head as ONE 32-row MLX call (ref2 above 16 rows)
        if S["head1"] and int(a[1].shape[0]) == S["vocab"]:
            return v32._W["qmm"](*a, **k)
        return qmm0(*a, **k)
    mx.quantized_matmul = qmm_ko
    dec0 = Q.DecoderLayer.__call__

    def dec_spin(self, *a, **k):                         # A16cpu / C32cpu: extra Python CPU in every layer call
        if S["spin"] > 0:
            t = time.thread_time()
            while time.thread_time() - t < S["spin"]:
                pass
        return dec0(self, *a, **k)
    Q.DecoderLayer.__call__ = dec_spin
    ae0 = mx.async_eval

    def ae_t(*a, **k):                                   # encode CPU inside SB's mid-loop submits
        u = time.thread_time()
        r = ae0(*a, **k)
        S["enc"] += time.thread_time() - u
        return r
    mx.async_eval = ae_t
    S["ae0"] = ae0
    # the PLE stage timer on MTPLX's own n-gram stage (ref2's deferral calls it through _ORIG either way)
    if "ple_stage" in ref2._ORIG:
        st0 = ref2._ORIG["ple_stage"]

        def st_t(*a, **k):
            w, u = time.perf_counter(), time.thread_time()
            r = st0(*a, **k)
            S["ple_w"] += time.perf_counter() - w
            S["ple_u"] += time.thread_time() - u
            S["ple_n"] += 1
            return r
        ref2._ORIG["ple_stage"] = st_t
        S["ple_where"] = "ref2._ORIG"
    else:
        st0 = Q.NGramEmbedding.stage

        def st_t(self, *a, **k):
            w, u = time.perf_counter(), time.thread_time()
            r = st0(self, *a, **k)
            S["ple_w"] += time.perf_counter() - w
            S["ple_u"] += time.thread_time() - u
            S["ple_n"] += 1
            return r
        Q.NGramEmbedding.stage = st_t
        S["ple_where"] = "NGramEmbedding.stage"
    S["seam0"] = int(ref2.CFG.get("seam_hi", 16))
    S["ok"] = True
    _say(G, f"C55 engine: setup done (v32one installed off; vocab {S['vocab']}; {S['nl']} layers; PLE timer on {S['ple_where']}; "
            f"ref2 sb_first {ref2.CFG.get('sb_first')} ple_late {ref2.CFG.get('ple_late')} seam_hi {S['seam0']})")


def knobs(G, arm):
    ref2 = G["ref2"]
    ref2.CFG["seam_hi"] = 32 if arm in ("C32sb", "C32one") else S["seam0"]
    S["v32one"].ON[0] = arm == "C32one"
    S["head1"] = arm == "C32h1"
    S["spin"] = 1e-3 * SPIN_MS / S["nl"] if arm in ("A16cpu", "C32cpu") else 0.0


def check(G, P, toks, n, arm, key, phase, rnd=-1):
    """one verify + read + full-accept commit, profiled; the caller rolls back"""
    model, ref2, CS, states, stats = G["model"], G["ref2"], G["CS"], G["states"], G["stats"]
    knobs(G, arm if arm in TAGS else "A16")
    sn = CS.snapshot_untrimmable_cache_lazy(P)
    mx.synchronize()
    o0 = G["BG"].ext.overflow_count()                    # BIGMOE_OVERFLOW_FLAG=1 (the env): an over-C gate call is flagged,
    MP.tag(TAGS.get(arm, 40))                            # not aborted; a flagged check is garbage and the report drops it
    s0 = np.array(stats(), np.float64)
    c0 = MP.counts()
    S["enc"] = 0.0
    S["ple_w"] = S["ple_u"] = 0.0
    S["ple_n"] = 0
    tt0 = time.time()
    t0, w0, u0 = MP.now(), time.perf_counter(), time.thread_time()
    with ref2.verify_scope(model):
        lg = model(mx.array([toks]), cache=P)
    tf, w1, u1 = MP.now(), time.perf_counter(), time.thread_time()
    enc_f = S["enc"]
    lg = lg[0].astype(mx.float32)
    mx.eval(lg)
    tr, w2, u2 = MP.now(), time.perf_counter(), time.thread_time()
    assert model.commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
    mx.eval(states(P))
    mx.synchronize()
    t3, w3, u3 = MP.now(), time.perf_counter(), time.thread_time()
    c1 = MP.counts()
    d = np.array(stats(), np.float64) - s0
    ovf = int(G["BG"].ext.overflow_count() - o0)
    if ovf:
        _say(G, f"C55 {key} {arm} {phase} round {rnd}: OVERFLOW flagged ({ovf} gate calls over C): this check is dropped")
    MP.tag(63)
    knobs(G, "A16")
    REC.append(dict(arm=arm, key=key, phase=phase, round=rnd, n=n, ovf=ovf, tt=(tt0, time.time()), t=(t0, tf, tr, t3), w=(w1 - w0, w2 - w1, w3 - w2),
                    u=(u1 - u0, u2 - u1, u3 - u2), enc_f=enc_f, enc_all=S["enc"], c={k: c1[k] - c0[k] for k in c1},
                    cb=(c0["ncb"], c1["ncb"]), misses=float(d[0]), ssd_ms=1e3 * float(d[1]), slow_ms=1e3 * float(d[2]),
                    ple=(1e3 * S["ple_w"], 1e3 * S["ple_u"], S["ple_n"])))
    return lg


def back(G, P, S0, n):
    G["CS"].rollback_after_verify(P, S0, n)
    mx.eval(G["states"](P))


def cmp(a, b):
    e = mx.all(a == b, axis=-1)
    mx.eval(e)
    return int(np.array(e).sum()), int(e.shape[0])


def warm_span(key, gen, n=32):
    """the latest prompt span [s, s + n) whose distinct experts per layer (the routes file) fit the window's own gate_req,
    the bound G32TEST takes from the check rows (gen .. gen + 36), which the tier's C covers. Mon 00:25: the first engine
    attempt warmed on the prompt's last 32 rows, which nobody had screened: 179 experts at layer 0 > C 175, and BigMoE's ext
    aborted (bigmoe_ext.cpp ensure()). Returns (s, its max distinct, gate_req); s is None when no span fits."""
    R = np.load(os.path.expanduser(f"~/datasets/deepdraft/routes/{key}.npz"))["routes"]
    mu = lambda a, b: max(len(np.unique(R[a:b, l])) for l in range(R.shape[1]))
    req = mu(gen, gen + 36)
    for s in range(gen - n, -1, -1):
        m = mu(s, s + n)
        if m <= req:
            return s, m, req
    return None, None, req


def cold(G, key, P, S0, gids, gen):
    """the window's first check right after its prefill (before G32TEST's bitwise pair): fresh tokens, cold PLE rows and
    expert misses as a real check meets them"""
    setup(G)
    if not S.get("k32warm"):                             # the process's first 32-row forward builds its kernels (one-time;
        s, m, req = warm_span(key, gen)                  # 96 ms on the slice test): build them on prompt tokens that fit the
        if s is None:                                    # tier (their n-gram rows are warm from the prefill), roll back, then
            _say(G, f"C55 {key}: no prompt span fits gate_req {req}: 32-row kernels NOT pre-built (this cold check includes them)")
        else:                                            # the cold check on fresh tokens
            knobs(G, "C32")
            o0 = G["BG"].ext.overflow_count()
            with G["ref2"].verify_scope(G["model"]):
                lgw = G["model"](mx.array([gids[s:s + 32]]), cache=P)
            mx.eval(lgw[0].astype(mx.float32))
            o1 = G["BG"].ext.overflow_count()
            back(G, P, S0, 32)
            knobs(G, "A16")
            S["k32warm"] = True
            _say(G, f"C55 {key}: 32-row kernels built on prompt tokens {s}..{s + 31} (routes: <= {m} distinct experts a layer, "
                    f"gate_req {req}; overflow flags {o1 - o0}) before the cold check")
    check(G, P, gids[gen:gen + 32], 32, "C32", key, "cold32")
    back(G, P, S0, 32)


def window(G, key, P, S0, gids, gen, Greq, first):
    setup(G)
    TMR = int(os.environ.get("R57_G32TIME", "12") or 12)
    WU = int(os.environ.get("R57_G32TIME_WARM", "2") or 2)
    t_w = time.time()
    LG = {}
    for r in range(WU + TMR):
        rot = r % len(ARMS)
        for a in ARMS[rot:] + ARMS[:rot]:
            n = 16 if a.startswith("A16") else 32
            lg = check(G, P, gids[gen:gen + n], n, a, key, "same" if r >= WU else "warm", r - WU)
            back(G, P, S0, n)
            if r == WU:
                LG[a] = lg
    b = {f"{a}_vs_C32": cmp(LG[a], LG["C32"]) for a in ("C32sb", "C32one", "C32cpu", "C32h1")}
    b["A16cpu_vs_A16"] = cmp(LG["A16cpu"], LG["A16"])
    S["bits"][key] = b
    _say(G, f"C55 {key}: same-token rounds done in {time.time() - t_w:.0f} s; bits (rows equal / rows) {b} (C32h1 is not exact: info)")
    if first and os.environ.get("C55_PASSK", "1") == "1":
        try:
            passk(G, P, S0, gids, gen, key, LG)
        except Exception as e:                             # never lose the timing data to the attribution pass
            _say(G, f"C55 pass K FAILED on {key}: {e!r}; the timing data stands")
            uninstall_attr(G)
    dump(G)


# ---- pass K: op classes flushed into their own command buffers (as c55_profile.py; the engine's MoE block too)
CLASSES = ["other", "ple", "hc_read", "hc_write", "gdn_proj", "gdn_core", "att_proj", "att_core", "moe_router",
           "moe_experts", "moe_shared", "moe_tail", "head_mixer", "head_lm", "read", "commit"]
TI = {c: i for i, c in enumerate(CLASSES)}
K = dict(on=False, cur=0, layer=-1, lastu=0.0, acc={}, saved=[])


def _collect(objs, out, Q):
    for o in objs:
        if isinstance(o, mx.array):
            out.append(o)
        elif isinstance(o, (tuple, list)):
            _collect(o, out, Q)
        elif isinstance(o, dict):
            _collect(list(o.values()), out, Q)
        elif o is not None and not isinstance(o, nn.Module) and hasattr(o, "state"):
            out.extend(Q._midloop_state_arrays([o]))
    return out


def _switch(G, tag, objs):
    u = time.thread_time()
    key = (K["layer"], K["cur"])
    a = K["acc"].setdefault(key, [0.0, 0.0, 0, 0])
    a[0] += u - K["lastu"]
    arrs = _collect(objs, [], G["Q"])
    if arrs:
        c0 = MP.counts()
        u1 = time.thread_time()
        S["ae0"](*arrs)
        a[1] += time.thread_time() - u1
        a[2] += MP.counts()["disp"] - c0["disp"]
        a[3] += 1
    K["cur"] = tag
    MP.tag(tag)
    K["lastu"] = time.thread_time()


def _leaf(G, name, extra=None, after=None):
    def deco(f):
        def w(*a, **k):
            if not K["on"]:
                return f(*a, **k)
            nm = name(a) if callable(name) else name
            if nm is None:
                return f(*a, **k)
            prev = K["cur"]
            _switch(G, TI[nm], a)
            out = f(*a, **k)
            _switch(G, after(prev) if after else prev, (out,) + (tuple(extra(a, k)) if extra else ()))
            return out
        return w
    return deco


def install_attr(G):
    model, Q, ref2 = G["model"], G["Q"], G["ref2"]
    inner = model.language_model.model
    mixer = inner.hyper_connection_mixer
    lidx = {id(l): i for i, l in enumerate(inner.layers)}

    def put(obj, attr, w):
        had = attr in getattr(obj, "__dict__", {})
        K["saved"].append((obj, attr, getattr(obj, attr), had))
        setattr(obj, attr, w)
    qmm0 = mx.quantized_matmul

    def qmm_w(*a, **k):
        if not K["on"]:
            return qmm0(*a, **k)
        N, c = int(a[1].shape[0]), CLASSES[K["cur"]]
        nm = "head_lm" if N == S["vocab"] else "gdn_proj" if c == "gdn_core" else "att_proj" if c == "att_core" else \
            "moe_shared" if c in ("moe_router", "moe_tail") and N == 1 else None
        if nm is None:
            return qmm0(*a, **k)
        prev = K["cur"]
        _switch(G, TI[nm], a)
        out = qmm0(*a, **k)
        _switch(G, prev, (out,))
        return out
    put(mx, "quantized_matmul", qmm_w)
    put(Q.PLELayer, "__call__", _leaf(G, "ple")(Q.PLELayer.__call__))
    put(Q.GatedResidual, "__call__", _leaf(G, lambda a: "head_mixer" if a[0] is mixer else "hc_read")(Q.GatedResidual.__call__))
    put(Q, "_hyper_residual_write", _leaf(G, "hc_write")(Q._hyper_residual_write))
    put(Q.GatedDeltaNet, "__call__", _leaf(G, "gdn_core", extra=lambda a, k: tuple(a[3:4]) + ((k["cache"],) if "cache" in k else ()))(
        Q.GatedDeltaNet.__call__))
    put(Q.Attention, "__call__", _leaf(G, "att_core", extra=lambda a, k: tuple(a[2:3]) + ((k["cache"],) if "cache" in k else ()))(
        Q.Attention.__call__))
    put(Q.SparseMoeBlock, "__call__", _leaf(G, "moe_router")(Q.SparseMoeBlock.__call__))
    sw_cls = type(inner.layers[0].mlp.switch_mlp)
    put(sw_cls, "__call__", _leaf(G, "moe_experts", after=lambda p: TI["moe_tail"] if CLASSES[p] in ("moe_router", "moe_experts")
                                  else p)(sw_cls.__call__))
    import glue_fused
    sc = _leaf(G, lambda a: "moe_shared")(glue_fused.shared_cat)
    put(glue_fused, "shared_cat", sc)                    # v32's 17-32-row tier block imports it at call time
    if "shared_cat" in G:
        K["saved"].append((G, "__dict_shared_cat__", G["shared_cat"], True))
        G["shared_cat"] = _leaf(G, lambda a: "moe_shared")(G["shared_cat"])   # K-73's <= 16-row block: the script's global
    dec0 = Q.DecoderLayer.__call__

    def dec_w(self, *a, **k):
        if not K["on"]:
            return dec0(self, *a, **k)
        prevL = K["layer"]
        _switch(G, K["cur"], a + tuple(k.values()))
        K["layer"] = lidx.get(id(self), -1)
        out = dec0(self, *a, **k)
        _switch(G, K["cur"], (out,))
        K["layer"] = prevL
        return out
    put(Q.DecoderLayer, "__call__", dec_w)


def uninstall_attr(G):
    K["on"] = False
    for obj, attr, f, had in reversed(K["saved"]):
        if attr == "__dict_shared_cat__":
            obj["shared_cat"] = f
        elif had or not isinstance(obj, type):
            setattr(obj, attr, f)
        else:
            try:
                delattr(obj, attr)
            except AttributeError:
                setattr(obj, attr, f)
    K["saved"].clear()
    MP.tag(63)


def passk(G, P, S0, gids, gen, key, LG):
    install_attr(G)
    out = {}
    for a in ("A16", "C32"):
        n = 16 if a == "A16" else 32
        rows = []
        for r in range(1 + int(os.environ.get("C55_KROUNDS", "3"))):
            K["acc"] = {}
            g0 = MP.tag_counts()
            i0 = MP.counts()["ncb"]
            K["on"] = True
            K["cur"], K["layer"] = TI["other"], -1
            MP.tag(K["cur"])
            K["lastu"] = time.thread_time()
            knobs(G, a)
            sn = G["CS"].snapshot_untrimmable_cache_lazy(P)
            with G["ref2"].verify_scope(G["model"]):
                lg = G["model"](mx.array([gids[gen:gen + n]]), cache=P)
            _switch(G, TI["read"], (lg,))
            lg = lg[0].astype(mx.float32)
            mx.eval(lg)
            _switch(G, TI["commit"], ())
            assert G["model"].commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
            mx.eval(G["states"](P))
            mx.synchronize()
            _switch(G, TI["other"], ())
            K["on"] = False
            knobs(G, "A16")
            i1 = MP.counts()["ncb"]
            g1 = MP.tag_counts()
            back(G, P, S0, n)
            if r == 0:
                continue
            rows.append(dict(acc={f"{L}:{CLASSES[t]}": v for (L, t), v in K["acc"].items()}, cb=(i0, i1),
                             tagdisp=(g1 - g0)[:len(CLASSES)].tolist(), bits=cmp(lg, LG[a])))
        out[a] = rows
    uninstall_attr(G)
    S["passk"][key] = out
    _say(G, f"C55 pass K {key}: done; bits vs the timing pass " + str({a: [x["bits"] for x in v] for a, v in out.items()}))


def finish(G):
    """fresh 16-row checks on C55_FRESH windows (cold, then warm re-check of the same tokens), then the report"""
    setup(G)
    keys = [k for k in os.environ.get("C55_FRESH", "").split(",") if k]
    BG, BS, SW_ = G["BG"], G["BS"], G["SW_"]
    for k in keys:
        try:
            bz = np.load(os.path.expanduser(f"~/datasets/deepdraft/bank/{k}.npz"))
            rz = np.load(os.path.expanduser(f"~/datasets/deepdraft/routes/{k}.npz"))
            gids, gen, RT = [int(t) for t in bz["ids"]], int(bz["gen"]), rz["routes"]
            g16 = max(len(np.unique(RT[gen:gen + 20, l])) for l in range(RT.shape[1]))
            for sw in SW_:
                sw._bm["gate_req"] = g16
            G["set_arm"]("slot")
            P, _l, _T, _w = G["prefill"](gids[:gen], False, chunk=G["PFCHUNK"])
            mx.eval(G["states"](P))
            G["set_arm"]("gate+t")
            BS.GATE_N[0] = BG.MAXK
            S0 = G["CS"].snapshot_untrimmable_cache_lazy(P)
            for ph in ("fresh_cold", "fresh_warm", "fresh_warm2"):
                check(G, P, gids[gen:gen + 16], 16, "A16", k, ph)
                back(G, P, S0, 16)
            del P, S0
            mx.clear_cache()
        except Exception as e:
            _say(G, f"C55 fresh {k}: FAILED {e!r}")
    _say(G, f"C55 fresh: {len(keys)} windows done")
    dump(G, final=True)


def heat_intervals(t_from, path=os.path.expanduser("~/Library/Logs/fanwatch.log"), tail_bytes=4_000_000):
    """fanwatch's heat episodes as (start, end) epochs: 'heat pause ... PAUSE set' (or a 'froze' line) until the next
    'PAUSE cleared' (open episodes run to now). Jobs are frozen (SIGSTOP) inside these; the orchestrator 14:5x: drop or
    re-run any timed round one overlaps."""
    out, start = [], None
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            n = fh.tell()
            fh.seek(max(0, n - tail_bytes))
            lines = fh.read().decode(errors="replace").splitlines()
    except OSError:
        return [], "fanwatch.log unreadable"
    for ln in lines:
        if len(ln) < 19 or not ln[:4].isdigit():
            continue
        try:
            t = time.mktime(time.strptime(ln[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if t < t_from - 3600:
            continue
        if ("PAUSE set" in ln or " froze " in ln) and start is None:
            start = t
        elif "PAUSE cleared" in ln and start is not None:
            out.append((start, t + 1.0))
            start = None
    if start is not None:
        out.append((start, time.time() + 1.0))
    return out, "ok"


def mark_heat():
    t_from = min((r["tt"][0] for r in REC), default=time.time())
    iv, st = heat_intervals(t_from)
    for r in REC:
        a, b = r["tt"]
        r["hot"] = any(x0 < b and a < x1 for x0, x1 in iv)
    S["heat"] = dict(intervals=iv, status=st, dropped=sum(r["hot"] for r in REC))
    return iv


def dump(G, final=False):
    time.sleep(0.2)
    cb = MP.cbs(0, MP.counts()["ncb"])
    for r in REC:
        if "gpu" in r:
            continue
        t0, tf, tr, t3 = r["t"]
        c = cb[r["cb"][0]:r["cb"][1]]
        if len(c) and not c["ready"].all():
            continue
        st = [s for s, rd in zip(c["start"], c["ready"]) if rd and s >= t0]
        r.update(gpu=MP.busy(c, t0, t3), gpu_fwd=MP.busy(c, t0, tf), gpu_read=MP.busy(c, tf, tr), gpu_commit=MP.busy(c, tr, t3),
                 gpu_first=(min(st) - t0) if st else float("nan"))
    K_out = {}
    for key, d in S["passk"].items():
        K_out[key] = {}
        for a, rows in d.items():
            agg = {c: [0.0, 0.0, 0.0, 0.0] for c in CLASSES}   # disp, gpu ms, build ms, enc ms per check
            for x in rows:
                for c in CLASSES:
                    agg[c][0] += x["tagdisp"][TI[c]]
                cc = cb[x["cb"][0]:x["cb"][1]]
                ok = cc["ready"] == 1
                for c in CLASSES:
                    sel = ok & (cc["tag"] == TI[c])
                    agg[c][1] += 1e3 * float(np.sum((cc["end"] - cc["start"])[sel]))
                for kk, v in x["acc"].items():
                    c = kk.split(":", 1)[1]
                    agg[c][2] += 1e3 * v[0]
                    agg[c][3] += 1e3 * v[1]
            n = max(len(rows), 1)
            K_out[key][a] = {c: [v / n for v in vals] for c, vals in agg.items()}
    psos = {k: v for k, v in MP.psos().items()}
    kern = {}
    for a, t in TAGS.items():
        tp = MP.tag_pso(t)
        kern[a] = {psos.get(int(sl), ("?", 0))[0]: int(tp[sl]) for sl in np.nonzero(tp)[0]}
    mark_heat()
    out = os.path.join(G["OUT"], f"{G['TAG']}_c55engine.json")
    json.dump(dict(when=time.strftime("%a %d %b %Y %H:%M:%S"), final=final, records=REC, bits=S["bits"], passk=K_out, heat=S.get("heat"),
                   kernels_cum=kern, notes=S["notes"], spin_ms=SPIN_MS, seam0=S.get("seam0")),
              open(out, "w"), indent=0, default=str)
    if final:
        report(G, out)


def report(G, path):
    Rall = [r for r in REC if "gpu" in r]
    hot_rounds = {(r["key"], r["round"]) for r in Rall if r.get("hot") and r["phase"] == "same"}
    R = [r for r in Rall if not r.get("hot") and not r.get("ovf") and not (r["phase"] == "same" and (r["key"], r["round"]) in hot_rounds)]
    m = lambda v: float(np.median(v)) if len(v) else float("nan")
    hs = S.get("heat", {})
    L = [f"C55 engine profile ({time.strftime('%a %d %b %H:%M')}): exact-45 loop, C 160; medians per check (ms)",
         f"heat (fanwatch): {len(hs.get('intervals', []))} episode(s) in the run; {hs.get('dropped', 0)} checks overlapped -> dropped, "
         f"and {len(hot_rounds)} same-token round(s) dropped whole ({hs.get('status')})",
         f"overflow guard (BIGMOE_OVERFLOW_FLAG=1): {sum(1 for r in REC if r.get('ovf'))} check(s) flagged -> dropped"]
    L.append("| arm | phase | n | wall | CPU build | CPU encode (SB) | CPU read + commit | GPU busy | GPU idle | GPU first start | "
             "dispatches | misses | SSD ms | PLE stage ms (wall / CPU) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ph in ("same", "cold32", "fresh_cold", "fresh_warm", "fresh_warm2"):
        for a in ARMS:
            x = [r for r in R if r["phase"] == ph and r["arm"] == a]
            if not x:
                continue
            W = [1e3 * sum(r["w"]) for r in x]
            L.append(f"| {a} | {ph} | {len(x)} | {m(W):.2f} | {m([1e3 * (r['u'][0] - r['enc_f']) for r in x]):.2f} | "
                     f"{m([1e3 * r['enc_f'] for r in x]):.2f} | {m([1e3 * (r['u'][1] + r['u'][2]) for r in x]):.2f} | "
                     f"{m([1e3 * r['gpu'] for r in x]):.2f} | {m([1e3 * (sum(r['w']) - r['gpu']) for r in x]):.2f} | "
                     f"{m([1e3 * r['gpu_first'] for r in x]):.2f} | {m([r['c']['disp'] for r in x]):.0f} | "
                     f"{m([r['misses'] for r in x]):.0f} | {m([r['ssd_ms'] for r in x]):.2f} | {m([r['ple'][0] for r in x]):.2f} / "
                     f"{m([r['ple'][1] for r in x]):.2f} |")
    L.append("paired per round and window (same tokens), median [IQR] ms of wall; GPU busy delta:")
    for a, b in (("C32sb", "C32"), ("C32one", "C32"), ("C32one", "C32sb"), ("C32cpu", "C32"), ("A16cpu", "A16"), ("C32h1", "C32"),
                 ("C32", "A16")):
        d, g = [], []
        for key in sorted(set(r["key"] for r in R if r["phase"] == "same")):
            xa = {r["round"]: r for r in R if r["phase"] == "same" and r["arm"] == a and r["key"] == key}
            xb = {r["round"]: r for r in R if r["phase"] == "same" and r["arm"] == b and r["key"] == key}
            for rr in sorted(set(xa) & set(xb)):
                d.append(1e3 * (sum(xa[rr]["w"]) - sum(xb[rr]["w"])))
                g.append(1e3 * (xa[rr]["gpu"] - xb[rr]["gpu"]))
        if d:
            L.append(f"  {a} - {b}: wall {np.median(d):+.2f} [{np.percentile(d, 25):+.2f}, {np.percentile(d, 75):+.2f}] over {len(d)}; "
                     f"GPU busy {np.median(g):+.2f}")
    fr = {}
    for r in R:
        if r["phase"].startswith("fresh"):
            fr.setdefault(r["key"], {})[r["phase"]] = r
    dl = [(1e3 * (sum(v["fresh_cold"]["w"]) - sum(v["fresh_warm"]["w"])), v["fresh_cold"]["ple"][0] - v["fresh_warm"]["ple"][0],
           v["fresh_cold"]["ssd_ms"] - v["fresh_warm"]["ssd_ms"], v["fresh_cold"]["misses"] - v["fresh_warm"]["misses"])
          for v in fr.values() if "fresh_cold" in v and "fresh_warm" in v]
    if dl:
        D = np.array(dl)
        L.append(f"fresh 16-row checks, cold - warm per window (n {len(D)}): wall {np.median(D[:, 0]):+.2f} ms [IQR "
                 f"{np.percentile(D[:, 0], 25):+.2f}, {np.percentile(D[:, 0], 75):+.2f}]; PLE stage {np.median(D[:, 1]):+.2f} ms; "
                 f"SSD {np.median(D[:, 2]):+.2f} ms; misses {np.median(D[:, 3]):+.0f}")
    L.append(f"bits: {S['bits']}")
    for key, d in S["passk"].items():
        L.append(f"pass K {key}: bits vs timing {[[x['bits'] for x in v] for v in d.values()]}")
    txt = path.replace(".json", ".txt")
    open(txt, "w").write("\n".join(L) + "\n")
    for s in L:
        G["say"](s)
