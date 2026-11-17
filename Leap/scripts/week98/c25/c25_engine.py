"""c25_engine.py (Leap, Mon 16 Nov 2026 12:5x; CHECK25, data/week98/check25_plan.md): the LOSSY 32-row check's engine A/Bs.
Called from c25_loop.py (a copy of BigMoE's r57_screen_fast.py, scripts/week98/c25/make_c25_loop.py), whose G32T time mode
calls window() per R57_G32TEST window and finish() at the end. Env: c25_env.sh = BigMoE's 1c-tiers set (t1cx45t1c1:
lossy-45 + R57_TIERS=1c) + the G32T test mode (MAXK 320, build_g32, V32_MOE1, v32 over ref2) minus REF2_GDN_LAZY (v32
refuses it; no bits change outside GDNB).
Per window: R57_G32TIME_WARM warm-ups + R57_G32TIME timed rounds of the arms, rotated. Each check = a verify, the logits'
read and a full-accept commit, on a rollback to the window's prefill: the bank's exact continuation (every draft right),
warm (the pool holds the window's experts after the warm-ups: misses ~0, counted). 1c-tiers engaged on every check (FG t1c,
its rows), the per-layer union counts by tier read after each check -> expert bytes per check.
Arms (C25_ARMS selects), each a set of components (CFG below):
  F16      the 16-row lossy check (lossy-45 + 1c at lambda 1.6e-2): the base
  F32s     32 rows through v32's split (every op 2 x 16, cores chunked): today's > 16-row FAST route
  F32o     + f32one (one pass: qmm32n dense, HC8 read32, one-call GDN / attention, fused SDPA, SVH qmm16c2a; bitwise forms)
  F32d     F32o with the dense form dn32 (bf16 x uint4 from threadgroup memory; FAST)
  F32df    F32d + the fused 1c-tiers kernel (t1cf.py; == BigMoE's glue on 384,000 / 384,000 slots)
  F32if    F32df + int8d (int8 dense activations; G1's lever)
  F32if6   F32if at lambda 1.8e-2 (C6's experts) + int8
  F32c5    F32df + V3 (HC mixers at 4-bit: hc4i -> hc4r / hc4r32) + V1 (a 16K screened head): C5's bytes (Friday's number 1)
  F32c6i   F32c5 + int8d at lambda 1.8e-2: C6 + int8 (Friday's number 2)
  F32g     F32o + f32one.FASTGDN (gdn_fast.py: the GDN core bit for bit == mlx_lm's; must equal F32o in every row)
  F32c6g   F32c6i + FASTGDN (C25_ARMS selects these two; the default stays the 11-16 nine)
  F16g     F16 at lambda 1.8e-2: G1's 16-row check (int8-free, 8-bit HC mixers; Saturday's TREE-16 path; 11-17)
  F16gd    F16g + GDN16 (m16.py: gdn_fast at 2-16 rows; bitwise: must equal F16g in every row)
  F32c6p   F32c6i + HC32P (m16.py: the 32-row HC read in one launch, per-call barrier slots; bitwise == F32c6i)
  F32c6x   F32c6i + the bitwise stack: FASTGDN + HC32P + seams at 32 rows (ref2 seam_hi 32) + int8fold (MODE bit)
  F16gk / F32c6k   F16g / F32c6i + Breakthrough's expert one-call tgl16e (knob_v1 xknob; EXACT: == ooo1cq bit for bit)
  F16gu / F32c6u   F16g / F32c6i + tgl16eu (tgl16e + the TIER-4 UPGRADE: FAST, batch-dependent; its own gate)
  F16gx    F16g + GDN16 + tgl16eu: Saturday's projected 16-row check (the flags BigMoE would get)
  F32c6xu  F32c6x + tgl16eu: the projected 32-row check (the whole stack)
  V1 / V3 here are TIMING emulations (4-bit mixers requantized from the 8-bit copies; the 16K head = FR-Spec's top 16,384
  ids, not the code list): their bytes and kernels are C5's, their tokens are not EvalSuite's V1 / V3.
  (the chunked gated delta rule, gdn32c.py, is correct but 3x slower as built (Mon 12:58): not an arm until redesigned)
Bits / agreement on the first timed round: F32o vs F32s (rows bitwise + argmax), F32d / F32i / F32*6 vs F32s (argmax rows,
max |logit diff|). Report: per-arm median wall (and its parts: build + submit, wait, commit), ratio to F16, paired diffs."""
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

REC = []
S = dict(ok=False, bits={}, notes=[])
ALL = ["F16", "F32s", "F32o", "F32d", "F32df", "F32if", "F32if6", "F32c5", "F32c6i", "F32g", "F32c6g", "F16g", "F16gd",
       "F32c6p", "F32c6x", "F16gk", "F16gu", "F32c6k", "F32c6u", "F16gx", "F32c6xu"]
CFG = {"F16": {}, "F32s": {}, "F32o": dict(one=True, form="qmm32n"), "F32d": dict(one=True, form="dn32"),
       "F32df": dict(one=True, form="dn32", t1cf=True), "F32if": dict(one=True, form="dn32", t1cf=True, int8=True),
       "F32if6": dict(one=True, form="dn32", t1cf=True, int8=True, lam=0.018),
       "F32c5": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True),
       "F32c6i": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018),
       "F32g": dict(one=True, form="qmm32n", fastgdn=True),          # F32o + gdn_fast (bitwise: == F32o in every row)
       "F32c6g": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, fastgdn=True),
       "F16g": dict(lam=0.018), "F16gd": dict(lam=0.018, gdn16=True),
       "F32c6p": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, hc32p=True),
       "F32c6x": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, fastgdn=True, hc32p=True,
                     seam=True, fold=True),
       "F16gk": dict(lam=0.018, xk="tgl16e"), "F16gu": dict(lam=0.018, xk="tgl16eu"),
       "F32c6k": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, xk="tgl16e"),
       "F32c6u": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, xk="tgl16eu"),
       "F16gx": dict(lam=0.018, gdn16=True, xk="tgl16eu"),
       "F32c6xu": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, fastgdn=True, hc32p=True,
                      seam=True, fold=True, xk="tgl16eu")}
ARMS = [a for a in os.environ.get("C25_ARMS", ",".join(ALL[:9])).split(",") if a]   # the 11-16 nine by default
assert ARMS and all(a in ALL for a in ARMS) and "F32s" in ARMS, ARMS
PAIRS = [("F32s", "F16", "ratio"), ("F32o", "F16", "ratio"), ("F32df", "F16", "ratio"), ("F32if6", "F16", "ratio"),
         ("F32o", "F32s", "diff"), ("F32d", "F32o", "diff"), ("F32df", "F32d", "diff"), ("F32if", "F32df", "diff"),
         ("F32if6", "F32if", "diff"), ("F32df", "F32s", "diff"), ("F32if6", "F32s", "diff"), ("F32c5", "F32df", "diff"),
         ("F32c6i", "F32c5", "diff"), ("F32c5", "F16", "ratio"), ("F32c6i", "F16", "ratio"), ("F32g", "F32o", "diff"),
         ("F32c6g", "F32c6i", "diff"), ("F32c6g", "F16", "ratio"), ("F16g", "F16", "diff"), ("F16gd", "F16g", "diff"),
         ("F32c6p", "F32c6i", "diff"), ("F32c6x", "F32c6i", "diff"), ("F32c6x", "F16", "ratio"), ("F16gd", "F16", "diff"),
         ("F16gk", "F16g", "diff"), ("F16gu", "F16g", "diff"), ("F32c6k", "F32c6i", "diff"), ("F32c6u", "F32c6i", "diff"),
         ("F16gx", "F16g", "diff"), ("F16gx", "F16", "diff"), ("F32c6xu", "F32c6i", "diff"), ("F32c6xu", "F16", "ratio")]
E4_MB, E3_MB = 2.784, 2.064                              # MB per expert READ: records3 pieces, 4-bit = MSB + LSB part, MSB-3 =
                                                         # the MSB part (corrected 11-16 13:4x from 2.135 / 1.661: check25_bound_real.py)


def _say(G, s):
    G["say"](s)
    S["notes"].append(s)


def setup(G):
    if S["ok"]:
        return
    sys.path.insert(1, os.path.expanduser("~/Leap/scripts/week98"))
    ref2, v32 = G["ref2"], G["v32"]
    import f32one
    import int8d
    import hc4i
    import mlx.nn as nn
    f32one.install("qmm32n")                           # order: v32 (the loop) -> f32one -> hc4i -> int8d (outermost)
    hc4i.install()
    int8d.install()
    import int8fold                                    # 11-17 (slot #3): after int8d; then m16 (wraps whatever is installed)
    int8fold.install()
    int8fold.ON[0] = False
    import m16
    m16.install()
    xk = None
    if any(CFG[a].get("xk") for a in ARMS):              # Breakthrough's expert one-call (frozen releases, read-only)
        for kv in (os.environ.get("C25_XKNOB", "knob_v2"), "knob_v1"):   # 11-17 11:2x: v2 = v1's outputs, O(1) grouping
            kd = os.path.expanduser(f"~/Breakthrough/data/xrw/{kv}")
            try:
                sys.path.insert(0, kd)
                import xknob as xk
                assert f"/{kv}/" in xk.__file__, xk.__file__
                xk.install()
                xk.ON[0] = False
                S["xk_rel"] = kv
                break
            except Exception as e:                       # fall back to the next release (never a mixed import)
                S["notes"].append(f"C25: xknob {kv} failed: {e!r}"[:300])
                for m_ in ("xknob", "tgl16", "tgl16v", "w4a8", "xgrp"):
                    sys.modules.pop(m_, None)
                if kd in sys.path:
                    sys.path.remove(kd)
                xk = None
        assert xk is not None, "no xknob release imports (knob_v2, knob_v1)"
    f32one.ON[0] = False
    int8d.ON[0] = False
    # V3 emulation: 4-bit copies of every trunk GatedResidual's 8-bit mixers (dequantize -> quantize at 4-bit, g64)
    Q = sys.modules["mtplx.models.qwen4_exp"]
    TM = getattr(G["model"], "language_model", G["model"])
    HC = []
    for name, mod in TM.named_modules():
        if name.startswith("mtp") or not isinstance(mod, Q.GatedResidual):
            continue
        pair = []
        for attr in ("input_mix_weight_down", "input_mix_weight_up"):
            q8 = getattr(mod, attr)
            if not (isinstance(q8, nn.QuantizedLinear) and q8.bits == 8):
                pair = None
                break
            w = mx.dequantize(q8["weight"], q8["scales"], q8["biases"], group_size=q8.group_size, bits=8).astype(mx.bfloat16)
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
            lin.weight = w
            q4 = nn.QuantizedLinear.from_linear(lin, 64, 4)
            mx.eval(q4.parameters())
            pair.append((attr, q8, q4))
        if pair:
            HC.append((mod, pair))
    # V1 emulation: a second screened-head state with a 16K base (FR-Spec's top ids), swapped in per arm
    sh_ = dict(ref2._SH)
    v1 = None
    if sh_:
        lm = G["model"].language_model.lm_head
        ref2.screened_head_install(lm, base_n=16384)
        v1 = dict(ref2._SH)
        ref2._SH.clear(); ref2._SH.update(sh_)
    S.update(hc4i=hc4i, HC=HC, sh0=sh_, sh16=v1)
    S.update(int8fold=int8fold, m16=m16, seam0=int(ref2.CFG["seam_hi"]), xk=xk)
    if xk is not None:
        _say(G, f"C25: xknob release {S.get('xk_rel')} ({xk.__file__})")
    S.update(f32one=f32one, int8d=int8d, lam0=float(G["TIERS_LAM"]), G=G,
             vocab=int(G["model"].language_model.lm_head.weight.shape[0]))
    S["ae0"] = mx.async_eval                           # the original (pass K flushes with it; c25_passk.py)
    if os.environ.get("C25_CPUSPLIT") or os.environ.get("C25_PROF"):   # 11-16 18:1x: the forward's wall split into Python
        import c25_cpusplit                            # build / MLX encode / blocked (wraps mx.async_eval + mx.eval; after ae0)
        c25_cpusplit.install()
    S["ok"] = True
    _say(G, f"C25: setup done (arms {ARMS}; f32one / hc4i / int8d installed off; V3 4-bit mixer copies {len(S['HC'])} modules; "
            f"V1 16K head {'ready' if S['sh16'] else 'NOT AVAILABLE (no SVH state)'}; lambda {S['lam0']}; TIERS {G.get('TIERS')!r}; "
            f"FAST {G.get('FAST')!r}; ref2 dn_ks {ref2.CFG.get('dn_ks')}, small_native {ref2.CFG.get('small_native')}, "
            f"hc8_fast {ref2.CFG.get('hc8_fast')}, att_single {ref2.CFG.get('att_single')}, screened_head "
            f"{ref2.CFG.get('screened_head')}, gdn_lazy {ref2.CFG.get('gdn_lazy_commit')}; DNR weights {len(ref2._DNR)})")


def knobs(G, arm, n):
    f, i, c = S["f32one"], S["int8d"], CFG[arm]
    f.ON[0] = bool(c.get("one"))
    f.FORM[0] = c.get("form", "qmm32n")
    f.CHUNK[0] = bool(c.get("chunk"))
    f.FASTGDN[0] = bool(c.get("fastgdn"))
    G["C25_T1CF"][0] = bool(c.get("t1cf"))
    i.ON[0] = bool(c.get("int8"))
    S["m16"].GDN16[0] = bool(c.get("gdn16"))
    S["m16"].HC32P[0] = bool(c.get("hc32p"))
    S["int8fold"].ON[0] = bool(c.get("fold"))
    G["ref2"].CFG["seam_hi"] = 32 if c.get("seam") else S["seam0"]
    if S["xk"] is not None:
        S["xk"].ON[0] = bool(c.get("xk"))
        if c.get("xk"):
            S["xk"].MODE[0] = c["xk"]
    G["TIERS_LAM"] = c.get("lam", S["lam0"])
    G["FG"].update(t1c=True, rows=n, t1cn=[])
    _swap(G, bool(c.get("hc4")), bool(c.get("v1")))


def _swap(G, hc4, v1):
    """V3 / V1 emulations on (4-bit mixers, 16K head) or off (8-bit mixers, the base head)"""
    if S.get("hc_on") != hc4:
        for mod, pair in S["HC"]:
            for attr, q8, q4 in pair:
                setattr(mod, attr, q4 if hc4 else q8)
        S["hc_on"] = hc4
    want = S["sh16"] if (v1 and S["sh16"]) else S["sh0"]
    if want and S.get("sh_on") != v1:
        G["ref2"]._SH.clear(); G["ref2"]._SH.update(want)
        S["sh_on"] = v1


def unknob(G):
    S["f32one"].ON[0] = False
    S["f32one"].CHUNK[0] = False
    S["f32one"].FASTGDN[0] = False
    G["C25_T1CF"][0] = False
    _swap(G, False, False)
    S["int8d"].ON[0] = False
    S["m16"].GDN16[0] = False
    S["m16"].HC32P[0] = False
    S["int8fold"].ON[0] = False
    G["ref2"].CFG["seam_hi"] = S["seam0"]
    if S.get("xk") is not None:
        S["xk"].ON[0] = False
    G["TIERS_LAM"] = S["lam0"]
    G["FG"].update(t1c=False, rows=0, t1cn=None)


def check(G, P, toks, n, arm, key, phase, rnd=-1):
    """one verify + the logits' read + a full-accept commit; the caller rolls back"""
    model, ref2, CS, states = G["model"], G["ref2"], G["CS"], G["states"]
    knobs(G, arm, n)
    MP = _mp()
    prof = None
    try:
        sn = CS.snapshot_untrimmable_cache_lazy(P)
        st0 = G["stats"]()
        ext = G["BG"].ext
        ov0 = ext.overflow_count() if hasattr(ext, "overflow_count") else 0
        mx.synchronize()
        pso = MP is not None and phase == "same" and rnd == 0       # the kernels by name, once per arm and window
        if MP is not None:
            p0 = MP.psos() if pso else None
            c0 = MP.counts()
            m0, tc0 = MP.now(), time.thread_time()
        CSP = sys.modules.get("c25_cpusplit")             # installed by setup() under C25_CPUSPLIT or C25_PROF
        sa0 = CSP.snap() if CSP is not None else None
        w0 = time.perf_counter()
        with ref2.verify_scope(model):
            lg = model(mx.array([toks]), cache=P)
        w1 = time.perf_counter()
        sa1 = CSP.snap() if CSP is not None else None
        if MP is not None:
            m1, tc1 = MP.now(), time.thread_time()
        lg = lg[0].astype(mx.float32)
        mx.eval(lg)
        w2 = time.perf_counter()
        if MP is not None:
            m2, tc2 = MP.now(), time.thread_time()
        assert model.commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
        mx.eval(states(P))
        mx.synchronize()
        w3 = time.perf_counter()
        if MP is not None:
            m3, tc3 = MP.now(), time.thread_time()
            c1 = MP.counts()
            cb = MP.cbs(c0["ncb"], c1["ncb"])
            rd = cb["ready"] > 0
            prof = dict(busy=MP.busy(cb, m0, m3), busy_p=[MP.busy(cb, m0, m1), MP.busy(cb, m1, m2), MP.busy(cb, m2, m3)],
                        cpu=[tc1 - tc0, tc2 - tc1, tc3 - tc2], disp=c1["disp"] - c0["disp"], ncb=c1["ncb"] - c0["ncb"],
                        bar=c1["bar"] - c0["bar"], enc=c1["enc"] - c0["enc"], unready=int((~rd).sum()),
                        first=float(cb["start"][rd].min() - m0) if rd.any() else float("nan"),
                        last=float(m3 - cb["end"][rd].max()) if rd.any() else float("nan"))
            if CSP is not None:                         # ms: Python build / MLX encode / blocked in eval / other waits
                prof["split"] = CSP.parts(CSP.delta(sa0, sa1), w1 - w0, tc1 - tc0)
            if pso:
                S.setdefault("pso", {}).setdefault(key, {})[arm] = MP.pso_delta(p0, MP.psos())
        st = G["stats"]() - st0
        ov = (ext.overflow_count() if hasattr(ext, "overflow_count") else 0) - ov0   # a 17+-row gate call that overflowed
        tn = G["FG"]["t1cn"] or []
        if tn:
            c = np.array(mx.stack(tn).sum(axis=0)).tolist()      # union experts by tier over the 48 layers: e4, e3, edrop
        else:
            c = [0, 0, 0]
    finally:
        unknob(G)
    gb = (c[0] * E4_MB + c[1] * E3_MB) / 1e3
    REC.append(dict(arm=arm, key=key, phase=phase, round=rnd, n=n, w=(w1 - w0, w2 - w1, w3 - w2),
                    miss=float(st[0]), ssd_ms=1e3 * float(st[1]), t1c=c, exp_gb=gb, layers=len(tn), ovf=int(ov), prof=prof))
    return lg


def _mp():
    """libmtlprof, if the loop loaded it (C25_PROF; patch_c25_prof.py)"""
    return sys.modules.get("mtlprof") if os.environ.get("C25_PROF") else None


def back(G, P, S0, n):
    G["CS"].rollback_after_verify(P, S0, n)
    mx.eval(G["states"](P))


def cmp(a, b):
    e = mx.all(a == b, axis=-1)
    fin = mx.isfinite(a) & mx.isfinite(b)
    d = mx.max(mx.where(fin, mx.abs(a - b), 0))
    am = mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)
    mx.eval(e, d, am)
    return int(np.array(e).sum()), int(np.array(am).sum()), int(e.shape[0]), round(float(d.item()), 4)


def window(G, key, P, S0, gids, gen, Greq, first):
    setup(G)
    TMR = int(os.environ.get("R57_G32TIME", "12") or 12)
    WU = int(os.environ.get("R57_G32TIME_WARM", "2") or 2)
    t_w = time.time()
    LG = {}
    for r in range(WU + TMR):
        rot = r % len(ARMS)
        for a in ARMS[rot:] + ARMS[:rot]:
            n = 16 if a.startswith("F16") else 32
            lg = check(G, P, gids[gen:gen + n], n, a, key, "same" if r >= WU else "warm", r - WU)
            back(G, P, S0, n)
            if r == WU:
                LG[a] = lg
    b = {}
    for x, y in (("F32o", "F32s"), ("F32d", "F32s"), ("F32df", "F32d"), ("F32if", "F32df"), ("F32if6", "F32if"),
                 ("F32c5", "F32df"), ("F32c6i", "F32c5"), ("F32s", "F16"), ("F32g", "F32o"), ("F32c6g", "F32c6i"),
                 ("F16g", "F16"), ("F16gd", "F16g"), ("F32c6p", "F32c6i"), ("F32c6x", "F32c6i"), ("F16gk", "F16g"),
                 ("F16gu", "F16g"), ("F32c6k", "F32c6i"), ("F32c6u", "F32c6i"), ("F16gx", "F16gu"), ("F32c6xu", "F32c6u")):
        if x in LG and y in LG:
            A, B = (LG[x][:16], LG[y]) if y == "F16" else (LG[x], LG[y])
            b[f"{x}_vs_{y}"] = cmp(A, B)                        # (rows bitwise, argmax rows, rows, max |diff|)
    S["bits"][key] = b
    if os.environ.get("C25_PASSK") and first and _mp() is not None:   # per-class GPU time (first window only; guarded)
        try:
            import c25_passk
            t_k = time.time()
            ka = tuple(os.environ.get("C25_KARMS", "F16,F32s,F32o,F32c6i").split(","))   # 11-17: per-run pass K arms
            S.setdefault("passk", {})[key] = c25_passk.run(sys.modules[__name__], G, P, S0, gids, gen, key, LG, arms=ka)
            _say(G, f"C25 pass K {key}: done in {time.time() - t_k:.0f} s")
        except Exception as e_:
            _say(G, f"C25 pass K {key}: FAILED {e_!r} (the timing pass is unaffected)")
    _say(G, f"C25 {key}: {WU} warm-up + {TMR} timed rounds of {len(ARMS)} arms in {time.time() - t_w:.0f} s; (rows bitwise, "
            f"argmax rows, rows, max |logit diff|): {b}; f32one {dict((k, v) for k, v in S['f32one'].COUNTS.items() if v)}; "
            f"int8d {dict((k, v) for k, v in S['int8d'].COUNTS.items() if v and k != 'passthru')}; "
            f"hc4i {dict(S['hc4i'].COUNTS) if S.get('hc4i') is not None and hasattr(S['hc4i'], 'COUNTS') else None}")
    dump(G)


def _walls():
    W = {}
    for r in REC:
        if r["phase"] == "same":
            W.setdefault((r["key"], r["round"]), {})[r["arm"]] = r
    return W


def finish(G):
    setup(G)
    W = _walls()
    m = lambda xs: float(np.median(xs)) if xs else float("nan")
    q = lambda xs: (float(np.percentile(xs, 25)), float(np.percentile(xs, 75))) if xs else (float("nan"),) * 2
    _say(G, f"C25 report: {len(W)} (window, round) samples; arms {ARMS}")
    if S.get("m16") is not None:                         # 11-17: engagement + the barrier check (must be 0)
        _say(G, f"C25: m16 COUNTS {S['m16'].COUNTS}; one-launch HC spin-cap fails {S['m16'].fails()} (must be 0); "
                f"int8fold COUNTS {S['int8fold'].COUNTS}; xknob COUNTS "
                f"{S['xk'].COUNTS if S.get('xk') is not None else None}")
    for a in ARMS:
        rs = [v[a] for v in W.values() if a in v]
        tot = [sum(r["w"]) for r in rs]
        _say(G, f"  {a:6s} median {1e3 * m(tot):8.2f} ms over {len(rs)} checks (build+submit {1e3 * m([r['w'][0] for r in rs]):.2f}, "
                f"wait {1e3 * m([r['w'][1] for r in rs]):.2f}, commit {1e3 * m([r['w'][2] for r in rs]):.2f}); expert GB "
                f"{m([r['exp_gb'] for r in rs]):.2f}; misses {m([r['miss'] for r in rs]):.1f}, SSD {m([r['ssd_ms'] for r in rs]):.2f} ms; "
                f"overflowed checks {sum(1 for r in rs if r.get('ovf'))} (invalid if > 0)")
    out = {}
    for x, y, kind in PAIRS:
        if x in ARMS and y in ARMS:
            xs = [(sum(v[x]["w"]) / sum(v[y]["w"])) if kind == "ratio" else 1e3 * (sum(v[x]["w"]) - sum(v[y]["w"]))
                  for v in W.values() if x in v and y in v]
            lo, hi = q(xs)
            out[f"{x}_{kind}_{y}"] = dict(median=m(xs), iqr=(lo, hi), n=len(xs))
            _say(G, f"  {x} {'/' if kind == 'ratio' else '-'} {y}: median {m(xs):+.3f}{'' if kind == 'ratio' else ' ms'} "
                    f"[IQR {lo:+.3f}, {hi:+.3f}] over {len(xs)} paired rounds")
    S["pairs"] = out
    _say(G, f"  forms over the run: f32one {dict(S['f32one'].COUNTS)}; int8d {dict(S['int8d'].COUNTS)}; hc4i "
            f"{dict(S['hc4i'].COUNTS) if S.get('hc4i') is not None and hasattr(S['hc4i'], 'COUNTS') else None}")
    if any(r.get("prof") for r in REC):                   # libmtlprof (C25_PROF): GPU busy vs wall, CPU, dispatches
        _say(G, "  PROFILE (libmtlprof; per check medians; ms): wall | GPU busy (fwd / read / commit) | GPU idle | CPU thread "
                "(fwd / read / commit) | dispatches | command buffers | first GPU start | GPU end -> check end")
        for a in ARMS:
            rs = [v[a] for v in W.values() if a in v and v[a].get("prof")]
            if not rs:
                continue
            P_ = [r["prof"] for r in rs]
            wall = 1e3 * m([sum(r["w"]) for r in rs])
            bz = 1e3 * m([p["busy"] for p in P_])
            bzp = " / ".join(f"{1e3 * m([p['busy_p'][j] for p in P_]):.2f}" for j in range(3))
            cpu = 1e3 * m([sum(p["cpu"]) for p in P_])
            cpup = " / ".join(f"{1e3 * m([p['cpu'][j] for p in P_]):.2f}" for j in range(3))
            nd = m([p["disp"] for p in P_])
            _say(G, f"  {a:6s} wall {wall:7.2f} | GPU {bz:7.2f} ({bzp}) = {bz / wall:.0%} | idle {wall - bz:6.2f} | CPU "
                    f"{cpu:6.2f} ({cpup}) | {nd:.0f} disp ({1e3 * cpu / max(nd, 1):.1f} us CPU each) | "
                    f"{m([p['ncb'] for p in P_]):.0f} cbs | first {1e3 * m([p['first'] for p in P_]):.2f} | "
                    f"last {1e3 * m([p['last'] for p in P_]):.2f}; unready cbs {sum(p['unready'] for p in P_)}")
        _cpusplit_report(G, W, m)
        for a in ("F16", "F32s", "F32o", "F32df", "F32if", "F32c6i", "F32c6g", "F16g", "F16gd", "F32c6p", "F32c6x", "F16gk",
                  "F16gu", "F32c6k", "F32c6u", "F16gx", "F32c6xu"):
            ks = [d[a] for d in S.get("pso", {}).values() if a in d]
            if ks:
                tot = {}
                for d in ks:
                    for k_, v_ in d.items():
                        tot[k_] = tot.get(k_, 0) + v_
                top = sorted(tot.items(), key=lambda kv: -kv[1])[:20]
                _say(G, f"  kernels {a} (dispatches per check, mean over {len(ks)} windows): "
                        + "; ".join(f"{k_[:48]} {v_ / len(ks):.0f}" for k_, v_ in top))
    if S.get("passk"):
        try:
            import c25_passk
            for key_, PK in S["passk"].items():
                _say(G, f"  PASS K (window {key_}; classes flushed into their own command buffers, scaled to pass T):")
                c25_passk.report(sys.modules[__name__], G, PK)
        except Exception as e_:
            _say(G, f"  PASS K report FAILED {e_!r}")
    dump(G)


def _cpusplit_report(G, W, m):
    """C25_CPUSPLIT: per arm, the forward call's wall (build + submit) = Python build + MLX encode + blocked in eval +
    other waits, next to the GPU's busy time in the same window (medians over checks, ms)"""
    for a in ARMS:
        rs = [v[a] for v in W.values() if a in v and (v[a].get("prof") or {}).get("split")]
        if not rs:
            continue
        P_ = [r["prof"] for r in rs]
        sp = lambda k: m([p["split"][k] for p in P_])
        fw, fc = 1e3 * m([r["w"][0] for r in rs]), 1e3 * m([p["cpu"][0] for p in P_])
        bf, bz = 1e3 * m([p["busy_p"][0] for p in P_]), 1e3 * m([p["busy"] for p in P_])
        wall = 1e3 * m([sum(r["w"]) for r in rs])
        _say(G, f"  CPU SPLIT {a:6s}: build + submit {fw:6.2f} = Python build {sp('py'):6.2f} + MLX encode {sp('enc'):6.2f} "
                f"+ blocked in eval {sp('blk'):6.2f} + other waits {sp('other'):6.2f} (thread CPU {fc:.2f}; "
                f"{sp('n_ae'):.0f} async_eval + {sp('n_ev'):.0f} eval calls) | GPU busy during it {bf:6.2f} ({bf / max(fw, 1e-9):.0%}); "
                f"whole check {wall:.2f}, GPU busy {bz:.2f} ({bz / max(wall, 1e-9):.0%}); n {len(rs)}")


def dump(G):
    out = os.path.join(G.get("OUT", os.path.expanduser("~/Leap/data/week98/c25")), "c25.json")
    with open(out, "w") as f:
        json.dump(dict(arms=ARMS, rec=REC, bits=S["bits"], pairs=S.get("pairs", {}), notes=S["notes"],
                       pso=S.get("pso", {}), passk=S.get("passk", {})), f)
