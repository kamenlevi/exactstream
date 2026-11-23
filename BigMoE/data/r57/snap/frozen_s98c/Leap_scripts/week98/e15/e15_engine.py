"""e15_engine.py (Leap, Tue 17 Nov 2026 06:3x): E15 = E14's engine (below, unchanged) + NewIdeas' ADAPTIVE K DOWN kill test
(adaptk_k8.txt, Tue 06:24 / 06:32) + the ref2 switch bit tests (E15_MODE=sw).
E15_MODE=time (default): new arms A8 (an 8-row REF2-T check) and A8b (A8 + BOF), timed next to A16 / A16b, with C32one in the
same run, so R8 and R32 come from one run. Reported as RATIOS OF MEANS over the paired timed rounds (NewIdeas' rule), with
paired medians beside. Pre-registered (data/week98/check55_plan.md 'E15', before the run): R8 = A8 / A16 >= 0.82 -> KILL;
else ALIVE. Bits: A8 == A16's first 8 rows; A8b == A8. Lab, check-level (warm n-gram rows, rotated arms), like E14.
E15_MODE=sw (with REF2_BOF=1 REF2_HC_B=1, after ref2_switches_v3.patch is applied): BigMoE's test list (a) / (b) and on / off,
through ref2's own switch path (CFG + apply()); E14's setup / knobs are NOT used in this mode:
  (c) the 16- and 32-row checks with both switches on == off, bitwise; BOF / BOF2 / hc_b fired when on, not when off;
  (b) SD == AR with both on: 16 ref2.ar_step()s == the 16-row check, bitwise; the check again after the AR rollback;
  (a) the loop's own prefill with the last chunk of r rows (17-32: BOF2 must NOT fire; <= 16: BOF, REF2-T bits), on vs off:
      bitwise cache states + the first logits (chunk 64 on the window's prompt; chunk PFCHUNK on short prompts).
  Verdict: SW PASS iff every compare is bit-equal, no DRIFT, BOF and hc_b fired when on and nothing when off (BOF2's count
  is info: a default 32-row check may split o_proj at its seams), and BOF2 never fired in a prefill.
  E15_V3TREE=<dir>: e15_loop imports ref2 / bofi / bof from that dir's v3-patched COPIES (the live files untouched).

E14's docstring:
e14_engine.py (Leap, Sun 15 Nov 2026 18:3x; Monday night's E1-E4 in Leap's slot, Tue 00:30 -> 03:00): the exact check's
engine A/Bs. Called from e14_loop.py, a copy of Breakthrough's r57_c55.py (itself BigMoE's r57_screen_fast.py with the C55
patches), whose G32T time mode calls window() per G32T window and finish() at the end. Env as BigMoE's G32T runner
(e14_env.sh, from Breakthrough's c55_engine_env.sh): the exact-45 loop, C 160, MAXK 320, build_g32, v32 + V32_MOE1,
SB_FIRST + PLE_LATE at 5-16 rows.
Per window: R57_G32TIME_WARM warm-up + R57_G32TIME timed rounds of the arms, rotated. Each check is a verify, the logits'
read and a full-accept commit, on a rollback to the window's prefill: G32T's same tokens, so the n-gram rows are warm
after the warm-up (the cold-read lesson of C9 / C10). Wall per phase.
Arms (E14_ARMS selects; E2 first by value):
  A16      REF2-T 16-row check (the base);
  C32      REF2-T 32-row check through v32's split (seams at <= 16);
  C32one   step 1 complete: v32one's one-dispatch forms + seams to 32;
  C32oneB  C32one + BOF2 on o_proj / out_proj (bofi.TABLE32; REF2-T's bits; its own pre-registered line);
  A16b     A16 + BOF (bofi: o_proj / out_proj, QSA, shared down, lm_head; REF2-T's bits);
  A16h     A16 with ref2's hc_b toggled (D1-B; exact);
  A16c     A16 under the reference contract (refc: a NEW reference, timing only);
  C32c     C32one under the contract.
Bits on the first timed round's logits: C32one, C32oneB == C32 (32 rows); A16b, A16h == A16; C32's first 16 rows == A16
(REF2-T's SD == AR at 32 rows); C32c's first 16 rows == A16c (the contract's own SD == AR); A16c vs A16 differ by design.
The pre-registered lines live in data/week98/check55_plan.md 'E14' (written before the run); finish() prints the numbers
they read."""
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

REC = []
S = dict(ok=False, bits={}, notes=[])
ALL = ["A16", "C32", "C32one", "C32oneB", "A16b", "A16h", "A16c", "C32c", "A16m", "C32onem", "A8", "A8b"]   # E15: + A8, A8b
ARMS = [a for a in os.environ.get("E14_ARMS", ",".join(ALL)).split(",") if a]
MODE = os.environ.get("E15_MODE", "time")   # E15: time | sw
assert MODE in ("time", "sw"), MODE
assert ARMS and all(a in ALL for a in ARMS) and "A16" in ARMS, ARMS
PAIRS = [("C32", "A16", "ratio"), ("C32one", "A16", "ratio"), ("C32one", "C32", "diff"), ("C32oneB", "C32one", "diff"),
         ("A16b", "A16", "diff"), ("A16h", "A16", "diff"), ("A16c", "A16", "diff"), ("C32c", "C32one", "diff"),
         ("A16m", "A16", "diff"), ("C32onem", "C32one", "diff"),
         ("A8", "A16", "ratio"), ("A8b", "A16b", "ratio"), ("A8b", "A8", "diff")]   # E15
# Amendment 1 (Sun 21:3x): A16m / C32onem = A16 / C32one with CHECK55 lever 2, Breakthrough's moe_dense.dense_pre in place
# of K-73's router_fused + shared_cat pair (bitwise, 108 / 108): <= 16 rows in e14_loop's _exact_block (E14_L2), 17-32
# rows through the _tier_block wrapper below. Any error falls back to the pair and is reported.
L2 = dict(on=False, n=0, err=[])


def _tier_block_l2(self, x, kind):
    """v32._tier_block with lever 2 for the K-73 kind (the same body otherwise; v32 Sun 02:4x)"""
    v32 = S["v32"]
    if not (L2["on"] and kind == "k73"):
        return S["tier0"](self, x, kind)
    try:
        md = S["G"]["_e14_md"]()
        inds, scores, g, s = md.dense_pre(self, x)
        if L2["n"] == 0:
            mx.eval(inds, scores, g, s)                  # the first call (a warm-up round): a kernel build error is caught
    except Exception as e:
        L2["err"].append(repr(e)[:300])
        return S["tier0"](self, x, kind)
    L2["n"] += 1
    from glue_fused import moe_tail
    bm = self.switch_mlp._bm
    EM = sys.modules.get("evalsuite.model")
    if EM is not None and getattr(EM.STATE, "capture", None) is not None:
        EM.STATE.capture.setdefault(bm["L"], []).append(inds)
    bm["tier"]["scores"], bm["tier"]["shared"] = scores, s
    try:
        y = self.switch_mlp(x, inds)
    finally:
        bm["tier"].pop("scores", None)
        bm["tier"].pop("shared", None)
    if y.ndim == inds.ndim + 1:
        y = (y * scores[..., None]).sum(axis=-2)
    v32.CNT["tierblk"] += 1
    return moe_tail(y, g, s, exp="precise")


def _say(G, s):
    G["say"](s)
    S["notes"].append(s)


def setup(G):
    if S["ok"]:
        return
    sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98c/Leap_scripts/week98"))
    model, ref2, v32 = G["model"], G["ref2"], G["v32"]
    import v32one
    import refc
    import bofi
    v32one.install(v32, ref2)                          # order: v32 (the loop) -> v32one -> refc -> bofi (outermost)
    refc.install(ref2)
    bofi.install(ref2)
    v32one.ON[0] = refc.ON[0] = bofi.ON[0] = False
    bofi.TABLE32[(2560, 6144)] = (4, 1)                # BOF2 on o_proj / out_proj at 17-32 rows, gated by bofi.ON
    S.update(v32one=v32one, refc=refc, bofi=bofi, seam0=int(ref2.CFG.get("seam_hi", 16)),
             hcb0=bool(ref2.CFG.get("hc_b", False)), vocab=int(model.language_model.lm_head.weight.shape[0]),
             v32=v32, G=G, tier0=v32._tier_block)
    v32._tier_block = _tier_block_l2                   # amendment 1: lever 2 at 17-32 rows (off unless an m arm runs)
    S["ok"] = True
    _say(G, f"E15: setup done (arms {ARMS}; v32one / refc / bofi installed off; seam_hi {S['seam0']}; hc_b {S['hcb0']}; "
            f"vocab {S['vocab']})")


def knobs(G, arm):
    ref2 = G["ref2"]
    one = arm in ("C32one", "C32oneB", "C32c", "C32onem")
    ref2.CFG["seam_hi"] = 32 if one else S["seam0"]
    S["v32one"].ON[0] = one
    S["bofi"].ON[0] = arm in ("A16b", "C32oneB", "A8b")   # E15: + A8b
    S["refc"].ON[0] = arm in ("A16c", "C32c")
    ref2.CFG["hc_b"] = (not S["hcb0"]) if arm == "A16h" else S["hcb0"]
    m = arm in ("A16m", "C32onem")                     # amendment 1: lever 2 (<= 16 rows: the loop's flag; 17-32: L2)
    L2["on"] = m
    if "E14_L2" in G:
        G["E14_L2"][0] = m


def check(G, P, toks, n, arm, key, phase, rnd=-1):
    """one verify + the logits' read + a full-accept commit; the caller rolls back"""
    model, ref2, CS, states = G["model"], G["ref2"], G["CS"], G["states"]
    knobs(G, arm)
    try:
        sn = CS.snapshot_untrimmable_cache_lazy(P)
        mx.synchronize()
        w0 = time.perf_counter()
        with ref2.verify_scope(model):
            lg = model(mx.array([toks]), cache=P)
        w1 = time.perf_counter()
        lg = lg[0].astype(mx.float32)
        mx.eval(lg)
        w2 = time.perf_counter()
        assert model.commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
        mx.eval(states(P))
        mx.synchronize()
        w3 = time.perf_counter()
    finally:
        knobs(G, "A16")
    REC.append(dict(arm=arm, key=key, phase=phase, round=rnd, n=n, w=(w1 - w0, w2 - w1, w3 - w2)))
    return lg


def back(G, P, S0, n):
    G["CS"].rollback_after_verify(P, S0, n)
    mx.eval(G["states"](P))


def cmp(a, b):
    e = mx.all(a == b, axis=-1)
    mx.eval(e)
    return int(np.array(e).sum()), int(e.shape[0])


def cold(G, key, P, S0, gids, gen):
    """the loop's first-check hook: setup only (the warm-up rounds build the kernels and warm the window's n-gram rows)"""
    if MODE == "sw":
        return sw_setup(G)
    setup(G)


def window(G, key, P, S0, gids, gen, Greq, first):
    if MODE == "sw":
        return sw_window(G, key, P, S0, gids, gen)
    setup(G)
    TMR = int(os.environ.get("R57_G32TIME", "12") or 12)
    WU = int(os.environ.get("R57_G32TIME_WARM", "2") or 2)
    t_w = time.time()
    LG = {}
    for r in range(WU + TMR):
        rot = r % len(ARMS)
        for a in ARMS[rot:] + ARMS[:rot]:
            n = 8 if a.startswith("A8") else 16 if a.startswith("A16") else 32   # E15: A8 / A8b = 8 rows
            lg = check(G, P, gids[gen:gen + n], n, a, key, "same" if r >= WU else "warm", r - WU)
            back(G, P, S0, n)
            if r == WU:
                LG[a] = lg
    b = {}
    for x, y, rows in (("C32one", "C32", None), ("C32oneB", "C32", None), ("A16b", "A16", None), ("A16h", "A16", None),
                       ("C32", "A16", 16), ("C32c", "A16c", 16), ("A16c", "A16", None), ("A16m", "A16", None),
                       ("C32onem", "C32", None), ("A8", "A16", 8), ("A8b", "A8", None), ("A8b", "A16b", 8),
                       ("C32one", "A16", 16)):   # E15 (C32one[:16] == A16: SD == AR at 32 rows)
        if x in LG and y in LG:
            A, B = (LG[x][:rows], LG[y][:rows]) if rows else (LG[x], LG[y])
            b[f"{x}_vs_{y}" + (f"[:{rows}]" if rows else "")] = cmp(A, B)
    S["bits"][key] = b
    _say(G, f"E15 {key}: {WU} warm-up + {TMR} timed rounds of {len(ARMS)} arms in {time.time() - t_w:.0f} s; bits (rows "
            f"equal / rows; A16c vs A16 differ by design) {b}")
    dump(G)


def _walls():
    """{(key, round): {arm: whole wall s}} over the timed rounds"""
    W = {}
    for r in REC:
        if r["phase"] == "same":
            W.setdefault((r["key"], r["round"]), {})[r["arm"]] = sum(r["w"])
    return W


def finish(G):
    if MODE == "sw":
        return sw_finish(G)
    setup(G)
    W = _walls()
    m = lambda xs: float(np.median(xs)) if xs else float("nan")
    q = lambda xs: (float(np.percentile(xs, 25)), float(np.percentile(xs, 75))) if xs else (float("nan"),) * 2
    _say(G, f"E15 report: {len(W)} (window, round) samples; arms {ARMS}")
    for a in ARMS:
        xs = [v[a] for v in W.values() if a in v]
        _say(G, f"  {a:8s} median {1e3 * m(xs):8.2f} ms over {len(xs)} checks")
    out = {}
    for x, y, kind in PAIRS:
        if x in ARMS and y in ARMS:
            xs = [(v[x] / v[y]) if kind == "ratio" else 1e3 * (v[x] - v[y]) for v in W.values() if x in v and y in v]
            lo, hi = q(xs)
            out[f"{x}_{kind}_{y}"] = dict(median=m(xs), iqr=(lo, hi), n=len(xs))
            _say(G, f"  {x} {'/' if kind == 'ratio' else '-'} {y}: median {m(xs):+.3f}{'' if kind == 'ratio' else ' ms'} "
                    f"[IQR {lo:+.3f}, {hi:+.3f}] over {len(xs)} paired rounds")
    S["pairs"] = out
    rom = {}                                           # E15: RATIOS OF MEANS (NewIdeas' rule for the R(K) table)
    for x, y in (("A8", "A16"), ("A8b", "A16b"), ("C32one", "A16"), ("C32", "A16"), ("A16b", "A16"), ("A8b", "A8")):
        if x in ARMS and y in ARMS:
            ps = [(v[x], v[y]) for v in W.values() if x in v and y in v]
            if ps:
                mx_, my_ = float(np.mean([p[0] for p in ps])), float(np.mean([p[1] for p in ps]))
                rom[f"{x}/{y}"] = dict(ratio_of_means=mx_ / my_, mean_ms=(1e3 * mx_, 1e3 * my_), n=len(ps))
                _say(G, f"  RATIO OF MEANS {x} / {y}: {mx_ / my_:.4f} over {len(ps)} paired rounds (means {1e3 * mx_:.2f} / "
                        f"{1e3 * my_:.2f} ms)")
    S["rom"] = rom
    if "A8/A16" in rom:
        r8 = rom["A8/A16"]["ratio_of_means"]
        bt = {k: v for k, v in S["bits"].items()}
        _say(G, f"  E15 VERDICT (pre-registered, check55_plan.md 'E15'; NewIdeas' line): R8 = A8 / A16 (ratio of means) "
                f"{r8:.3f} -> {'KILL (>= 0.82)' if r8 >= 0.82 else 'ALIVE (< 0.82)'}; R8b = A8b / A16b "
                f"{rom.get('A8b/A16b', {}).get('ratio_of_means', float('nan')):.3f}; R32 = C32one / A16 "
                f"{rom.get('C32one/A16', {}).get('ratio_of_means', float('nan')):.3f}; bits per window {bt}")
    _say(G, f"  forms fired over the run: bofi {dict(S['bofi'].CNT)}; refc {dict(S['refc'].CNT)}; v32one {dict(S['v32one'].CNT)}")
    _say(G, f"  lever 2 (amendment 1): <= 16 rows calls {G.get('E14_L2N', ['n/a'])[0]}, errors {G.get('E14_L2ERR', [])[:3]}; "
            f"17-32 rows calls {L2['n']}, errors {L2['err'][:3]} (an error falls back to K-73's pair: that arm then times the base)")
    dump(G)


def dump(G):
    out = os.path.join(G.get("OUT", os.path.expanduser("~/Leap/data/week98/e15")), "e15.json" if MODE == "time" else "e15_sw.json")
    with open(out, "w") as f:
        json.dump(dict(mode=MODE, arms=ARMS, rec=REC, bits=S["bits"], pairs=S.get("pairs", {}), rom=S.get("rom", {}),
                       sw=SW, notes=S["notes"]), f, default=str)


# ---------------------------------------------------------------- E15_MODE=sw: the ref2 switch bit tests (v3; BigMoE's list)
SW = dict(win={}, fails=[], setup={})
_UT = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}


def _beq(a, b):
    """bit-equal: shape, dtype and every bit (unsigned views of the same width)"""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.size == 0:
        return True
    ut = _UT.get(a.dtype.size)
    if ut is None:
        return bool(mx.all(a == b).item())
    return bool(mx.all(mx.view(a.reshape(-1), ut) == mx.view(b.reshape(-1), ut)).item())


def _rows(A, B):
    """[rows bit-equal, rows] for two f32 [n, V] logits"""
    if A.shape != B.shape:
        return [0, max(int(A.shape[0]), int(B.shape[0]))]
    e = mx.all(mx.view(A, mx.uint32) == mx.view(B, mx.uint32), axis=-1)
    mx.eval(e)
    return [int(np.array(e).sum()), int(e.shape[0])]


def _leaves(st):
    from mlx.utils import tree_flatten
    return [v for _, v in tree_flatten(st) if isinstance(v, mx.array)]


def _states(A, B):
    fa, fb = _leaves(A), _leaves(B)
    if len(fa) != len(fb):
        return [0, max(len(fa), len(fb))]
    return [sum(_beq(x, y) for x, y in zip(fa, fb)), len(fa)]


def _cnt():
    b = sys.modules.get("bofi")
    return dict(b.CNT) if b is not None else {}


def _dc(c1, c0):
    return {k: c1.get(k, 0) - c0.get(k, 0) for k in sorted(set(c1) | set(c0))}


def _sw(G, on):
    """both switches through ref2's own path: CFG + apply() (v3: apply() syncs bofi.ON via _bof_on / _BOF_OWN)"""
    ref2 = G["ref2"]
    ref2.CFG["bof"] = bool(on)
    ref2.CFG["hc_b"] = bool(on)
    ref2.apply()
    b = sys.modules.get("bofi")
    st = (bool(ref2.CFG["bof"]), bool(ref2.CFG["hc_b"]), bool(b is not None and b.ON[0]), bool(ref2._ON[0]))
    if st != (bool(on), bool(on), bool(on), True):
        SW["fails"].append(f"switch state (bof, hc_b, bofi.ON, ref2 on) {st} after _sw({on})")
    return st


def _chk(G, P, toks, n):
    """E14's check() without its knobs: one verify + the logits' read + a full-accept commit; the caller rolls back"""
    model, ref2, CS, states = G["model"], G["ref2"], G["CS"], G["states"]
    sn = CS.snapshot_untrimmable_cache_lazy(P)
    with ref2.verify_scope(model):
        lg = model(mx.array([toks]), cache=P)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg)
    assert model.commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
    mx.eval(states(P))
    return lg


def _ar(G, P, toks):
    """REF2's AR reference: ref2.ar_step per token (a 1-row verify + commit keep 1); the caller rolls back len(toks)"""
    out = []
    for t in toks:
        lg = G["ref2"].ar_step(G["model"], P, int(t))
        mx.eval(lg, G["states"](P))
        out.append(lg.astype(mx.float32))
    lg = mx.stack(out)
    mx.eval(lg)
    return lg


def _pf(G, ids, chunk):
    """the loop's own prefill, as G32T's window prefill: arm 'slot', then the loop's check arm back"""
    G["set_arm"]("slot")
    try:
        P2, lg, _t, _w = G["prefill"](ids, False, chunk=chunk)
        st = G["states"](P2)
        lg = lg.astype(mx.float32)
        mx.eval(lg, st)
    finally:
        G["set_arm"]("gate+t")
        G["BS"].GATE_N[0] = G["BG"].MAXK
    return P2, lg, st


def sw_setup(G):
    if S["ok"]:
        return
    ref2 = G["ref2"]
    b = sys.modules.get("bofi")
    SW["setup"] = dict(env=(os.environ.get("REF2_BOF"), os.environ.get("REF2_HC_B")), cfg=(ref2.CFG.get("bof"), ref2.CFG.get("hc_b")),
                       ref2_on=bool(ref2._ON[0]), bofi=(None if b is None else dict(on=bool(b.ON[0]), file=b.__file__,
                       table32={str(k): v for k, v in b.TABLE32.items()}, self_test=hasattr(b, "self_test"))),
                       bof_on=hasattr(ref2, "_bof_on"), files=[getattr(sys.modules.get(m), "__file__", None)
                                                               for m in ("ref2", "bofi", "bof")], v3tree=os.environ.get("E15_V3TREE"))
    v3 = os.environ.get("E15_V3TREE")
    if v3 and not all(f and f.startswith(v3) for f in SW["setup"]["files"]):
        SW["fails"].append(f"setup: E15_V3TREE set but the modules are not its copies: {SW['setup']['files']}")
    if not (ref2.CFG.get("bof") is True and ref2.CFG.get("hc_b") is True and b is not None and b.ON[0] and hasattr(ref2, "_bof_on")):
        SW["fails"].append(f"setup: the switches are not on through v3's path: {SW['setup']}")
    S["ok"] = True
    _say(G, f"E15 SW: setup {SW['setup']}")


def sw_window(G, key, P, S0, gids, gen):
    sw_setup(G)
    hc = lambda: int(G["ref2"].COUNTS.get("hc_b", 0))
    R = {}
    t0 = time.time()
    t16, t32 = gids[gen:gen + 16], gids[gen:gen + 32]
    try:
        _sw(G, False)                                  # (c) off
        c0, h0 = _cnt(), hc()
        off16 = _chk(G, P, t16, 16); back(G, P, S0, 16)
        off32 = _chk(G, P, t32, 32); back(G, P, S0, 32)
        R["off fired"] = dict(bofi=_dc(_cnt(), c0), hc_b=hc() - h0)
        _sw(G, True)                                   # (c) on
        c0, h0 = _cnt(), hc()
        on16 = _chk(G, P, t16, 16); back(G, P, S0, 16)
        on32 = _chk(G, P, t32, 32); back(G, P, S0, 32)
        R["on fired"] = dict(bofi=_dc(_cnt(), c0), hc_b=hc() - h0)
        ar = _ar(G, P, t16); back(G, P, S0, 16)        # (b) SD == AR, both on
        on16b = _chk(G, P, t16, 16); back(G, P, S0, 16)
        R["c on16 == off16"] = _rows(on16, off16)
        R["c on32 == off32"] = _rows(on32, off32)
        R["b AR16 == on16"] = _rows(ar, on16)
        R["b on16 after the AR rollback == on16"] = _rows(on16b, on16)
        R["x on32[:16] == on16"] = _rows(on32[:16], on16)
        del off16, off32, on16, on32, ar, on16b
        cases = [(64, r) for r in ((24, 9, 17, 32, 16, 1) if not SW["win"] else (24, 9))]
        if not SW["win"]:
            cases += [(int(G["PFCHUNK"]), 24), (int(G["PFCHUNK"]), 9)]
        for ch, r in cases:                            # (a) prefills, on vs off
            L = ((gen - r) // ch) * ch + r if gen >= r else 0
            if L <= 0:
                continue
            _sw(G, True)
            c0 = _cnt()
            Pa, la, sa = _pf(G, gids[:L], ch)
            fa = _dc(_cnt(), c0)
            _sw(G, False)
            Pb, lb, sb = _pf(G, gids[:L], ch)
            R[f"a prefill L {L} chunk {ch} (last chunk {r} rows)"] = dict(states=_states(sa, sb), logits=_beq(la, lb),
                                                                         on_fired=fa)
            del Pa, Pb, sa, sb, la, lb
            mx.clear_cache()
    except Exception as e:
        SW["fails"].append(f"{key}: {type(e).__name__}: {str(e)[:300]}")
        import traceback
        traceback.print_exc()
    finally:
        _sw(G, True)
    SW["win"][key] = R
    _say(G, f"E15 SW {key}: {time.time() - t0:.0f} s; {R}")
    dump(G)


def sw_finish(G):
    bad = list(SW["fails"])
    for key, R in SW["win"].items():
        for k, v in R.items():
            if k[:2] in ("b ", "c ", "x ") and v[0] != v[1]:
                bad.append(f"{key} {k} {v}")
            if k.startswith("a "):
                if v["states"][0] != v["states"][1] or not v["logits"] or v["on_fired"].get("bof2", 0) or v["on_fired"].get("drift", 0):
                    bad.append(f"{key} {k} {v}")
        on, off = R.get("on fired", {}), R.get("off fired", {})
        if not (on and on["bofi"].get("bof", 0) > 0 and on["hc_b"] > 0):   # BOF2: info only (a default 32-row check may split
            bad.append(f"{key}: on did not fire BOF / hc_b: {on}")            # o_proj at its seams; E14: C32oneB == C32 anyway)
        if not off or off["bofi"].get("bof", 0) or off["bofi"].get("bof2", 0) or off["hc_b"]:
            bad.append(f"{key}: off fired something: {off}")
    d = _cnt().get("drift", -1)
    if d != 0:
        bad.append(f"bofi drift count {d}")
    if not SW["win"]:
        bad.append("no window ran")
    SW["verdict"] = "PASS" if not bad else "FAIL"
    _say(G, f"E15 SW VERDICT: {SW['verdict']} over {len(SW['win'])} windows; bofi counters {_cnt()}; "
            f"{'all compares bit-equal, no drift, counters as designed' if not bad else 'FAILS: ' + '; '.join(bad)[:2000]}")
    dump(G)
