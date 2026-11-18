"""e14_engine.py (Leap, Sun 15 Nov 2026 18:3x; Monday night's E1-E4 in Leap's slot, Tue 00:30 -> 03:00): the exact check's
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
ALL = ["A16", "C32", "C32one", "C32oneB", "A16b", "A16h", "A16c", "C32c", "A16m", "C32onem"]
ARMS = [a for a in os.environ.get("E14_ARMS", ",".join(ALL)).split(",") if a]
assert ARMS and all(a in ALL for a in ARMS) and "A16" in ARMS, ARMS
PAIRS = [("C32", "A16", "ratio"), ("C32one", "A16", "ratio"), ("C32one", "C32", "diff"), ("C32oneB", "C32one", "diff"),
         ("A16b", "A16", "diff"), ("A16h", "A16", "diff"), ("A16c", "A16", "diff"), ("C32c", "C32one", "diff"),
         ("A16m", "A16", "diff"), ("C32onem", "C32one", "diff")]
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
    sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/week98"))
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
    _say(G, f"E14: setup done (arms {ARMS}; v32one / refc / bofi installed off; seam_hi {S['seam0']}; hc_b {S['hcb0']}; "
            f"vocab {S['vocab']})")


def knobs(G, arm):
    ref2 = G["ref2"]
    one = arm in ("C32one", "C32oneB", "C32c", "C32onem")
    ref2.CFG["seam_hi"] = 32 if one else S["seam0"]
    S["v32one"].ON[0] = one
    S["bofi"].ON[0] = arm in ("A16b", "C32oneB")
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
    setup(G)


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
    b = {}
    for x, y, rows in (("C32one", "C32", None), ("C32oneB", "C32", None), ("A16b", "A16", None), ("A16h", "A16", None),
                       ("C32", "A16", 16), ("C32c", "A16c", 16), ("A16c", "A16", None), ("A16m", "A16", None),
                       ("C32onem", "C32", None)):
        if x in LG and y in LG:
            A, B = (LG[x][:rows], LG[y][:rows]) if rows else (LG[x], LG[y])
            b[f"{x}_vs_{y}" + (f"[:{rows}]" if rows else "")] = cmp(A, B)
    S["bits"][key] = b
    _say(G, f"E14 {key}: {WU} warm-up + {TMR} timed rounds of {len(ARMS)} arms in {time.time() - t_w:.0f} s; bits (rows "
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
    setup(G)
    W = _walls()
    m = lambda xs: float(np.median(xs)) if xs else float("nan")
    q = lambda xs: (float(np.percentile(xs, 25)), float(np.percentile(xs, 75))) if xs else (float("nan"),) * 2
    _say(G, f"E14 report: {len(W)} (window, round) samples; arms {ARMS}")
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
    _say(G, f"  forms fired over the run: bofi {dict(S['bofi'].CNT)}; refc {dict(S['refc'].CNT)}; v32one {dict(S['v32one'].CNT)}")
    _say(G, f"  lever 2 (amendment 1): <= 16 rows calls {G.get('E14_L2N', ['n/a'])[0]}, errors {G.get('E14_L2ERR', [])[:3]}; "
            f"17-32 rows calls {L2['n']}, errors {L2['err'][:3]} (an error falls back to K-73's pair: that arm then times the base)")
    dump(G)


def dump(G):
    out = os.path.join(G.get("OUT", os.path.expanduser("~/Leap/data/week98/e14")), "e14.json")
    with open(out, "w") as f:
        json.dump(dict(arms=ARMS, rec=REC, bits=S["bits"], pairs=S.get("pairs", {}), notes=S["notes"]), f)
