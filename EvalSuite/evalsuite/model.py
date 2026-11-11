"""Loading the 125B (BigMoE's SSD-streamed MTPLX runtime, read-only) and the numerics patches that define variants.

Variants (VARIANTS / variant()):
  ref        the harness reference: exact 4-bit weights, top-k combined in gate-rank order (sorted by score)
  floor      same weights, MTPLX/mlx-lm's stock combine order (argpartition order): a same-weights path that differs
             only in float summation order -> the noise floor between two runs of the same weights
  ar4        accepted build: the routed experts of Intel's AutoRound int4 quant (models/ar4, tools/fetch_autoround.py),
             reference trunk; rank-sorted combine like ref
  plan<bits> rank tiers (MoonShots planes125 "plan"): one digit per gate rank, 4 = exact 4-bit, 3/2 = the MSB-3/MSB-2
             tier of the same codes (low bits cleared, bias += scale * per-group mean of the cleared bits)
  w2f_r10/05 Breakthrough's frugal wait-top-2 (~/Breakthrough/scripts/bt_frugal.py, read-only): 10 % / 5 % of each
             layer's experts resident, ranks 0-1 waited for, missing ranks 2-9 replaced by resident ranks 10-19. Its
             own runtime (MoEPrefetch's fork125 streamer, EVALSUITE_FRUGAL=1); one-token calls only, multi-row exact.
  b12 / b8 / b8fast  Breakthrough's small-RAM configs as they ship (~/Breakthrough/scripts/bt_b12.py, read-only): frugal
             routing on its pinned v3 tier engine (v3f; plan 4433333333 MSB-3 + LSB sidecar, 8-bit HC mixers), C_m 51 /
             26 / 26 (B8: cold MSB-3 ranks 2-3 half a layer late; B8-fast: waits for ranks 0-2), MTP on (spec125
             3:3:0, FRSPEC) after the release prefill (D1i). EVALSUITE_SHIP=1 and the switches come from cli.gpu.

Scope: prefill tests apply a variant to every token ("everywhere", conservative: it also perturbs prompt processing).
Decode sets use the deployed semantics of decode-only variants (exact prefill, tiers at decode), as MoonShots ran them.
The tfstep stage feeds the teacher-forced sequences one token per call, for variants that act only there (frugal).
The tier math is vendored from ~/MoonShots/patches/planes125/planes125.py and checked against it (selftest())."""
from __future__ import annotations
import importlib.util, json, os, re, sys
from dataclasses import dataclass, field
from . import env  # noqa: F401  (environment first)
import mlx.core as mx

AR4_MODEL = os.path.join(env.ROOT, "models", "ar4")


@dataclass
class State:
    sorted: bool = True            # rank-sorted top-k combine (False = stock order: the floor)
    segs: list | None = None       # [(rank0, rank1, bits)] of a plan, None = exact
    tiers_prefill: bool = False    # apply tiers on multi-row (prefill) calls
    tiers_decode: bool = False     # apply tiers on single-token decode calls
    capture: dict | None = None    # layer -> list of rank-ordered top-k ids (lazy), when routing capture is on
    pass_id: int = 0               # bumped by the driver per forward pass (invalidates the masked-scratch memo)
    fast_decode: bool = False      # decode tiers through one compiled gather+mask+bias call per layer (fast_tier_check)
    memo: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"tier_prefill_calls": 0, "tier_decode_calls": 0, "masked_scratch": 0})


STATE = State()
VARIANTS = {
    "ref": dict(model="ref", sorted=True, plan=None),
    "floor": dict(model="ref", sorted=False, plan=None),
    "ar4": dict(model="ar4", sorted=True, plan=None),
    "ref2": dict(model="ref", sorted=True, plan=None, ref2=True),   # R-40: the reference + Leap's REF2 kernels, decoded
                                                                   # by ref2.generate (docs/PREREG_ref2_2026-10-28.md)
    "ref2b": dict(model="ref", sorted=True, plan=None, ref2=True, ref2_code="leap-faff8e5"),   # REF2b: REF2 + key
                                      # buckets past 512 keys (Leap c4b6be9; docs/PREREG_ref2b_2026-10-30.md)
    "ref2t": dict(model="ref", sorted=True, plan=None, ref2=True, ref2_code="leap-faff8e5", tier="4433333333"),  # R-58:
                                      # REF2b on BigMoE's rank tiers, 3c / C_l 112 (docs/PREREG_r58_2026-10-31.md)
    "ref2ct": dict(model="ref", sorted=True, plan=None, ref2=True, ref2_code="leap-91cd316", tier="4433333333",
                   ref2_cfg={"att_single": True, "gdn_inproj": True}),   # REF2c-T: REF2-T on Leap's REF2c freeze 91cd316
                                      # (one fused SDPA per 16-row call; padded fused GDN in_proj; PREREG_ref2ct_2026-11-01)
    "ref2th8": dict(model="ref", sorted=True, plan=None, ref2=True, ref2_code="leap-faff8e5", tier="4433333333", hc8=True),
}                                     # Q3 HC8 (ROUTE_350, day 9): REF2-T with the HC mixers + PLE key/value at 8 bits


def _w4(hc8r=False, **on):
    """REF2-T on Leap 52cf2c9 (PREREG_nax6, PREREG_f1 rev. 3): every switch the two PREREGs name is set explicitly, off
    unless `on` turns it on (no env leak can flip one); hc8r: the snapshot's hc8r.hc8_install before ref2.apply (HC8-1P)"""
    cfg = dict(tier_nax_ooo=False, nax_native=False, nax_hybrid=False, fatt_verify=False, fatt_r8=False, hc8_fast=False,
               hc8_v=4, dense_native=False)
    cfg.update(on)
    return dict(model="ref", sorted=True, plan=None, ref2=True, ref2_code="leap-52cf2c9", tier="4433333333",
                ref2_cfg=cfg, **({"hc8r": True} if hc8r else {}))


VARIANTS.update({
    "ref2tf0": _w4(),                                                  # the control: REF2-T on the new code (step 0)
    "ref2tnax": _w4(tier_nax_ooo=True),                                # NAX6, the EXACT reference candidate
    "ref2tfh": _w4(True, fatt_verify=True, fatt_r8=True, hc8_fast=True),     # the FAST batch on REF2-T: F1 + r8 + HC8-1P
    "ref2tf": _w4(fatt_verify=True, fatt_r8=True),                     # split: F1 + r8
    "ref2th8p": _w4(True, hc8_fast=True),                              # split: HC8-1P
    "ref2tnfh": _w4(True, tier_nax_ooo=True, fatt_verify=True, fatt_r8=True, hc8_fast=True),   # the batch on NAX6
    "ref2tnf": _w4(tier_nax_ooo=True, fatt_verify=True, fatt_r8=True),
    "ref2tnh8": _w4(True, tier_nax_ooo=True, hc8_fast=True),
    # the NAX6 base on the CONVERTED store (Leap 18:1x): + nax_native, NAX6-N (bitwise == NAX6; tierglu_nax6n), only
    # when all 48 layers are converted; non-NAX tier calls route to 3c-N by themselves (BigMoE's layout router)
    "ref2tnaxn": _w4(tier_nax_ooo=True, nax_native=True),
    "ref2tnfhn": _w4(True, tier_nax_ooo=True, nax_native=True, fatt_verify=True, fatt_r8=True, hc8_fast=True),
    "ref2tnfn": _w4(tier_nax_ooo=True, nax_native=True, fatt_verify=True, fatt_r8=True),
    "ref2tnh8n": _w4(True, tier_nax_ooo=True, nax_native=True, hc8_fast=True),
})
SCRATCH_RE = re.compile(r"(ref2tnfhn|ref2tnh8n|ref2tnaxn|ref2tnfn|ref2tnfh|ref2tnh8|ref2tnax|ref2tnf|ref2tfh|ref2tf0|ref2th8p|"
                        r"ref2th8|ref2tf|ref2ct|ref2t|ref2b|ref2|ref)(\.r\d+)")   # scratch runs of a VARIANTS entry
REF2_CODE = os.path.join(env.ROOT, "snap", "leap-86ca915")   # the REF2 the R-40 verdict judges: Leap 86ca915 (ref2 + its
                                                  # kernels, GREEN in our harness 16:52), frozen: Leap edits them daily
FRUGAL = {"w2f_r10": 0.10, "w2f_r05": 0.05}        # resident fraction of each layer's experts
B12 = {"b12": 0.10, "b8": 0.05, "b8fast": 0.05,      # load_b12(frac); the configs' switches: cli.B12_ENV
       "b12c192": 0.10,                           # Breakthrough's D1 (B12's frugal wait-top-2) at C_m 192 (B12_CM)
       "b12l": 0.10, "b12ll": 0.10,               # B12 + cold ranks 2-3 / 2-9 half a layer late (Breakthrough 10-23)
       "b12llb16": 0.10, "b12llb18": 0.10, "b12llb20": 0.10,   # B12LL + a per-pass expert budget (Fit12's hook)
       "b12llugb": 0.10, "b12llugbx": 0.10,       # b16 only on MTP-confident passes (UGB-0.5); x = gate 1.01 (never on)
       "b12llu2": 0.10,                           # U2 = B12LL + Fit12's PR1 (keep-1 commits replay the PLE layer on 2 rows)
       "b12llk8": 0.10, "b12llka80": 0.10,        # L1-17 (Leap; Fit12's hook): verify rows keep top-8 / >= 80 % of top-10
       "b12llk10": 0.10, "b12llka100": 0.10,      # their no-ops (bitwise = B12LL): the hook's identity checks
       "b12llk8u2": 0.10, "b12llk10u2": 0.10,     # top-8 on U2 (PR1 on: Fit12's TK2 stack) + its no-op (= U2 bitwise)
       "b12llfoff": 0.10, "b12llfon": 0.10,       # B12_FOLD off / on (Breakthrough 10-27): B12LL in today's shipped env
       "b12llox": 0.10, "b12llfx": 0.10, "b12llff": 0.10,   # the fold confirmation (R-41): old shipped / fold-exact / fold-fast
       "b12llfx2": 0.10, "b12llff2": 0.10,        # R-41D: fold-exact / fold-fast (FFX) + R-45's fix (B12_FRS_NOCYCLE)
       "b12llff3": 0.10,                          # decision 5: b12llff2 on v3o + B12_PFR_MIN 512
       "b12llff4": 0.10, "b12llff4p": 0.10,       # PMIN12: fold-fast on bt_b12 8db71cc537 (v3p), without / with B12_PMIN 0.5
       "b12llff4t0c": 0.10, "b12llff4t0h": 0.10}  # T0 (PREREG_t0): Fit12's top-2 two-pass at theta 0.8 / 0.95, MTP off
FOLD_ENGINE = {"b12llfoff": ("v3l", "1", "0"),    # (B12_ENGINE, B12_TRIMS, B12_FOLD) of the fold verdict's arms: bt_b12's
               "b12llfon": ("v3m", "1", "1"),     # shipped engine and trims, pinned (cli.B12_ENV), not the v3f verdict pin
               "b12llox": ("v3l", "1", "0"), "b12llfx": ("v3m", "1", "1"), "b12llff": ("v3m", "1", "1"),
               "b12llfx2": ("v3m", "1", "1"), "b12llff2": ("v3m", "1", "1"), "b12llff3": ("v3o", "1", "1"),
               "b12llff4": ("v3p", "1", "1"), "b12llff4p": ("v3p", "1", "1")}
FOLD_SWITCHES = {"b12llox": ("balanced", None, None),   # (B12_PFR_GRID, B12_FOLD_AW, B12_FOLD_CM) of the confirmation's
                 "b12llfx": ("keep", None, None),       # arms (Breakthrough's R-41 envs, cli.B12_ENV)
                 "b12llff": ("balanced", "1", "1"),
                 "b12llfx2": ("keep", None, None),      # R-41D (docs/PREREG_r41d_2026-10-29.md, amendment 1)
                 "b12llff2": ("balanced", "1", "2700:60,8600:60,16500:56,32800:47"),
                 "b12llff3": ("balanced", "1", "2700:60,8600:60,16500:56,32800:47"),
                 "b12llff4": ("balanced", "1", "2700:60,8600:60,16500:56,32800:47"),   # PMIN12
                 "b12llff4p": ("balanced", "1", "2700:60,8600:60,16500:56,32800:47")}
BT_B12_SHA = {"b12llfoff": "664b010b0d", "b12llfon": "664b010b0d",   # sha256[:10] of ~/Breakthrough/scripts/bt_b12.py:
              "b12llox": "c3f957cba6", "b12llfx": "c3f957cba6", "b12llff": "c3f957cba6",   # the code each arm's rows
              "b12llfx2": "ce6d4c63d2", "b12llff2": "ce6d4c63d2",   # R-41D: Breakthrough's swap with R-45's fix
              "b12llff3": "677c6558b6",   # decision 5: the live file with v3o + B12_PFR_MIN
              "b12llff4": "8db71cc537", "b12llff4p": "8db71cc537",   # PMIN12: Breakthrough 3676db3 (v3p default + B12_PMIN)
              "b12llff4t0c": "8db71cc537", "b12llff4t0h": "8db71cc537"}   # T0: Fit12's gate file (= the live sha)
#                                                  are made on (31b63af changed the defaults: the old arms mean other code)
B12_MTP = {"b12": True, "b8": False, "b8fast": True, "b12c192": True,   # B8 (eager) was judged as Breakthrough measured it; B12L / B12LL:
           "b12l": True, "b12ll": True,           # MTP (late writes fixed under MTP: ad8779c + 1a5eee1)
           "b12llb16": True, "b12llb18": True, "b12llb20": True, "b12llugb": True, "b12llugbx": True, "b12llu2": True,
           "b12llk8": True, "b12llka80": True, "b12llk10": True, "b12llka100": True, "b12llk8u2": True, "b12llk10u2": True,
           "b12llfoff": True, "b12llfon": True, "b12llox": True, "b12llfx": True, "b12llff": True,
           "b12llfx2": True, "b12llff2": True, "b12llff3": True, "b12llff4": True, "b12llff4p": True,
           "b12llff4t0c": False, "b12llff4t0h": False}   # T0 has no drafter
FRS_NOCYCLE = ("b12llfx2", "b12llff2", "b12llff3", "b12llff4", "b12llff4p")   # R-41D: R-45's fix on, asserted at load
PFR_MIN = {"b12llff3": "512", "b12llff4": "512", "b12llff4p": "512"}   # decision 5: B12_PFR_MIN, asserted at load
PMIN = {"b12llff4p": "0.5"}               # PMIN12 (Breakthrough 11-01): B12_PMIN, asserted at load (unset elsewhere)
R53A = {"b12llff4": "0", "b12llff4p": "0"}
T0 = {"b12llff4t0c": 0.8, "b12llff4t0h": 0.95}   # T0's theta per arm (PREREG_t0: Breakthrough's code / chat settings)
T0_ENV = {'MTPLX_PREFILL_CHUNK_SIZE': '512', 'BIGMOE_TIER_PF_GROUP': '64', 'BIGMOE_RESET_GEN': '0', 'EVALSUITE_SHIP': '1', 'B12_LATE3': '2,10', 'B12_NORM': 'mass', 'BIGMOE_PF_GROUP': '64', 'BIGMOE_ASYNC_EVERY': '1', 'B12_TRIMS': '1', 'B12_PFR': '1', 'B12_ALLWAIT': '0', 'B12_CACHE_MB': '64', 'BIGMOE_LOOP_GUARD': '0', 'MTPLX_NGRAM_PREWARM': 'off', 'MTPLX_COMPILED_VERIFY': '0', 'MTPLX_QWEN4EXP_COMPILE': '0', 'MTPLX_COMPILE_AR_FORWARD': '0', 'MTPLX_COMPILED_TARGET_PREFIX': '0', 'BIGMOE_SPARE': '0', 'BIGMOE_PF_NOPUBLISH': '1'}   # Fit12's T0 gate env (21:43), asserted at load
T0_POP = ('B12_ENGINE', 'B12_FOLD', 'B12_PFR_GRID', 'B12_CM', 'B12_CL', 'B12_FOLD_CM', 'B12_FOLD_AW', 'B12_W', 'B12_PMIN')   # unset: the live file's fast defaults   # PMIN12: v3p's R-53 part a off (F12_R53A, bt_b12's default since 12:4x)
B12_RE = re.compile(r"(b12llff4t0c|b12llff4t0h|b12llff4p|b12llff4|b12llff3|b12llfx2|b12llff2|b12c192|b12llfoff|b12llfon|b12llox|b12llfx|b12llff|b12llugbx|b12llugb|b12llu2|b12llk10u2|b12llk8u2|b12llka100|b12llka80|b12llk10|b12llk8|b12llb16|b12llb18|b12llb20|b12ll|b12l|b12|b8fast|b8)(g?)((?:\.r\d+)?)")   # g = guard
F24_RE = re.compile(r"f24(b16|b18|b20|nat)((?:\.r\d+)?)")   # BigMoE's 24 GB fast mode (scripts/fast24.py); nat = the v3
#                                                    tier plan's natural routing (exact-to-v3), the speed-up's baseline


def variant(name):
    if name in VARIANTS:
        return dict(VARIANTS[name], name=name)
    m = SCRATCH_RE.fullmatch(name)
    if m:
        return dict(VARIANTS[m.group(1)], name=name)
    if name in FRUGAL:
        return dict(model="frugal", sorted=False, plan=None, frac=FRUGAL[name], name=name)
    m = F24_RE.fullmatch(name)
    if m:                                         # load_fast24's own env and decode (guard off, fixed MTP widths 3:3:0)
        return dict(model="fast24", sorted=False, plan=None, budget="" if m.group(1) == "nat" else m.group(1),
                    mtp=True, fast24=True, name=name)
    m = B12_RE.fullmatch(name)
    if m:                                         # the verdict runs guard-off (it derails long math: 10-22 15:43)
        base = m.group(1)
        return dict(model="b12", sorted=False, plan=None, frac=B12[base], mtp=B12_MTP[base], guard=bool(m.group(2)),
                    base=base, name=name, t0=T0.get(base))
    m = re.fullmatch(r"plan([234]{2,16})", name)
    if m:
        return dict(model="ref", sorted=True, plan=m.group(1), name=name)
    raise SystemExit(f"unknown variant {name!r}: ref, floor, ar4, w2f_r10, w2f_r05, b12 / b8 / b8fast[g][.rN] or plan<bits>, "
                     f"e.g. plan4433333333")


def plan_segments(plan):
    segs = []
    for r, c in enumerate(plan):
        n = int(c)
        if segs and segs[-1][2] == n:
            segs[-1][1] = r + 1
        else:
            segs.append([r, r + 1, n])
    return [tuple(s) for s in segs]


# ---------------------------------------------------------------------------------------------- tier math (vendored)
_KEEP = {3: 0xEEEEEEEE, 2: 0xCCCCCCCC}            # keep the top n bits of every nibble


def _make_mask(n):
    low, keep = (1 << (4 - n)) - 1, mx.array(_KEEP[n], dtype=mx.uint32)

    @mx.compile
    def f(w):                                     # (codes with the low 4-n bits cleared, per-word sum of those bits)
        lsum = None
        for j in range(8):
            part = (w >> (4 * j)) & low
            lsum = part if lsum is None else lsum + part
        return w & keep, lsum
    return f


_MASK = {n: _make_mask(n) for n in _KEEP}


def tier_weights(w, s, b, gs, n):
    """MSB-n tier of 4-bit affine codes in 4-bit format: (codes, bias); scales unchanged."""
    wm, lsum = _MASK[n](w)
    off = lsum.reshape(*s.shape, gs // 8).sum(-1).astype(mx.float32) / gs
    bias = (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)
    return wm, bias


def _tier_proj(lin, h, sl, idx, n):
    w, s, b = lin.weight[sl], lin.scales[sl], lin.biases[sl]
    wm, bias = tier_weights(w, s, b, lin.group_size, n)
    return mx.gather_qmm(h, wm, s, bias, rhs_indices=idx, transpose=True, group_size=lin.group_size, bits=4)


def _make_fast_tier(n, gs):
    """gather the slots of all three projections, clear the low 4-n bits, bias += scale * mean(cleared bits), as ONE
    compiled call (decode). Bit-identical to tier_weights by construction: the masking and the bit sums are integer
    ops, the mean is k/64 (k <= 64) and scale is bf16, so scale * mean is exact in float32 and the single rounding of
    bias + scale * mean is the same with or without an FMA; fast_tier_check() also compares the two on data."""
    low, keep = (1 << (4 - n)) - 1, _KEEP[n]

    def one(w, s, b):
        lsum = None
        for j in range(8):
            part = (w >> (4 * j)) & low
            lsum = part if lsum is None else lsum + part
        off = lsum.reshape(*s.shape, gs // 8).sum(-1).astype(mx.float32) / gs
        return w & keep, (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)

    @mx.compile
    def f(gw, gsc, gb, uw, usc, ub, dw, dsc, db, sl):
        out = []
        for w, s, b in ((gw, gsc, gb), (uw, usc, ub), (dw, dsc, db)):
            s_ = s[sl]
            wm, bias = one(w[sl], s_, b[sl])
            out += [wm, s_, bias]
        return tuple(out)
    return f


_FAST = {}


def _fast_tiers(sw, sl, n):
    gs = sw.up_proj.group_size
    f = _FAST.get((n, gs)) or _FAST.setdefault((n, gs), _make_fast_tier(n, gs))
    p = [getattr(sw, q) for q in ("gate_proj", "up_proj", "down_proj")]
    return f(p[0].weight, p[0].scales, p[0].biases, p[1].weight, p[1].scales, p[1].biases,
             p[2].weight, p[2].scales, p[2].biases, sl)


def fast_tier_check(n=3, slots=16, take=8, seed=3):
    """compiled decode tiers == tier_weights on random pool-shaped data (125B expert shapes, group 64)"""
    k = mx.random.key(seed)
    ks = mx.random.split(k, 9)
    pools = []
    for i, (o, c) in enumerate(((640, 320), (640, 320), (2560, 80))):
        w = mx.random.randint(0, 2 ** 31, (slots, o, c), key=ks[3 * i]).astype(mx.uint32) * 2 + mx.random.randint(0, 2, (slots, o, c), key=ks[3 * i + 1]).astype(mx.uint32)
        s = (mx.random.uniform(1e-3, 2e-2, (slots, o, c * 8 // 64), key=ks[3 * i + 2])).astype(mx.bfloat16)
        b = (-8 * s.astype(mx.float32) + mx.random.normal(s.shape, key=k) * 1e-3).astype(mx.bfloat16)
        pools.append((w, s, b))
    sl = mx.array([3, 7, 0, 15, 9, 2, 11, 5][:take], dtype=mx.uint32)
    f = _make_fast_tier(n, 64)
    got = f(*[a for p in pools for a in p], sl)
    ok = True
    for i, (w, s, b) in enumerate(pools):
        wm, bias = tier_weights(w[sl], s[sl], b[sl], 64, n)
        ok &= mx.array_equal(got[3 * i], wm).item() and mx.array_equal(got[3 * i + 1], s[sl]).item() \
            and mx.array_equal(got[3 * i + 2], bias).item()
    return bool(ok)


def selftest():
    """vendored tier math == MoonShots' planes125 (when present), and == unpack / clear / repack."""
    w = mx.random.randint(0, 2 ** 31, (3, 8, 32), key=mx.random.key(7)).astype(mx.uint32) * 2 + 1
    q = ((w[..., None] >> mx.arange(0, 32, 4).astype(mx.uint32)) & 15).reshape(3, 8, -1)
    for n, low in ((3, 1), (2, 3)):
        wm, ls = _MASK[n](w)
        qm = ((wm[..., None] >> mx.arange(0, 32, 4).astype(mx.uint32)) & 15).reshape(3, 8, -1)
        assert mx.array_equal(qm, q & (15 - low)).item()
        assert mx.array_equal(ls, (q & low).reshape(3, 8, -1, 8).sum(-1).astype(mx.uint32)).item()
    p = os.path.expanduser("~/MoonShots/patches/planes125")
    if os.path.isdir(p):
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("_ms_planes125", os.path.join(p, "planes125.py"))
        m = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(m)
        except Exception as e:                      # their module imports mtplx; fine to skip if it moved
            return f"planes125 not importable ({type(e).__name__}); vendored math self-checked only"
        for n in (3, 2):
            a, b_ = _MASK[n](w), m._MASK[n](w)
            assert mx.array_equal(a[0], b_[0]).item() and mx.array_equal(a[1], b_[1]).item(), "tier math drifted"
        return "tier math == MoonShots planes125"
    return "vendored tier math self-checked"


# ---------------------------------------------------------------------------------------------- patches
def _text(model):
    return model.language_model if hasattr(model, "language_model") else model


def install(model):
    """Patch MTPLX's qwen4_exp SparseMoeBlock (rank-sorted combine, routing capture, the scores for BigMoE's fused tier
    combine) and the streamed switch's base class (rank tiers). Both read STATE, so one loaded model serves ref / floor
    / plan variants."""
    from mtplx.models import qwen4_exp
    layers = _text(model).model.layers
    sws = [l.mlp.switch_mlp for l in layers if getattr(getattr(l, "mlp", None), "switch_mlp", None) is not None]
    sw0 = next(sw for sw in sws if getattr(sw, "_bm", None) is not None)
    base = type(sw0).__mro__[1]                   # the class bigmoe.stream._streamed_class wrapped
    for p in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(sw0, p)
        assert lin.bits == 4 and lin.group_size % 16 == 0, (p, lin.bits, lin.group_size)
    if getattr(base, "_evalsuite", False):
        return
    orig_sw = base.__call__

    def switch(self, x, indices, *a, **kw):
        segs = STATE.segs
        bm = getattr(self, "_bm", None)
        if segs is None or bm is None:
            return orig_sw(self, x, indices, *a, **kw)
        single = indices.size == indices.shape[-1]
        if (single and not STATE.tiers_decode) or (not single and not STATE.tiers_prefill):
            return orig_sw(self, x, indices, *a, **kw)
        assert segs[-1][1] == indices.shape[-1], (segs, indices.shape)
        outs = []
        if single:                                # decode: slots into the pool, tier the gathered slots (planes125)
            xe = mx.expand_dims(x, (-2, -3))
            for a0, b0, n in segs:
                ids = indices[..., a0:b0]
                if n == 4:
                    outs.append(orig_sw(self, x, ids, *a, **kw))
                    continue
                sl = ids.reshape(-1)
                idx = mx.arange(sl.size).astype(mx.uint32).reshape(ids.shape)
                if STATE.fast_decode:
                    gw, gsc, gb, uw, usc, ub, dw, dsc, db = _fast_tiers(self, sl, n)
                    gsz = self.up_proj.group_size
                    qmm = lambda h_, w_, s_, b_: mx.gather_qmm(h_, w_, s_, b_, rhs_indices=idx, transpose=True,
                                                               group_size=gsz, bits=4)
                    h = self.activation(qmm(xe, uw, usc, ub), qmm(xe, gw, gsc, gb))
                    outs.append(qmm(h, dw, dsc, db).squeeze(-2))
                else:
                    h = self.activation(_tier_proj(self.up_proj, xe, sl, idx, n), _tier_proj(self.gate_proj, xe, sl, idx, n))
                    outs.append(_tier_proj(self.down_proj, h, sl, idx, n).squeeze(-2))
            STATE.stats["tier_decode_calls"] += 1
        else:                                     # prefill: the full-E scratch (FULLSCR), tiered once per layer and pass
            E = self.up_proj.weight.shape[0]
            assert E == bm.get("E_full", E) and E >= 256, "tiered prefill needs the full-E scratch (BIGMOE_PF_FULLSCR=1)"
            key = (bm["L"], STATE.pass_id)        # every tier level of the plan at once: one level per memo made a
            if STATE.memo.get("key") != key:      # 3+2-bit plan rebuild both on every call (10x slower, 2026-10-20)
                STATE.memo.clear()
                tiered = {}
                for n in sorted({n for *_, n in segs if n < 4}):
                    for p in ("gate_proj", "up_proj", "down_proj"):
                        lin = getattr(self, p)
                        wm, bias = tier_weights(lin.weight, lin.scales, lin.biases, lin.group_size, n)
                        mx.eval(wm, bias)         # the scratch buffer is refilled for the next layer
                        tiered[n, p] = (wm, bias)
                STATE.memo.update(key=key, tiered=tiered)
                STATE.stats["masked_scratch"] += 1
            for a0, b0, n in segs:
                ids = indices[..., a0:b0]
                if n == 4:
                    outs.append(orig_sw(self, x, ids, *a, **kw))
                    continue
                saved = {p: (getattr(self, p).weight, getattr(self, p).biases) for p in ("gate_proj", "up_proj", "down_proj")}
                try:
                    for p in ("gate_proj", "up_proj", "down_proj"):
                        getattr(self, p).weight, getattr(self, p).biases = STATE.memo["tiered"][n, p]
                    outs.append(orig_sw(self, x, ids, *a, **kw))
                finally:
                    for p, (w0, b0_) in saved.items():
                        getattr(self, p).weight, getattr(self, p).biases = w0, b0_
            STATE.stats["tier_prefill_calls"] += 1
        return mx.concatenate([o.astype(outs[0].dtype) for o in outs], axis=-2)

    base.__call__ = switch
    base._evalsuite = True
    blk = qwen4_exp.SparseMoeBlock
    orig_block = blk.__call__

    def block(self, x):
        # mlx-lm Qwen3NextSparseMoeBlock's math with the top-k sorted by gate score, so the streamed switch gets
        # rank-ordered ids (planes125's decode block, here for every width); STATE.sorted False = the stock order
        if getattr(self.switch_mlp, "_bm", None) is None or not STATE.sorted:
            if STATE.capture is not None and getattr(self.switch_mlp, "_bm", None) is not None:
                g = mx.softmax(self.gate(x), axis=-1, precise=True)          # separate ops: the stock path is untouched
                STATE.capture.setdefault(self.switch_mlp._bm["L"], []).append(
                    mx.argpartition(g, kth=-self.top_k, axis=-1)[..., -self.top_k:])
            return orig_block(self, x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        if STATE.capture is not None:
            STATE.capture.setdefault(self.switch_mlp._bm["L"], []).append(inds)
        # R-69 (Leap): with BigMoE's BIGMOE_TIER_COMBINE (3c; + BIGMOE_OOO) its tier call takes the scores and returns
        # the combined rows, bitwise == the combine below. BigMoE's own block (tier.install_block) hands them over the
        # same way; this patch replaces it, so without this the two switches never engage (R-69D).
        t, bt = self.switch_mlp._bm.get("tier"), sys.modules.get("bigmoe.tier")
        fused = t is not None and bt is not None and bt._COMBINE and bt._GLU == "3c"
        if fused:
            t["scores"] = scores
        try:
            y = self.switch_mlp(x, inds)
        finally:
            if fused:                             # a call that never reached the tier call (prefill) leaves them
                t.pop("scores", None)
                t.pop("shared", None)
        if y.ndim > x.ndim:                       # [..., K, D]: not combined in the switch
            y = (y * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    blk.__call__ = block


def configure(v, prefill=True):
    """Set STATE for variant dict v. prefill=True: tiers on every token (prefill tests). False: deployed semantics
    of a decode-only variant (exact prefill, tiers at decode)."""
    STATE.sorted = v["sorted"]
    STATE.segs = plan_segments(v["plan"]) if v.get("plan") else None
    STATE.tiers_prefill = bool(STATE.segs) and prefill
    STATE.tiers_decode = bool(STATE.segs)
    STATE.memo.clear()


class _SnapFinder:
    """Imports the top-level names a snapshot holds from the snapshot, whatever sys.path says by then. Its qmmg25.py puts
    ~/Leap/scripts first before `from gmr24 import ...` (so until 2026-11-01 gmr24 came from Leap's live tree: the same
    bytes since 10-24, 8e56a11), and ref2's CFG gdn_exact puts ~/Leap/third_party/ms_gdn first before its GDN imports."""
    def __init__(self, dirs):
        self.dirs = dirs

    def find_spec(self, name, path=None, target=None):
        for d in self.dirs if path is None else ():
            p = os.path.join(d, name + ".py")
            if os.path.isfile(p):
                return importlib.util.spec_from_file_location(name, p)
        return None


def ref2_import(v):
    """-> ref2 from a REF2 variant's frozen snapshot (snap/<ref2_code>, else REF2_CODE), with the variant's ref2_cfg set
    in ref2.CFG (REF2c: switches the snapshot has but leaves off); call it before ref2.apply(). Every module the snapshot
    holds loads from it (_SnapFinder), its ms_gdn/ too: the copy of MoonShots' exact GDN files that Leap's ref2 since
    129c86e imports for CFG gdn_exact. The GDN kernels' header (tierglu, maybe imported earlier by BigMoE) must be its bytes."""
    code = os.path.join(env.ROOT, "snap", v["ref2_code"]) if v.get("ref2_code") else REF2_CODE
    gdn = os.path.join(code, "ms_gdn")
    dirs = [d for d in (code, gdn) if os.path.isdir(d)]
    sys.meta_path.insert(0, _SnapFinder(dirs))
    import ref2
    if gdn in dirs:
        import hashlib
        import gdn_exact_patch  # noqa: F401  (ref2's _gdnx() then finds it imported)
        sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
        tg = sys.modules["tierglu"].__file__      # gdn_exact's `from tierglu import _HDR`: maybe BigMoE's planes_engine copy
        assert sha(tg) == sha(os.path.join(gdn, "tierglu.py")), f"tierglu {tg} differs from the snapshot's"
    for d in dirs:                                # every snapshot module imported so far came from the snapshot
        for f in sorted(os.listdir(d)):
            m = sys.modules.get(f[:-3]) if f.endswith(".py") and f != "tierglu.py" else None
            assert m is None or os.path.dirname(os.path.realpath(m.__file__)) == os.path.realpath(d), f"{f} from {m.__file__}"
    cfg = v.get("ref2_cfg") or {}
    assert set(cfg) <= set(ref2.CFG), f"ref2.CFG has no {sorted(set(cfg) - set(ref2.CFG))}"
    ref2.CFG.update(cfg)
    return ref2


def hc8_fake(model, bits=8, group_size=64):
    """Q3 HC8 (ROUTE_350, Kamen's day 9): the weights Breakthrough's hc8.install takes to 8 bits (v3*/hc8.py: every trunk
    GatedResidual's input_mix_weight_down / up, the PLE layer's key_proj / value_proj; not the MTP head's), MLX affine,
    group 64, as fake quantization: each becomes dequantize(quantize(W)) in bf16 and stays an nn.Linear, so REF2-T's
    kernels (leapk.hc_read reads the bf16 mixers) run unchanged and the arm differs from REF2-T only in those weights'
    values. Call it before ref2.apply(). -> (weights swapped, largest |W - W8|)"""
    import mlx.nn as nn
    import mtplx.models.qwen4_exp as Q
    n, err = 0, 0.0
    for name, mod in _text(model).named_modules():
        if name.startswith("mtp"):
            continue
        attrs = (("input_mix_weight_down", "input_mix_weight_up") if isinstance(mod, Q.GatedResidual) else
                 ("key_proj", "value_proj") if isinstance(mod, Q.PLELayer) else ())
        for a in attrs:
            lin = getattr(mod, a)
            w = lin.weight
            assert isinstance(lin, nn.Linear) and w.dtype == mx.bfloat16 and w.shape[1] % group_size == 0, (name, a, lin)
            w8 = mx.dequantize(*mx.quantize(w, group_size=group_size, bits=bits), group_size=group_size, bits=bits)
            assert w8.dtype == mx.bfloat16 and w8.shape == w.shape, (name, a, w8.dtype, w8.shape)
            err = max(err, mx.abs(w8.astype(mx.float32) - w.astype(mx.float32)).max().item())
            lin.weight = w8
            n += 1
    mx.eval(_text(model).parameters())
    return n, err


def load(v, C=128):
    """-> (rt, streamer). v: variant dict. C: expert slots per layer (numerics do not depend on it: exact streaming)."""
    if v["model"] == "frugal":                    # Breakthrough's own hook and runtime; none of our patches
        assert os.environ.get("EVALSUITE_FRUGAL") == "1", "frugal runs need EVALSUITE_FRUGAL=1 (cli.gpu sets it)"
        sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
        from bt_frugal import load_frugal
        rt, st = load_frugal(v["frac"])
        configure(v)
        return rt, st
    if v["model"] == "fast24":                    # BigMoE's hook (pinned in cli.FAST24_COMMIT); none of our patches
        assert os.environ.get("EVALSUITE_FAST24") == "1", "fast24 variants need EVALSUITE_FAST24=1 (cli.gpu sets it)"
        sys.path.insert(0, os.path.expanduser("~/BigMoE/scripts"))
        from fast24 import load_fast24
        rt, st = load_fast24(0.25, mtp=True, budget=v["budget"])
        configure(v)
        return rt, st
    if v["model"] == "b12":                       # Breakthrough's hook and runtime (v3f), MTP on; none of our patches
        assert os.environ.get("EVALSUITE_SHIP") == "1", "b12 / b8 / b8fast need EVALSUITE_SHIP=1 (cli.gpu sets it)"
        want = {"b12": {"B12_NORM": "mass"}, "b8": {"B12_LATE3": "2,4", "B12_NORM": "mass"},
                "b8fast": {"B12_W": "3", "B12_NORM": "mass"}, "b12l": {"B12_LATE3": "2,4", "B12_NORM": "mass"},
                "b12ll": {"B12_LATE3": "2,10", "B12_NORM": "mass"},
                "b12c192": {"B12_CM": "192", "B12_NORM": "mass"}}[v["base"][:5] if v["base"].startswith("b12ll") else v["base"]]
        for k in ("B12_LATE3", "B12_W", "B12_CM", "B12_CL", "B12_HC", "B12_LATE1", "B12_PRE", "B12_PF", "B12_NORM"):
            assert os.environ.get(k) == want.get(k), f"{v['name']}: {k}={os.environ.get(k)!r}, want {want.get(k)!r}"
        if v.get("t0") is not None:               # T0 (PREREG_t0): Fit12's gate env exactly, on the pinned live file
            bad = {k: os.environ.get(k) for k, x in T0_ENV.items() if os.environ.get(k) != x}
            bad.update({k: os.environ[k] for k in T0_POP if k in os.environ})
            assert not bad, f"{v['name']}: env differs from Fit12's T0 gate: {bad}"
            import hashlib
            sha = hashlib.sha256(open(os.path.expanduser("~/Breakthrough/scripts/bt_b12.py"), "rb").read()).hexdigest()[:10]
            assert sha == BT_B12_SHA[v["base"]], f"{v['name']}: bt_b12.py is {sha}; T0 is gated on {BT_B12_SHA[v['base']]}"
        elif v["base"] in FOLD_ENGINE:            # the fold verdict: both arms in bt_b12's shipped engine + trims
            got = tuple(os.environ.get(k) for k in ("B12_ENGINE", "B12_TRIMS", "B12_FOLD"))
            assert got == FOLD_ENGINE[v["base"]], f"{v['name']}: (B12_ENGINE, B12_TRIMS, B12_FOLD) = {got}, want {FOLD_ENGINE[v['base']]}"
            if v["base"] in FOLD_SWITCHES:
                got = tuple(os.environ.get(k) for k in ("B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_CM"))
                assert got == FOLD_SWITCHES[v["base"]], \
                    f"{v['name']}: (B12_PFR_GRID, B12_FOLD_AW, B12_FOLD_CM) = {got}, want {FOLD_SWITCHES[v['base']]}"
            if v["base"] in FRS_NOCYCLE:
                assert os.environ.get("B12_FRS_NOCYCLE") == "1", \
                    f"{v['name']}: B12_FRS_NOCYCLE={os.environ.get('B12_FRS_NOCYCLE')!r}, want '1' (R-45's fix)"
            assert os.environ.get("B12_PFR_MIN") == PFR_MIN.get(v["base"]), \
                f"{v['name']}: B12_PFR_MIN={os.environ.get('B12_PFR_MIN')!r}, want {PFR_MIN.get(v['base'])!r}"
            assert os.environ.get("B12_PMIN") == PMIN.get(v["base"]), \
                f"{v['name']}: B12_PMIN={os.environ.get('B12_PMIN')!r}, want {PMIN.get(v['base'])!r}"
            if v["base"] in R53A:
                assert os.environ.get("F12_R53A") == R53A[v["base"]], f"{v['name']}: F12_R53A={os.environ.get('F12_R53A')!r}"
            import hashlib
            sha = hashlib.sha256(open(os.path.expanduser("~/Breakthrough/scripts/bt_b12.py"), "rb").read()).hexdigest()[:10]
            assert sha == BT_B12_SHA[v["base"]], \
                f"{v['name']}: bt_b12.py is {sha}; this arm's rows are made on {BT_B12_SHA[v['base']]}"
        else:
            assert (os.environ.get("B12_ENGINE"), os.environ.get("B12_TRIMS")) == ("v3f", "0"), \
                "the verdicts run bt_b12's v3f engine without trims (cli.SHIP_AR); bt_b12's defaults moved to v3g + trims"
        sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
        sys.path.insert(1, os.path.expanduser("~/SpecDecode"))          # spec125 (MTP decode)
        from bt_b12 import load_b12
        assert os.environ.get("BIGMOE_LOOP_GUARD") == (None if v["guard"] else "0"), "guard switch (cli.gpu sets it)"
        if v["base"].startswith(("b12llb", "b12llugb", "b12llu2", "b12llk")):   # Fit12's hook: B12LL + budget / PR1 / top-k
            sys.path.insert(1, os.path.expanduser("~/Fit12/scripts"))
            from fit_hook import load_b12ll_budget
            if v["base"] == "b12llu2":           # U2: PR1 only, no budget (Fit12 10-24 12:57; engine-independent)
                rt, st = load_b12ll_budget("", pr1=True)
            elif v["base"].startswith("b12llk") and v["base"].endswith("u2"):   # top-k on U2 (PR1 on; Fit12 16:30)
                rt, st = load_b12ll_budget("", pr1=True, topk=int(v["base"][6:-2]))
            elif v["base"].startswith("b12llka"):   # L1-17 adaptive k: the smallest rank prefix holding >= p of the top-10
                rt, st = load_b12ll_budget("", topp=int(v["base"][7:]) / 100)   # weight, renormalized (Fit12 15:35)
            elif v["base"].startswith("b12llk"):    # L1-17 top-k, renormalized over the kept experts (verify rows only)
                rt, st = load_b12ll_budget("", topk=int(v["base"][6:]))
            elif v["base"].startswith("b12llugb"):  # UGB-0.5: b16 on confident passes only; x: a gate that never opens
                rt, st = load_b12ll_budget("b16", ugb=1.01 if v["base"].endswith("x") else 0.5)
            else:
                rt, st = load_b12ll_budget(v["base"][5:])
        else:
            if v.get("t0") is not None:           # T0: Fit12's t0_twopass patches bt_b12's decode block before load_b12
                sys.path.insert(1, os.path.expanduser("~/Fit12/scripts"))   # (t0_hook.load, keeping the streamer)
                import bt_b12, t0_twopass
                t0_twopass.apply(bt_b12)
            rt, st = load_b12(v["frac"], mtp=v["mtp"])
            if v["base"] in FOLD_ENGINE:          # PMIN12: spec125.generate takes B12_PMIN's p_min iff the arm sets it
                import spec125
                want = float(PMIN[v["base"]]) if v["base"] in PMIN else None
                assert getattr(spec125.generate, "_b12_pmin", None) == want, f"{v['name']}: spec125 p_min wrapper != {want}"
        configure(v)
        return rt, st
    from bigmoe.loader import load_streamed
    import bigmoe
    assert os.path.realpath(bigmoe.__file__).startswith(os.path.realpath(env.BIGMOE_CODE)), \
        f"bigmoe imported from {bigmoe.__file__}, not {env.BIGMOE_CODE}"
    md = AR4_MODEL if v["model"] == "ar4" else env.REF_MODEL
    if v["model"] == "ar4":
        idx = json.load(open(os.path.join(md, "model.safetensors.index.json")))
        assert "evalsuite_anchor" in idx.get("metadata", {}), "models/ar4 is not linked yet (tools/fetch_autoround.py --link)"
    rt, st = load_streamed(md, C, mtp=False)
    model = rt.model if hasattr(rt, "model") else rt
    for L, sw in enumerate(l.mlp.switch_mlp for l in _text(model).model.layers):
        sw._bm["E_full"] = st.hdr[st.prefix_fmt.format(L=L) + ".gate_proj.weight"][3][0]
    install(model)
    if v.get("hc8"):                              # before ref2.apply(): ref2 caches each mixer hc_read reads (_WCAT)
        assert not any(os.environ.get(k) == "1" for k in ("BIGMOE_COMPILE_HC", "BIGMOE_HC_VREAD", "BIGMOE_HC_PACK"))
        print("hc8_fake: %d weights at 8 bits, largest |W - W8| %.3g" % hc8_fake(model), flush=True)
    if v.get("ref2"):                             # after install: ref2's class patches wrap the suite's pass-through
        ref2 = ref2_import(v)
        if v.get("ref2_code"):                    # REF2b: key buckets on; tier pools only for REF2-T (R-58), as S45-T ran
            tier = {k: os.environ.get(k) for k in ("BIGMOE_TIER", "BIGMOE_TIER_GLU", "BIGMOE_TIER_CL", "BIGMOE_TIER_COMBINE")}
            want = (dict(BIGMOE_TIER=v["tier"], BIGMOE_TIER_GLU="3c", BIGMOE_TIER_CL="112", BIGMOE_TIER_COMBINE=None)
                    if v.get("tier") else dict.fromkeys(tier))
            assert ref2.CFG.get("key_buckets") is True and tier == want, (ref2.CFG, tier)
        if ref2.CFG.get("gdn_inproj"):            # Leap 91cd316: kept unset (if set, ref2 pads MTPLX's concatenation)
            assert "MTPLX_FUSED_GDN_INPROJ" not in os.environ, "MTPLX_FUSED_GDN_INPROJ is set"
        if v.get("hc8r"):                         # HC8-1P (Leap c805fd0): the trunk mixers to 8-bit g64 before ref2.apply,
            import hc8r                           # as Leap's hc8_wire_check does; the snapshot's copy (_SnapFinder)
            snap = os.path.realpath(os.path.join(env.ROOT, "snap", v["ref2_code"]))
            assert os.path.dirname(os.path.realpath(hc8r.__file__)) == snap, f"hc8r from {hc8r.__file__}"
            n, mb0, mb1, _ = hc8r.hc8_install(model)
            assert n > 0, "hc8r.hc8_install swapped nothing"
            print(f"hc8r.hc8_install: {n} linears at 8 bits, {mb0:.0f} -> {mb1:.0f} MB", flush=True)
        ref2.apply()
        st_ = ref2.self_test()
        assert all(st_.values()), f"ref2.self_test failed: {st_}"
        assert all(ref2.CFG[k] == x for k, x in v.get("ref2_cfg", {}).items()), ref2.CFG
        ref2.reset_counts()                       # self_test bumps qmm16c / mlx16 / linear
        if v.get("hc8"):                          # no cached concatenation of a model mixer from before the swap
            gr = {id(m): m for _, m in _text(model).named_modules() if type(m).__name__ == "GatedResidual"}
            for k, w in ref2._WCAT.items():
                d = gr[k].input_mix_weight_down.weight if k in gr else None
                assert d is None or mx.array_equal(w[:d.shape[0]], d).item(), "ref2._WCAT holds a pre-swap mixer"
        print(f"ref2 applied; CFG {v['ref2_cfg']}; self_test {st_}" if v.get("ref2_cfg") else f"ref2 applied; self_test {st_}",
              flush=True)
    configure(v)
    return rt, st
