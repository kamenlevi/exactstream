"""REF2 (R-40; Kamen approved decision 4, Tue 2026-10-27 22:40): a batch-invariant exact reference for the 125B's
45 GB engine (MTPLX, MLX 0.32.2, M5). Every row's bits are the same whether it runs alone (AR decode) or inside a
16-row verify check, so speculative decoding against REF2's own AR output is exact by construction (INV1: all 16
rows, every layer record and the logits bit-identical on CA's 12-layer slice).
Prior art (the principle): Thinking Machines Lab, H. He, "Defeating Nondeterminism in LLM Inference" (Sep 2025;
batch-invariant RMSNorm / matmul / attention with fixed split sizes; repo batch_invariant_ops); vLLM
VLLM_BATCH_INVARIANT=1; SGLang --enable-deterministic-inference. Ours is the MLX / Metal port for streamed 16-row
speculative checks; the MLX-specific breaks below (the 128-key mode switch, NAX attention past 512 keys, the
commit-phase sigmoid) are our own findings. The key buckets are TML's fixed split size, applied to MLX's key length.
What changes (monkeypatches on MTPLX, read-only):
  - quantized matmuls with <= 16 rows: padded to 16 rows; 4-bit group-64 with N >= 1024 (N % 64 == 0) through qmm16c
    (L1-15 / L1-23: one fixed 16-row NAX tile, fp32-exact dequant); everything else through MLX's own 16-row call
    (padding-independent, L1-23)
  - bf16 nn.Linear with < 16 rows: padded to 16 rows (MLX's 16-row GEMM)
  - GatedResidual reads (<= 16 rows): with block inject -> leapk.hc_read (R-09b's S4) at 16 rows; the head mixer ->
    the eager 16-row chain; both padded
  - QSA attention with < 16 rows: padded to 16 rows (copies of the last row) through the verify path, then the QSA cache
    is trimmed back (QSACache.trim)
  - routed experts (<= 16 rows): qmmg25's kernel (L1-25: batch-invariant, 2.3x more accurate than MLX's path, -22 ms
    per 16-row check). On BigMoE's streamed pools / an unfused SwitchGLU: gate and up as two calls (experts_split, =
    the fused halves bit for bit); with MTPLX_FUSED_GATE_UP=1 (resident): experts_nax on the fused weights.
    BIGMOE_TIER (base + sidecar pools) is not supported.
  - BigMoE's plain _sync_call (calls with > MAXK ids: 16-row checks, big commit replays): <= 16-row calls run the
    experts kernel on the pool with slot ids instead of a per-layer scratch gather (R-50 SLOTCALL: bit-equal, -384
    ms per check at C 144, ~200 vs 600-1600 ms at C 192). Under PF1 the prefill's staging is dropped first.
    CFG["slot_call"] = False keeps the shipped gather.
  - The commit (commit_verified_window) runs in the verify forward's phase (decode_verify + target_verify): its lone
    PLE-layer replay then takes the same sigmoid lowering as the AR and the checks (R-40 STEP 3t). CFG["commit_phase"].
  - 16-row attention: the call's key length N picks how MLX rounds a row. Below 128 keys and from 128 up to 512 each
    side is N-invariant, with a switch at 128 (STEP 3m / 3l). Past 512 keys, MLX 0.32.2's full (NAX) attention kernel
    gives a row one of two bit patterns depending on N, flipping irregularly; at equal N it is equal (STEP 4a / 4b).
    - So every 16-row call with N >= 512 runs at the next multiple of 512 (k / v zero-padded, masked): the key
      buckets, CFG["key_buckets"]. This is REF2b, R-40 STEP 4c.
    - A check row whose AR step (16 padded rows, p + 16 keys) falls in another class than the check's call is
      recomputed in that class and selected in exactly. CFG["sdpa_classes"], CFG["sdpa_bounds"], CFG["kb_from"],
      CFG["kb_step"].
  - GDN mixers with 1-16 rows: MoonShots' exact fused prework (conv + silu + l2norm + scale + conv state) and gated
    norm, each one dispatch and bitwise == MTPLX's eager chain (R-61a, 2026-11-01: tokens equal in the live loop at
    C 224, -3 ms per cycle). CFG["gdn_exact"]; frozen copy in ~/Leap/third_party/ms_gdn. No bits change.
  - R-62 (REF2c candidate, off by default): CFG["att_single"] runs each 16-row attention call as one fused SDPA instead
    of MTPLX's 12 head chunks (the key buckets stay; past 512 keys the bits differ from the chunked calls).
  - F1 (FAST-K, off by default; Leap Tue 2026-11-03, TABLE_350 'F1-SWITCH', EvalSuite PREREG_f1): CFG["fatt_verify"]
    (or REF2_FATT_VERIFY=1), read per call, runs each bool-mask 16- / 8-row verify call as fs4 (fatt16s, KS 4; batch-
    and N-invariant, so no key buckets or class re-runs) instead of MTPLX's 12 head chunks. Not bitwise == REF2-T
    (~2e-3); SD == AR holds by fs4's invariance (the AR pads to 16 rows, so it takes fs4 too). COUNTS fatt_verify /
    verify_old / verify_old_eligible. fs4 is imported on its first call (lazy modules: fatt16s.py, fatt16.py).
  - R-68 (REF2c candidate, off by default): CFG["gdn_inproj"] runs each GDN layer's qkv / z / b / a projections as one
    4-bit matmul on the row-concatenated packs + 32 zero rows (N 16480 -> 16512, so it takes qmm16c; MTPLX's own
    MTPLX_FUSED_GDN_INPROJ fusion is N 16480 and falls to MLX's call). qkv / z keep their bits, b's (and maybe a's)
    change. Built once per layer at its first call with the switch on (the four modules are then dropped); with the
    switch off the built layer runs the four calls on row slices of the same packs (= the stock modules, bitwise).
  - R-72 (exact-safe, off by default; bitwise == REF2-T at the layer level): CFG["att_rope"] runs each <= 16-row QSA
    attention call's RoPE as MTPLX's one-dispatch kernel (mtplx.kernels.qwen4_m4_rope.rope_qk; MTPLX wires it for 1-8
    rows behind MTPLX_QWEN4_VERIFY_GLUE) instead of the stock ~15-kernel chain; CFG["idx_prep"] runs the indexer's
    RMSNorm + partial RoPE (queries, pooled block keys) through MTPLX's shipped one-dispatch kernel
    (qsa_indexer_prepare) without MTPLX_FUSED_QSA_INDEXER, which would also switch the block selector.
  - R-73 (exact-safe, off by default): CFG["hc_write"] runs the layer's hyper-connection residual write on <= 16 rows as
    one compiled kernel (MTPLX's op-diet 'resid' view spelling under mx.compile; bf16 rounds after each op either way)
    instead of Multiply + Add. The MoE block's exact pieces (R-70 router, R-73 tail) live in the block owners' code.
  - GDNB (exact-safe, off by default; Mon 2026-11-02): CFG["gdn_body"] runs a 1-16-row GDN call's body (gdn_exact's
    prework + the recurrence + the exact gated norm) as ONE dispatch (scripts/gdnbody.py), bitwise == REF2-T.
  - R-83 (exact-safe, off by default): CFG["gdn_inproj_x"] is R-68's fusion without the reference change: qkv | z as ONE
    qmm16c call (N 16384; qmm16c's per-column bits do not depend on N) and b | a as ONE MLX 16-row call (N 96, below
    dense_min_n, so it stays on MLX's kernel). Every part keeps REF2-T's bits; two dispatches less per GDN layer.
  - R-82 (exact-safe, off by default): CFG["hc_gate"] runs the S4 HC read's partition sum and its gate1 lookups as one
    kernel (leapk.hc_read fuse_gate): the same mix / inject bits, one dispatch less per read.
  - D1-B (exact-safe, off by default; Mon 2026-11-02): CFG["hc_b"] runs the S4 HC read's partition sum + gate, MLX's
    K-320 up GEMM (a bitwise replica) and the tail as ONE kernel (leapk.hc_read kernel_b): the same mixed / inject bits,
    5 -> 3 dispatches per read (quiet: -0.7 to -1.0 ms per check). Takes precedence over hc_gate.
  - K-62 (REF2c-T candidate, off by default): CFG["tier_nax4"] runs every BigMoE tier GLU call on the matrix-unit kernel
    (~/Leap/scripts/tierglu_nax4.py): batch-invariant, more accurate than 3c, but its bits differ from 3c's, so it is
    a new reference that needs the same-scores test.
  - C1 (Mon 2026-11-02, REF2c-T candidate, off by default): CFG["tier_nax_ooo"] runs every tier GLU call on NAX6
    (~/Leap/scripts/tierglu_nax6.py, NAX4 with a fixed split-K) WITH BigMoE's OOO: bigmoe.tier._ooo_call's phase 1
    (resident pairs, encoded before the spins) and phase 2 (the late pairs) each take NAX6 on a phase-aware GPU grouping,
    then the same combine_phase; the other calls take NAX6 in one call (+ combine_phase when scores come). A pair's bits
    do not depend on its group (INV), so the two phases == one NAX6 call bitwise. New reference (NAX6 != 3c bits).
  - H-1i (Mon 2026-11-02, FAST, off by default; BigMoE's fastBEh arm): CFG["nax_hybrid"] runs C1's NAX6 calls (both the
    plain call and OOO's two phases) on NAX6-H (~/Leap/scripts/tierglu_nax6h.py): NAX6's kernels with one-pair groups on
    the SIMD ALUs (fp32 FMA dot) instead of the 16-row matrix op set. Needs tier_nax_ooo. Outputs differ from NAX6's in
    ~1 % of pair rows (bf16 last bits), and a pair's bits can depend on its group's size: FAST, not a reference.
  - N7-13 (Mon 2026-11-02, off by default): CFG["nax_native"] runs C1's NAX6 calls on the fragment-native record layout
    (~/Leap/scripts/tierglu_nax6n.py; bitwise == NAX6, quiet +18.7 % at S 16). The pools MUST have been converted with
    tierglu_nax6n.repack (3c cannot read them). With nax_hybrid too: NAX6-NH (== NAX6-H bitwise, +23.3 % at S 16).
  - N7-15 (Tue 2026-11-03, BigMoE's load path; inert without registered flags): native_flags(base, dirty_m, side,
    dirty_l) registers a tier layer's per-slot dirty flags (int32; 1 = an in-service demand load left the slot row-major).
    With CFG nax_native every NAX6-N call first rewrites its dirty slots in place (tierglu_nax6n.repack_slots): in
    _nax6_glu, in _ooo_nax6 before phase 2, and before phase 1 too with CFG nax_repack_p1 (off: BigMoE gathers every
    other load native on the CPU, so phase 1 never meets a dirty slot).
  - N3C (Tue 2026-11-03, off by default; the exact fallback for records converted to NAX6-N's native layout):
    CFG["native_3c"] runs REF2-T's 3c calls on native records with tierglu_3cn (3c with native addressing; == 3c on
    row-major records bitwise, TABLE_350 NATIVE-3C): tier_call's 3c branch and _tier_slots via _tier_glu_3cn, OOO's two
    phases via _ooo_3cn (BigMoE's _ooo_call, step for step). Raises if combined with a row-major reader (tier_nax4, or
    tier_nax_ooo without nax_native).
  - Per-layer layouts (Tue 2026-11-03 17:3x, BigMoE's router, inert until bigmoe.tier has _LAYOUT): a layer's records are
    0 row-major, 1 native or 2 native + p11 (11-bit scales, scripts/p11.py). _tier_glu_sel / _ooo_sel route by t["layout"]
    or bigmoe.tier._LAYOUT[0]: 0 -> BigMoE's 3c, 1 / 2 -> 3c-N (p11 found from the pools' uint32 scale pieces); without a
    router CFG native_3c decides as before. NAX6-N reads p11 pools the same way; _native_repack skips them.
Bits change once vs today's engine (qmm16c / experts_nax round differently); this is the approved reference change.
REF2b (Fri 2026-10-30, key buckets on) changes REF2's AR bits past 512 keys only; CFG["key_buckets"] = False gives
REF2's original AR (EvalSuite's results/ref2 rows), whose live loop is exact only below 512 keys.
The scheduling lever SB (R-11c: async_eval every 4 layers + the commit replay submitted alone) is bit-identical and
lives in the engine loop: see HANDOFF_EXACT16.md.
Usage (EvalSuite, greedy AR on the standard sets):
  os.environ["MTPLX_FUSED_GATE_UP"] = "1"   # before loading the model (experts_nax needs the fused weights)
  import ref2; ref2.apply(); assert all(ref2.self_test().values())
  new_ids = ref2.generate(model, prompt_ids, max_new, stop=eos_ids)   # each token = a 1-row verify + commit keep 1
  ref2.remove()
REF2's AR runs every token through the engine's verify scopes with 16 padded rows (attention included), so AR
decoding is slower than today's. The spec loop (16-row checks) runs the same kernels, so spec == AR, bit for bit.
"""
import os, sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leapk as LK
from qmm16 import qmm16c
from qmmg25 import experts_nax, qmmg
from gmr24 import segments
from mlx_lm.models import switch_layers as SL
from mtplx.models import qwen4_exp as Q

R = 16
CFG = {"dense_min_n": 1024, "experts": True, "experts_cfg_a": (4, 1), "experts_cfg_b": (1, 1), "slot_call": True,
       "commit_phase": True, "sdpa_classes": True, "sdpa_bounds": (128,), "key_buckets": True, "kb_from": 512,
       "kb_step": 512, "tiers": True, "gdn_exact": True, "tier_nax4": False, "tier_nax_ooo": False, "att_single": False,
       "gdn_inproj": False, "att_rope": False, "idx_prep": False, "hc_write": False, "hc_gate": False,
       "gdn_inproj_x": False, "gdn_body": False, "hc_b": False,
       "nax_hybrid": False, "nax_native": False, "nax_repack_p1": False, "native_3c": False,
       "fatt_verify": os.environ.get("REF2_FATT_VERIFY") == "1"}
_ON = [False]
COUNTS = dict(qmm16c=0, mlx16=0, linear=0, hc_read=0, hc_eager16=0, attention=0, experts=0, slot_calls=0, sdpa_split=0,
              kb_pad=0, tier_calls=0, gdnx=0, nax4=0, att_single=0, gdn_inproj=0, att_rope=0, idx_prep=0, hc_write=0, gdn_inproj_x=0, nax6=0, nax6_ooo=0, gdnb=0, hc_b=0, nax_hybrid=0, nax_native=0, nax_repack=0, native_3c=0,
              fatt_verify=0, verify_old=0, verify_old_eligible=0)
_ORIG = {}
_WCAT = {}

def _rows(x):
    lead = x.shape[:-1]
    return lead, int(np.prod(lead)) if lead else 1

def _pad16(x2):
    m = x2.shape[0]
    return x2 if m == R else mx.concatenate([x2, mx.zeros((R - m,) + x2.shape[1:], x2.dtype)], axis=0)

def _qmm(x, w, *args, **kw):
    orig = _ORIG["qmm"]
    if not _ON[0]:
        return orig(x, w, *args, **kw)
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    lead, rows = _rows(x)
    if not (transpose and mode == "affine" and biases is not None and rows <= R and x.dtype == mx.bfloat16):
        return orig(x, w, *args, **kw)
    K_, N = x.shape[-1], w.shape[0]
    x16 = _pad16(x.reshape(rows, K_))
    if bits == 4 and group_size == 64 and N >= CFG["dense_min_n"] and N % 64 == 0 and K_ % 64 == 0:
        y = qmm16c(x16, w, scales, biases)
        COUNTS["qmm16c"] += 1
    else:
        y = orig(x16, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode)
        COUNTS["mlx16"] += 1
    return y[:rows].reshape(*lead, N)

def _lin(self, x):
    lead, rows = _rows(x)
    if _ON[0] and rows < R and x.dtype == mx.bfloat16:
        y = _ORIG["lin"](self, _pad16(x.reshape(rows, x.shape[-1])))
        COUNTS["linear"] += 1
        return y[:rows].reshape(*lead, y.shape[-1])
    return _ORIG["lin"](self, x)

def _gr(self, hyper_input, pending=None):
    orig = _ORIG["gr"]
    if not _ON[0]:
        return orig(self, hyper_input, pending)
    lead, rows = _rows(hyper_input)
    if rows > R:
        return orig(self, hyper_input, pending)
    h16 = _pad16(hyper_input.reshape(rows, -1)).reshape(1, R, -1)
    p16 = None if pending is None else tuple(_pad16(p.reshape(rows, -1)).reshape(1, R, -1) for p in pending)
    if "block_inject_weight" in self:
        if id(self) not in _WCAT:
            _WCAT[id(self)] = mx.concatenate([self.input_mix_weight_down.weight, self.block_inject_weight.weight], axis=0)
        mixed, written, inject = LK.hc_read(self, _WCAT[id(self)], h16, p16, rows=R, fuse_gate=CFG["hc_gate"],
                                            kernel_b=CFG["hc_b"])
        COUNTS["hc_read"] += 1
        COUNTS["hc_b"] += bool(CFG["hc_b"])
        return (mixed[0, :rows].reshape(*lead, -1), written[0, :rows].reshape(hyper_input.shape),
                inject[0, :rows].reshape(*lead, -1))
    out = orig(self, h16, p16)
    COUNTS["hc_eager16"] += 1
    return out[0, :rows].reshape(*lead, -1)

def _att(self, x, cache):
    S = x.shape[1]
    if _ON[0] and S < R and isinstance(cache, Q.QSACache):
        xp = mx.concatenate([x, mx.repeat(x[:, -1:], R - S, axis=1)], axis=1)
        out = _ORIG["att"](self, xp, cache)
        cache.trim(R - S)
        COUNTS["attention"] += 1
        return out[:, :S]
    return _ORIG["att"](self, x, cache)

def _switch(self, x, indices):
    if _ON[0] and CFG["experts"] and hasattr(self, "gu_weight"):
        lead, rows = _rows(x)
        if rows <= R:
            k = indices.shape[-1]
            y = experts_nax(x.reshape(rows, x.shape[-1]), (self.gu_weight, self.gu_scales, self.gu_biases),
                            (self.down_proj.weight, self.down_proj.scales, self.down_proj.biases),
                            indices.reshape(rows, k).astype(mx.int32), CFG["experts_cfg_a"], CFG["experts_cfg_b"])
            COUNTS["experts"] += 1
            return y.reshape(*lead, k, -1)
    return _ORIG["switch"](self, x, indices)

def experts_split(x2d, gate, up, down, ids, cfg_a=None, cfg_b=None):
    """experts_nax with separate gate / up weights (BigMoE's pools or an unfused SwitchGLU): two qmmg calls whose
    outputs equal the fused call's halves bit for bit (qmmg's per-output order does not depend on N)"""
    cfg_a = cfg_a or CFG["experts_cfg_a"]; cfg_b = cfg_b or CFG["experts_cfg_b"]
    R_ = x2d.shape[0]
    order, sexp, prow, seg, P = segments(ids)
    g = qmmg(x2d, *gate, prow, sexp, seg, P, *cfg_a)
    u = qmmg(x2d, *up, prow, sexp, seg, P, *cfg_a)
    h = nn.silu(g) * u
    y = qmmg(h, *down, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b)
    inv = mx.argsort(order)
    return y[inv].reshape(R_, ids.shape[-1], -1)

def _q4(m):
    """(weight, scales, biases) of a 4-bit / group-64 quantized switch linear, else None"""
    if getattr(m, "bits", None) == 4 and getattr(m, "group_size", None) == 64 and "biases" in m:
        return (m.weight, m.scales, m.biases)
    return None

def _split_call(self, x, indices):
    lead, rows = _rows(x)
    k = indices.shape[-1]
    W = [_q4(self.gate_proj), _q4(self.up_proj), _q4(self.down_proj)] if hasattr(self, "gate_proj") else [None]
    if rows > R or any(w is None for w in W):
        return None
    y = experts_split(x.reshape(rows, x.shape[-1]), *W, indices.reshape(rows, k).astype(mx.int32))
    COUNTS["experts"] += 1
    return y.reshape(*lead, k, -1)

def _switch_base(self, x, indices, *a, **kw):
    if _ON[0] and CFG["experts"]:
        y = _split_call(self, x, indices)
        if y is not None:
            return y
    return _ORIG["swbase"](self, x, indices, *a, **kw)

def _switch_sorted(self, x, indices):
    if _ON[0] and CFG["experts"]:
        y = _split_call(self, x, indices)
        if y is not None:
            return y
    return _ORIG["swsorted"](self, x, indices)

_NAX4 = [None]
def _nax4mod():
    """K-62 (REF2c-T candidate, CFG tier_nax4, off by default): ~/Leap/scripts/tierglu_nax4.py, the tier GLU on the M5's
    matrix units (dequantized once per (slot, tier, sidecar) group; batch-invariant bitwise; NOT bit-identical to 3c)"""
    if _NAX4[0] is None:
        p = os.path.expanduser("~/Leap/scripts")
        if p not in sys.path:
            sys.path.insert(0, p)
        import tierglu_nax4 as N4
        _NAX4[0] = N4
    return _NAX4[0]

def _nax4_glu(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", scores=None, **kw):
    """tierglu.tier_glu's signature on NAX4 with its GPU pair grouping (no host sync, so the sync-free gate path keeps
    it): [M, R, D], or with scores the block's combine in 3c's fused order (tierglu.combine_phase, every pair phase 1)"""
    N4 = _nax4mod()
    M, R = x.shape[0], len(plan)
    y = N4.tier_glu_nax4(x, msb, lsb, msb_slot, lsb_slot, plan=plan, b2=kw.get("b2"), grp=N4.gpu_grp(msb_slot, lsb_slot, plan))
    COUNTS["nax4"] += 1
    if scores is None:
        return y
    import tierglu as TG
    return TG.combine_phase(y.reshape(M * R, -1), y.reshape(M * R, -1), mx.ones((M, R), mx.int32), scores.reshape(M, R))

_NAX6 = [None]
def _nax6mod():
    """C1 (CFG tier_nax_ooo, off by default): ~/Leap/scripts/tierglu_nax6.py (NAX4 with a fixed split-K)"""
    if _NAX6[0] is None:
        try:                                           # BigMoE's tierglu (MoonShots' planes_engine) first, so it is THE
            import bigmoe.tier as BT                   # "tierglu" module (Leap's third_party copy is identical, 11-02)
            BT._tier_glu_mod()
        except Exception:
            pass
        p = os.path.expanduser("~/Leap/scripts")
        if p not in sys.path:
            sys.path.insert(0, p)
        import tierglu_nax6 as N6
        _NAX6[0] = N6
    return _NAX6[0]

def _nax_glu_fn():
    """the NAX6 call C1 makes: NAX6, NAX6-H (H-1i, FAST) with CFG nax_hybrid, NAX6-N (N7-13, exact, native pools only)
    with CFG nax_native, NAX6-NH with both"""
    N6 = _nax6mod()
    if CFG["nax_native"]:
        COUNTS["nax_native"] += 1
        if CFG["nax_hybrid"]:
            import tierglu_nax6h as NH
            COUNTS["nax_hybrid"] += 1
            return NH.tier_glu_nax6nh
        import tierglu_nax6n as NN
        return NN.tier_glu_nax6n
    if CFG["nax_hybrid"]:
        import tierglu_nax6h as NH
        COUNTS["nax_hybrid"] += 1
        return NH.tier_glu_nax6h
    return N6.tier_glu_nax6

_NATIVE_FLAGS = {}
def native_flags(base, dirty_m, side=None, dirty_l=None):
    """N7-15 (BigMoE's ext, BIGMOE_NAX_NATIVE=1): register one tier layer's per-slot dirty flags, int32 MLX arrays the ext
    owns (it sets 1 after an in-service demand load wrote a slot row-major, before the slot's ready status; only the GPU
    repack clears them). Call once per tier layer after install_sidecars with t["base"] / t["sideP"]."""
    _NATIVE_FLAGS[id(base)] = (base, dirty_m, side, dirty_l)

def _native_repack(msb, lsb, msb_slot, lsb_slot):
    """N7-15: the dirty slots among a call's pairs -> native, in place, before a NAX6-N call reads them; returns the slot
    arrays the call must take (pools without registered flags, or nax_native off: unchanged)"""
    f = _NATIVE_FLAGS.get(id(msb)) if CFG["nax_native"] else None
    if f is not None and msb["gate_proj"]["s3"].dtype == mx.uint32:  # p11 pools come from native files: nothing to repack
        f = None
    if f is None:
        return msb_slot, lsb_slot
    import tierglu_nax6n as NN
    COUNTS["nax_repack"] += 1
    side = f[3] is not None and lsb_slot is not None
    sm2, sl2 = NN.repack_slots(msb, lsb if side else None, msb_slot, lsb_slot if side else None, f[1], f[3] if side else None)
    return sm2, (sl2 if side else lsb_slot)                 # no sidecar flags: the sidecar slots pass through as they are

def _nax6_glu(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", scores=None, **kw):
    """C1: tierglu.tier_glu's signature on NAX6 with its GPU pair grouping: [M, R, D], or with scores the block's combine
    in 3c's fused order (combine_phase, every pair phase 1)"""
    N6 = _nax6mod()
    M, R = x.shape[0], len(plan)
    msb_slot, lsb_slot = _native_repack(msb, lsb, msb_slot, lsb_slot)
    y = _nax_glu_fn()(x, msb, lsb, msb_slot, lsb_slot, plan=plan, b2=kw.get("b2"), grp=N6.gpu_grp(msb_slot, lsb_slot, plan))
    COUNTS["nax6"] += 1
    if scores is None:
        return y
    import bigmoe.tier as BT
    TG = BT._tier_glu_mod()
    return TG.combine_phase(y.reshape(M * R, -1), y.reshape(M * R, -1), mx.ones((M, R), mx.int32), scores.reshape(M, R))

def _ooo_nax6(x, ids, bm, t, post, sc, shared):
    """C1: bigmoe.tier._ooo_call (BigMoE's OOO at e3210bf) with NAX6 in both phases: the resident pairs during the loader
    spin, then the rest, then the same LY=8 combine_phase. == _nax6_glu with scores, bitwise (INV)."""
    import bigmoe.tier as BT
    N6 = _nax6mod()
    TG = BT._tier_glu_mod()
    from tierglu_nax import rowsum64
    BT.OOO_STATS["calls"] += 1
    COUNTS["nax6_ooo"] += 1
    PLAN = BT.PLAN
    K = ids.shape[-1]
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    lo4 = [s_ for s_ in t["segs"] if s_[2] == 4]
    R4 = lo4[-1][1] if lo4 else 0
    assert not lo4 or lo4[0][0] == 0, t["segs"]
    gate, L = bm["gate"], bm["L"]
    dep_m, n_m = post(L, ids)
    pre_m = dep_m[1:1 + n_m].reshape(M, K)
    ok = pre_m >= 0
    if R4:
        dep_l, n_l = post(t["side"], ids[..., :R4], after=dep_m)
        pre_l = dep_l[1:1 + n_l].reshape(M, R4)
        ok = ok & mx.concatenate([pre_l >= 0, mx.ones((M, K - R4), mx.bool_)], axis=-1)
    phase = mx.where(ok, 1, 2).astype(mx.int32)
    xs = rowsum64(x2)
    if CFG["nax_repack_p1"]:                       # N7-15 option (c): phase 1's slots too (negative = not resident: skipped)
        pre_m, pre_l = _native_repack(t["base"], t["sideP"], pre_m, pre_l if R4 else None)
    sl1 = mx.maximum(pre_l, 0) if R4 else None
    glu = _nax_glu_fn()
    y1 = glu(x2, t["base"], t["sideP"], pre_m, sl1, plan=PLAN, xs=xs,
                          grp=N6.gpu_grp_phase(pre_m, sl1 if R4 else pre_m, PLAN, phase, 1)).reshape(M * K, -1)
    after = [y1] + ([shared] if shared is not None else [])
    sm = gate.spin(mx.depends(dep_m, after), L, n_m).reshape(M, K)
    sl = gate.spin(mx.depends(dep_l, after), t["side"], n_l).reshape(M, R4) if R4 else None
    sm, sl = _native_repack(t["base"], t["sideP"], sm, sl)   # N7-15: the spin's demand loads -> native before phase 2
    y2 = glu(x2, t["base"], t["sideP"], sm, sl, plan=PLAN, xs=xs,
                          grp=N6.gpu_grp_phase(sm, sl if R4 else sm, PLAN, phase, 2)).reshape(M * K, -1)
    return TG.combine_phase(y1, y2, phase, sc.reshape(-1, K)).reshape(*x.shape[:-1], -1)

def _layout(t=None):
    """the layer's record layout from BigMoE's per-layer router: t["layout"] or bigmoe.tier._LAYOUT[0] (0 row-major, 1 native,
    2 native + p11); None without a router"""
    if t is not None and "layout" in t:
        return t["layout"]
    BT = sys.modules.get("bigmoe.tier")
    lay = getattr(BT, "_LAYOUT", None) if BT is not None else None
    return lay[0] if lay is not None else None

def _native(t=None):
    """does this layer's tier GLU run on native records? (the router's layout; without one, CFG native_3c)"""
    lay = _layout(t)
    if lay is None:
        return CFG["native_3c"]
    if lay == 0 and CFG["native_3c"]:
        raise RuntimeError("REF2 native_3c on a row-major layer (layout 0): the router decides per layer")
    return lay in (1, 2)

def _native_guard(t=None):
    """N3C: records in the native layout must not reach a row-major reader"""
    if _native(t) and (CFG["tier_nax4"] or (CFG["tier_nax_ooo"] and not CFG["nax_native"])):
        raise RuntimeError("REF2 native_3c: native records need 3c-N, or NAX6-N (tier_nax_ooo + nax_native); not tier_nax4")

def _tier_glu_3cn(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", scores=None, **kw):
    """N3C (CFG native_3c): tierglu.tier_glu's signature on native records (tierglu_3cn.tier_glu_n, == 3c bitwise)"""
    import tierglu_3cn as C3
    COUNTS["native_3c"] += 1
    return C3.tier_glu_n(x, msb, lsb, msb_slot, lsb_slot, plan=plan, scores=scores, **kw)

def _ooo_3cn(x, ids, bm, t, post, sc, shared):
    """N3C: bigmoe.tier._ooo_call (BigMoE's 3c OOO) step for step, its phase kernels from tierglu_3cn (native records)"""
    import bigmoe.tier as BT
    TG = BT._tier_glu_mod()
    import tierglu_3cn as C3
    BT.OOO_STATS["calls"] += 1
    COUNTS["native_3c"] += 1
    PLAN = BT.PLAN
    K = ids.shape[-1]
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    lo4 = [s_ for s_ in t["segs"] if s_[2] == 4]
    R4 = lo4[-1][1] if lo4 else 0
    assert not lo4 or lo4[0][0] == 0, t["segs"]
    gate, L = bm["gate"], bm["L"]
    dep_m, n_m = post(L, ids)
    pre_m = dep_m[1:1 + n_m].reshape(M, K)
    ok = pre_m >= 0
    if R4:
        dep_l, n_l = post(t["side"], ids[..., :R4], after=dep_m)
        pre_l = dep_l[1:1 + n_l].reshape(M, R4)
        ok = ok & mx.concatenate([pre_l >= 0, mx.ones((M, K - R4), mx.bool_)], axis=-1)
    phase = mx.where(ok, 1, 2).astype(mx.int32)
    y1 = C3.tier_glu_phase_n(x2, t["base"], t["sideP"], pre_m, mx.maximum(pre_l, 0) if R4 else None, phase, 1, plan=PLAN)
    after = [y1] + ([shared] if shared is not None else [])
    sm = gate.spin(mx.depends(dep_m, after), L, n_m).reshape(M, K)
    sl = gate.spin(mx.depends(dep_l, after), t["side"], n_l).reshape(M, R4) if R4 else None
    y2 = C3.tier_glu_phase_n(x2, t["base"], t["sideP"], sm, sl, phase, 2, plan=PLAN)
    return TG.combine_phase(y1, y2, phase, sc.reshape(-1, K)).reshape(*x.shape[:-1], -1)

def _ooo_sel(x, ids, bm, t, post, sc, shared):
    """bigmoe.tier._ooo_call with REF2's C1 switch (off: BigMoE's 3c phases). A converted layer (layout 1 / 2) takes 3c-N,
    the native twin of BigMoE's 3c, even with REF2 removed (BigMoE 17:4x: r57_screen toggles REF2 around drafts)"""
    if _ON[0]:
        _native_guard(t)
        if CFG["tier_nax_ooo"]:
            return _ooo_nax6(x, ids, bm, t, post, sc, shared)
        if _native(t):
            return _ooo_3cn(x, ids, bm, t, post, sc, shared)
    elif _layout(t) in (1, 2):
        return _ooo_3cn(x, ids, bm, t, post, sc, shared)
    return _ORIG["ooo_call"](x, ids, bm, t, post, sc, shared)

def _tier_glu_sel():
    """bigmoe.tier._tier_glu with REF2's switch: every tier GLU call (tier_call's 1-4-row and gated verify rows,
    _tier_slots' 5-16-row calls) takes the same kernel, so a pair's bits do not depend on the call size"""
    if _ON[0]:
        _native_guard()
    if _ON[0] and CFG["tier_nax_ooo"]:
        return _nax6_glu
    if _ON[0] and CFG["tier_nax4"]:
        import bigmoe.tier as BT
        if getattr(BT, "_OOO", False):
            raise RuntimeError("REF2 tier_nax4 needs BIGMOE_OOO off (its phase kernels are 3c's)")
        return _nax4_glu
    if (_native() if _ON[0] else _layout() in (1, 2)):  # a converted layer takes 3c-N even with REF2 removed
        _ORIG["tier_glu"]()                            # BigMoE's own hook first (its path; LP1 patches tierglu's sources)
        return _tier_glu_3cn
    return _ORIG["tier_glu"]()

_GDNX = [None]
def _gdnx():
    """R-61: MoonShots' exact fused GDN pieces (a frozen copy in ~/Leap/third_party/ms_gdn): bitwise ==
    MTPLX's eager chain (theirs: S 1-8, model-level 4-row logits; Leap 2026-11-01: S 1-16, tensor level). Installed once:
    MTPLX's fused conv-norm entry points then run the exact kernel."""
    if _GDNX[0] is None:
        p = os.path.expanduser("~/Leap/third_party/ms_gdn")   # frozen copy of MoonShots' files (README there)
        if p not in sys.path:
            sys.path.insert(0, p)
        import gdn_exact_patch as GP
        GP.install(norm=False)
        _GDNX[0] = GP
    return _GDNX[0]

def _gdnx_ok(self, B, mask, cache):
    return (B == 1 and mask is None and cache is not None and not self.training and getattr(cache, "lengths", None) is None
            and self.conv_dim == 10240 and self.key_dim == 2048 and self.conv_kernel_size == 4 and self.head_k_dim == 128
            and getattr(self.conv1d, "bias", None) is None)

def _gdn_rows_applies(self, B, S, mask, cache):
    """R-61: with CFG gdn_exact, 2-16-row GDN calls take MTPLX's fused rows branch (its own kernel: 2-6 rows, not exact),
    which then runs MoonShots' exact prework: conv + silu + l2norm + scale + the conv state in one dispatch, bitwise"""
    if _ON[0] and CFG["gdn_exact"] and 2 <= S <= R and _gdnx_ok(self, B, mask, cache):
        _gdnx(); COUNTS["gdnx"] += 1
        return True
    return _ORIG["gdn_rows"](self, B, S, mask, cache)

def _gdn_row_applies(self, B, S, mask, cache):
    """R-61: the same for the AR's 1-row calls (MTPLX's S = 1 conv-norm branch -> the exact kernel)"""
    if _ON[0] and CFG["gdn_exact"] and S == 1 and _gdnx_ok(self, B, mask, cache):
        _gdnx(); COUNTS["gdnx"] += 1
        return True
    return _ORIG["gdn_row"](self, B, S, mask, cache)

def _gated_norm(self, hidden_states, gate=None):
    """R-61: with CFG gdn_exact, SigmoidRMSNormGated on <= 16 rows x 48 heads as MoonShots' exact one-dispatch kernel"""
    if (_ON[0] and CFG["gdn_exact"] and gate is not None and hidden_states.shape[-1] == 128
            and hidden_states.size // 128 <= R * 48 and hidden_states.shape == gate.shape
            and hidden_states.dtype == mx.bfloat16 and gate.dtype == mx.bfloat16 and self.weight.dtype == mx.bfloat16):
        return _gdnx().gated_norm_exact(hidden_states, gate, self.weight, self.eps)
    return _ORIG["gnorm"](self, hidden_states, gate)

_GDN_IN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")

class _InProjPad:
    """R-68 (REF2c, CFG gdn_inproj): one GDN layer's four input projections on the row-concatenated 4-bit packs, zero
    rows appended up to N % 64 == 0 (a zero pack row gives exactly 0 and is sliced off). A plain object, not a Module:
    MTPLX's GDN forward picks it up as in_proj_fused and calls it -> (qkv, z, b, a)."""
    def __init__(self, gdn):
        old = getattr(gdn, "in_proj_fused", None)
        if old is not None:                            # MTPLX_FUSED_GDN_INPROJ=1 already concatenated them (unpadded)
            arrs, splits = (old.weight, old.scales, old.biases), list(old._splits)
            self.gs, self.bits, self.mode = old.group_size, old.bits, old.mode
            n = old.weight.shape[0]
        else:
            ps = [getattr(gdn, k) for k in _GDN_IN]
            assert all(isinstance(p, nn.QuantizedLinear) and "bias" not in p for p in ps), [type(p) for p in ps]
            assert len({(p.group_size, p.bits, getattr(p, "mode", "affine")) for p in ps}) == 1
            self.gs, self.bits, self.mode = ps[0].group_size, ps[0].bits, getattr(ps[0], "mode", "affine")
            rows = [p.weight.shape[0] for p in ps]
            splits = list(np.cumsum(rows)[:3])
            arrs = tuple(mx.concatenate([p[key] for p in ps]) for key in ("weight", "scales", "biases"))
            n = int(sum(rows))
        pad = (-n) % 64
        if pad:
            arrs = tuple(mx.concatenate([a, mx.zeros((pad,) + a.shape[1:], a.dtype)]) for a in arrs)
        self.w, self.s, self.b = arrs
        self.o = [int(v) for v in splits] + [n]
        mx.eval(self.w, self.s, self.b)

    def _q(self, x, w, s, b):
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=self.gs, bits=self.bits, mode=self.mode)

    def __call__(self, x):
        o = self.o
        if _ON[0] and CFG["gdn_inproj"]:
            COUNTS["gdn_inproj"] += 1
            y = self._q(x, self.w, self.s, self.b)
            return y[..., :o[0]], y[..., o[0]:o[1]], y[..., o[1]:o[2]], y[..., o[2]:o[3]]
        return tuple(self._q(x, self.w[a:e], self.s[a:e], self.b[a:e]) for a, e in zip([0] + o[:3], o))

class _InProjExact:
    """R-83 (exact-safe, CFG gdn_inproj_x): qkv | z row-concatenated (one qmm16c call) and b | a row-concatenated (one
    MLX 16-row call); a plain object picked up by MTPLX's GDN forward as in_proj_fused -> (qkv, z, b, a). Built once per
    layer (the four modules are dropped, so no extra memory); with the switch off it runs the four calls on row slices."""
    def __init__(self, gdn):
        ps = [getattr(gdn, k) for k in _GDN_IN]
        assert all(isinstance(p, nn.QuantizedLinear) and "bias" not in p for p in ps), [type(p) for p in ps]
        self.gs, self.bits, self.mode = ps[0].group_size, ps[0].bits, getattr(ps[0], "mode", "affine")
        self.n = [p.weight.shape[0] for p in ps]
        assert (self.n[0] + self.n[1]) % 64 == 0 and self.n[0] + self.n[1] >= CFG["dense_min_n"], self.n
        assert self.n[2] + self.n[3] < CFG["dense_min_n"], self.n
        cat = lambda a, b: tuple(mx.concatenate([a[k], b[k]]) for k in ("weight", "scales", "biases"))
        self.qz, self.ba = cat(ps[0], ps[1]), cat(ps[2], ps[3])
        mx.eval(self.qz, self.ba)

    def __call__(self, x):
        q = lambda w: mx.quantized_matmul(x, *w, transpose=True, group_size=self.gs, bits=self.bits, mode=self.mode)
        n0, n2 = self.n[0], self.n[2]
        if _ON[0] and CFG["gdn_inproj_x"] and _rows(x)[1] <= R:          # prefill widths keep the four calls (stock bits)
            COUNTS["gdn_inproj_x"] += 1
            yqz, yba = q(self.qz), q(self.ba)
            return yqz[..., :n0], yqz[..., n0:], yba[..., :n2], yba[..., n2:]
        sl = lambda w, a, e: tuple(t[a:e] for t in w)                    # off: the four stock calls on row slices
        return (q(sl(self.qz, 0, n0)), q(sl(self.qz, n0, n0 + self.n[1])), q(sl(self.ba, 0, n2)),
                q(sl(self.ba, n2, n2 + self.n[3])))

_GB = [None]
def _gbmod():
    if _GB[0] is None:
        d = os.path.dirname(os.path.abspath(__file__))
        if d not in sys.path:
            sys.path.insert(0, d)
        import gdnbody
        _GB[0] = gdnbody
    return _GB[0]

def _gdn_body_call(self, inputs, mask, cache):
    """GDNB (Mon 2026-11-02, CFG gdn_body, off by default): a 1-16-row GDN call's body (gdn_exact's prework + the
    recurrence + the exact gated norm) as ONE dispatch (scripts/gdnbody.py: bitwise == REF2-T at piece and layer level,
    16-row == 16 x 1-row; gdnbody_test.py). in_proj / out_proj, g / beta and the capture-commit stash are unchanged.
    None = not applicable (MTPLX's own forward runs)."""
    B, S = inputs.shape[0], inputs.shape[1]
    if not (CFG["gdn_exact"] and B == 1 and mask is None and cache is not None and 1 <= S <= R
            and inputs.dtype == mx.bfloat16 and cache[0] is not None and cache[1] is not None
            and cache[1].dtype == mx.float32 and _gdnx_ok(self, B, mask, cache)
            and isinstance(self.norm, Q.SigmoidRMSNormGated) and self.norm.weight.dtype == mx.bfloat16
            and not self._fused_out_applies(B, S) and not self._fused_step_applies(B, S, mask, cache)
            and not Q._boundary_capture_offsets(S, B, cache, mask)):
        return None
    fused_in = getattr(self, "in_proj_fused", None)
    if fused_in is not None:
        qkv, z, b, a = fused_in(inputs)
    else:
        qkv, z = self.in_proj_qkv(inputs), self.in_proj_z(inputs)
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
    z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
    y, q, k, v, cs, st = _gbmod().gdn_body(qkv, z, a, b, cache[0], self.conv1d.weight, self.A_log, self.dt_bias,
                                          cache[1], self.norm.weight, self.norm.eps, _gdnx()._scale())
    if Q._VERIFY_CAPTURE.get():                        # MTPLX's capture-commit stash: the same tensors, the same bits
        cache._mtplx_verify_rows = (qkv, q, k, v, a, b)
    cache[0] = cs
    cache[1] = st
    cache.advance(S)
    COUNTS["gdnb"] += 1
    return Q._linear(self.out_proj, y.reshape(B, S, -1))

def _gdn_call(self, inputs, mask=None, cache=None):
    """R-68: with CFG gdn_inproj, a GDN layer's first call builds its padded fused projection (once); GDNB: with
    CFG gdn_body, 1-16-row calls run the body as one dispatch (_gdn_body_call)"""
    if _ON[0] and CFG["gdn_inproj"] and not isinstance(self.__dict__.get("in_proj_fused"), _InProjPad):
        f = _InProjPad(self)
        self.pop("in_proj_fused", None)
        for k in _GDN_IN:
            self.pop(k, None)
        object.__setattr__(self, "in_proj_fused", f)
    elif (_ON[0] and CFG["gdn_inproj_x"] and not CFG["gdn_inproj"] and "in_proj_fused" not in self.__dict__
          and "in_proj_fused" not in self and all(k in self for k in _GDN_IN)):
        f = _InProjExact(self)                                          # R-83: built once, the four modules dropped
        for k in _GDN_IN:
            self.pop(k, None)
        object.__setattr__(self, "in_proj_fused", f)
    if _ON[0] and CFG["gdn_body"]:
        out = _gdn_body_call(self, inputs, mask, cache)
        if out is not None:
            return out
    return _ORIG["gdn_call"](self, inputs, mask, cache)

_BS = [None]
def _tier_slots(self, x, indices, bm):
    """R-58 REF2-T: a 5-16-row call on BigMoE's rank-tier pools (BIGMOE_TIER, e.g. 4433333333 with TIER_GLU 3c) runs its
    routed experts at the plan's tiers on pool slots, as the gated path's tier_call does for 1-4 rows: base-pool (MSB-3)
    slots for every rank, LSB-sidecar slots for the 4-bit ranks, MoonShots' fused 3c kernel. That kernel computes each
    (row, rank) pair in its own threadgroups, so a row's bits do not depend on the call's other rows. BigMoE's own
    _sync_call would run these calls as the exact 4-bit prefill instead (tier.prefill_call)."""
    from bigmoe import tier as T
    BS, t = _BS[0], bm["tier"]
    ids = np.array(indices)                                           # [..., K] in gate-rank order
    K = ids.shape[-1]
    uniq = np.unique(ids)
    assert len(uniq) <= bm["C"], ("REF2-T: more experts than base slots", len(uniq), bm["C"])
    if BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS.SYNC["calls"] += 1; BS.SYNC["rows"] += ids.size // K
    slots = np.array(BS.ext.ensure(bm["L"], mx.array(uniq.astype(np.int32))))
    sm = mx.array(slots[np.searchsorted(uniq, ids)].astype(np.int32).reshape(-1, K))
    lo4 = [s for s in t["segs"] if s[2] == 4]
    sl = None
    if lo4:
        ids4 = ids[..., lo4[0][0]:lo4[-1][1]]
        u4 = np.unique(ids4)
        assert len(u4) <= t["C_l"], ("REF2-T: more 4-bit experts than sidecar slots", len(u4), t["C_l"])
        s4 = np.array(BS.ext.ensure(t["side"], mx.array(u4.astype(np.int32))))
        sl = mx.array(s4[np.searchsorted(u4, ids4)].astype(np.int32).reshape(-1, ids4.shape[-1]))
    if hasattr(T, "_LAYOUT"):                                         # BigMoE's per-layer router (layout 0 / 1 / 2)
        T._LAYOUT[0] = t.get("layout", 0)
    y = T._tier_glu()(x.reshape(-1, x.shape[-1]), t["base"], t["sideP"], sm, sl, plan=T.PLAN)
    y = y.reshape(*x.shape[:-1], K, -1)
    mx.eval(y)                                                        # before a later ensure of L can evict slots
    COUNTS["tier_calls"] += 1
    return y

def _sync_slots(cls, self, x, indices, bm, *a, **kw):
    """R-50 SLOTCALL: BigMoE's plain _sync_call gathers each layer's experts into a fresh scratch pool to keep MLX's
    sorted gather on the expert ORDER; REF2's expert kernel computes every (row, expert) pair on its own, so a
    <= 16-row call runs on the pool itself with slot ids, as the gated path does"""
    BS = _BS[0]
    lead, rows = _rows(x)
    if _ON[0] and CFG["tiers"] and "tier" in bm and rows <= R and not bm.get("mtp"):
        return _tier_slots(self, x, indices, bm)
    if (not _ON[0] or not CFG["experts"] or not CFG["slot_call"] or rows > R or "tier" in bm or bm.get("mtp")):
        return _ORIG["sync"](cls, self, x, indices, bm, *a, **kw)    # > 16 rows: stock MLX needs the expert order
    ids = np.array(indices)                                           # the shipped path's id sync
    uniq = np.unique(ids)
    if len(uniq) > int(bm["C"] * 0.9):                                # the shipped path would go in groups
        return _ORIG["sync"](cls, self, x, indices, bm, *a, **kw)
    if BS._STAGE["bufs"] is not None:                                 # PF1's prefill staging: drop it, as the
        BS._stage_drop()                                              # engine's first gated call does
    BS.SYNC["calls"] += 1; BS.SYNC["rows"] += ids.size // ids.shape[-1]
    slots = np.array(BS.ext.ensure(bm["L"], mx.array(uniq.astype(np.int32))))
    sl = mx.array(slots[np.searchsorted(uniq, ids)].astype(np.int32)).astype(indices.dtype)
    y = cls.__call__(self, x, sl, *a, **kw)                           # -> _switch_base -> experts_split
    mx.eval(y)                                                        # before a later ensure of L can evict slots
    COUNTS["slot_calls"] += 1
    return y

def _klass(N):
    """the class of a 16-row attention call with N keys (R-40 STEP 3m / 3l / 4a / 4b, measured on real data): rows are
    bit-stable within a class and change across classes. 0 / 1 / ...: the sdpa_bounds bands (N-invariant inside);
    with key buckets, from kb_from keys up: the length the bucket's calls run at (the next multiple of kb_step above N)"""
    if CFG["key_buckets"] and N >= CFG["kb_from"]:
        return (N // CFG["kb_step"] + 1) * CFG["kb_step"]
    return sum(1 for b in CFG["sdpa_bounds"] if N >= b)

def _sdpa_one(q, k, v, *, scale, mask):
    """R-62 (REF2c candidate, CFG att_single, off by default): the 16-row call as ONE fused SDPA (MLX's full NAX kernel
    takes GQA natively at q_len > 8) instead of MTPLX's 12 head-chunked calls: 2.1-2.8x faster; bitwise == the chunked
    calls below 512 keys, different bits past it (a new reference: the AR pads to 16 rows, so it takes this path too)"""
    COUNTS["att_single"] += 1
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

def _fatt16s():
    """F1: fs4 (scripts/fatt16s.py, which imports fatt16.py), loaded on the first switched call only"""
    from fatt16s import fatt16s
    return fatt16s

def _sdpa_at(q, k, v, mask, scale, n):
    """MTPLX's own verify SDPA with k / v zero-padded to n keys and the mask extended (False / -inf)"""
    orig = _sdpa_one if CFG["att_single"] else _ORIG["vsdpa"]
    N = k.shape[2]
    if n == N:
        return orig(q, k, v, scale=scale, mask=mask)
    pad = n - N
    kp = mx.concatenate([k, mx.zeros(k.shape[:2] + (pad, k.shape[3]), k.dtype)], axis=2)
    vp = mx.concatenate([v, mx.zeros(v.shape[:2] + (pad, v.shape[3]), v.dtype)], axis=2)
    fill = mx.zeros(mask.shape[:-1] + (pad,), mx.bool_) if mask.dtype == mx.bool_ else \
        mx.full(mask.shape[:-1] + (pad,), -float("inf"), mask.dtype)
    return orig(q, kp, vp, scale=scale, mask=mx.concatenate([mask, fill], axis=-1))

def _verify_sdpa(q, k, v, *, scale, mask):
    """R-40 STEP 3m / 4c: a 16-row call runs in its class: at its own N in the bands, at the bucket's length past
    kb_from (zero-padded, masked). REF2's AR pads attention to 16 rows, so the AR step at position p calls it with
    p + 16 keys; in a 16-row check with N keys row j sits at N - 16 + j, so its AR step had N + j keys. Rows whose AR
    class differs from the check's are recomputed in one more call in their class and selected in exactly.
    F1 (CFG fatt_verify, off by default): a bool-mask 16- / 8-row call runs fs4 instead (batch- and N-invariant)."""
    elig = (mask is not None and mask.dtype == mx.bool_ and q.shape[2] in (16, 8) and q.shape[0] == 1
            and q.shape[3] % 32 == 0 and q.shape[1] % k.shape[1] == 0 and (mask.ndim == 2 or mask.shape[-3] == 1))
    if _ON[0] and CFG["fatt_verify"] and elig:
        COUNTS["fatt_verify"] += 1
        return _fatt16s()(q, k, v, scale, mask, ks=4)
    COUNTS["verify_old"] += 1
    COUNTS["verify_old_eligible"] += int(elig)
    orig = _ORIG["vsdpa"]
    S, N = q.shape[2], k.shape[2]
    if not _ON[0] or not CFG["sdpa_classes"] or S != R or mask is None:
        return (_sdpa_one if _ON[0] and CFG["att_single"] and S == R else orig)(q, k, v, scale=scale, mask=mask)
    run = lambda c: c if c >= CFG["kb_from"] else N if c == 0 else max(N, CFG["sdpa_bounds"][c - 1])
    c0 = _klass(N)
    cls = [_klass(N + j) for j in range(S)]
    out = _sdpa_at(q, k, v, mask, scale, run(c0))
    if c0 >= CFG["kb_from"]:
        COUNTS["kb_pad"] += 1
    for c in sorted(set(cls) - {c0}):
        sel = mx.array([cls[j] == c for j in range(S)])[None, None, :, None]
        out = mx.where(sel, _sdpa_at(q, k, v, mask, scale, run(c)), out)
        COUNTS["sdpa_split"] += 1
    return out

def _glue_rope(self, rows):
    """R-72 (exact-safe, CFG att_rope): MTPLX's QSA attention takes its one-dispatch RoPE (rope_qk) on REF2's <= 16-row
    calls; bitwise == the stock chain (R-72: layer 3's real weights, prefill 383-4100 keys)"""
    if _ON[0] and CFG["att_rope"] and 1 <= rows <= R:
        COUNTS["att_rope"] += 1
        return True
    return _ORIG["glue_rope"](self, rows)

def _idx_prep_ok(self, values, norm_weight, *, expected_ndim):
    """R-72 (exact-safe, CFG idx_prep): the indexer's RMSNorm + partial RoPE through MTPLX's shipped one-dispatch kernel
    (qsa_indexer_prepare_queries / _keys, pinned bit-exact vs the eager chain at head_dim 128); the selector stays stock"""
    if _ON[0] and CFG["idx_prep"] and not self._uses_vision_positions():
        from mtplx.kernels.qsa_indexer_prepare import qsa_indexer_prepare_supported
        ok = bool(qsa_indexer_prepare_supported(values, norm_weight, self._inv_freq, expected_ndim=expected_ndim))
        COUNTS["idx_prep"] += int(ok)
        return ok
    return _ORIG["idx_prep"](self, values, norm_weight, expected_ndim=expected_ndim)

@mx.compile
def _hc_write_c(hyper, block_out, inject):
    grouped = hyper.reshape(*hyper.shape[:-1], inject.shape[-1], block_out.shape[-1])
    return (grouped + block_out[..., None, :] * inject[..., :, None]).reshape(*hyper.shape)

def _hc_write(hyper, block_out, inject):
    """R-73 (exact-safe, CFG hc_write): hyper + block_out x inject as ONE compiled kernel on <= 16-row bf16 calls
    (bitwise == the stock Multiply + Add: R-73, layer 3's real weights, 6 scales)"""
    lead, rows = _rows(hyper)
    if _ON[0] and CFG["hc_write"] and rows <= R and hyper.dtype == mx.bfloat16:
        COUNTS["hc_write"] += 1
        return _hc_write_c(hyper, block_out, inject)
    return _ORIG["hc_write"](hyper, block_out, inject)

def _commit(self, cache, snapshot_states, *, keep_tokens, verified_tokens):
    """R-40 STEP 3t: the commit's lone PLE-layer replay must run in the verify forward's phase. MTPLX's attention_gate
    (the PLE gate) and GatedResidual's inject lower sigmoid differently in decode_verify (the compiled fused form) than
    elsewhere (the stock one); the two differ in the last bit at some bf16 inputs, so a replay outside the verify
    scope left layer 1's state off the AR's at rare positions (prealgebra/1640, relative 39-40). REF2's AR and its
    checks run in decode_verify; a 1-row AR commit is a full accept and replays nothing."""
    if not _ON[0] or not CFG["commit_phase"]:
        return _ORIG["commit"](self, cache, snapshot_states, keep_tokens=keep_tokens, verified_tokens=verified_tokens)
    from mtplx.attention_context import attention_phase, model_forward_kind
    with attention_phase("decode_verify"), model_forward_kind("target_verify"):
        return _ORIG["commit"](self, cache, snapshot_states, keep_tokens=keep_tokens, verified_tokens=verified_tokens)

def apply():
    """install the patches (idempotent) and switch them on"""
    if not _ORIG:
        _ORIG.update(qmm=mx.quantized_matmul, lin=nn.Linear.__call__, gr=Q.GatedResidual.__call__,
                     att=Q.Attention.__call__, switch=Q._FusedGateUpSwitchGLU.__call__, swbase=SL.SwitchGLU.__call__)
        mx.quantized_matmul = _qmm
        nn.Linear.__call__ = _lin
        Q.GatedResidual.__call__ = _gr
        Q.Attention.__call__ = _att
        Q._FusedGateUpSwitchGLU.__call__ = _switch
        SL.SwitchGLU.__call__ = _switch_base           # BigMoE's streamed subclass calls this with slot ids
        _ORIG["commit"] = Q.Qwen4ExpTextModel.commit_verified_window   # R-40 3t: replays in the verify phase
        Q.Qwen4ExpTextModel.commit_verified_window = _commit
        _ORIG["vsdpa"] = Q._verify_sdpa                # R-40 3m: each check row in its AR step's SDPA class
        Q._verify_sdpa = _verify_sdpa
        _ORIG["gdn_rows"] = Q.GatedDeltaNet._fused_conv_norm_rows_applies   # R-61: CFG gdn_exact (off by default)
        Q.GatedDeltaNet._fused_conv_norm_rows_applies = _gdn_rows_applies
        _ORIG["gdn_row"] = Q.GatedDeltaNet._fused_conv_norm_applies
        Q.GatedDeltaNet._fused_conv_norm_applies = _gdn_row_applies
        _ORIG["gnorm"] = Q.SigmoidRMSNormGated.__call__
        Q.SigmoidRMSNormGated.__call__ = _gated_norm
        _ORIG["gdn_call"] = Q.GatedDeltaNet.__call__   # R-68: CFG gdn_inproj (off by default)
        Q.GatedDeltaNet.__call__ = _gdn_call
        _ORIG["glue_rope"] = Q.Attention._verify_glue_rope         # R-72: CFG att_rope (exact-safe, off by default)
        Q.Attention._verify_glue_rope = _glue_rope
        _ORIG["idx_prep"] = Q.QSAIndexer._prepare_kernel_supported  # R-72: CFG idx_prep (exact-safe, off by default)
        Q.QSAIndexer._prepare_kernel_supported = _idx_prep_ok
        _ORIG["hc_write"] = Q._hyper_residual_write                # R-73: CFG hc_write (exact-safe, off by default)
        Q._hyper_residual_write = _hc_write
        try:                                           # BIGMOE_SORT_VERIFY=1's verify-width path, if BigMoE is loaded
            import bigmoe.stream as BS
            _ORIG["swsorted"] = BS._switch_sorted
            BS._switch_sorted = _switch_sorted
            _ORIG["sync"] = BS._sync_call              # R-50: <= 16-row calls on pool slots
            BS._sync_call = _sync_slots
            _BS[0] = BS
        except Exception:
            pass
        try:                                           # K-62: CFG tier_nax4 picks the tier GLU kernel (off: 3c)
            import bigmoe.tier as BT
            _ORIG["tier_glu"] = BT._tier_glu
            BT._tier_glu = _tier_glu_sel
            _ORIG["ooo_call"] = BT._ooo_call               # C1: CFG tier_nax_ooo runs OOO's phases on NAX6 (off: 3c)
            BT._ooo_call = _ooo_sel
            BT.LAYOUT_ROUTER = True                        # BigMoE's guards: REF2 routes layouts 1 / 2 (per layer)
        except Exception:
            pass
    _ON[0] = True

def counts():
    """calls of each REF2 path since the last reset (counted when the graph is built)"""
    return dict(COUNTS)

def reset_counts():
    for k in COUNTS:
        COUNTS[k] = 0

def remove():
    _ON[0] = False

def enabled():
    return _ON[0]

# ---------------------------------------------------------------- REF2's AR decoding (the reference itself)
import contextlib
@contextlib.contextmanager
def verify_scope(model):
    """the engine's verify scopes (as ca_anat.verify_scope): REF2's AR runs every token through this path"""
    from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield

def ar_step(model, cache, token):
    """REF2's reference AR step (INV1 / R-40 Part 1): one token as a 1-row verify + commit keep 1; returns [V] logits"""
    from mtplx import cache_state as CS
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope(model):
        logits = model(mx.array([[int(token)]]), cache=cache)
    model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=1)
    return logits[0, -1]

def prefill(model, ids, chunk=1024):
    """the plain prefill (as ca_anat.prefill); its rows > 16 run today's kernels in AR and in spec alike"""
    cache = model.make_cache()
    lg = None
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, [c.state for c in cache if hasattr(c, "state")])
    return cache, lg[0, -1]

def generate(model, ids, max_new, stop=()):
    """greedy AR decoding under REF2: prefill, then ar_step per token; returns the new token ids"""
    assert _ON[0], "call ref2.apply() first"
    cache, last = prefill(model, list(ids))
    out = []
    tok = int(mx.argmax(last).item())
    for _ in range(max_new):
        out.append(tok)
        if tok in stop:
            break
        tok = int(mx.argmax(ar_step(model, cache, tok)).item())
    return out

def self_test(seed=0):
    """batch invariance of each kernel on random data: a row's bits alone (padded) == inside a 16-row call"""
    out = {}
    was = _ON[0]
    _ON[0] = True
    try:
        mx.random.seed(seed)
        eq = lambda a, b: bool(mx.all(mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())
        for name, N, K in (("qmm16c 6144x2560", 6144, 2560), ("MLX 16-row 512x2560", 512, 2560)):
            W = mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
            X = (mx.random.normal((R, K)) * 0.5).astype(mx.bfloat16)
            full = _qmm(X, *W, transpose=True, group_size=64, bits=4)
            out[name] = all(eq(_qmm(X[j:j + m], *W, transpose=True, group_size=64, bits=4), full[j:j + m])
                            for j, m in ((0, 1), (5, 1), (3, 4), (0, 12)))
        E, D_, I_ = 8, 2560, 640
        g4 = mx.quantize((mx.random.normal((E, I_, D_)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
        u4 = mx.quantize((mx.random.normal((E, I_, D_)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
        d4 = mx.quantize((mx.random.normal((E, D_, I_)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
        gu = tuple(mx.concatenate([a, b], axis=1) for a, b in zip(g4, u4))
        X = (mx.random.normal((R, D_)) * 0.5).astype(mx.bfloat16)
        ids = mx.random.randint(0, E, (R, 4)).astype(mx.int32)
        ys = experts_split(X, g4, u4, d4, ids)
        out["experts split == fused"] = eq(ys, experts_nax(X, gu, d4, ids, CFG["experts_cfg_a"], CFG["experts_cfg_b"]))
        out["experts rows alone"] = all(eq(experts_split(X[j:j + m], g4, u4, d4, ids[j:j + m]), ys[j:j + m])
                                        for j, m in ((0, 1), (9, 1), (4, 3)))
        lin = nn.Linear(2560, 2560, bias=False)
        lin.weight = (lin.weight * 1).astype(mx.bfloat16)
        X = (mx.random.normal((R, 2560)) * 0.5).astype(mx.bfloat16)
        full = _lin(lin, X)
        out["bf16 Linear"] = all(eq(_lin(lin, X[j:j + m]), full[j:j + m]) for j, m in ((0, 1), (7, 1), (2, 5)))
    finally:
        _ON[0] = was
    return out
