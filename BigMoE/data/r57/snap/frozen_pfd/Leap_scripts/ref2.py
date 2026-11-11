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
    C 224, -3 ms per cycle). CFG["gdn_exact"]; frozen copy in /Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/ms_gdn. No bits change.
  - R-62 (REF2c candidate, off by default): CFG["att_single"] runs each 16-row attention call as one fused SDPA instead
    of MTPLX's 12 head chunks (the key buckets stay; past 512 keys the bits differ from the chunked calls).
  - F1 (FAST-K, off by default; Leap Tue 2026-11-03, TABLE_350 'F1-SWITCH', EvalSuite PREREG_f1): CFG["fatt_verify"]
    (or REF2_FATT_VERIFY=1), read per call, runs each bool-mask 16- / 8-row verify call as fs4 (fatt16s, KS 4; batch-
    and N-invariant, so no key buckets or class re-runs) instead of MTPLX's 12 head chunks. Not bitwise == REF2-T
    (~2e-3); SD == AR holds by fs4's invariance (the AR pads to 16 rows, so it takes fs4 too). COUNTS fatt_verify /
    verify_old / verify_old_eligible. fs4 is imported on its first call (lazy modules: fatt16s.py, fatt16.py).
    F1-R8 (CFG["fatt_r8"] or REF2_FATT_R8=1, off by default, only with fatt_verify): a padded call with <= 8 real rows
    (8-row checks, AR steps) runs fs4's R 8 form on its first 8 rows: bitwise == F1's rows (B6 INV (e)); COUNTS fatt_r8.
  - DENSE-N (Wed 2026-11-04, exact-safe; CFG["dense_native"] or REF2_DENSE_NATIVE=1, off by default): the 4-bit dense
    calls that take qmm16c run L1-26's fragment-native layout instead (qmm26.qmm16n at ks 1: bitwise == qmm16c). A
    module's resident weight is repacked on the GPU at its first switched call (_dn_repack, MLX ops, no sync) and kept
    in _DN (+~1.5 GB for the 125B's dense set; MLX's layout stays for prefill); only weights marked by the patched
    nn.QuantizedLinear call (or R-83's concat) are repacked, any other array (a slice) keeps qmm16c. CFG
    dense_native_nt (REF2_DENSE_NATIVE_NT, default 1) only groups tiles per threadgroup (bits unchanged). COUNTS
    dense_native / dense_native_repack; remove() drops the copies. qmm26 is imported on the first switched call.
  - HC8-1P (Wed 2026-11-04, FAST, NOT exact; CFG["hc8_fast"] or REF2_HC8_FAST=1, off by default; CFG hc8_np /
    REF2_HC8_NP, default 20): a <= 16-row HC read of a GatedResidual whose mixers are 8-bit g64 nn.QuantizedLinear
    (scripts/hc8r.py hc8_install, the FAST arm's model change, the HC8 recipe) runs hc8r.read v2 at 16 padded rows: one
    pass over the streams, the norm after the GEMM, fp32 gates (6.69e-3 rel rms vs the exact read = HC8-eager's
    6.70e-3; 5.22 vs 9.67 ms per 96-read chain, quiet). Row-invariant (16 padded rows), so SD == AR. Without the
    switch, 8-bit mixers take MTPLX's eager chain at 16 padded rows. COUNTS hc8_fast. hc8r is imported on first use.
    CFG hc8_v / REF2_HC8_V (default 4, Wed 14:5x): hc8r's kernel version; v4 is bitwise == v2 (96 / 96 reads) and
    4.97 vs 5.27 ms per 96-read chain (quiet); v2 is what EvalSuite's snap/leap-c805fd0 runs.
  - FAST-NAX (Thu 2026-11-05, FAST, off by default; CFG["nax_1cq"] or REF2_NAX_1CQ=1, with ooo_1c + tier_nax_ooo +
    nax_native, no hybrid; TABLE_350 'FAST-NAX'): U3's sure rows and QOM's drops on NAX6-N's one-call (scripts/ooo1cq.py:
    late groups keyed by sidecar state too, a dropped group's rows are zeros, each pair writes its own sidecar entry).
    COUNTS nax_1cq. GPU checks: scripts/ooo1cq_check.py.
  - QOM-SPLIT-1C (Fri 2026-11-06, FAST, off by default; CFG["qom_split"] = REF2_QOM_SPLIT, else R57_QOM_SPLIT, as a
    float T; "" / "off" = off; TABLE_350 'QOM-SPLIT-1C'): Breakthrough's qom_keep_mx split on the one-call. A strict row
    (QOM-sure, draft cf >= T, t["qom"][2]) keeps no resident expert, so its optional uses are dropped BEFORE the post:
    the row's rank-0 id (a required use, read anyway) at weight 0. Other rows keep the bit-21 post. Both one-call forms
    (3c / 3c-N QOM, NAX6-N 1C-Q); no kernel change. COUNTS qom_split. Before it, R57_QOM_SPLIT did nothing here.
  - QCNT_1C (Fri 2026-11-06): set ref2.QCNT_1C = [] and every one-call appends, lazily (graph arrays, no sync),
    int32 [uses kept, uses dropped, groups computed, uses the split dropped]. Groups = distinct base slots after the
    drop over the call's rows (the union per block). A dropped pair reads its row's rank-0 slot (3c QOM, the split) or
    is -2 (NAX6-N 1C-Q). Unset (None): nothing is added to the graph.
  - OOO-1C (Wed 2026-11-04, exact, off by default; CFG["ooo_1c"] or REF2_OOO_1C=1): BigMoE's OOO in ONE expert call per
    MoE layer (scripts/ooo1c.py): the late pairs go last in the grid and resolve their slots inside kernel A from the
    loader's release region (coherent(system) polls of BigMoE's coherent-spin layout; no ext change). On BigMoE's 3c path
    (== tier_glu with scores == its two phases + combine_phase, bitwise) and on C1's NAX path (== _ooo_nax6, bitwise);
    needs the gate in spin mode; native records and nax_hybrid keep the two phases. Harness, ~4 % late pairs: -3.2 (3c)
    / -4.1 (NAX) ms per check. COUNTS ooo_1c. U3's sure rows (t["nolsb"], BigMoE's _ooo_call_t3) on the 3c path: the
    same dup-id sidecar post and sidecar slot -2 (final, LP1 -> tier 3); the NAX path declines them. Native records
    (Friday's converted store) take the native forms (3c-N / NAX6-N, bf16 native or p11 pools); layers with N7-15's
    registered dirty flags keep the two phases.
  - QOM-OOO (Wed 2026-11-04 17:1x): with CFG ooo_1c, a call carrying BigMoE's QOM (t["qom"], U3's t3 path, row-major 3c)
    runs one-call too: optional uses posted with bit 21; a late base slot resolving to -2 (not read) is dropped (its
    row's rank-0 slot, weight 0 in kernel B) == the QOM two phases. NAX / native / no-t3 QOM calls keep the two phases.
  - SVH (Wed 2026-11-04, FAST, off by default; CFG["screened_head"] or REF2_SCREENED_HEAD=1): the screened verifier head.
    screened_head_install(lm_head) registers the trunk's 4-bit head; its <= 16-row calls (REF2 on) run qmm16c over
    FR-Spec's top-N (CFG screened_head_n, 98,304) + up to 4,096 session ids (screened_head_add(ids) / _reset()), and
    the logits are scattered into a full-vocabulary -inf array. Inside the subset == REF2's full head bitwise; rows
    whose true argmax is outside change. COUNTS screened_head.
  - D5 (Wed 2026-11-04, EXACT, off by default; CFG["hc_pend_mlp"] or REF2_HC_PEND=1): a layer's MLP-side residual write
    goes to the next layer's attention-side HC read as pending (as the attention write goes to the MLP read). The
    write (_hc_write) stashes (output, hyper, block_out, inject) in one carry slot; an HC read whose input IS that
    output reads (hyper, pending) and folds the write into its norm pass, bitwise. The write's kernel then runs only
    for other consumers (taps, mid-loop evals). COUNTS hc_pend.
  - SMALL-N (Wed 2026-11-04, FAST, off by default; CFG["small_native"] or REF2_SMALL_NATIVE=1): the small 4-bit g64 calls
    that take MLX's 16-row call (N < dense_min_n or N % 64 != 0: R-74's shared gate | up | gate, k / v, R-83's b | a)
    run on qmm16n with rows zero-padded to a multiple of 32, for marked resident weights (QuantizedLinear calls, R-83's
    concats, R-74's concat). Bits differ from MLX's call; 2.94 -> 1.80 ms per check on the default's shapes. COUNTS
    small_native. remove() drops the padded copies.
  - DENSE-N-RAM (Wed 2026-11-04, exact, inert until called): dense_native_convert(model, head=False) replaces every
    eligible 4-bit dense weight (QuantizedLinear, bits 4, group 64, N >= dense_min_n; R-83's qkv | z concat too) IN PLACE
    by its native copy, 4-D on purpose so that a direct reader fails loudly; one copy, no RAM cost. _qmm: a converted
    weight takes qmm16n at <= 16 rows with ref2 on (bitwise == qmm16c); any other call (prefill, ref2 off) rebuilds MLX's
    layout per call (_dn_unpack, the exact inverse) for MLX's own kernel (stock bits). The 4-bit lm_head stays unless
    head=True. COUNTS dense_native / dense_native_unpack. dense_native_unconvert(model) restores MLX's layout.
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
  - LAZY COMMIT (Fable B, ~/FableGod/fableb/build_1105, Thu 2026-11-05; a SKETCH, never run on a GPU; off by default:
    CFG["gdn_lazy_commit"] or REF2_GDN_LAZY=1; needs CFG gdn_body): a partly accepted check no longer pays MTPLX's
    commit replay on the pure-GDN layers (a second kernel per layer that reads the snapshot state and writes the new
    one, pulled by the next check's eval). The cache entry's state becomes [conv, base, pack]: the true recurrent
    state is the pack's rows (q | k | v | a | b of the kept rows, 10,336 bf16 each) replayed on base, and the next
    <= 16-row call's body (gdnbody.gdn_body_lazy) replays them in registers before its own rows, writes that committed
    state once and emits its own rows' pack. The pack is a state leaf, so MTPLX's snapshot / restore carry it. Any other
    reader of entry.state[1] must flush first (gdn_lazy_flush) or ask gdn_lazy_state; _gdn_call flushes for the calls
    the body declines. The PLE layer (4 slots) is never lazy. COUNTS gdn_lazy / gdn_lazy_rows / gdn_lazy_flush. The
    protocol is fuzzed on CPU (build_1105/lazy_commit/lazy_sim_check.py); the bits need scripts/gdnb_lazy_check.py.
  - DN-KS (Fri 2026-11-06, FAST, off by default: CFG["dn_ks"] or REF2_DN_KS=1): the native 4-bit calls (DENSE-N, DENSE-N-RAM,
    SMALL-N) take a K split per shape (_dn_ks) instead of ks 1. Part B found the narrow calls K-serial; the chained bench
    (scripts/dnks_bench.py) saves 7.7 ms per 2-row check over out_proj / o_proj (ks 6), a|b (10), k / v (10), shared
    gate|up (8) and shared down (5). Bits change in the K order (<= 1 bf16 ulp per output): FAST only. COUNTS dn_ks.
    REF2_DN_KS=2 (v2, Fri 08:5x): + in_proj qkv|z at ks 2 (scripts/wide_bench.py: -13 %), fallback ks <= 10; =1 unchanged.
  - R-83 (exact-safe, off by default): CFG["gdn_inproj_x"] is R-68's fusion without the reference change: qkv | z as ONE
    qmm16c call (N 16384; qmm16c's per-column bits do not depend on N) and b | a as ONE MLX 16-row call (N 96, below
    dense_min_n, so it stays on MLX's kernel). Every part keeps REF2-T's bits; two dispatches less per GDN layer.
  - R-82 (exact-safe, off by default): CFG["hc_gate"] runs the S4 HC read's partition sum and its gate1 lookups as one
    kernel (leapk.hc_read fuse_gate): the same mix / inject bits, one dispatch less per read.
  - D1-B (exact-safe, off by default; Mon 2026-11-02): CFG["hc_b"] runs the S4 HC read's partition sum + gate, MLX's
    K-320 up GEMM (a bitwise replica) and the tail as ONE kernel (leapk.hc_read kernel_b): the same mixed / inject bits,
    5 -> 3 dispatches per read (quiet: -0.7 to -1.0 ms per check). Takes precedence over hc_gate.
  - K-62 (REF2c-T candidate, off by default): CFG["tier_nax4"] runs every BigMoE tier GLU call on the matrix-unit kernel
    (/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/tierglu_nax4.py): batch-invariant, more accurate than 3c, but its bits differ from 3c's, so it is
    a new reference that needs the same-scores test.
  - C1 (Mon 2026-11-02, REF2c-T candidate, off by default): CFG["tier_nax_ooo"] runs every tier GLU call on NAX6
    (/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/tierglu_nax6.py, NAX4 with a fixed split-K) WITH BigMoE's OOO: bigmoe.tier._ooo_call's phase 1
    (resident pairs, encoded before the spins) and phase 2 (the late pairs) each take NAX6 on a phase-aware GPU grouping,
    then the same combine_phase; the other calls take NAX6 in one call (+ combine_phase when scores come). A pair's bits
    do not depend on its group (INV), so the two phases == one NAX6 call bitwise. New reference (NAX6 != 3c bits).
  - H-1i (Mon 2026-11-02, FAST, off by default; BigMoE's fastBEh arm): CFG["nax_hybrid"] runs C1's NAX6 calls (both the
    plain call and OOO's two phases) on NAX6-H (/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/tierglu_nax6h.py): NAX6's kernels with one-pair groups on
    the SIMD ALUs (fp32 FMA dot) instead of the 16-row matrix op set. Needs tier_nax_ooo. Outputs differ from NAX6's in
    ~1 % of pair rows (bf16 last bits), and a pair's bits can depend on its group's size: FAST, not a reference.
  - N7-13 (Mon 2026-11-02, off by default): CFG["nax_native"] runs C1's NAX6 calls on the fragment-native record layout
    (/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/tierglu_nax6n.py; bitwise == NAX6, quiet +18.7 % at S 16). The pools MUST have been converted with
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
       "fatt_verify": os.environ.get("REF2_FATT_VERIFY") == "1", "fatt_r8": os.environ.get("REF2_FATT_R8") == "1",
       "dense_native": os.environ.get("REF2_DENSE_NATIVE") == "1",
       "dense_native_nt": int(os.environ.get("REF2_DENSE_NATIVE_NT", "1")),
       "hc8_fast": os.environ.get("REF2_HC8_FAST") == "1", "hc8_np": int(os.environ.get("REF2_HC8_NP", "20")),
       "keep_caches": os.environ.get("REF2_KEEP_CACHES") == "1",   # KEEP-C: remove() keeps _DN / _SN / _HC8 (default off)
       "eprep": os.environ.get("REF2_EPREP") == "1",   # PLAN 220 #3e: _ooo_1c_call's id glue as one dispatch (eprep_k; default off)
       "gdn_tg": os.environ.get("REF2_GDN_TG") == "1",   # PLAN 220 1a: the 1-16-row GDN step as gdn_step_r's TG 2 (bitwise; default off)
       "sb_first": os.environ.get("REF2_SB_FIRST") == "1",   # PLAN 220 seam (b): layer 0's output sent before layer 1 (default off)
       "ple_late": os.environ.get("REF2_PLE_LATE") == "1",   # PLAN 220 seam (d): the PLE stage at layer 1, not before layer 0 (default off)
       "commit_sub": int(os.environ.get("REF2_COMMIT_SUB", "0") or 0),   # PLAN 220 seam (e): send the commit's GDN replays every N (0 off)
       "tier_chunk": os.environ.get("REF2_TIER_CHUNK") == "1",   # F24: a tier call whose expert union exceeds the slots runs in row groups (default off)
       "tier_chunk_cap": int(os.environ.get("REF2_TIER_CHUNK_CAP", "0") or 0),   # test only: group at this union cap (0: the pools' own)
       "hc8_fold": os.environ.get("REF2_HC8_FOLD") == "1",   # CHAT-150 G1 / B1: HC8 reads with <= 4 rows fold K2 into K3 (hc8f; bitwise; default off)
       "hc8_k1m": os.environ.get("REF2_HC8_K1M") == "1",   # CHAT-150 G1: HC8 reads with <= 4 rows: K1 reads the M-row h / pending (no pad launches), then k2b + k3c, or K3F with hc8_fold (hc8f; bitwise; default off)
       "hc8_g": int(os.environ.get("REF2_HC8_G", "0") or 0),   # CHAT-150 G1: HC8 reads with <= N (1-4) rows in GEMV form (hc8g; FAST: sums reordered, IFEval-gated; 0 off)
       "g2_rows": int(os.environ.get("REF2_G2_ROWS", "0") or 0),   # CHAT-150 G2: FAST-NAX one-calls with <= N rows run 3c-N's GEMV one-call (g2k, U3-capable; FAST, IFEval-gated; 0 off)
       "nax_coh": os.environ.get("REF2_NAX_COH", "1") == "1",   # COH-1CQ, DEFAULT ON since Sun 2026-11-08 (CHS4: == AR 12/12 in two runs vs 3/12 plain, no cost; BigMoE OK 07:5x): FAST-NAX one-call kernels read the expert pools coherent(system) (coh1cq); REF2_NAX_COH=0 = the old plain reads (stale slot lines)
       "ngemv_rows": int(os.environ.get("REF2_NGEMV_ROWS", "0") or 0),   # CHAT-150 G1: native 4-bit dense calls with <= N (1-4) rows run NGEMV (ngemv.py; FAST, IFEval-gated; 0 off)
       "pad_in": os.environ.get("REF2_PAD_IN") == "1",   # CHAT-150 G1: native <= 16-row calls pad in the kernel (qmm26m; bitwise; default off)
       "hc8_v": int(os.environ.get("REF2_HC8_V", "4")),
       "ooo_1c": os.environ.get("REF2_OOO_1C") == "1", "small_native": os.environ.get("REF2_SMALL_NATIVE") == "1",
       "nax_1cq": os.environ.get("REF2_NAX_1CQ") == "1",
       "qom_split": (lambda v: None if v in ("", "off") else float(v))(
           os.environ.get("REF2_QOM_SPLIT", os.environ.get("R57_QOM_SPLIT", ""))),   # QOM-SPLIT-1C (Fri 11-06)
       "hc_pend_mlp": os.environ.get("REF2_HC_PEND") == "1",
       "screened_head": os.environ.get("REF2_SCREENED_HEAD") == "1",
       "screened_head_n": int(os.environ.get("REF2_SCREENED_HEAD_N", "98304")),
       "gdn_lazy_commit": os.environ.get("REF2_GDN_LAZY") == "1",
       "dn_ks": int(os.environ.get("REF2_DN_KS", "0") or "0"),
       "neverwait": os.environ.get("REF2_NEVERWAIT") == "1",
       "nw_force": os.environ.get("REF2_NW_FORCE", "")}
_ON = [False]
COUNTS = dict(qmm16c=0, mlx16=0, linear=0, hc_read=0, hc_eager16=0, attention=0, experts=0, slot_calls=0, sdpa_split=0,
              kb_pad=0, tier_calls=0, gdnx=0, nax4=0, att_single=0, gdn_inproj=0, att_rope=0, idx_prep=0, hc_write=0, gdn_inproj_x=0, nax6=0, nax6_ooo=0, gdnb=0, hc_b=0, nax_hybrid=0, nax_native=0, nax_repack=0, native_3c=0,
              fatt_verify=0, verify_old=0, verify_old_eligible=0, fatt_r8=0, dense_native=0, dense_native_repack=0, hc8_fast=0, dense_native_unpack=0, ooo_1c=0, small_native=0, hc_pend=0, screened_head=0, ooo_1c_qom=0, nax_1cq=0, qom_split=0,
              gdn_lazy=0, gdn_lazy_rows=0, gdn_lazy_flush=0, dn_ks=0, nw_calls=0, nw_late_calls=0, nw_writes=0, eprep=0, gdn_tg=0, sb_first=0, ple_late=0, ple_late_stale=0, commit_sub=0, tier_chunk=0, tier_chunk_groups=0, pad_in=0, hc8_fold=0, hc8_k1m=0, hc8_g=0, g2_3cn=0, ngemv=0, nax_coh=0)
QCNT_1C = None                                       # Fri 11-06: a list -> each one-call's post-drop work (see the notes)
NW_ON = False                                        # NEVER-WAIT (Fri 11-06): runtime switch, read per one-call (armed by
NWCNT = None                                         # CFG["neverwait"]); NWCNT: a list -> per one-call float32 counters
NW_LAST = 47                                         # the last MoE layer: a cold rank 0 waits there, other late pairs drop
NW = {"k": 0, "pend": None, "due": None, "after": False, "inj": {}}


def NW_RESET():
    """NEVER-WAIT: per check (r57 calls it before each verify pass): the layer counter to 0 and nothing carried over"""
    NW.update(k=0, pend=None, due=None, after=False, inj={})


def _nw_force():
    """REF2_NW_FORCE 'L:e1,e2;L2:e3' -> {L: (e1, e2), ...} (test only: these groups are late whatever their residency)"""
    out = {}
    for part in (CFG["nw_force"] or "").split(";"):
        if ":" in part:
            l_, es = part.split(":", 1)
            out[int(l_)] = tuple(int(e) for e in es.split(",") if e.strip())
    return out
_ROWS = [R]                                          # F1-R8: the real rows of the attention call being built (_att)
_ORIG = {}
_WCAT = {}

def _rows(x):
    lead = x.shape[:-1]
    return lead, int(np.prod(lead)) if lead else 1

def _pad16(x2):
    m = x2.shape[0]
    return x2 if m == R else mx.concatenate([x2, mx.zeros((R - m,) + x2.shape[1:], x2.dtype)], axis=0)

_DN = {}                                             # DENSE-N: id(weight) -> (weight, wn, sn, bn), native copies
_DN_SRC = [None]                                     # DENSE-N: the resident weight of the call being built (_qlin)
_DN_MAPS = []

def _qmm16n():
    """DENSE-N: qmm26.qmm16n, loaded on the first switched call only"""
    from qmm26 import qmm16n
    return qmm16n

def _dn_repack(w, scales, biases):
    """DENSE-N: qmm26.repack's layout built with MLX ops on the GPU (lazy, no host sync): wn [N/32, K/64, 32, 8]
    uint32 (lane l's word 2 st + p = rows 16 p + fm(l) (low half) and 16 p + fm(l) + 8 (high half), codes of step st),
    sn / bn [N/32, K/64, 32]"""
    if not _DN_MAPS:
        lane = np.arange(32)
        qid = lane >> 2
        fm, fq = (qid & 4) | ((lane >> 1) & 3), (qid & 2) | (lane & 1)
        i0, i1, sh = (np.zeros((32, 8), np.int64) for _ in range(3))
        for l in range(32):
            for st in range(4):
                for pp in range(2):
                    ws = 2 * int(fq[l]) + (st >> 1)
                    i0[l, 2 * st + pp] = (16 * pp + int(fm[l])) * 8 + ws
                    i1[l, 2 * st + pp] = (16 * pp + int(fm[l]) + 8) * 8 + ws
                    sh[l, 2 * st + pp] = 16 * (st & 1)
        _DN_MAPS.extend([mx.array(i0.reshape(-1).astype(np.int32)), mx.array(i1.reshape(-1).astype(np.int32)),
                         mx.array(sh.reshape(-1).astype(np.uint32))])
    i0, i1, sh = _DN_MAPS
    N, KW = w.shape
    T, KG = N // 32, KW // 8
    w3 = w.reshape(T, 32, KG, 8).transpose(0, 2, 1, 3).reshape(T, KG, 256)          # [tile, group, row * 8 + word]
    lo = mx.bitwise_and(mx.right_shift(mx.take(w3, i0, axis=2), sh), 0xFFFF)
    hi = mx.bitwise_and(mx.right_shift(mx.take(w3, i1, axis=2), sh), 0xFFFF)
    wn = mx.contiguous(mx.bitwise_or(lo, mx.left_shift(hi, 16)).reshape(T, KG, 32, 8))
    sn = mx.contiguous(scales.reshape(T, 32, KG).transpose(0, 2, 1))
    bn = mx.contiguous(biases.reshape(T, 32, KG).transpose(0, 2, 1))
    return wn, sn, bn

_DNR = {}                                            # DENSE-N-RAM: id(native weight) -> (native weight, N, K)
_DNR_ON = [False]
_DN_IMAPS = []

def _dn_unpack(wn, sn, bn, N, K):
    """DENSE-N-RAM: _dn_repack's exact inverse (MLX ops): native [N/32, K/64, 32, 8] + [N/32, K/64, 32] ->
    MLX's [N, K/8] uint32 + [N, K/64] scales / biases"""
    if not _DN_IMAPS:
        lane = np.arange(32)
        qid = lane >> 2
        fm, fq = (qid & 4) | ((lane >> 1) & 3), (qid & 2) | (lane & 1)
        lane_of = {(int(fm[l]), int(fq[l])): l for l in range(32)}
        ilo, ihi, sh = (np.zeros(256, np.int64) for _ in range(3))
        for row in range(32):
            pp, r16 = row >> 4, row & 15
            for ws in range(8):
                l = lane_of[(r16 & 7, ws >> 1)]
                stb = 2 * (ws & 1)
                ilo[row * 8 + ws] = l * 8 + 2 * stb + pp
                ihi[row * 8 + ws] = l * 8 + 2 * (stb + 1) + pp
                sh[row * 8 + ws] = 16 * int(r16 >= 8)
        _DN_IMAPS.extend([mx.array(ilo.astype(np.int32)), mx.array(ihi.astype(np.int32)), mx.array(sh.astype(np.uint32))])
    ilo, ihi, sh = _DN_IMAPS
    T, KG = N // 32, K // 64
    n3 = wn.reshape(T, KG, 256)                                        # [tile, group, lane * 8 + word]
    lo = mx.bitwise_and(mx.right_shift(mx.take(n3, ilo, axis=2), sh), 0xFFFF)
    hi = mx.bitwise_and(mx.right_shift(mx.take(n3, ihi, axis=2), sh), 0xFFFF)
    w = mx.bitwise_or(lo, mx.left_shift(hi, 16)).reshape(T, KG, 32, 8).transpose(0, 2, 1, 3).reshape(N, K // 8)
    return (w, sn.transpose(0, 2, 1).reshape(N, KG), bn.transpose(0, 2, 1).reshape(N, KG))

def _dnr_arrays(w, scales, biases):
    """DENSE-N-RAM: the native copy, evaluated (so the caller can drop the originals right away)"""
    wn, sn, bn = _dn_repack(w, scales, biases)
    mx.eval(wn, sn, bn)
    return wn, sn, bn

def _dn_mlx(m):
    """DENSE-N-RAM: a QuantizedLinear's (weight, scales, biases) in MLX's layout (unpacked if converted)"""
    w = m["weight"]
    e = _DNR.get(id(w))
    if e is not None and e[0] is w:
        return _dn_unpack(w, m["scales"], m["biases"], e[1], e[2])
    return w, m["scales"], m["biases"]

def dense_native_convert(model, head=False):
    """DENSE-N-RAM: replace every eligible 4-bit dense weight of model (and R-83's built qkv | z concats) by its native
    copy in place; returns (weights converted, MB). Call after load, outside mx.compile; ref2.apply() first."""
    n, mb = 0, 0.0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear) and "biases" in m and m.bits == 4 and m.group_size == 64 \
                and getattr(m, "mode", "affine") == "affine":
            w = m["weight"]
            e = _DNR.get(id(w))
            if w.ndim != 2 or (e is not None and e[0] is w) or m["scales"].dtype != mx.bfloat16:
                continue
            N, K = int(w.shape[0]), int(w.shape[1]) * 8
            if N < CFG["dense_min_n"] or N % 64 or K % 64 or (N > 65536 and not head):
                continue
            wn, sn, bn = _dnr_arrays(w, m["scales"], m["biases"])
            m["weight"], m["scales"], m["biases"] = wn, sn, bn
            _DNR[id(wn)] = (wn, N, K)
            n += 1
            mb += (wn.nbytes + sn.nbytes + bn.nbytes) / 1e6
        f = getattr(m, "__dict__", {}).get("in_proj_fused")
        if isinstance(f, _InProjExact) and f._convert():
            n += 1
            mb += sum(t.nbytes for t in f.qz) / 1e6
    _DNR_ON[0] = True
    return n, mb

def dense_native_unconvert(model):
    """DENSE-N-RAM: MLX's layout back (every converted module and R-83 concat of model); returns the count"""
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            w = m["weight"]
            e = _DNR.pop(id(w), None)
            if e is not None and e[0] is w:
                wm, sm, bm = _dn_unpack(w, m["scales"], m["biases"], e[1], e[2])
                mx.eval(wm, sm, bm)
                m["weight"], m["scales"], m["biases"] = wm, sm, bm
                n += 1
        f = getattr(m, "__dict__", {}).get("in_proj_fused")
        if isinstance(f, _InProjExact):
            e = _DNR.pop(id(f.qz[0]), None)
            if e is not None and e[0] is f.qz[0]:
                f.qz = _dn_unpack(*f.qz, e[1], e[2])
                mx.eval(f.qz)
                n += 1
    _DNR_ON[0] = False
    return n

_DNKS = {(2560, 6144): 6, (96, 2560): 10, (512, 2560): 10, (1312, 2560): 8, (2560, 640): 5}
_DNKS2 = {**_DNKS, (16384, 2560): 2}              # DN-KS v2 (REF2_DN_KS=2): + in_proj qkv|z at ks 2 (WIDE-NT, -13 %)


def _dn_ks(N, K):
    """DN-KS (Leap, Fri 2026-11-06, FAST, off by default; CFG["dn_ks"] or REF2_DN_KS=1): the K split of a native 4-bit
    call (qmm26.qmm16n's ks) per shape. ks 1 walks the whole K in one simdgroup per 32-column tile, so a narrow call is
    latency-bound (STEP 0 PART B: out_proj / o_proj 112 us, a|b 45, k / v 48, shared gate|up 47). The 5 measured kinds
    take their best ks (scripts/dnks_bench.py, PASS 08:01: -7.7 ms per 2-row check); any other shape takes the smallest
    ks dividing K / 64 (<= 16, threadgroup memory) that gives >= 320 simdgroups. Bits change (the K order): FAST only."""
    v = CFG["dn_ks"]
    if not v:
        return 1
    ks = (_DNKS2 if v >= 2 else _DNKS).get((N, K))
    if ks is None:
        g, tiles, ks = K // 64, N // 32, 1
        for k in ((1, 2, 4, 5, 8, 10) if v >= 2 else (1, 2, 4, 5, 8, 10, 16)):   # v2: <= 20 KB of partials (NAX-OCC)
            if g % k == 0:
                ks = k
                if tiles * k >= 320:
                    break
    if ks > 1:
        COUNTS["dn_ks"] += 1
    return ks


def _dnr_qmm(x, w, orig, args, kw):
    """DENSE-N-RAM: a call on a converted weight"""
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    _, N, K = _DNR[id(w)]
    lead, rows = _rows(x)
    if (_ON[0] and transpose and mode == "affine" and bits == 4 and group_size == 64 and rows <= R
            and x.dtype == mx.bfloat16 and x.shape[-1] == K and scales.dtype == mx.bfloat16):
        ks = _dn_ks(N, K)
        nt = CFG["dense_native_nt"] if ks == 1 and N % (32 * CFG["dense_native_nt"]) == 0 else 1
        if rows <= CFG["ngemv_rows"]:                  # CHAT-150 G1 NGEMV (FAST): the GEMV form at few rows
            import ngemv as _NG
            y = _NG.ngemv(x.reshape(rows, K), w, scales, biases, N)
            COUNTS["ngemv"] += 1
        elif CFG["pad_in"]:                            # CHAT-150 G1: no materialized 16-row pad (qmm26m, bitwise)
            from qmm26m import qmm16n_m
            y = qmm16n_m(x.reshape(rows, K), w, scales, biases, N, ks, nt)
            COUNTS["pad_in"] += 1
        else:
            y = _qmm16n()(_pad16(x.reshape(rows, K)), w, scales, biases, N, ks, nt)
        COUNTS["dense_native"] += 1
        return y[:rows].reshape(*lead, N)
    wm, sm, bm = _dn_unpack(w, scales, biases, N, K)
    COUNTS["dense_native_unpack"] += 1
    extra = {k: v for k, v in kw.items() if k == "stream"}
    return orig(x, wm, sm, bm, transpose=transpose, group_size=group_size, bits=bits, mode=mode, **extra)

def _dn_get(w, scales, biases):
    """DENSE-N: the native copy of a resident weight (repacked at its first marked call), or None (keep qmm16c)"""
    e = _DN.get(id(w))
    if e is not None and e[0] is w:
        return e
    if w is not _DN_SRC[0] or w.shape[0] % 32 != 0:
        return None
    e = _DN[id(w)] = (w,) + _dn_repack(w, scales, biases)
    COUNTS["dense_native_repack"] += 1
    return e

_SN = {}                                             # SMALL-N: id(weight) -> (weight, wn, sn, bn, padded N)

def _sn_get(w, scales, biases):
    """SMALL-N: the padded native copy of a marked resident small weight (built at its first marked call), or None"""
    e = _SN.get(id(w))
    if e is not None and e[0] is w:
        return e
    if w is not _DN_SRC[0] or w.ndim != 2:
        return None
    N = int(w.shape[0])
    Np = (N + 31) // 32 * 32
    pad = lambda a: mx.concatenate([a, mx.zeros((Np - N,) + tuple(a.shape[1:]), a.dtype)]) if Np > N else a
    e = _SN[id(w)] = (w,) + _dn_repack(pad(w), pad(scales), pad(biases)) + (Np,)
    return e

_SH = {}                                             # SVH: id(lm_head weight) -> its subset state

def screened_head_install(lm_head, base_n=None, cap=4096, rank_path=None):
    """SVH: register the trunk's 4-bit g64 lm_head (bf16 scales) for the screened verifier head: the base subset
    (FR-Spec's top base_n ids, sorted) gathered once, plus room for cap session ids. Returns the subset's base rows."""
    w, sc, bi = lm_head["weight"], lm_head["scales"], lm_head["biases"]
    assert lm_head.bits == 4 and lm_head.group_size == 64 and sc.dtype == mx.bfloat16 and w.ndim == 2, "4-bit g64 head"
    base_n = base_n or CFG["screened_head_n"]
    rank = np.load(rank_path or os.path.expanduser("~/Leap/data/frspec_rank.npy"))
    ids = np.unique(np.asarray(rank[:base_n], np.int64)).astype(np.int32)
    base = mx.array(ids)
    e = dict(w=w, s=sc, b=bi, V=int(w.shape[0]), ids=base, set=set(ids.tolist()), extra=[], cap=int(cap),
             wb=mx.contiguous(mx.take(w, base, axis=0)), sb=mx.contiguous(mx.take(sc, base, axis=0)),
             bb=mx.contiguous(mx.take(bi, base, axis=0)),
             xids=mx.zeros((cap,), mx.int32), xw=mx.zeros((cap, w.shape[1]), w.dtype), xs=mx.zeros((cap, sc.shape[1]), sc.dtype),
             xb=mx.zeros((cap, bi.shape[1]), bi.dtype))
    mx.eval(e["ids"], e["wb"], e["sb"], e["bb"], e["xids"], e["xw"], e["xs"], e["xb"])
    _SH.clear()
    _SH[id(w)] = e
    return int(ids.size)

def screened_head_add(token_ids):
    """SVH: the session's ids (prompt, committed tokens) outside the base join the subset (once each, up to cap)"""
    for e in _SH.values():
        new = [t for t in dict.fromkeys(int(t) for t in np.asarray(token_ids).reshape(-1)) if t not in e["set"]]
        new = new[:max(0, e["cap"] - len(e["extra"]))]
        if not new:
            continue
        a, n = len(e["extra"]), len(new)
        idx = mx.array(np.asarray(new, np.int32))
        e["xids"][a:a + n] = idx
        e["xw"][a:a + n] = mx.take(e["w"], idx, axis=0)
        e["xs"][a:a + n] = mx.take(e["s"], idx, axis=0)
        e["xb"][a:a + n] = mx.take(e["b"], idx, axis=0)
        mx.eval(e["xids"], e["xw"], e["xs"], e["xb"])
        e["extra"] += new
        e["set"].update(new)
    return sum(len(e["extra"]) for e in _SH.values())

def screened_head_reset():
    """SVH: drop the session ids (a new prompt)"""
    for e in _SH.values():
        e["set"] -= set(e["extra"])
        e["extra"] = []

def _screened(e, x, scales, biases):
    lead, rows = _rows(x)
    if rows > R or x.dtype != mx.bfloat16:
        return None
    x16 = _pad16(x.reshape(rows, x.shape[-1]))
    y = qmm16c(x16, e["wb"], e["sb"], e["bb"])[:rows]
    ids = e["ids"]
    n = len(e["extra"])
    if n:
        yx = qmm16c(x16, e["xw"], e["xs"], e["xb"])[:rows, :n]
        y = mx.concatenate([y, yx], axis=1)
        ids = mx.concatenate([ids, e["xids"][:n]])
    out = mx.put_along_axis(mx.full((rows, e["V"]), -float("inf"), y.dtype),
                            mx.broadcast_to(ids[None].astype(mx.int32), y.shape), y, axis=1)
    COUNTS["screened_head"] += 1
    return out.reshape(*lead, e["V"])

def _qlin(self, x):
    """DENSE-N / SMALL-N: nn.QuantizedLinear's call with its weight marked as resident for _qmm (inert unless one of
    the switches is on); SVH: the registered lm_head's screened call (CFG screened_head)"""
    if _SH and _ON[0] and CFG["screened_head"]:
        e = _SH.get(id(self["weight"]))
        if e is not None and e["w"] is self["weight"]:
            y = _screened(e, x, self["scales"], self["biases"])
            if y is not None:
                return y
    if not (_ON[0] and (CFG["dense_native"] or CFG["small_native"])):
        return _ORIG["qlin"](self, x)
    prev = _DN_SRC[0]
    _DN_SRC[0] = self.get("weight")
    try:
        return _ORIG["qlin"](self, x)
    finally:
        _DN_SRC[0] = prev

def _qmm(x, w, *args, **kw):
    orig = _ORIG["qmm"]
    if _DNR:
        e = _DNR.get(id(w))
        if e is not None and e[0] is w:                  # DENSE-N-RAM: a converted weight (any phase, ref2 on or off)
            return _dnr_qmm(x, w, orig, args, kw)
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
    x2 = x.reshape(rows, K_)
    pin = CFG["pad_in"]                                  # CHAT-150 G1: the native forms pad in the kernel (qmm26m, bitwise)
    x16 = None if pin else _pad16(x2)
    if bits == 4 and group_size == 64 and N >= CFG["dense_min_n"] and N % 64 == 0 and K_ % 64 == 0:
        e = _dn_get(w, scales, biases) if CFG["dense_native"] else None
        if e is not None:                                # DENSE-N: bitwise == qmm16c (ks 1); nt only groups tiles
            ks = _dn_ks(N, K_)
            nt = CFG["dense_native_nt"] if ks == 1 and N % (32 * CFG["dense_native_nt"]) == 0 else 1
            if rows <= CFG["ngemv_rows"]:              # CHAT-150 G1 NGEMV (FAST)
                import ngemv as _NG
                y = _NG.ngemv(x2, e[1], e[2], e[3], N)
                COUNTS["ngemv"] += 1
            elif pin:
                from qmm26m import qmm16n_m
                y = qmm16n_m(x2, e[1], e[2], e[3], N, ks, nt)
                COUNTS["pad_in"] += 1
            else:
                y = _qmm16n()(x16, e[1], e[2], e[3], N, ks, nt)
            COUNTS["dense_native"] += 1
        else:
            y = qmm16c(_pad16(x2) if x16 is None else x16, w, scales, biases)
            COUNTS["qmm16c"] += 1
    else:
        e = (_sn_get(w, scales, biases) if (CFG["small_native"] and bits == 4 and group_size == 64 and K_ % 64 == 0
                                           and scales is not None and scales.dtype == mx.bfloat16) else None)
        if e is not None:                                # SMALL-N (FAST): qmm16n on the padded native copy
            if rows <= CFG["ngemv_rows"]:                # CHAT-150 G1 NGEMV (FAST)
                import ngemv as _NG
                y = _NG.ngemv(x2, e[1], e[2], e[3], e[4])[:, :N]
                COUNTS["ngemv"] += 1
            elif pin:
                from qmm26m import qmm16n_m
                y = qmm16n_m(x2, e[1], e[2], e[3], e[4], _dn_ks(e[4], K_), 1)[:, :N]
                COUNTS["pad_in"] += 1
            else:
                y = _qmm16n()(x16, e[1], e[2], e[3], e[4], _dn_ks(e[4], K_), 1)[:, :N]
            COUNTS["small_native"] += 1
        else:
            y = orig(_pad16(x2) if x16 is None else x16, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode)
            COUNTS["mlx16"] += 1
    return y[:rows].reshape(*lead, N)

def _lin(self, x):
    lead, rows = _rows(x)
    if _ON[0] and rows < R and x.dtype == mx.bfloat16:
        y = _ORIG["lin"](self, _pad16(x.reshape(rows, x.shape[-1])))
        COUNTS["linear"] += 1
        return y[:rows].reshape(*lead, y.shape[-1])
    return _ORIG["lin"](self, x)

_HC8 = {}
def _hc8r():
    """HC8-1P: scripts/hc8r.py, loaded on first use"""
    import hc8r
    return hc8r

def _hc8_params(g):
    """HC8-1P: the 8-bit mixers' arrays of a GatedResidual as hc8r.read's P (cached per module), or None when its mixers
    are not 8-bit g64 bf16-scaled nn.QuantizedLinear"""
    d, u = g.input_mix_weight_down, g.input_mix_weight_up
    if not (isinstance(d, nn.QuantizedLinear) and isinstance(u, nn.QuantizedLinear)):
        return None
    e = _HC8.get(id(g))
    if e is not None and e[0] is d["weight"] and e[1] is u["weight"]:
        return e[2]
    if not (d.bits == 8 and u.bits == 8 and d.group_size == 64 and u.group_size == 64 and "biases" in d and "biases" in u
            and d["scales"].dtype == mx.bfloat16 and u["scales"].dtype == mx.bfloat16 and "block_inject_weight" in g):
        return None
    P = dict(wq=d["weight"], ws=d["scales"], wb=d["biases"], uq=u["weight"], us=u["scales"], ub=u["biases"],
             wi=g.block_inject_weight.weight.astype(mx.bfloat16), wn=g.hc_norm.weight.astype(mx.bfloat16),
             hc=int(g.hc_count), d=int(g.hidden_size), lr=int(d["weight"].shape[0]), eps=float(g.hc_norm.eps))
    _HC8[id(g)] = (d["weight"], u["weight"], P)
    return P

def _gr(self, hyper_input, pending=None):
    orig = _ORIG["gr"]
    if not _ON[0]:
        return orig(self, hyper_input, pending)
    c = _CARRY[0]
    if pending is None and c is not None and c[0] is hyper_input and CFG["hc_pend_mlp"] and "block_inject_weight" in self:
        _CARRY[0] = None                                 # D5: read the last write's operands, the write folded in
        hyper_input, pending = c[1], (c[2], c[3])
        COUNTS["hc_pend"] += 1
    lead, rows = _rows(hyper_input)
    if rows > R:
        return orig(self, hyper_input, pending)
    if ((rows <= CFG["hc8_g"] or (CFG["hc8_k1m"] and CFG["hc8_v"] in (4, 5))) and CFG["hc8_fast"] and rows <= 4
            and "block_inject_weight" in self and hyper_input.dtype == mx.bfloat16
            and (pending is None or all(t.dtype == mx.bfloat16 for t in pending))):
        q8 = _hc8_params(self)
        if q8 is not None:
            bo, inj = (None, None) if pending is None else (pending[0].reshape(rows, -1), pending[1].reshape(rows, -1))
            if rows <= CFG["hc8_g"]:                 # CHAT-150 G1 HC8-G: GEMV form (FAST: sums reordered)
                import hc8g
                mixed, written, inject = hc8g.read_g(q8, hyper_input.reshape(rows, -1), bo, inj, rps=4 if rows <= 2 else 8)
                COUNTS["hc8_g"] += 1
                COUNTS["hc8_fast"] += 1
                return mixed.reshape(*lead, -1), written.reshape(hyper_input.shape), inject.reshape(*lead, -1)
            import hc8f                              # CHAT-150 K1M: K1 on the M-row inputs (bitwise == the padded read)
            if CFG["hc8_fold"]:
                mixed, written, inject = hc8f.read_fold(q8, hyper_input.reshape(rows, -1), bo, inj, NP=CFG["hc8_np"],
                                                        v=CFG["hc8_v"], M=rows)
                COUNTS["hc8_fold"] += 1
            else:
                mixed, written, inject = hc8f.read_k1m(q8, hyper_input.reshape(rows, -1), bo, inj, NP=CFG["hc8_np"],
                                                       v=CFG["hc8_v"])
            COUNTS["hc8_k1m"] += 1
            COUNTS["hc8_fast"] += 1
            return (mixed[:rows].reshape(*lead, -1), written[:rows].reshape(hyper_input.shape),
                    inject[:rows].reshape(*lead, -1))
    h16 = _pad16(hyper_input.reshape(rows, -1)).reshape(1, R, -1)
    p16 = None if pending is None else tuple(_pad16(p.reshape(rows, -1)).reshape(1, R, -1) for p in pending)
    q8 = _hc8_params(self) if "block_inject_weight" in self else None
    if q8 is not None:                               # HC8-1P: 8-bit mixers (the FAST arm) never take the bf16 S4 path
        bf = mx.bfloat16
        if CFG["hc8_fast"] and h16.dtype == bf and (p16 is None or all(t.dtype == bf for t in p16)):
            bo, inj = (None, None) if p16 is None else (p16[0].reshape(R, -1), p16[1].reshape(R, -1))
            if CFG["hc8_fold"] and rows <= 4 and CFG["hc8_v"] in (4, 5):   # CHAT-150: K2 folded into K3 (hc8f, bitwise)
                import hc8f
                mixed, written, inject = hc8f.read_fold(q8, h16.reshape(R, -1), bo, inj, NP=CFG["hc8_np"], v=CFG["hc8_v"], M=rows)
                COUNTS["hc8_fold"] += 1
            else:
                mixed, written, inject = _hc8r().read(q8, h16.reshape(R, -1), bo, inj, NP=CFG["hc8_np"], v=CFG["hc8_v"])
            COUNTS["hc8_fast"] += 1
            return (mixed[:rows].reshape(*lead, -1), written[:rows].reshape(hyper_input.shape),
                    inject[:rows].reshape(*lead, -1))
        m_, w_, i_ = orig(self, h16, p16)
        COUNTS["hc_eager16"] += 1
        return (m_[0, :rows].reshape(*lead, -1), w_[0, :rows].reshape(hyper_input.shape),
                i_[0, :rows].reshape(*lead, -1))
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
        _ROWS[0] = S                                  # F1-R8: the verify SDPA may compute only the real rows
        try:
            out = _ORIG["att"](self, xp, cache)
        finally:
            _ROWS[0] = R
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
    """K-62 (REF2c-T candidate, CFG tier_nax4, off by default): /Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/tierglu_nax4.py, the tier GLU on the M5's
    matrix units (dequantized once per (slot, tier, sidecar) group; batch-invariant bitwise; NOT bit-identical to 3c)"""
    if _NAX4[0] is None:
        p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts")
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
    """C1 (CFG tier_nax_ooo, off by default): /Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts/tierglu_nax6.py (NAX4 with a fixed split-K)"""
    if _NAX6[0] is None:
        try:                                           # BigMoE's tierglu (MoonShots' planes_engine) first, so it is THE
            import bigmoe.tier as BT                   # "tierglu" module (Leap's third_party copy is identical, 11-02)
            BT._tier_glu_mod()
        except Exception:
            pass
        p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts")
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

def _ooo1c():
    """OOO-1C: scripts/ooo1c.py, loaded on first use"""
    import ooo1c
    return ooo1c

def _ooo1c_offs(bm, t, R4):
    """OOO-1C: [base region offset, sidecar region offset, seq] of this pass for the gate's release regions (the ext's
    ((seq * nl + L) % reg_n) * pairs * rs), or None when the gate has no regions (event mode)"""
    import bigmoe.gated as BG
    gate = bm["gate"]
    if getattr(BG, "MODE", "spin") != "spin" or getattr(gate, "region", None) is None:
        return None
    rs = 32 * (2 + BG.XLINES)
    off = lambda Lx: ((gate.seq * gate.nl + Lx) % gate.nreg) * BG.PAIRS * rs
    return mx.array([off(bm["L"]), off(t["side"]) if R4 else 0, gate.seq], mx.int32)

def _ooo_1c_call(x, ids, bm, t, post, sc, shared, nax):
    """OOO-1C: the post as today, then ONE expert call (3c or NAX6) whose late pairs resolve their slots in kernel A;
    None when not applicable (the caller then runs its two phases)"""
    import bigmoe.tier as BT
    PLAN = BT.PLAN
    K = ids.shape[-1]
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    lo4 = [s_ for s_ in t["segs"] if s_[2] == 4]
    R4 = lo4[-1][1] if lo4 else 0
    assert not lo4 or lo4[0][0] == 0, t["segs"]
    offs = _ooo1c_offs(bm, t, R4)
    nlz = t.get("nolsb")                             # U3's 3-bit sure rows (BigMoE r57_screen _ooo_call_t3)
    g2 = bool(nax and CFG["g2_rows"] and M <= CFG["g2_rows"] and CFG["nax_1cq"] and CFG["nax_native"] and _native(t))
    if g2:                                           # CHAT-150 G2 (FAST): 3c-N's GEMV one-call (g2k, U3-capable) at few rows
        nax = False
    qn = bool(nax and CFG["nax_1cq"] and CFG["nax_native"])   # FAST-NAX (Thu 11-05): U3 + QOM on NAX6-N's one-call (ooo1cq)
    if offs is None or (nlz is not None and nax and not qn):   # (NAX + t3 without FAST-NAX: not here; the two phases
        return None                                  # ignore nolsb too: BigMoE told)
    if _NATIVE_FLAGS.get(id(t["base"])) is not None:  # N7-15's dirty-slot repack runs only on the two-phase path
        return None
    native = bool(CFG["nax_native"]) if nax else _native(t)   # Friday's converted store: NAX6-N / 3c-N one-call forms
    qm = t.get("qom")                                # QOM-OOO (Wed 17:1x): QOM's optional uses (BigMoE _ooo_call_t3);
    if qm is not None and ((nax and not qn) or nlz is None):   # QOM: the 3c / 3c-N one-call, or NAX6-N's with FAST-NAX
        return None                                  # (QOM rides on U3's t3)
    sdrop = None
    if qm is not None and CFG["qom_split"] is not None:   # QOM-SPLIT-1C (Fri 11-06): a strict row (QOM-sure, cf >= T) keeps
        if qm[2] is None or qm[2].size != M:              # no resident expert (qom_keep_mx's split), so its optional uses
            raise RuntimeError("REF2: the QOM split needs each row's draft cf in t['qom'][2]")   # drop before the post
        sdrop = qm[0].reshape(M, K) & (qm[1].reshape(M) & (qm[2].reshape(M) >= CFG["qom_split"]))[:, None]
    gate, L = bm["gate"], bm["L"]
    BT.OOO_STATS["calls"] += 1
    COUNTS["ooo_1c"] += 1
    ids_se = None
    if sdrop is None and CFG["eprep"] and (qm is not None or (R4 and nlz is not None)):   # PLAN 220 #3e: one dispatch,
        import eprep_k                               # bitwise == the ops below (scripts/eprep_k.py check(); dtypes kept)
        nle = nlz[0] if (R4 and nlz is not None) else None
        ids_p, ids_se = eprep_k.glue(ids, qm[0] if qm is not None else None, nle, R4 if nle is not None else 0)
        if qm is None:
            ids_p = ids                              # no QOM: the loader's ids are the ids (as below)
        ids_k = ids.reshape(M, K)
        COUNTS["eprep"] += 1
    elif sdrop is None:
        ids_p = ids if qm is None else mx.where(qm[0].reshape(ids.shape), ids.astype(mx.int32) | (1 << 21),
                                                ids.astype(mx.int32))   # QOM: optional uses marked for the loader only
        ids_k = ids.reshape(M, K)
    else:                                            # the split: a strict row's optional use takes its row's rank-0 id (a
        idm = ids.reshape(M, K).astype(mx.int32)     # required use, read anyway) at weight 0; other optional uses: bit 21
        ids_k = mx.where(sdrop, idm[:, :1], idm)
        ids_p = mx.where(qm[0].reshape(M, K) & ~sdrop, ids_k | (1 << 21), ids_k).reshape(ids.shape)
        sc = mx.where(sdrop.reshape(sc.shape), mx.zeros_like(sc), sc)
        COUNTS["qom_split"] += 1
    dep_m, n_m = post(L, ids_p)
    pre_m = dep_m[1:1 + n_m].reshape(M, K)
    if qm is not None:
        t.pop("qom")
        COUNTS["ooo_1c_qom"] += 1
    if R4:
        if nlz is not None:                          # _ooo_call_t3 verbatim: sure rows post an unsure row's rank-0 expert
            t.pop("nolsb")                           # (read anyway) and take sidecar slot -2 (final: LP1 -> tier 3)
            nl, nl_np = nlz
            nl2 = nl.reshape(M, 1)
            if ids_se is not None:                   # PLAN 220 #3e: computed with ids_p in eprep_k's one dispatch
                ids_s = ids_se
            else:
                un = np.where(~np.asarray(nl_np))[0]
                dup = ids.reshape(M, K)[int(un[0]) if len(un) else 0, 0]
                ids_s = mx.where(nl2, dup, ids.reshape(M, K)[:, :R4]).reshape(*ids.shape[:-1], R4)
        else:
            ids_s = ids[..., :R4]
        dep_l, n_l = post(t["side"], ids_s, after=dep_m)
        pre_l = dep_l[1:1 + n_l].reshape(M, R4)
        if nlz is not None:
            pre_l = mx.where(nl2, -2, pre_l).astype(mx.int32)
    else:
        if nlz is not None:
            t.pop("nolsb")
        pre_l = mx.zeros((M, 1), mx.int32)
    xin = mx.depends(x2, [shared]) if shared is not None else x2   # the shared expert first, as today's spins wait after it
    nw = bool(NW_ON and CFG["neverwait"])
    if nw:                                           # NEVER-WAIT (Fri 11-06, Leap; data/nw/NW_DESIGN.md): a cold DEMAND pair
        k_ = NW["k"]                                 # (post slot -1, not QOM's optional use) runs one layer late
        NW["k"] = k_ + 1
        COUNTS["nw_calls"] += 1
        cold = (pre_m.reshape(M, K) == -1)
        late = cold
        dropo = None
        if qm is not None:                           # NW v3 (Fri 21:2x): a cold OPTIONAL use is dropped now (weight 0), never
            dropo = cold & qm[0].reshape(M, K)       # waited for (v1 left it at -1, so the call still waited on the release,
            late = cold & ~qm[0].reshape(M, K)       # which lands with the layer's demand reads: BigMoE's KILL, stalls unchanged)
        fz = _nw_force().get(k_)
        if fz:                                       # test only: forced groups are late whatever their residency
            late = late | mx.any(ids_k.reshape(M, K, 1).astype(mx.int32) == mx.array(list(fz), mx.int32).reshape(1, 1, -1),
                                 axis=-1)
        stall = None
        if k_ >= NW_LAST:                            # the last MoE layer: a cold rank 0 waits, the other late pairs drop
            rk = mx.arange(K).reshape(1, K) > 0
            stall = late & ~rk
            late = late & rk
        sc2 = sc.reshape(M, K)
        lw = mx.where(late, sc2, mx.zeros_like(sc2))
        skip = late if dropo is None else (late | dropo)
        fill = -2 if (nax and qn) else 0             # NW v2 (Fri 09:4x): FAST-NAX's 1C-Q skips -2 pairs (zero rows, no read);
        pm_late = mx.where(late, pre_m.reshape(M, K), fill).reshape(pre_m.shape)   # 3c-N's QOM path reads rank 0's slot for
        pre_m = mx.where(skip, fill, pre_m.reshape(M, K)).reshape(pre_m.shape).astype(pre_m.dtype)   # -2: slot 0 there
        sc = mx.where(skip, mx.zeros_like(sc2), sc2).reshape(sc.shape)
        if R4:
            lr = late[:, :R4]
            pl_late = mx.where(lr, pre_l.reshape(M, R4), -2).astype(pre_l.dtype).reshape(pre_l.shape)
            cl = (pre_l.reshape(M, R4) == -1) & ~lr      # NW v3: a cold LSB sidecar of a resident base runs 3-bit now (U3's -2)
            pre_l = mx.where(lr | cl, -2, pre_l.reshape(M, R4)).astype(pre_l.dtype).reshape(pre_l.shape)
        else:
            pl_late = pre_l
        if NWCNT is not None:                        # [late groups, late uses, late mass, layer-47 drops, layer-47 stalls, forced]
            f32 = lambda a: a.astype(mx.float32)
            li = mx.sort(mx.where(late, ids_k.reshape(M, K).astype(mx.int32), -1).reshape(-1))
            ng = ((li[1:] != li[:-1]) & (li[1:] >= 0)).sum() + (li[0] >= 0)
            NWCNT.append(mx.stack([f32(ng), f32(late.sum()), f32(lw.sum()),
                                   f32(late.sum()) * (1.0 if k_ >= NW_LAST else 0.0),
                                   f32(stall.sum()) if stall is not None else mx.array(0.0),
                                   f32(mx.array(len(fz) if fz else 0)),
                                   f32(dropo.sum()) if dropo is not None else mx.array(0.0),   # v3: optional uses dropped now
                                   f32(cl.sum()) if R4 else mx.array(0.0)]))   # v3: cold sidecars run 3-bit
    tb, tsp = t["base"], t["sideP"]
    OC = _ooo1c()
    if nax:
        TG = BT._tier_glu_mod()
        if qn:                                       # FAST-NAX: U3's -2 sidecars, QOM's dropped groups (zero rows)
            import ooo1cq as OQ
            f1c = OQ.tier_glu_nax6n_1cq
            COUNTS["nax_1cq"] += 1
            if CFG["nax_coh"]:                       # COH-1CQ (test): every device read coherent(system) (S7B-PROBE B4 / B6)
                import coh1cq
                f1c = coh1cq.tier_glu_nax6n_1cq_coh
                COUNTS["nax_coh"] += 1
        else:
            f1c = OC.tier_glu_nax6n_1c if native else OC.tier_glu_nax6_1c

        def _kern(xin_, pm_, pl_, sc_):
            y_, rs_, _ = f1c(xin_, tb, tsp, pm_, pl_, ids_k, PLAN, gate.region, offs, gate.status)
            return (TG.combine_phase(y_.reshape(M * K, -1), y_.reshape(M * K, -1), mx.ones((M, K), mx.int32),
                                     sc_.reshape(-1, K)), rs_)
        COUNTS["nax6_ooo"] += 1
    else:
        if g2:
            import g2k
            f1c = g2k.tier_glu_3cn_1c_u3
            COUNTS["g2_3cn"] += 1
        else:
            f1c = OC.tier_glu_3cn_1c if native else OC.tier_glu_3c_1c
        qom_ = qm is not None                        # QOM-OOO: dropped optional uses (-2) read rank 0's slot at weight 0

        def _kern(xin_, pm_, pl_, sc_):
            if qom_:
                y_, rs_, _ = f1c(xin_, tb, tsp, pm_, pl_, PLAN, gate.region, offs, gate.status, sc_.reshape(-1, K), qom=True)
            else:
                y_, rs_, _ = f1c(xin_, tb, tsp, pm_, pl_, PLAN, gate.region, offs, gate.status, sc_.reshape(-1, K))
            return y_, rs_
    y, rs_m = _kern(xin, pre_m, pre_l, sc)
    if QCNT_1C is not None:
        _qcnt_1c(rs_m, M, K, sdrop)
    if nw:                                           # the previous layer's late pairs, on its saved input and pool; written
        p_ = NW["pend"]                              # after this layer's MoE write through its own inject (_hc_write)
        if p_ is not None:
            d_, _ = p_["kern"](p_["x"], p_["pm"], p_["pl"], p_["sc"])
            NW["due"] = (d_, p_["k"])
            NW["pend"] = None
            COUNTS["nw_late_calls"] += 1
        if k_ < NW_LAST:
            NW["pend"] = dict(k=k_, x=xin, pm=pm_late, pl=pl_late, sc=lw, kern=_kern)
        NW["after"] = True
    return y.reshape(*x.shape[:-1], -1)

def _qcnt_1c(rs_m, M, K, sdrop):
    """QCNT_1C (Fri 11-06, the orchestrator for BigMoE's C3-R8): one one-call's post-drop work, lazily (no sync): int32
    [uses kept, uses dropped, groups computed = distinct base slots after the drop over the call's rows, uses the split
    dropped]. A dropped pair holds its row's rank-0 slot (3c QOM, the split) or -2 (NAX6-N 1C-Q); rank 0 is always kept.
    Same count as r57_screen's qcnt[2] on the two phases (sort, distinct, >= 0)."""
    assert rs_m.size == M * K, (rs_m.shape, M, K)
    rs = rs_m.reshape(M, K)
    r0 = rs[:, :1]
    dr = mx.concatenate([mx.zeros((M, 1), mx.bool_), (rs[:, 1:] < 0) | (rs[:, 1:] == r0)], axis=1)
    s2 = mx.sort(mx.where(dr, r0, rs).reshape(-1))
    f2 = mx.concatenate([mx.array([True]), s2[1:] != s2[:-1]])
    i32 = lambda a: a.astype(mx.int32)
    QCNT_1C.append(mx.stack([i32((~dr).sum()), i32(dr.sum()), i32((f2 & (s2 >= 0)).sum()),
                             i32(sdrop.sum()) if sdrop is not None else mx.array(0, mx.int32)]))

def _ooo_nax6(x, ids, bm, t, post, sc, shared):
    """C1: bigmoe.tier._ooo_call (BigMoE's OOO at e3210bf) with NAX6 in both phases: the resident pairs during the loader
    spin, then the rest, then the same LY=8 combine_phase. == _nax6_glu with scores, bitwise (INV)."""
    if CFG["ooo_1c"] and not (CFG["nax_hybrid"] or CFG["nax_repack_p1"]):
        y1c = _ooo_1c_call(x, ids, bm, t, post, sc, shared, nax=True)   # OOO-1C: one NAX6 call
        if y1c is not None:
            return y1c
        if NW_ON and CFG["neverwait"]:
            raise RuntimeError("REF2 NEVER-WAIT: the one-call declined a layer (NW needs it on every MoE layer of the pass)")
    if t.get("qom") is not None:                     # FAST-NAX declined (or off): the two phases below would ignore QOM
        raise RuntimeError("REF2: QOM on the NAX path needs FAST-NAX's one-call (ooo_1c + nax_1cq + nax_native, no hybrid)")
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
    qn = bool(_ON[0] and CFG["tier_nax_ooo"] and CFG["ooo_1c"] and CFG["nax_1cq"] and CFG["nax_native"]
              and not (CFG["nax_hybrid"] or CFG["nax_repack_p1"]))   # FAST-NAX (Thu 11-05): QOM on NAX6-N's one-call
    if t.get("qom") is not None and ((_ON[0] and CFG["tier_nax_ooo"] and not qn) or (_layout(t) in (1, 2) and not (_ON[0] and CFG["ooo_1c"]))):
        raise RuntimeError("REF2: QOM on the NAX path, or on native records without ooo_1c, is not built (those two phases "
                           "would ignore t['qom']); QOM runs on 3c (BigMoE's two phases or ref2's one-call) and 3c-N (one-call)")
    if _ON[0]:
        _native_guard(t)
        if CFG["tier_nax_ooo"]:
            return _ooo_nax6(x, ids, bm, t, post, sc, shared)
        if _native(t):
            if CFG["ooo_1c"]:                            # OOO-1C on native records: 3c-N in one call
                y1c = _ooo_1c_call(x, ids, bm, t, post, sc, shared, nax=False)
                if y1c is not None:
                    return y1c
                if NW_ON and CFG["neverwait"]:
                    raise RuntimeError("REF2 NEVER-WAIT: the one-call declined a layer (NW needs it on every MoE layer of the pass)")
            if t.get("qom") is not None:                 # the one-call declined: 3c-N's two phases would ignore QOM
                raise RuntimeError("REF2: QOM on native records needs the one-call path (ooo_1c, a gate with regions, t3)")
            return _ooo_3cn(x, ids, bm, t, post, sc, shared)
        if CFG["ooo_1c"]:                                # OOO-1C: BigMoE's 3c in one call
            y1c = _ooo_1c_call(x, ids, bm, t, post, sc, shared, nax=False)
            if y1c is not None:
                return y1c
    if NW_ON and CFG["neverwait"]:                       # NEVER-WAIT: every other route is a decline while NW is on
        raise RuntimeError("REF2 NEVER-WAIT: the one-call declined a layer (NW needs it on every MoE layer of the pass)")
    if not _ON[0] and _layout(t) in (1, 2):              # (== the original 'elif' of 'if _ON[0]')
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
    """R-61: MoonShots' exact fused GDN pieces (a frozen copy in /Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/ms_gdn): bitwise ==
    MTPLX's eager chain (theirs: S 1-8, model-level 4-row logits; Leap 2026-11-01: S 1-16, tensor level). Installed once:
    MTPLX's fused conv-norm entry points then run the exact kernel."""
    if _GDNX[0] is None:
        p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/ms_gdn")   # frozen copy of MoonShots' files (README there)
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
        mw = [_dn_mlx(p) for p in ps]                    # DENSE-N-RAM: converted modules unpacked (transient)
        self.n = [int(a[0].shape[0]) for a in mw]
        assert (self.n[0] + self.n[1]) % 64 == 0 and self.n[0] + self.n[1] >= CFG["dense_min_n"], self.n
        assert self.n[2] + self.n[3] < CFG["dense_min_n"], self.n
        cat = lambda a, b: tuple(mx.concatenate([a[i], b[i]]) for i in range(3))
        self.qz, self.ba = cat(mw[0], mw[1]), cat(mw[2], mw[3])
        mx.eval(self.qz, self.ba)
        if _DNR_ON[0]:
            self._convert()

    def _convert(self):
        """DENSE-N-RAM: the qkv | z concat replaced by its native copy (registered); False if already converted"""
        w = self.qz[0]
        e = _DNR.get(id(w))
        if (e is not None and e[0] is w) or w.ndim != 2:
            return False
        N, K = int(w.shape[0]), int(w.shape[1]) * 8
        wn, sn, bn = _dnr_arrays(*self.qz)
        _DNR[id(wn)] = (wn, N, K)
        self.qz = (wn, sn, bn)
        return True

    def _qz_mlx(self):
        e = _DNR.get(id(self.qz[0]))
        return _dn_unpack(*self.qz, e[1], e[2]) if e is not None and e[0] is self.qz[0] else self.qz

    def __call__(self, x):
        q = lambda w: mx.quantized_matmul(x, *w, transpose=True, group_size=self.gs, bits=self.bits, mode=self.mode)
        n0, n2 = self.n[0], self.n[2]
        if _ON[0] and CFG["gdn_inproj_x"] and _rows(x)[1] <= R:          # prefill widths keep the four calls (stock bits)
            COUNTS["gdn_inproj_x"] += 1
            prev, _DN_SRC[0] = _DN_SRC[0], self.qz[0]   # DENSE-N: the concat is resident (inert unless dense_native)
            try:
                yqz = q(self.qz)
            finally:
                _DN_SRC[0] = prev
            prev, _DN_SRC[0] = _DN_SRC[0], self.ba[0]   # SMALL-N: b | a is resident too (inert unless small_native)
            try:
                yba = q(self.ba)
            finally:
                _DN_SRC[0] = prev
            return yqz[..., :n0], yqz[..., n0:], yba[..., :n2], yba[..., n2:]
        sl = lambda w, a, e: tuple(t[a:e] for t in w)                    # off: the four stock calls on row slices
        qz = self._qz_mlx()                                              # DENSE-N-RAM: MLX's layout for the slices
        return (q(sl(qz, 0, n0)), q(sl(qz, n0, n0 + self.n[1])), q(sl(self.ba, 0, n2)),
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
    lz = _lz_form(cache)                               # LAZY COMMIT: 3 = [conv, base, pack], 2 = the true state, 0 = not ours
    if CFG["gdn_lazy_commit"] and lz:
        pend = cache.state[2] if lz == 3 else None
        y, q, k, v, cs, st, pack = _gbmod().gdn_body_lazy(qkv, z, a, b, cache[0], self.conv1d.weight, self.A_log,
                                                          self.dt_bias, cache[1], self.norm.weight, self.norm.eps,
                                                          _gdnx()._scale(), pending=pend)
        if Q._VERIFY_CAPTURE.get():                    # MTPLX's commit still validates and reads the stash (conv state)
            cache._mtplx_verify_rows = (qkv, q, k, v, a, b)
        cache.state = [cs, st, pack]                   # the post-window state = st (committed) + the pack's S rows
        cache._ref2_lz_c = (st, S)                     # for THIS window's commit only: _gdn_call clears it on any call
        cache.advance(S)
        COUNTS["gdnb"] += 1
        COUNTS["gdn_lazy"] += 1
        COUNTS["gdn_lazy_rows"] += 0 if pend is None else int(pend.shape[-2])
        return Q._linear(self.out_proj, y.reshape(B, S, -1))
    if lz == 3:                                        # the switch went off with rows pending: today's replay first
        gdn_lazy_flush(self, cache)
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
    if cache is not None and getattr(cache, "_ref2_lz_c", None) is not None:
        cache._ref2_lz_c = None                        # LAZY COMMIT: only the lazy body's own forward leaves it set
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
    if _lz_form(cache) == 3:                           # LAZY COMMIT: MTPLX's own forward reads cache[1] itself
        gdn_lazy_flush(self, cache)
    return _ORIG["gdn_call"](self, inputs, mask, cache)


def _lz_form(entry):
    """LAZY COMMIT: a pure-GDN cache entry's form. 2 = [conv, state], state the true recurrent state; 3 = [conv, base,
    pack], the true state = the pack's rows replayed on base; 0 = anything else (None, QSA caches, the PLE layer's 4
    slots). The arity is the whole protocol: MTPLX's snapshots take a view of every leaf and its restores install clones
    with the snapshot's arity (cache_state._lazy_state_view / _restore_state_preserving_container), so the pack follows
    the state through rollbacks with no hook and no reliance on array identity."""
    st = getattr(entry, "state", None)
    if not isinstance(entry, Q.ArraysCache) or type(st) is not list or len(st) not in (2, 3):
        return 0
    if len(st) == 3 and not (st[2] is not None and st[2].ndim == 3 and st[2].shape[-1] == _gbmod().PW):
        return 0
    return len(st)


def _lz_true(gdn, base, pack):
    """LAZY COMMIT: the true state of (base, pack) = MTPLX's own commit replay (mlx_lm gated_delta_update on the kept rows'
    q / k / v / a / b from the snapshot state), so a flush is today's computation on today's values"""
    from mlx_lm.models.gated_delta import gated_delta_update
    q, k, v, a, b = _gbmod().unpack_rows(pack)
    _, st = gated_delta_update(q, k, v, a, b, gdn.A_log, gdn.dt_bias, base, None, use_kernel=not gdn.training)
    return st


def gdn_lazy_flush(gdn, entry):
    """LAZY COMMIT: [conv, base, pack] -> [conv, true state] (lazy MLX ops, no eval); False when there was nothing to do"""
    if _lz_form(entry) != 3:
        return False
    conv, base, pack = entry.state
    entry.state = [conv, _lz_true(gdn, base, pack)]
    entry._ref2_lz_c = None
    COUNTS["gdn_lazy_flush"] += 1
    return True


def gdn_lazy_state(model, cache, flush=False):
    """LAZY COMMIT, for readers of the recurrent state (digests, dumps, a cache save): every layer's TRUE recurrent
    state (None for QSA layers); flush=True also puts the entries back in today's 2-leaf form. model: anything with
    .layers (Qwen4ExpTextModel) or .model.layers."""
    lays = getattr(model, "layers", None) or model.model.layers
    out = []
    for layer, e in zip(lays, cache):
        f = _lz_form(e)
        if f == 3:
            gdn = layer.linear_attn
            if flush:
                gdn_lazy_flush(gdn, e)
                out.append(e.state[1])
            else:
                out.append(_lz_true(gdn, e.state[1], e.state[2]))
        else:
            st = getattr(e, "state", None)
            out.append(st[1] if isinstance(e, Q.ArraysCache) and type(st) is list and len(st) > 1 else None)
    return out


def _lz_commit_pre(self, cache, snapshot_states, verified):
    """LAZY COMMIT, before MTPLX's commit: wins = the entries whose LAST forward was this window's lazy body (their
    replay by MTPLX is discarded afterwards, whatever it starts from). Every other pure-GDN entry whose snapshot is in
    the 3-leaf form gets its snapshot's TRUE state handed to MTPLX in place of the bare base (MTPLX replays from
    pre[1]); the caller's snapshot is not modified."""
    wins, ss = {}, None
    for i, (layer, e) in enumerate(zip(self.layers, cache)):
        w = getattr(e, "_ref2_lz_c", None) if e is not None else None
        if w is not None and w[1] == verified and _lz_form(e) == 3:
            wins[i] = w
            continue
        pre = snapshot_states[i] if snapshot_states is not None else None
        if (pre is not None and getattr(layer, "ple", None) is None and isinstance(pre, (list, tuple)) and len(pre) == 3
                and pre[2] is not None and getattr(pre[2], "ndim", 0) == 3 and pre[2].shape[-1] == _gbmod().PW):
            if ss is None:
                ss = list(snapshot_states)
            ss[i] = [pre[0], _lz_true(layer.linear_attn, pre[1], pre[2])]
    return (snapshot_states if ss is None else tuple(ss)), wins


def _lz_commit_post(cache, wins, ok, keep, verified):
    """LAZY COMMIT, after MTPLX's commit: on a partial accept MTPLX has set each entry's conv state (kept) and its
    recurrent state to a replay node (never evaluated: dropped here). The committed base goes back and the pack keeps
    its first `keep` rows (a view). A full accept changes nothing: base + the whole pack is the post-window state."""
    for i, (st, S) in wins.items():
        e = cache[i]
        if ok and keep < verified:
            e.state = [e.state[0], st, e.state[2][:, :keep]]
        e._ref2_lz_c = None

_BS = [None]
def _tier_row_groups(ids2, C, C_l, r4):
    """F24 row-chunking: contiguous row groups, in order, each with <= C distinct experts and (r4 = the 4-bit ranks'
    column range) <= C_l distinct 4-bit-rank experts; a single row (K <= C ids) always fits"""
    groups, a, ub, u4 = [], 0, set(), set()
    for i in range(ids2.shape[0]):
        rb = set(ids2[i].tolist())
        r4s = set(ids2[i, r4[0]:r4[1]].tolist()) if r4 else set()
        if i > a and (len(ub | rb) > C or len(u4 | r4s) > C_l):
            groups.append((a, i))
            a, ub, u4 = i, set(), set()
        ub |= rb
        u4 |= r4s
    groups.append((a, ids2.shape[0]))
    return groups

def _tier_slots(self, x, indices, bm):
    """F24 (CFG tier_chunk / REF2_TIER_CHUNK=1, default off; BigMoE Sat 18:5x): a call whose distinct experts exceed the
    base slots (or its 4-bit ranks' the sidecar slots) runs as contiguous row groups that fit, each through _tier_rows
    (its own ensure + kernel + eval, so a later group may evict an earlier group's experts), the rows written back in
    order. The kernel computes each (row, rank) pair in its own threadgroups, so every row keeps its bits. Never engages
    when the union fits (C 256: never). REF2_TIER_CHUNK_CAP=N (test only) groups at N to prove the bits at a C that fits."""
    t = bm["tier"]
    ids2 = np.array(indices).reshape(-1, indices.shape[-1])
    K = ids2.shape[-1]
    if CFG["tier_chunk"]:
        lo4 = [s_ for s_ in t["segs"] if s_[2] == 4]
        r4 = (lo4[0][0], lo4[-1][1]) if lo4 else None
        cb = min(bm["C"], CFG["tier_chunk_cap"]) if CFG["tier_chunk_cap"] else bm["C"]
        cl = min(t["C_l"], CFG["tier_chunk_cap"]) if CFG["tier_chunk_cap"] else t["C_l"]
        if len(np.unique(ids2)) > cb or (r4 and len(np.unique(ids2[:, r4[0]:r4[1]])) > cl):
            groups = _tier_row_groups(ids2, cb, cl, r4)
            x2 = x.reshape(-1, x.shape[-1])
            ys = [_tier_rows(self, x2[a:b], ids2[a:b], bm) for a, b in groups]
            COUNTS["tier_chunk"] += 1
            COUNTS["tier_chunk_groups"] += len(groups)
            return mx.concatenate(ys, axis=0).reshape(*x.shape[:-1], K, -1)
    return _tier_rows(self, x.reshape(-1, x.shape[-1]), ids2, bm).reshape(*x.shape[:-1], K, -1)

def _tier_rows(self, x, ids, bm):
    """R-58 REF2-T: a 5-16-row call on BigMoE's rank-tier pools (BIGMOE_TIER, e.g. 4433333333 with TIER_GLU 3c) runs its
    routed experts at the plan's tiers on pool slots, as the gated path's tier_call does for 1-4 rows: base-pool (MSB-3)
    slots for every rank, LSB-sidecar slots for the 4-bit ranks, MoonShots' fused 3c kernel. That kernel computes each
    (row, rank) pair in its own threadgroups, so a row's bits do not depend on the call's other rows. BigMoE's own
    _sync_call would run these calls as the exact 4-bit prefill instead (tier.prefill_call). x [n, D], ids [n, K] numpy
    -> y [n, K, Dout], evaluated (F24: _tier_slots's body, now per row group)."""
    from bigmoe import tier as T
    BS, t = _BS[0], bm["tier"]
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
    y = y.reshape(x.shape[0], K, -1)
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
        if CFG["fatt_r8"] and q.shape[2] == 16 and _ROWS[0] <= 8 and (q.shape[1] // k.shape[1]) % 2 == 0:
            COUNTS["fatt_r8"] += 1                    # F1-R8: <= 8 real rows (padded by _att): fs4's R 8 form on the first 8,
            y8 = _fatt16s()(q[:, :, :8], k, v, scale, mask[..., :8, :], ks=4)   # == the R 16 call's rows (B6 INV (e));
            return mx.concatenate([y8, mx.zeros_like(y8)], axis=2)            # the padding rows' outputs are discarded
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

_CARRY = [None]                                      # D5: (output, hyper, block_out, inject) of the last write

@mx.compile
def _nw_add_c(hyper, d, inject):
    """NEVER-WAIT's late term: hyper + d x inject, exactly hyper wherever the term is +-0 (so a warm pass stays bitwise)"""
    grouped = hyper.reshape(*hyper.shape[:-1], inject.shape[-1], d.shape[-1])
    t_ = d[..., None, :] * inject[..., :, None]
    return mx.where(t_ == 0, grouped, grouped + t_).reshape(*hyper.shape)


def _hc_write(hyper, block_out, inject):
    """R-73 (exact-safe, CFG hc_write): hyper + block_out x inject as ONE compiled kernel on <= 16-row bf16 calls
    (bitwise == the stock Multiply + Add: R-73, layer 3's real weights, 6 scales). D5 (CFG hc_pend_mlp): every write's
    operands are stashed for the next HC read of its output. NEVER-WAIT: the first write after a one-call is that layer's
    MLP-side write; its inject is kept for the next layer, and the previous layer's late term goes into hyper first (so
    the D5 carry still describes one write)."""
    if NW["after"] and NW_ON and CFG["neverwait"]:
        NW["after"] = False
        k_ = NW["k"] - 1
        NW["inj"][k_] = inject
        due = NW["due"]
        if due is not None:
            d_, kd = due
            NW["due"] = None
            hyper = _nw_add_c(hyper, d_.reshape(block_out.shape).astype(block_out.dtype), NW["inj"].pop(kd))
            COUNTS["nw_writes"] += 1
    lead, rows = _rows(hyper)
    if _ON[0] and CFG["hc_write"] and rows <= R and hyper.dtype == mx.bfloat16:
        COUNTS["hc_write"] += 1
        y = _hc_write_c(hyper, block_out, inject)
    else:
        y = _ORIG["hc_write"](hyper, block_out, inject)
    if _ON[0] and CFG["hc_pend_mlp"]:
        _CARRY[0] = (y, hyper, block_out, inject)
    return y

_IN_COMMIT = [False]                                   # PLAN 220 seams (d) / (e): inside MTPLX's commit
_PLE_DEFER = [None]                                    # (d): a forward's deferred PLE stage (the embedding, its args)
_CSUB = []                                             # (e): the commit's replayed GDN states not sent yet

def _dec_call(self_, x, *a, **k):
    """PLAN 220 seams (b) / (d) (CFG sb_first / ple_late; installed only when one is on). At the PLE-carrying layer
    (layer 1 here; the only DecoderLayer with .ple) of a 5-16-row pass outside the commit: (b) layer 0's output goes to
    the GPU first (mx.async_eval: MTPLX's cadence sends after layer 1 with SB 2), so the GPU works while Python stages
    and builds; (d) the PLE rows MTPLX's _forward deferred are staged now, with the same inputs and the same history
    (layer 0 does not touch layer 1's cache), so the same values. Scheduling only: bitwise."""
    if _ON[0] and getattr(self_, "ple", None) is not None and not _IN_COMMIT[0] and 5 <= x.shape[1] <= 16:
        if CFG["sb_first"]:
            mx.async_eval(x)
            COUNTS["sb_first"] += 1
        if _PLE_DEFER[0] is not None:
            emb_, args_ = _PLE_DEFER[0]
            _PLE_DEFER[0] = None
            _ORIG["ple_stage"](emb_, *args_)
            COUNTS["ple_late"] += 1
    return _ORIG["dec_call"](self_, x, *a, **k)

def _ple_stage(self_, input_ids, cache, state_idx):
    """PLAN 220 seam (d): MTPLX's _forward stages the PLE rows before layer 0's graph (memmap gathers + a page-touch
    pool: GPU idle); a 5-16-row pass outside the commit defers it to the PLE layer's call (_dec_call)"""
    if _PLE_DEFER[0] is not None:                      # a deferral its forward never consumed: drop it (that forward took
        _PLE_DEFER[0] = None                           # MTPLX's exact graph path for the rows)
        COUNTS["ple_late_stale"] += 1
    if _ON[0] and CFG["ple_late"] and not _IN_COMMIT[0] and 5 <= input_ids.shape[1] <= 16:
        _PLE_DEFER[0] = (self_, (input_ids, cache, state_idx))
        return None
    return _ORIG["ple_stage"](self_, input_ids, cache, state_idx)

def _gdu(*a, **k):
    """PLAN 220 seam (e): mlx_lm's gated_delta_update; inside the commit, every CFG commit_sub replays' new states are
    sent to the GPU (mx.async_eval) while Python builds the rest. Scheduling only: bitwise."""
    out = _ORIG["gdu"](*a, **k)
    if _IN_COMMIT[0] and CFG["commit_sub"]:
        _CSUB.append(out[1])
        if len(_CSUB) >= CFG["commit_sub"]:
            mx.async_eval(*_CSUB)
            _CSUB.clear()
            COUNTS["commit_sub"] += 1
    return out

def _commit(self, cache, snapshot_states, *, keep_tokens, verified_tokens):
    """R-40 STEP 3t: the commit's lone PLE-layer replay must run in the verify forward's phase. MTPLX's attention_gate
    (the PLE gate) and GatedResidual's inject lower sigmoid differently in decode_verify (the compiled fused form) than
    elsewhere (the stock one); the two differ in the last bit at some bf16 inputs, so a replay outside the verify
    scope left layer 1's state off the AR's at rare positions (prealgebra/1640, relative 39-40). REF2's AR and its
    checks run in decode_verify; a 1-row AR commit is a full accept and replays nothing."""
    lazy = any(e is not None and (getattr(e, "_ref2_lz_c", None) is not None or _lz_form(e) == 3) for e in cache) or \
        any(isinstance(p, (list, tuple)) and len(p) == 3 for p in (snapshot_states or ()))
    if lazy:                                           # LAZY COMMIT (off: no entry is ever in the 3-leaf form)
        snapshot_states, wins = _lz_commit_pre(self, cache, snapshot_states, int(verified_tokens))
    _IN_COMMIT[0] = True                               # PLAN 220 seams (d) / (e)
    try:
        if not _ON[0] or not CFG["commit_phase"]:
            ok = _ORIG["commit"](self, cache, snapshot_states, keep_tokens=keep_tokens, verified_tokens=verified_tokens)
        else:
            from mtplx.attention_context import attention_phase, model_forward_kind
            with attention_phase("decode_verify"), model_forward_kind("target_verify"):
                ok = _ORIG["commit"](self, cache, snapshot_states, keep_tokens=keep_tokens, verified_tokens=verified_tokens)
    finally:
        _IN_COMMIT[0] = False
        if _CSUB:                                      # (e): the last group goes with the caller's own submit
            _CSUB.clear()
    if lazy:
        _lz_commit_post(cache, wins, ok, int(keep_tokens), int(verified_tokens))
    return ok

def _gdk(q, k, v, g, beta, state, mask=None):
    """PLAN 220 1a (CFG gdn_tg / REF2_GDN_TG=1, installed only then): mlx_lm's gated_delta_kernel for 1-16 rows as
    gdn_step_r's TG 2 form (every row's inputs in threadgroup memory first, 2 dv rows per simdgroup, one kernel per T).
    Bitwise == mlx_lm's kernel (y and state) on every T 1-16 at the engine's shapes (gdn_step_r.check_tg_all: 48 / 48);
    -9 % per 16-row call in a chain (61.9 vs 68.3 us; ~-0.23 ms per check). A mask, T > 16, a vector g or other dtypes
    take mlx_lm's own kernel. COUNTS gdn_tg."""
    if (_ON[0] and CFG["gdn_tg"] and mask is None and g.ndim == 3 and 1 <= k.shape[1] <= 16 and q.dtype == mx.bfloat16
            and k.dtype == q.dtype and v.dtype == q.dtype and beta.dtype == q.dtype and g.dtype == mx.float32
            and state.dtype == mx.float32 and tuple(k.shape[2:]) == (16, 128) and tuple(v.shape[2:]) == (48, 128)):
        import gdn_step_r
        COUNTS["gdn_tg"] += 1
        return gdn_step_r.kernel_tg(q, k, v, g, beta, state, R=2)
    return _ORIG["gdk"](q, k, v, g, beta, state, mask)

def apply():
    """install the patches (idempotent) and switch them on"""
    if not _ORIG:
        _ORIG.update(qmm=mx.quantized_matmul, lin=nn.Linear.__call__, gr=Q.GatedResidual.__call__,
                     att=Q.Attention.__call__, switch=Q._FusedGateUpSwitchGLU.__call__, swbase=SL.SwitchGLU.__call__)
        mx.quantized_matmul = _qmm
        nn.Linear.__call__ = _lin
        _ORIG["qlin"] = nn.QuantizedLinear.__call__      # DENSE-N: marks resident weights (inert unless dense_native)
        nn.QuantizedLinear.__call__ = _qlin
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
        if CFG["sb_first"] or CFG["ple_late"]:         # PLAN 220 seams (b) / (d): only when switched on
            _ORIG["dec_call"] = Q.DecoderLayer.__call__
            Q.DecoderLayer.__call__ = _dec_call
        if CFG["ple_late"]:
            _ORIG["ple_stage"] = Q.NGramEmbedding.stage
            Q.NGramEmbedding.stage = _ple_stage
        if CFG["commit_sub"]:                          # PLAN 220 seam (e)
            import mlx_lm.models.gated_delta as _GDU
            _ORIG["gdu"] = _GDU.gated_delta_update
            _GDU.gated_delta_update = _gdu
        if CFG["gdn_tg"]:                              # PLAN 220 1a: only when switched on (the default path untouched)
            import mlx_lm.models.gated_delta as _GD
            import gdn_step_r as _GSR
            _ORIG["gdk"] = _GD.gated_delta_kernel
            _GD.gated_delta_kernel = _gdk
            _GSR.precompile_tg()                       # T 1-16 built now, not inside the first checks
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
    if CFG["keep_caches"]:                           # KEEP-C (Sat 11-07, Leap S2c): r57 calls remove() before every draft and
        _CARRY[0] = None                             # apply() before every check; the per-weight caches below are
        return                                       # identity-validated (e[0] is w), so keeping them changes no bit
    _DN.clear()                                      # DENSE-N: drop the native copies (and their weight references)
    _HC8.clear()                                     # HC8-1P: drop the cached parameter views
    _SN.clear()                                      # SMALL-N: drop the padded copies
    _CARRY[0] = None                                 # D5: no carried write across apply / remove

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
