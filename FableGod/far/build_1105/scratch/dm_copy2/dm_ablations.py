"""Perturbations for the decision map. Each setup registers ACTIVE[kind] = post(layer_idx, self, args, y) -> y'
(or ACTIVE[kind+"_pre"] = pre(layer_idx, self, args) -> args') for kinds att, gdn, hc, and PERT (dm_common) for the MoE
block (routed experts + shared expert; see dm_common's WIRED note A5).
WIRED (Breakthrough, Mon 2026-11-02; Fable's original in dm_ablations.py.orig):
- topk{k}: SCORE masking, not index masking (the block's ids are in argpartition order): ranks >= k get weight 0 and the
  kept experts keep the model's own weights ("mass"); topk{k}r also rescales the kept weights to the row's original routed
  weight ("renorm"). Q1 = topk2 (and topk2r) on code-sure / code-mid.
- hc_static: every HC read uses its gates' mean over the chunk's positions (per stream and channel) instead of the
  per-token learned gates, on the normed streams, as the model's own read; the input scale stays the model's.
  hc_mean_raw: Fable's literal version (the plain mean of the 4 written residual streams, no norm, no gates).
- shared_zero, moe reuse / fake-quant: inside the MoE block wrapper (routed part only for reuse and quant).
- *_q4 on att / gdn / hc: 4-bit fake quantisation of the layer OUTPUT (per row, max-abs scale), as Fable wrote it:
  an activation test, not the weight test HC4 was at 12 GB."""
import os
import mlx.core as mx
import mlx.nn as nn
from dm_common import ACTIVE, PREV, LAYERS, PERT, _shift, _fq

def topk_experts(k, renorm=False):
    def setup(): PERT.update(topk=k, renorm=renorm)
    return setup

def hc_static():
    def post(li, self, a, y):
        if not isinstance(y, tuple): return y
        hyper = y[1]                                     # the written stream the read used
        hc, D = self.hc_count, self.hidden_size
        normed = self.hc_norm(hyper)
        mix = nn.silu(self.input_mix_weight_down(normed) / hc)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], hc, D)
        mix = mx.broadcast_to(mix.mean(axis=-3, keepdims=True), mix.shape) if mix.ndim >= 3 else mix
        grouped = normed.reshape(*normed.shape[:-1], hc, D)
        return (mx.mean(mix * grouped, axis=-2).astype(y[0].dtype),) + tuple(y[1:])
    def setup(): ACTIVE["hc"] = post
    return setup

def hc_mean_raw():
    def post(li, self, a, y):
        if not isinstance(y, tuple): return y
        hyper = y[1]
        mean = hyper.reshape(*hyper.shape[:-1], self.hc_count, self.hidden_size).mean(axis=-2)
        return (mean.astype(y[0].dtype),) + tuple(y[1:])
    def setup(): ACTIVE["hc"] = post
    return setup

def shared_zero():
    def setup(): PERT.update(shared_zero=True)
    return setup

def reuse_prev(kind, layers=None, lag=1, chain=False, chain_max=0):
    if kind == "moe":                                    # lag / chain (/ bounded chain): dm_common._reuse (night 2)
        def setup(): PERT.update(reuse="all" if layers is None else set(layers), lag=lag, chain=chain,
                                 chain_max=chain_max)
        return setup
    assert lag == 1 and not chain, "lag / chain are wired for the MoE block only"
    def post(li, self, a, y):
        if layers is not None and li not in layers: return y
        z = y[0] if isinstance(y, tuple) else y
        shifted = _shift(z, -2)
        return (shifted,) + tuple(y[1:]) if isinstance(y, tuple) else shifted
    def setup(): ACTIVE[kind] = post
    return setup

def fake_quant(kind, bits, layers=None):
    if kind == "moe":
        def setup(): PERT.update(q=bits)
        return setup
    def post(li, self, a, y):
        if layers is not None and li not in layers: return y
        return (_fq(y[0], bits).astype(y[0].dtype),) + tuple(y[1:]) if isinstance(y, tuple) else _fq(y, bits).astype(y.dtype)
    def setup(): ACTIVE[kind] = post
    return setup

def gate_k(spec):
    """DAY9 Q1 / Q2 (Breakthrough, Mon 2026-11-02): per-row expert counts from the REALIZABLE gate (dm_gate.py: DFlash2's
    cf for the draft each row verifies), the kept experts at the model's own weights (mass, no renorm)."""
    def setup(): PERT.update(gate=spec, renorm=False)
    return setup

def hc8w(ple=True, bits=8, group_size=64):
    """DAY9 Q3: the HC mixers (input_mix_weight_down / up of every trunk GatedResidual) and, with ple, the PLE key / value
    projections at MLX affine 8-bit, group 64: the 12 GB line's HC8 (~/Breakthrough/v3n/hc8.py's recipe) as a WEIGHT
    change. Only the new modules are evaluated (the streamed model is never forced whole); clear() restores bf16.
    Quantized mixers fail GatedResidual._fused_read_applies (it checks for scales), so reads take the eager chain."""
    def setup():
        from dm_common import RT, TEARDOWN, Q
        rt = RT["rt"]; model = rt.model if hasattr(rt, "model") else rt
        TM = getattr(model, "language_model", model); undo = []
        for name, mod in TM.named_modules():
            if name.startswith("mtp"): continue
            tg = (["input_mix_weight_down", "input_mix_weight_up"] if isinstance(mod, Q.GatedResidual) else
                  ["key_proj", "value_proj"] if ple and isinstance(mod, Q.PLELayer) else [])
            for attr in tg:
                lin = getattr(mod, attr)
                if not isinstance(lin, nn.Linear) or lin.weight.shape[1] % group_size: continue
                q = nn.QuantizedLinear.from_linear(lin, group_size, bits); mx.eval(q.parameters())
                setattr(mod, attr, q); undo.append((mod, attr, lin))
        def restore():
            for mod, attr, lin in undo: setattr(mod, attr, lin)
        TEARDOWN.append(restore)
        assert len(undo) >= 96, f"hc8w swapped only {len(undo)} linears"
        print(f"[dm] hc8w: {len(undo)} linears at {bits} bits, group {group_size} (ple {ple})", flush=True)
    return setup

def gdn_state_bf16():
    def setup(): raise NotImplementedError("gdn_state_bf16 needs the GDN cache patch (A3); not wired")
    return setup

ABLATIONS = {
    "topk2": topk_experts(2), "topk2r": topk_experts(2, renorm=True), "topk5": topk_experts(5), "topk8": topk_experts(8),
    "topk10": topk_experts(10),                          # identity through the wrapper (a self-check: 0 flips expected)
    "hc_static": hc_static(), "hc_mean_raw": hc_mean_raw(), "shared_zero": shared_zero(),
    "moe_reuse_prev_all": reuse_prev("moe"), "att_reuse_prev": reuse_prev("att"), "gdn_reuse_prev": reuse_prev("gdn"),
    "moe_q4_all": fake_quant("moe", 4), "hc_q4": fake_quant("hc", 4), "att_q4": fake_quant("att", 4), "gdn_q4": fake_quant("gdn", 4),
    "gdn_state_bf16": gdn_state_bf16(),
    "gate_bands": gate_k(("bands",)), "hc8w": hc8w(), "hc8w_mix": hc8w(ple=False),
}
def pframe_map(q1=False):
    """The layer-wise reuse map (Fable A's ASSUMPTION_HUNT.md, the orchestrator 22:5x; FAST): the previous position's routed
    output reused in layers 8-11 on EVERY answer row and in layers 12-35 on rows the drafter certifies sure (cf >= 0.999),
    both as a bounded chain (S = 1: a reused output is the previous row's freshly computed one; consecutive reusers
    alternate with a refresh); q1 stacks DAY9's Q1 (gate_top2_cf099: top-2 on cf >= 0.99 rows, mass)."""
    def setup():
        PERT.update(reuse=set(range(8, 36)), chain=True, chain_max=1, chain_gate=0.999, chain_all=set(range(8, 12)))
        if q1: PERT.update(gate=("top", 0.99, 2), renorm=False)
    return setup

ABLATIONS["pframe_map"] = pframe_map(); ABLATIONS["pframe_map_q1"] = pframe_map(q1=True)
def q1_union(tag):
    """Q1-UNION (Fable C's AUDIT_350.md section 4, the orchestrator 23:2x): sure rows keep ranks 0-1 + the rank 2-9 experts
    already in their check's union (dm_q1u_masks.py -> out/<tag>/<window>.npz), the model's own weights (mass)."""
    def setup(): PERT.update(rmask=tag, renorm=False)
    return setup
ABLATIONS["q1u_cf099"] = q1_union("q1u_cf099"); ABLATIONS["q1u_cf09"] = q1_union("q1u_cf09")
for _arm in ("oea_k8", "oea_k6", "hyb_0.9_u8", "hyb_0.9_u6"):   # NewIdeas Tue 07:0x: OEA (arXiv 2511.02237) and the union stack
    ABLATIONS[_arm] = q1_union(_arm)                      # OUT/<arm>: dm_q1u_masks.py <arm> (= NewIdeas' masks bit for bit)
def hyb_t3(arm, thr):
    """<arm>_t3 (the orchestrator Tue 07:1x, FLOOR step 5): the hybrid's sure rows (cf >= thr) take ranks 0-1 from records3
    at 3 bits (dm_t3, the gate ("t3", thr)) and their union-member ranks 2-9 at 4 bits (OUT/<arm>_t3 masks, ranks 0-1
    off on sure rows: dm_q1u_masks.py <arm>:t3); the other rows as the hybrid."""
    def setup(): PERT.update(rmask=arm + "_t3", gate=("t3mask",), t3=3, t3k=2, renorm=False)   # t3 rows from the mask
    return setup
ABLATIONS["hyb_0.9_u8_t3"] = hyb_t3("hyb_0.9_u8", 0.9); ABLATIONS["q1u_cf09_t3"] = hyb_t3("q1u_cf09", 0.9)   # NewIdeas 07:3x
ABLATIONS["hyb_cold4_t3"] = hyb_t3("hyb_cold4", 0.9)   # Tue 19:xx: Fable C's COLD-CUT screen (UNION_STACK sec 5), dm_cold_masks.py
ABLATIONS["cold6_all"] = q1_union("cold6_all")          # its chat form (every row: ranks 0-5 + popular 6-9 + members)
ABLATIONS["hyb_0.9_u8_live_t3"] = hyb_t3("hyb_0.9_u8_live", 0.9)   # its control: hyb_0.9_u8_t3 at live checks (dm_cold_masks selfcheck)
ABLATIONS["hyb_cold4_lowo_t3"] = hyb_t3("hyb_cold4_lowo", 0.9)   # Tue 19:5x (0a565a6): COLD with an in-domain leave-one-window-out prior
ABLATIONS["hyb_cold4_pmt_t3"] = hyb_t3("hyb_cold4_pmt", 0.9)     # Tue 20:2x (4236037): the table = the request's own prompt
ABLATIONS["hyb_cold4_pmtc_t3"] = hyb_t3("hyb_cold4_pmtc", 0.9)   # the same OR cap45 (a corpus table + the request's footprint)
ABLATIONS["hyb_cold4_len_t3"] = hyb_t3("hyb_cold4_len", 0.9)     # Tue 22:3x (12b3d63): pmt's table if the prompt has >= 128 rows, else pmtc's
for _arm in ("top_k2", "oea_k2", "oea_k1"):            # NewIdeas Tue 09:10 (37f8e37, QUEUE.md "P1-OEA"): all-rows pass-1 arms
    ABLATIONS["p1_" + _arm] = q1_union(os.path.expanduser(f"~/NewIdeas/data/dm_masks/{_arm}"))   # its masks, read-only
for _arm in ("p1c_oea_k1_p08", "p1c_top_k2_p08", "q1u_0.9_live", "reach_0.9_0.5_live", "top_k8"):   # NewIdeas 10:24 / 10:38 / 11:08
    ABLATIONS[_arm] = q1_union(os.path.expanduser(f"~/NewIdeas/data/dm_masks/{_arm}"))
for _arm in ("qom_0.9_live", "qom_0.5_live", "qom_0_live"):   # NewIdeas Tue 17:01 (QUEUE 554dfe0): Fable C's Q1-ON-MISS, live-cut checks
    ABLATIONS[_arm] = q1_union(os.path.expanduser(f"~/NewIdeas/data/dm_masks/{_arm}"))
ABLATIONS["res_c256"] = q1_union("res_c256")   # Tue 17:3x (SCOREBOARD 4229db8): the resident-only picker, dm_res_masks.py
def t3_all(bits):
    """t3_all / t4_all (the orchestrator 23:3x; 45 GB's plan 3333333333 = the 1-bit sidecar dropped): EVERY row (prompt and
    answer) takes all ten routed experts from records3 at `bits` (3: MSB-3; 4: the control, rebuilt 4-bit through the
    same path), the model's own weights; the stock routed output is zeroed (per-row k 0 everywhere)."""
    def setup(): PERT.update(gate=("t3all",), renorm=False, t3=bits, t3k=10)
    return setup
ABLATIONS["t3_all"] = t3_all(3); ABLATIONS["t4_all"] = t3_all(4)
def t3_top2_all():
    """The 3-bit pass 1's certificate run (Tue 10:5x; Fit12's item 4): every row's top-2 from records3 at 3 bits (the
    stock routed output zeroed), as topk2 is for the 4-bit pass 1 (all rows cut, mass)."""
    def setup(): PERT.update(gate=("t3all",), renorm=False, t3=3, t3k=2)
    return setup
ABLATIONS["t3_top2_all"] = t3_top2_all()
def t3cert(thr):
    """The 3-bit two-pass mixed run: rows whose own p1 under the 3-bit top-2 cut (t3_top2_all) is >= thr take their top-2
    from records3 at 3 bits; the rest exact (4-bit, all ten)."""
    def setup(): PERT.update(gate=("t3cert", "t3_top2_all", thr), renorm=False, t3=3, t3k=2)
    return setup
ABLATIONS["t3cert_top2_p08"] = t3cert(0.8)
def p1_t3(mask_dir):
    """The 12 GB pass-1 stack (NewIdeas 14:52, the orchestrator 14:5x): every row the mask cuts takes its KEPT ranks from
    records3 at 3 bits (dm_t3 routed_topk K 10 with the mask's ranks), nothing from the stock path; uncut rows exact."""
    def setup(): PERT.update(rmask=mask_dir, gate=("t3keep",), t3=3, t3k=10, t3keep=True, renorm=False)
    return setup
ABLATIONS["p1_oea_k1_ku4_t3"] = p1_t3(os.path.expanduser("~/NewIdeas/data/dm_masks/oea_k1_ku4"))
for g in range(4):                                   # per-group MoE reuse (12 layers each), then per layer on night 2
    ABLATIONS[f"moe_reuse_prev_g{g}"] = reuse_prev("moe", set(range(12 * g, 12 * g + 12)))
for L in range(48):
    ABLATIONS[f"moe_reuse_prev_L{L:02d}"] = reuse_prev("moe", {L})

import re
def ablation(name):
    """ABLATIONS[name], or night 2's MoE reuse family by name (NewIdeas stage 1, Mon 2026-11-02):
    moe_reuse_{prev|lag<k>|chain[<S>]}_L<a>_<b>[_topk<k>] reuses the routed output on layers a..b inclusive: prev =
    lag 1, lag<k> = copy from row t - k, chain = sure rows (baseline class) take the last computed row's output, chain<S>
    = the same with a refresh so no output is reused more than S rows on (stage 2); _topk<k> stacks top-k (mass) on
    every row and layer; e.g. moe_reuse_prev_L12_35, moe_reuse_lag4_L12_35, moe_reuse_chain1_L08_35_topk2."""
    if name in ABLATIONS: return ABLATIONS[name]
    if "+" in name:                                      # FLOOR night stacks (Tue 07:0x): a+b runs both setups in order
        parts = [ablation(p) for p in name.split("+")]
        def stack():
            for p in parts: p()
        return stack
    g = re.fullmatch(r"gate_top(\d+)_cf(0\d+)(?:_L(\d+)_(\d+))?", name)   # DAY9 Q1: gate_top2_cf099; _L08_35: those layers only
    if g:
        base = gate_k(("top", float("0." + g.group(2)[1:]), int(g.group(1))))
        if not g.group(3): return base
        lay = set(range(int(g.group(3)), int(g.group(4)) + 1))
        def layered(): base(); PERT.update(k_layers=lay)  # e.g. gate_top0_cf0999_L08_35: skip (k 0) the sure rows there
        return layered
    c = re.fullmatch(r"cert_top(\d+)_p(0\d+)", name)          # mission 20's two-pass check: e.g. cert_top2_p099, cert_top5_p09
    if c: return gate_k(("cert", "topk" + c.group(1), float("0." + c.group(2)[1:]), int(c.group(1))))
    c = re.fullmatch(r"acert_top(\d+)_p(0\d+)_p(0\d+)_cf(0\d+)", name)   # the adaptive certificate: e.g. acert_top2_p07_p09_cf099
    if c: return gate_k(("acert", "topk" + c.group(1), *(float("0." + c.group(i)[1:]) for i in (2, 3, 4)), int(c.group(1))))
    t = re.fullmatch(r"gate_top2t([34])_cf(0\d+)", name)      # M-18: gated rows' top-2 from records3 at 3 bits (t4 = control)
    if t:
        spec = ("t3", float("0." + t.group(2)[1:]))
        def setup(): PERT.update(gate=spec, renorm=False, t3=int(t.group(1)))
        return setup
    m = re.fullmatch(r"moe_reuse_(prev|lag(\d+)|chain(\d*))_L(\d+)_(\d+)(?:_topk(\d+))?(?:_dcf(0\d+))?", name)
    if not m: raise KeyError(name)
    lo, hi = int(m.group(4)), int(m.group(5)); assert 0 <= lo <= hi < 48, name
    base = reuse_prev("moe", set(range(lo, hi + 1)), lag=int(m.group(2) or 1), chain=m.group(1).startswith("chain"),
                      chain_max=int(m.group(3) or 0))
    if m.group(7):                                       # _dcf<thr>: the chain skips by the drafter's cf (realizable)
        assert m.group(1).startswith("chain"), name
        b0 = base; thr = float("0." + m.group(7)[1:])
        def base(): b0(); PERT.update(chain_gate=thr)
    if not m.group(6): return base
    def setup(): base(); PERT.update(topk=int(m.group(6)), renorm=False)
    return setup
for _arm in ("qom_0.5_live", "qom_0_live"):   # NewIdeas Tue 23:02 (QUEUE.md): QOM with ranks 0-1 of sure rows at 3 bits
    ABLATIONS[_arm + "_t3"] = hyb_t3(os.path.expanduser(f"~/NewIdeas/data/dm_masks/{_arm}"), 0.9)   # rmask <arm>_t3 (its masks)
ABLATIONS["qom_0.5_live_c164"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.5_live_c164"))   # NewIdeas 23:21 (fe52587): QOM at C 160 pools
ABLATIONS["qom_0_live_bonus"] = q1_union(os.path.expanduser("~/FableGod/fablec/masks/qom_0_live_bonus"))   # Fable C BYTES10 screen 1(a) (the orchestrator Wed 04:0x): qom_0_live, every bonus row gated
def t3k_mask(path, k):   # NewIdeas screen 2 (QUEUE.md:1181-1185, the orchestrator Wed 04:08): ranks < k of the gated rows at 3 bits (dm_t3), the rest from the masks
    def setup(): PERT.update(rmask=path, gate=("t3mask",), t3=3, t3k=k, renorm=False)
    return setup
ABLATIONS["hyb_0.9_k1_u10_live_t3"] = t3k_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/hyb_0.9_k1_u10_live_t3"), 1)   # sure rows: rank 0 only
ABLATIONS["q1u_0.9_live_t3"] = t3k_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/q1u_0.9_live_t3"), 2)   # its top-2 twin
ABLATIONS["qom_0_live_bq"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_bq"))   # NewIdeas: screen 1 in QOM's form (QUEUE.md:1214-1222; the orchestrator Wed 04:47): the gated last row also keeps resident ranks 2-9
ABLATIONS["qom_0_live_bq_t3"] = hyb_t3(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_bq"), 0.9)   # its 3-bit form (masks qom_0_live_bq_t3, t3k 2); pre-approved only if bq is not KILL (NewIdeas' read-out)
def tbits_mask(path, bits, k):   # Wed 06:3x (the orchestrator 06:2x): ranks < k of the gated rows from records3 at `bits` (dm_t3), the rest from <path>'s masks
    def setup(): PERT.update(rmask=path, gate=("t3mask",), t3=bits, t3k=k, renorm=False)
    return setup
ABLATIONS["qom_0_k1_live_t3"] = tbits_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_k1_live_t3"), 3, 1)   # NewIdeas screen 2b (QUEUE.md:1189 / 1250): QOM cf 0, sure rows top-1 at 3 bits
ABLATIONS["qom_0_live_bq_t2"] = tbits_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_bq_t3"), 2, 2)   # NewIdeas (QUEUE.md:1305-1320): bq_t3's masks, sure rows' ranks 0-1 at 2 bits (to2bit)
ABLATIONS["qom_0_live_bq_c147_t3"] = tbits_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_bq_c147_t3"), 3, 2)   # NewIdeas screen 4 (C) (QUEUE.md:1343-1390; the orchestrator Wed 07:5x): qom_0_live_bq_t3 at the 24 GB pool (147 base slots)
ABLATIONS["qom_0_k1_live_bq_c147_t3keep"] = p1_t3(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_k1_live_bq_c147"))   # screen 4 (B'), the cheap top-1 form: cut rows wholly from records3 at 3 bits, the 24 GB pool
ABLATIONS["qom_0_live_ball_c147_t3"] = tbits_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_ball_c147_t3"), 3, 2)   # NewIdeas screen 6 (QUEUE.md 14:0x; the orchestrator 14:3x): DRAFT17's quality at the 24 GB pool, C with every last row gated
def lsrc_mask(path, srcs, bits=3, k=2):
    """Map screen 7 (NewIdeas QUEUE.md 14:1x + amendment 3de406e; the orchestrator 14:3x): tbits_mask(path, bits, k) on every
    layer, except that at the layers of srcs {layer: src} the gated rows take EVERY kept expert (ranks < k + the mask's kept
    ranks k-9) from src and post nothing on the stock path (dm_t3.routed_kept): src = "trl:<form>" (Fit12's TRL view; refused
    unless all 512 experts of each such layer are in the store, or DM_TRL_FALLBACK=1, counted) or an int (records3 at those
    bits through dm_t3)."""
    def setup():
        PERT.update(rmask=path, gate=("t3mask",), t3=bits, t3k=k, renorm=False, t3L=dict(srcs))
        for s in sorted({v for v in srcs.values() if isinstance(v, str)}):
            import dm_t3
            miss = dm_t3.trl_gate(sorted(L for L, v in srcs.items() if v == s), s.split(":", 1)[1])
            print(f"[dm] {s}: " + (f"FALLBACK to records3 MSB-3 for missing experts {miss}" if miss else
                                   f"all 512 experts present at layers {sorted(L for L, v in srcs.items() if v == s)}"), flush=True)
    return setup
_S7M = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_ball_c239_t3"); _S7 = (7, 18, 30, 41)   # screen 7's masks and S
ABLATIONS["s7_ref"] = tbits_mask(_S7M, 3, 2)                          # 7-ref: C + DRAFT17 at the mix's 239-slot pool (C's form)
ABLATIONS["s7_trl"] = lsrc_mask(_S7M, {L: "trl:mix" for L in _S7})    # 7-trl: at S every kept expert from the TRL mix (2.40 b)
ABLATIONS["s7_cal"] = lsrc_mask(_S7M, {L: 2 for L in _S7})            # 7-cal: at S every kept expert at MSB-2 (to2bit)
ABLATIONS["s7_cal_all"] = lsrc_mask(_S7M, {L: 2 for L in range(48)})  # 7-cal-all: the same at all 48 layers
for _arm in ("q1u_0.9_live_bonus9g", "qom_0.5_live_bonus5g"):   # Fable C screens 1(b) / 1(c) (BYTES10.md:255-260; the orchestrator Wed 15:3x): the gate-file-cf rebuilds, NEVER the 00:34 _bonus9 / _bonus5
    ABLATIONS[_arm] = q1_union(os.path.expanduser(f"~/FableGod/fablec/masks/{_arm}"))   # union form, as qom_0_live_bonus (1(a)); twins q1u_0.9_live / qom_0.5_live
ABLATIONS["s7_ref3"] = lsrc_mask(_S7M, {L: 3 for L in range(48)})    # NewIdeas 14:5x: 7-ref3, every kept expert of the gated rows at MSB-3 at all 48 layers (the engine's FAST-pool frame)
# NewIdeas 16:5x: map arm for Leap's DENSE-3 (the orchestrator's FAST dense diet (b)); paste into
# ~/FableGod/decision_map/dm_ablations.py after the s7_ref3 line. Arm name: qom_0_live_ball_c147_t3+dense3
# (screen 6's FAST code stack + DENSE-3, via ablation()'s "+" stack); twin = screen 6 (qom_0_live_ball_c147_t3).
def dense3(which=None):
    """Leap's DENSE-3 sim (~/Leap/scripts/dense3.py): the trunk's dense 4-bit g64 projections (GDN in / out, attention
    q / k / v / o) at their MSB-3 values in the 4-bit format (codes & 0xEEEEEEEE, biases bf16(b + s / 2)); the router,
    shared expert, indexer, experts, embed and head untouched. Restored by clear() (TEARDOWN), as hc8w."""
    def setup():
        import sys
        assert os.environ.get("BIGMOE_PKD1") != "1" and os.environ.get("BIGMOE_SCOUT_BUNDLE") != "1", \
            "dense3: a fusion repacked the projections at load; the sim would not reach the forward"
        sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
        import dense3 as D3
        from dm_common import RT, TEARDOWN
        rt = RT["rt"]; model = rt.model if hasattr(rt, "model") else rt
        W = which or D3.WHICH
        saved = [(m, m["weight"], m["biases"]) for _, m in D3._targets(model, W)]
        n, mb = D3.sim(model, W)
        def restore():
            for m, w, b in saved: m.weight = w; m.biases = b
        TEARDOWN.append(restore)
        assert n == len(saved) and n >= 96, f"dense3 converted only {n} modules"
        print(f"[dm] dense3: {n} modules at MSB-3 ({mb:.0f} MB of 4-bit weights rewritten)", flush=True)
    return setup
ABLATIONS["dense3"] = dense3()
ABLATIONS["qom_0.5_live_c147_bonus5g"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.5_live_c147_bonus5g"))   # NewIdeas 20:1x chat route (QUEUE 20:1x 4fb08ca, masks 4324e09): 1(c)'s rule on a 147-slot pool, 4-bit stock path
ABLATIONS["qom_0.3_live_c147_bonus3g"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.3_live_c147_bonus3g"))   # NewIdeas 22:02 (QUEUE 22:0x 04ac744): the chat route at cf 0.3, 147-slot pool, 4-bit stock path
ABLATIONS["qom_0.3_live_c147_bonus4g"] = q1_union(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.3_live_c147_bonus4g"))   # NewIdeas 22:49 (QUEUE 22:4x fb39925): the chat route, bonus row gated at 0.4, 147-slot pool, 4-bit stock path
ABLATIONS["qom_0.5_u8_live_c147_bonus5g"] = q1_union("/Users/kamen/FableGod/decision_map/out/ni/qom_0.5_u8_live_c147_bonus5g")   # snapshot of /Users/kamen/NewIdeas/data/dm_masks/qom_0.5_u8_live_c147_bonus5g (sha256 beside it), installed by install_c3.sh
ABLATIONS["qom_0.5_u8_reach0.5_live_c147_bonus5g"] = q1_union("/Users/kamen/FableGod/decision_map/out/ni/qom_0.5_u8_reach0.5_live_c147_bonus5g")   # snapshot of /Users/kamen/NewIdeas/data/dm_masks/qom_0.5_u8_reach0.5_live_c147_bonus5g (sha256 beside it), installed by install_c3.sh
ABLATIONS["c3_b2d"] = q1_union("/Users/kamen/FableGod/decision_map/out/c3/c3_b2d")   # snapshot of /Users/kamen/FableGod/cloud3/live/cloud3/masks/c3_b2d (sha256 beside it), installed by install_c3.sh
ABLATIONS["c3_b2c"] = q1_union("/Users/kamen/FableGod/decision_map/out/c3/c3_b2c")   # snapshot of /Users/kamen/FableGod/cloud3/live/cloud3/masks/c3_b2c (sha256 beside it), installed by install_c3.sh
ABLATIONS["c3_b2d_reach07"] = q1_union("/Users/kamen/FableGod/decision_map/out/c3/c3_b2d_reach07")   # snapshot of /Users/kamen/FableGod/cloud3/live/cloud3/masks/c3_b2d_reach07 (sha256 beside it), installed by install_c3.sh
ABLATIONS["c3_b2b_band5"] = q1_union("/Users/kamen/FableGod/decision_map/out/c3/c3_b2b_band5")   # snapshot of /Users/kamen/FableGod/cloud3/live/cloud3/masks/c3_b2b_band5 (sha256 beside it), installed by install_c3.sh
ABLATIONS["c3_b2b_top1"] = q1_union("/Users/kamen/FableGod/decision_map/out/c3/c3_b2b_top1")   # snapshot of /Users/kamen/FableGod/cloud3/live/cloud3/masks/c3_b2b_top1 (sha256 beside it), installed by install_c3.sh
ABLATIONS["code"] = lambda: None   # (Thu 11-05) no-op part: "<arm>+code" runs <arm> on the code windows (DM_SETS=code, the s67 runner)
ABLATIONS["qom_0_split0.9_live_bq"] = q1_union("/Users/kamen/FableGod/decision_map/out/ni/qom_0_split0.9_live_bq")   # snapshot of /Users/kamen/NewIdeas/data/dm_masks/qom_0_split0.9_live_bq (sha256 beside it), installed by install_c3.sh
ABLATIONS["qom_0_split0.99_live_bq"] = q1_union("/Users/kamen/FableGod/decision_map/out/ni/qom_0_split0.99_live_bq")   # snapshot of /Users/kamen/NewIdeas/data/dm_masks/qom_0_split0.99_live_bq (sha256 beside it), installed by install_c3.sh
ABLATIONS["qom_0_split0.9_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/ni/qom_0_split0.9_live_ball")   # snapshot of /Users/kamen/NewIdeas/data/dm_masks/qom_0_split0.9_live_ball (sha256 beside it), installed by install_c3.sh
ABLATIONS["qom_0_split0.99_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/ni/qom_0_split0.99_live_ball")   # snapshot of /Users/kamen/NewIdeas/data/dm_masks/qom_0_split0.99_live_ball (sha256 beside it), installed by install_c3.sh
ABLATIONS["qom_0_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/ni/qom_0_live_ball")   # snapshot of /Users/kamen/NewIdeas/data/dm_masks/qom_0_live_ball (sha256 beside it), installed by install_c3.sh
ABLATIONS["qom_0_split0.99_k8_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/fc/qom_0_split0.99_k8_live_ball")   # snapshot of /Users/kamen/FableGod/fablec/build_1105/masks/qom_0_split0.99_k8_live_ball (sha256 beside it), installed by install_fc.sh
ABLATIONS["qom_0_split0.99_k6_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/fc/qom_0_split0.99_k6_live_ball")   # snapshot of /Users/kamen/FableGod/fablec/build_1105/masks/qom_0_split0.99_k6_live_ball (sha256 beside it), installed by install_fc.sh
ABLATIONS["qom_0_split0.99_r0.7_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/fc/qom_0_split0.99_r0.7_live_ball")   # snapshot of /Users/kamen/FableGod/fablec/build_1105/masks/qom_0_split0.99_r0.7_live_ball (sha256 beside it), installed by install_fc.sh
ABLATIONS["qom_0_split0.99_m0.9_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/fc/qom_0_split0.99_m0.9_live_ball")   # snapshot of /Users/kamen/FableGod/fablec/build_1105/masks/qom_0_split0.99_m0.9_live_ball (sha256 beside it), installed by install_fc.sh
ABLATIONS["qom_0_split0.99_r0.7_z0.999_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/fc/qom_0_split0.99_r0.7_z0.999_live_ball")   # snapshot of /Users/kamen/FableGod/fablec/build_1105/masks/qom_0_split0.99_r0.7_z0.999_live_ball (sha256 beside it), installed by install_fc.sh
ABLATIONS["qom_0_split0.99_k4_live_ball"] = q1_union("/Users/kamen/FableGod/decision_map/out/fc/qom_0_split0.99_k4_live_ball")   # snapshot of /Users/kamen/FableGod/fablec/build_1105/masks/qom_0_split0.99_k4_live_ball (sha256 beside it), installed by install_fc.sh
def nw_mask(path, d, zero=False):
    """NEVER-WAIT (Fable FAR + ZERO, ~/FableGod/far/NEVERWAIT_TEST.md; Thu 2026-11-05): q1_union(path) whose masks also
    hold `late` [rows, 48, 10] (rank-sorted like keep; ~/FableGod/far/build_1105/nw_masks.py). A kept-and-late rank's term
    of layer L is written d layers late through layer L's own inject gates (dm_common._nw_write); the 4-bit stock path,
    the model's own weights, no _t3. zero = the late flags forced off (self-check i: must equal the twin bit for bit)."""
    def setup():
        PERT.update(rmask=path, renorm=False, nw=d)
        if zero: PERT.update(nw_zero=True)
    return setup
_NWD = "/Users/kamen/FableGod/far/build_1105/scratch/dm_copy2/out/far/"   # snapshots of ~/FableGod/far/build_1105/masks/ (MD5SUMS beside them), installed by install_nw.sh
ABLATIONS["qom_0_live_ball_c208"] = q1_union(_NWD + "qom_0_live_ball_c208")   # tier 1: the twin, QOM cf 0 + DRAFT17 on 208 uniform slots
ABLATIONS["qom_0_live_ball_c208_nw1"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw1", 1)   # tier 1, decisive: every demand miss one layer late
ABLATIONS["qom_0_live_ball_c208_nw1r1"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw1r1", 1)   # tier 1, fallback: rank-0 experts wait
ABLATIONS["qom_0_live_ball_nw1"] = nw_mask(_NWD + "qom_0_live_ball_nw1", 1)   # tier 1: nw1 at the published pool (twin qom_0_live_ball)
ABLATIONS["qom_0_split0.99_live_ball_nw1"] = nw_mask(_NWD + "qom_0_split0.99_live_ball_nw1", 1)   # tier 1: the stack (twin qom_0_split0.99_live_ball)
ABLATIONS["qom_0_live_ball_c208_nwz"] = nw_mask(_NWD + "qom_0_live_ball_c208", 1, zero=True)   # self-check i: the twin's masks through the never-wait path
ABLATIONS["qom_0_live_ball_c208_nw0"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw0", 0)   # self-check ii: lateness 0
ABLATIONS["qom_0_live_ball_c208_nw2"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw2", 2)   # tier 2: two layers late
ABLATIONS["qom_0_live_ball_c208_nw1s"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw1s", 1)   # tier 2: late only on rows whose draft cf >= 0.9
ABLATIONS["qom_0_split0.99_live_ball_c208"] = q1_union(_NWD + "qom_0_split0.99_live_ball_c208")   # tier 2: the stack's twin at 208 slots
ABLATIONS["qom_0_split0.99_live_ball_c208_nw1"] = nw_mask(_NWD + "qom_0_split0.99_live_ball_c208_nw1", 1)   # tier 2: the stack at 208 slots
ABLATIONS["qom_0_split0.9_live_ball_nw1"] = nw_mask(_NWD + "qom_0_split0.9_live_ball_nw1", 1)   # tier 2: split 0.9 + nw1 (twin qom_0_split0.9_live_ball)
