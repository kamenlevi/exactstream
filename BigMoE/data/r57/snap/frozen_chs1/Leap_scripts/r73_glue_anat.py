"""R-73 GLUE-ANAT (Leap, Sun 2026-11-01; shared GPU, real weights, day-sized): the per-layer glue around attention / GDN and
the routed experts, as a 16-row check runs it under REF2-T (layer 3's real weights; the routed experts' combined rows
and the attention output are inputs):
    mixed, hyper, inject = attn_hyper_connection(hidden)                 REF2: leapk.hc_read at 16 rows
    mixed, hyper, inject = mlp_hyper_connection(hyper, (attn_out, inject))
    MoE block minus the experts: gate (8-bit) -> routing chain -> [experts] -> y + sigmoid(shared_gate(x)) * shared(x)
    hidden = _hyper_residual_write(hyper, moe_out, inject)
1. Probe: MLX's sigmoid on all 65,536 bf16 values vs glue_fused's exp spellings (which one replays mx.sigmoid).
2. Census per piece (kernel-launching primitives, R-65's method) and the exact-safe candidates, each bitwise vs base:
   router = R-70's fused routing; tail = glue_fused.moe_tail (3 -> 1); write = the residual write under mx.compile
   (MTPLX's own argument: in bf16 each op rounds to 16 bits fused or not); cat = R-74's shared_cat (3 small
   projections -> 1 MLX call); all = the four.
Usage: [BigMoE venv] r73_glue_anat.py -> ~/Leap/data/r73_glue_anat.txt"""
import os, sys, json, re, tempfile, collections, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from router_fused import router_fused, router_chain
from glue_fused import moe_tail, moe_tail_ref, shared_cat
from mtplx.models import qwen4_exp as Q

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
b16 = lambda a: np.array(a.view(mx.uint16))

# 1. sigmoid probe (exhaustive over bf16)
allb = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).reshape(1, 65536, 1)
ref = mx.sigmoid(allb)
mx.eval(ref)
rb = b16(ref).reshape(-1)
rn = np.isnan(np.array(ref.astype(mx.float32))).reshape(-1)
EXP = None
for exp in ("maybe", "precise", "fast"):
    ones = mx.ones((1, 65536, 1), mx.bfloat16)
    got = moe_tail(mx.zeros((1, 65536, 1), mx.bfloat16), allb, ones, exp=exp)   # 0 + sigmoid(g) * 1
    mx.eval(got)
    gb = b16(got).reshape(-1)
    gn = np.isnan(np.array(got.astype(mx.float32))).reshape(-1)
    bad = int(((gb != rb) & ~(gn & rn)).sum())
    say(f"sigmoid probe, exp = {exp}: {bad} of 65536 bf16 inputs differ from mx.sigmoid")
    if bad == 0 and EXP is None:
        EXP = exp
say(f"tail kernel uses exp = {EXP}")

D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
qc = cfg["quantization"]
args = Q.TextArgs.from_dict(cfg["text_config"])
L = int(os.environ.get("R73_LAYER", "3"))
def load(prefix, mod):
    keys = [k for k in wmap if k.startswith(prefix)]
    w = {}
    for f in sorted({wmap[k] for k in keys}):
        sh = mx.load(f"{D}/{f}")
        w.update({k[len(prefix):]: sh[k] for k in keys if wmap[k] == f})
    def qpred(p, m):
        if not (isinstance(m, nn.Linear) and f"{p}.scales" in w):
            return False
        c = qc.get(prefix + p, {})
        return {"group_size": c.get("group_size", qc["group_size"]), "bits": c.get("bits", qc["bits"])}
    nn.quantize(mod, group_size=qc["group_size"], bits=qc["bits"], class_predicate=qpred)
    mod.load_weights(list(w.items()), strict=False)
    mod.eval()
    return w
pre = f"language_model.model.layers.{L}."
ahc, mhc, blk = Q.GatedResidual(args), Q.GatedResidual(args), Q.SparseMoeBlock(args)
load(pre + "attn_hyper_connection.", ahc)
load(pre + "mlp_hyper_connection.", mhc)
wm = load(pre + "mlp.", blk)
mx.eval(ahc.parameters(), mhc.parameters(), blk.gate.parameters(), blk.shared_expert.parameters(),
        blk.shared_expert_gate.parameters())
say(f"layer {L}: HC weights {[k for k in ahc.keys()]}; MoE top_k {blk.top_k}, norm_topk_prob {blk.norm_topk_prob}")
ref2.apply()

@mx.compile
def _write_c(hyper, block_out, inject):
    # MTPLX's op-diet 'resid' spelling (the add on the [.., hc, hidden] view, so compile can fuse the two ops); its
    # eager form is value-identical to the stock write (MTPLX), and in bf16 each op rounds to 16 bits fused or not
    grouped = hyper.reshape(*hyper.shape[:-1], inject.shape[-1], block_out.shape[-1])
    return (grouped + block_out[..., None, :] * inject[..., :, None]).reshape(*hyper.shape)

def glue(hidden, att_out, experts, arm):
    fr, ft, fw = arm[:3]
    fc = len(arm) > 3 and arm[3]
    mixed, hyper, inject = ahc(hidden)
    mixed, hyper, inject = mhc(hyper, pending=(att_out, inject))
    x = mixed
    logits = blk.gate(x)
    inds, scores = router_fused(logits, blk.top_k) if fr else router_chain(logits, blk.top_k, blk.norm_topk_prob)
    if fc:
        g, s = shared_cat(blk, x)                 # R-74: gate_proj | up_proj | shared_expert_gate in one MLX call
    else:
        g = blk.shared_expert_gate(x)
        s = blk.shared_expert(x)
    y = experts                                   # stands for switch_mlp's combined rows (TIER_COMBINE)
    out = moe_tail(y, g, s, exp=EXP) if ft else y + mx.sigmoid(g) * s
    hid = _write_c(hyper, out, inject) if fw else Q._hyper_residual_write(hyper, out, inject)
    return dict(read_a=None, mixed=mixed, inds=inds, scores=scores, out=out, hidden=hid)

VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided",
         "Flatten", "Unflatten", "StopGradient", "Depends", "Split"}
def census(*arrs):
    cnt = collections.Counter()
    fd, p = tempfile.mkstemp(suffix=".dot"); os.close(fd)
    try:
        mx.export_to_dot(p, *arrs)
        for line in open(p):
            m = re.search(r'label\s*=\s*"([A-Z][A-Za-z0-9_]*)', line)
            if m:
                cnt[m.group(1)] += 1
    finally:
        os.remove(p)
    return cnt
nk = lambda c: sum(v for k_, v in c.items() if k_ not in VIEWS)

ARMS = {"base": (False, False, False), "router": (True, False, False), "tail": (False, True, False),
        "write": (False, False, True), "cat": (False, False, False, True), "all": (True, True, True, True)}
mx.random.seed(3)
fails = 0
for trial in range(6):
    sc = (0.5, 1.0, 2.0, 4.0, 1.0, 8.0)[trial]
    hidden = (mx.random.normal((1, 16, 4 * 2560)) * sc).astype(mx.bfloat16)
    att_out = (mx.random.normal((1, 16, 2560)) * sc).astype(mx.bfloat16)
    experts = (mx.random.normal((1, 16, 2560)) * sc * 0.3).astype(mx.bfloat16)
    mx.eval(hidden, att_out, experts)
    res = {}
    for a, arm in ARMS.items():
        o = glue(hidden, att_out, experts, arm)
        if trial == 0:
            # per-piece census: the HC reads alone, the MoE glue, the write
            ca = census(o["mixed"])
            cm = census(o["inds"], o["scores"], o["out"])
            ct = census(o["hidden"], o["inds"], o["scores"])
            say(f"[{a}] kernels: HC reads (to the MoE input) {nk(ca)}; + MoE glue {nk(cm) - nk(ca)}; whole glue "
                f"(+ write) {nk(ct)}")
            if a in ("base", "all"):
                say(f"   [{a}] whole glue: " + ", ".join(f"{k_} {v}" for k_, v in ct.most_common() if k_ not in VIEWS))
        mx.eval([v for v in o.values() if v is not None])
        res[a] = {k: (b16(v) if v.dtype == mx.bfloat16 else np.array(v)) for k, v in o.items() if v is not None}
    for a in ARMS:
        if a == "base":
            continue
        eq = all((res[a][k] == res["base"][k]).all() for k in res["base"])
        fails += not eq
        if not eq:
            say(f"trial {trial} (scale {sc}): {a} != base on " +
                ", ".join(k for k in res["base"] if not (res[a][k] == res["base"][k]).all()))
say(f"bits over 6 trials (scales 0.5-8): {'PASS' if fails == 0 else f'FAIL ({fails})'} (router, tail, write, cat, all == base)")

def bench(arm, n=30, chain=8):
    hidden = mx.random.normal((1, 16, 4 * 2560)).astype(mx.bfloat16)
    att_out = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16)
    experts = (mx.random.normal((1, 16, 2560)) * 0.3).astype(mx.bfloat16)
    mx.eval(hidden, att_out, experts)
    def go():
        h = hidden
        outs = []
        for _ in range(chain):                    # dependent layers: each layer's hidden feeds the next
            o = glue(h, att_out, experts, arm)
            h = o["hidden"]
            outs += [o["inds"], o["scores"]]
        return outs + [h]
    for _ in range(3):
        mx.eval(go())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(go()); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / chain
res = {a: [] for a in ARMS}
for _ in range(3):
    for a, arm in ARMS.items():
        res[a].append(bench(arm))
base = min(res["base"])
say("ms per layer of glue (8 dependent layers per eval; shared GPU, rough): "
    + ", ".join(f"{a} {1e3 * min(v):.3f} ({48e3 * (min(v) - base):+.2f} ms per 48-layer check)" for a, v in res.items()))
open(os.path.expanduser("~/Leap/data/r73_glue_anat.txt"), "w").write("\n".join(lines) + "\n")
