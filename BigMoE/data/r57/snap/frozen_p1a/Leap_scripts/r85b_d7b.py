"""R-85b D7b (Leap, Mon 2026-11-02 11:2x; DAY9 condition 6): D7's arm D drafted in 7.2-7.4 ms but lost 2.5 % kept on math.
Which parts can stay bf16 and keep the kept within -1 % at <= 8 ms? Arms (all with REF2 in the propose, R-76 conv kernel,
R-77 chain, qmm16c head, no forms): A = bf16 (the kept reference); D = layers + fc 4-bit (D7's D); E = layers 4-bit, fc
bf16; F = the MLPs 4-bit only; G = MLPs + q / o 4-bit (k / v, convs, fc bf16).
Was: R-85 = ROUTE_350 D7 (Leap, Mon 2026-11-02; DAY9 condition 6, kill > 8 ms): a 4-bit DFlash2 block on the fused pieces.
Arms (the L1-20 drafter, l120_ft; drafts verified by the target as always, so outputs stay exact; only kept can move):
  A  today's live drafter after K-83: bf16, R-76 conv kernel + R-78 forms (draft_fused), R-77 chain (chain_fused), MLX head
  B  A + R-84 (the head on qmm16c)
  C  4-bit layers (R-06b's quantize: every nn.Linear in layers + fc, g64), R-76 conv kernel, R-77 chain, qmm16c head
  D  C with REF2 applied during the propose: its 16-row 4-bit projections take qmm16c (REF2's 16-row NAX kernel)
Per arm, D-02b's teacher-forced replay (dfl2d_code 24 samples, dfl2d_math 60): ms per propose (median over the
replay's checks, each a real propose with its real context: mx.eval of the path), kept per check, identical drafts vs A.
Line (DAY9 #6): PASS if an arm's propose <= 8 ms with kept within -1 % of A; KILL if every arm stays > 8 ms.
Usage: [BigMoE venv] r85b_d7b.py -> ~/Leap/data/r85b_d7b.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
from qmm16 import qmm16c
import ref2

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/Leap/data")
V, K = 248320, 16
class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())
PRED = {"D": lambda p: p.startswith("layers") or p == "fc", "E": lambda p: p.startswith("layers"),
        "F": lambda p: p.startswith("layers") and ".mlp." in p,
        "G": lambda p: p.startswith("layers") and (".mlp." in p or p.endswith("q_proj") or p.endswith("o_proj"))}
def load(arm):
    d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
    if arm in PRED:
        nn.quantize(d, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and PRED[arm](p))
    d.bind(tgt)
    mx.eval(d.parameters())
    return d
ARM = {"head": "mlx", "ref2": False, "d": None}
def logits_of(hid):
    d = ARM["d"]
    lead, rows = hid.shape[:-1], hid.size // hid.shape[-1]
    lm = d.lm_head
    if ARM["head"] == "q16" and rows <= 16:
        x = hid.reshape(rows, -1)
        x16 = x if rows == 16 else mx.concatenate([x, mx.zeros((16 - rows, x.shape[-1]), x.dtype)])
        lg = qmm16c(x16, lm.weight, lm.scales, lm.biases)[:rows].reshape(*lead, -1)
    else:
        lg = lm(hid)
    lg = lg * d.config.output_multiplier
    if d.config.final_logit_softcapping:
        cap = d.config.final_logit_softcapping
        lg = mx.tanh(lg / cap) * cap
    return lg
def head2s(hid):
    lg = logits_of(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel_ = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel_, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)
def fn(blk, ctx, cache):
    d = ARM["d"]
    if ARM["ref2"]:
        ref2.apply()
    try:
        hid = d.hidden_states(blk, ctx, cache, 1)
        cand, unary = head2s(hid)
        out = chain_fused(blk, unary, cand, d.candidate_selector.hidden_projection(hid), d.candidate_selector)[0]
        mx.eval(out)
    finally:
        if ARM["ref2"]:
            ref2.remove()
    return out
def walk(sample, ts):
    d = ARM["d"]
    f, ids, off, gen = sample
    cache = d.make_cache()
    MASK = int(d.config.mask_token_id)
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, out, drafts, n = start, gen, [], [], len(ids)
    first = True
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * 15])
        mx.eval(ctx, blk)
        t0 = time.perf_counter()
        dr = np.array(fn(blk, ctx, cache)[0]).tolist()
        if not first:                                    # the window prefill (first call) is K-80's start-up, not D7's
            ts.append(time.perf_counter() - t0)
        first = False
        drafts.append(tuple(dr))
        fed = a
        true = ids[a - off + 1:a - off + 16]
        k = 0
        while k < len(true) and dr[k] == true[k]:
            k += 1
        out.append(k + 1)
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + k + 1
    return out, drafts
SETS = {}
for folder in ("dfl2d_code", "dfl2d_math"):
    smp = []
    for p in sorted(glob.glob(os.path.join(D, folder, "*.npz"))):
        z = np.load(p)
        smp.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
    SETS[folder] = smp
RES = {}
def run_arm(name):
    for folder, smp in SETS.items():
        kept, drafts, ts = [], [], []
        for sm in smp[:2]:                               # warm-up (kernels, shapes), not counted
            walk(sm, [])
        for sm in smp:
            o, dr = walk(sm, ts)
            kept += o; drafts.append(dr)
        RES[(name, folder)] = (kept, drafts, ts)
        say(f"  {name} {folder}: propose {1e3 * np.median(ts):.2f} ms (median of {len(ts)}; p25 {1e3 * np.percentile(ts, 25):.2f}, "
            f"p75 {1e3 * np.percentile(ts, 75):.2f}), kept per check {np.mean(kept):.3f} ({len(kept)} checks)")
DFU.install(forms=False)
for name in ("A", "D", "E", "F", "G"):
    ARM["d"] = load(name)
    ARM["head"], ARM["ref2"] = "q16", True
    run_arm(name)
    ARM["d"] = None
    mx.clear_cache()
DFU.uninstall()
for folder in SETS:
    base = RES[("A", folder)]
    for name in "DEFG":
        k_, dr_, ts_ = RES[(name, folder)]
        same = sum(int(x == y) for da, db in zip(base[1], dr_) for x, y in zip(da, db))
        tot = sum(min(len(da), len(db)) for da, db in zip(base[1], dr_))
        say(f"{folder} {name} vs A: propose {1e3 * np.median(ts_):.2f} vs {1e3 * np.median(base[2]):.2f} ms; kept "
            f"{np.mean(k_):.3f} vs {np.mean(base[0]):.3f} ({np.mean(k_) / np.mean(base[0]) - 1:+.2%}); identical drafts {same} / {tot}")
open(os.path.join(D, "r85b_d7b.txt"), "w").write("\n".join(lines) + "\n")
