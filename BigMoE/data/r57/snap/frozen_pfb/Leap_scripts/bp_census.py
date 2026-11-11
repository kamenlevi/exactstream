"""BP-11a (Leap, Wed 2026-11-04 17:0x): kernel census of the exact-safe propose (D8-NAX + screened head), R-65's method
(mx.export_to_dot of the propose's outputs; kernel-launching primitives counted, views excluded), by primitive.
-> ~/Leap/data/bp_census.txt"""
import os, sys, json, re, tempfile, collections, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU
import chain_fused as CF
import d7b_nax
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def get(k):
    return mx.load(f"{D}/{wmap[k]}")[k]
emb = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
for m, p in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
    m.weight, m.scales, m.biases = get(p + ".weight"), get(p + ".scales"), get(p + ".biases")
class _T: pass
tgt = _T(); tgt.embed_tokens = emb; tgt.lm_head = head
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
d7b_nax.apply8(draft); draft.bind(tgt); mx.eval(draft.parameters(), emb.parameters(), head.parameters())
DFU.install(forms=False, draft=draft)
SUB = d7b_nax.DraftSubset(head)
sel = draft.candidate_selector
MASK, R = int(draft.config.mask_token_id), 16
rng = np.random.default_rng(1)
dc = draft.make_cache()
ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)).astype(mx.bfloat16)[None]
blk = mx.array([[5] + [MASK] * (R - 1)])
mx.eval(ctx, blk)
hid = draft.hidden_states(blk, ctx, dc, 1)
cand, unary = d7b_nax.screened_head2s(draft, hid, SUB)
pth, cf = CF.chain_fused(blk, unary, cand, sel.hidden_projection(hid), sel)
VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided", "Flatten",
         "Unflatten", "StopGradient", "Depends", "Split"}
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
ch = census(hid)
ca = census(pth, cf)
kern = lambda c: sum(v for k, v in c.items() if k not in VIEWS)
line = (f"BP-11a {time.strftime('%a %H:%M')}: kernel-launching primitives: hidden_states {kern(ch)}, whole propose {kern(ca)}; "
        f"by primitive (whole): " + ", ".join(f"{k} {v}" for k, v in ca.most_common() if k not in VIEWS))
print(line); open(os.path.expanduser("~/Leap/data/bp_census.txt"), "a").write(line + "\n")
