"""BP-0 (Leap, Wed 2026-11-04 16:1x; TABLE_350 'BP-0'; the orchestrator's route 15:5x): where DFlash2's block pass (the
drafter's propose: 10.7 ms of a 100.2 ms code check in BigMoE's all-in C 160 arm, 6.3 with D7b quiet) spends its time.
Drafter as R-79 loads it (DFlash2 + l120_ft, bound to the real 4-bit embed / lm_head; ~3 GB), driven as r57_screen
drives it: hidden_states (5 sliding-window layers over the 16-row block + new context) -> head2s (compute_logits: the
4-bit lm_head at 16 rows, then the two-stage top-16) -> the selector's hidden_projection -> chain_fused (R-77).
Arms: bf16 + R-76 / R-78 forms (K83, the engine's exact drafter), and D7b (layers 4-bit RTN g64, no forms).
 (1) ms per propose (one eval + read-back), median over cycles;
 (2) per part, each evaluated alone (sync between parts, so the parts add ~0.1 ms each of sync);
 (3) the head alone at 16 rows: full 248K vocab vs a screened head (FR-Spec top-32K rows; ScreenedHead.logits).
usage: [BigMoE venv] bp_profile.py -> ~/Leap/data/bp_profile.txt (GPU, MEM ~3.5 GB; share lock)"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
from screened_head import ScreenedHead

OUT = os.path.expanduser("~/Leap/data/bp_profile.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nBP-0 {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def get(k):
    return mx.load(f"{D}/{wmap[k]}")[k]
emb = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
for m, p in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
    m.weight, m.scales, m.biases = get(p + ".weight"), get(p + ".scales"), get(p + ".biases")
class _T:
    pass
tgt = _T(); tgt.embed_tokens = emb; tgt.lm_head = head
def load(q4):
    d_ = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d_.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
    if q4:
        import d7b_live
        d_ = d7b_live.apply(d_) or d_
    d_.bind(tgt)
    mx.eval(d_.parameters())
    return d_
DR = {"bf16+forms": load(False), "D7b": load(True)}
mx.eval(emb.parameters(), head.parameters())
MASK, R, K = int(DR["D7b"].config.mask_token_id), 16, 16

def head2s(draft, hid):
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel_ = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel_, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)

def cycles(name, n=40, seed=7):
    draft = DR[name]
    DFU.uninstall()
    if name == "bf16+forms":
        DFU.install(forms=True, draft=draft)
    sel = draft.candidate_selector
    rng = np.random.default_rng(seed)
    dc = draft.make_cache()
    ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
    mx.eval(draft.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
    tot, parts = [], {k: [] for k in ("hidden_states", "compute_logits", "top-16", "hidden_projection", "chain_fused")}
    for c in range(n):
        a = dc[0].offset + 8
        ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)).astype(mx.bfloat16)[None]
        blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
        mx.eval(ctx, blk)
        if c % 2 == 0:                                   # the whole propose, one eval + read-back
            t0 = time.perf_counter()
            hid = draft.hidden_states(blk, ctx, dc, 1)
            cand, unary = head2s(draft, hid)
            hp = sel.hidden_projection(hid)
            pth, cf = chain_fused(blk, unary, cand, hp, sel)
            np.array(pth); np.array(cf)
            tot.append(time.perf_counter() - t0)
        else:                                            # parts, each evaluated alone
            t0 = time.perf_counter(); hid = draft.hidden_states(blk, ctx, dc, 1); mx.eval(hid); t1 = time.perf_counter()
            lg = draft.compute_logits(hid); mx.eval(lg); t2 = time.perf_counter()
            cand, unary = head2s(draft, hid); mx.eval(cand, unary); t3 = time.perf_counter()   # (recomputes lg: subtract)
            hp = sel.hidden_projection(hid); mx.eval(hp); t4 = time.perf_counter()
            pth, cf = chain_fused(blk, unary, cand, hp, sel); mx.eval(pth, cf); t5 = time.perf_counter()
            parts["hidden_states"].append(t1 - t0); parts["compute_logits"].append(t2 - t1)
            parts["top-16"].append((t3 - t2) - (t2 - t1)); parts["hidden_projection"].append(t4 - t3)
            parts["chain_fused"].append(t5 - t4)
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
    DFU.uninstall()
    return float(np.median(tot[3:])), {k: float(np.median(v[3:])) for k, v in parts.items()}

for rnd in range(2):
    for name in (DR if rnd == 0 else list(DR)[::-1]):
        t, p = cycles(name, seed=7 + rnd)
        say(f"round {rnd} {name:10s}: propose {1e3 * t:.2f} ms; parts alone: " + ", ".join(f"{k} {1e3 * v:.2f}" for k, v in p.items()))
rank = np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))
sh = ScreenedHead(head, rank[:32768], extra_cap=4096)
sh.add(rank[40000:42048])
h16 = (mx.random.normal((1, 16, 2560)) * 0.5).astype(mx.bfloat16)
mx.eval(h16)
ARMS = {"full head 248K": lambda: head(h16), "screened 32K": lambda: sh.logits(h16)}
for f in ARMS.values():
    mx.eval([f() for _ in range(4)])
T = {k: [] for k in ARMS}
for r in range(15):
    for k in (list(ARMS) if r % 2 == 0 else list(ARMS)[::-1]):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([ARMS[k]() for _ in range(8)]); mx.synchronize()
        T[k].append((time.perf_counter() - t0) / 8)
say("(3) the head at 16 rows (8 calls per eval, median of 15): " + ", ".join(f"{k} {1e3 * np.median(v):.3f} ms" for k, v in T.items())
    + f"; screened = 32,768 + {sh.n_extra} session rows")
