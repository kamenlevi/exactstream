"""L1-18 step B (Leap, Sat 2026-10-24, pre-registered in ~/Leap/TABLE_350.md): how much drafting headroom the 125B leaves on
code. Data: the 24 public HumanEval samples (data/dfl2d_code: the 125B's taps, its argmax, ids = prompt + its greedy code)
and their target top-32 log-probs (data/l118_tgt, from dfl2c_extract.py; row i predicts token i + 1).
1. Replay DFlash2 (incoai weights, temperature 0) at block 16 exactly as dfl2_eval.py / the official loop: per block,
   the accepted length and each depth's agreement with the true continuation.
2. Agreement by target confidence (top-1 p of the position's token) and by depth.
3. Oracle drafter: right wherever the target's top-1 p >= t, else right with DFlash2's observed agreement for that
   confidence bin (and depth); expected tokens per pass at block 16 and 32 by Monte Carlo over the real position sequence.
Usage: [BigMoE venv] headroom18.py  -> ~/Leap/data/headroom18.txt"""
import os, sys, glob, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_third_party"))
import dflash_model_mlx as D

class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
BS = 16
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

BINS = [(0.99, 1.01, "p >= 0.99"), (0.9, 0.99, "0.9-0.99"), (0.5, 0.9, "0.5-0.9"), (0.0, 0.5, "< 0.5")]
def bin_of(p):
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= p < hi:
            return i
    return len(BINS) - 1

agree = np.zeros((len(BINS), BS - 1)); count = np.zeros((len(BINS), BS - 1))
first_miss = np.zeros(len(BINS)); conf_all = []
seqs = []                                            # per sample: target top-1 p of every completion token, in order
tau_tok, tau_blocks = 0, 0
t0 = time.time()
for path in sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz"))):
    key = os.path.basename(path)[:-4]
    z = np.load(path); tz = np.load(os.path.expanduser(f"~/Leap/data/l118_tgt/{key}.npz"))
    f = z["f"].reshape(z["f"].shape[0], -1); ids = z["ids"]; off, gen = int(z["off"]), int(z["gen"]); n = len(ids)
    p1 = np.exp(tz["lp"][:, 0].astype(np.float64))     # p1[i] = target top-1 prob for token i + 1
    seqs.append(p1[gen - 1:n - 1]); conf_all += list(p1[gen - 1:n - 1])
    cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a = start, gen
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        block = mx.array([[int(ids[a - off])] + [MASK] * (BS - 1)])
        dp, cand, _ = draft.propose(block, ctx, cache, 0.0, logits_start=1)
        mx.eval(dp)
        fed = a
        d = np.array(dp[0]).tolist(); true = ids[a - off + 1:a - off + BS]
        k = 0
        while k < len(true) and k < len(d) and d[k] == true[k]:
            k += 1
        for j in range(len(true)):
            b = bin_of(p1[a - off + j]); agree[b, j] += int(d[j] == true[j]); count[b, j] += 1
        if k < len(true):
            first_miss[bin_of(p1[a - off + k])] += 1
        tau_tok += min(k + 1, off + n - 1 - a); tau_blocks += 1
        if (trim := cache[0].offset - a) > 0:
            D._trim_recent_cache(cache, trim)
        a = a + k + 1
conf_all = np.array(conf_all)
say(f"L1-18: 24 HumanEval samples, {len(conf_all)} completion tokens; DFlash2 (incoai) block 16 replay in {time.time() - t0:.0f} s")
say(f"  DFlash2 tau16 (tokens per pass) = {tau_tok / tau_blocks:.2f} over {tau_blocks} blocks")
say("  target top-1 p distribution over completion tokens: " +
    ", ".join(f"{nm} {np.mean((conf_all >= lo) & (conf_all < hi)):.3f}" for lo, hi, nm in BINS))
say("  DFlash2 agreement by target confidence (all depths) and first misses by bin:")
for i, (_, _, nm) in enumerate(BINS):
    tot = count[i].sum()
    say(f"    {nm:10s}: agreement {agree[i].sum() / max(1, tot):.3f} over {int(tot)} block positions; "
        f"first misses here {int(first_miss[i])} ({first_miss[i] / max(1, first_miss.sum()):.2f} of all)")
say("    by depth (p >= 0.99 bin): " + " ".join(f"{agree[0, j] / max(1, count[0, j]):.2f}" for j in range(BS - 1)))
say("    by depth (0.5-0.9 bin):   " + " ".join(f"{agree[2, j] / max(1, count[2, j]):.2f}" for j in range(BS - 1)))
q = agree / np.maximum(1, count)                     # observed agreement per bin and depth
rng = np.random.default_rng(0)
def oracle(t, B, runs=30):
    toks, blocks = 0, 0
    for _ in range(runs):
        for s in seqs:
            a = 0
            while a < len(s):
                k = 0
                while k < B - 1 and a + k < len(s):
                    p = s[a + k]
                    right = True if p >= t else rng.random() < q[bin_of(p), min(k, BS - 2)]
                    if not right:
                        break
                    k += 1
                toks += min(k + 1, len(s) - a); blocks += 1
                a += k + 1
    return toks / blocks
say("  oracle drafter (right where target top-1 p >= t, DFlash2's agreement elsewhere): tokens per pass")
for t in (0.9, 0.99):
    say(f"    t = {t}: block 16 {oracle(t, 16):5.2f}, block 32 {oracle(t, 32):5.2f}, block 64 {oracle(t, 64):5.2f}")
say(f"  perfect drafter (always the target's argmax): block 16 {oracle(0.0, 16):.2f}, block 32 {oracle(0.0, 32):.2f}")
open(os.path.expanduser("~/Leap/data/headroom18.txt"), "w").write("\n".join(lines) + "\n")
