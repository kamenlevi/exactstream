"""Leap L2 step 2 (2026-10-23): teacher-forced replay of the OFFICIAL DFlash2 decode loop (z-lab/dflash model_mlx.py,
vendored in /Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party, unmodified) on the 125B's features from dfl2_extract.py, greedy (temperature 0).
Per block, exactly as _stream_generate: the drafter cache holds the features of every committed position before the anchor
(first call: the last 2047 prompt rows, cache.offset = their start); block = [anchor, MASK x 7]; propose(...,
logits_start=1) -> 7 drafts; accepted = leading matches with the stored greedy continuation; next anchor after
accepted + 1; the next call feeds the features of the anchor and the accepted drafts. tau8 = generated tokens / blocks.
Also: the native MTP chain at K = 7 on the same agentic tokens (Chat1000 agent125 `mtp`, teacher-forced 15-step chains,
same block replay), per-depth P(first j right), and coverage of the true token in the drafter's top-16 candidates.
Usage: [BigMoE venv] dfl2_eval.py [--cpu] [--only PREFIX] [--smoke]"""
import os, sys, glob, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party"))
import dflash_model_mlx as D

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true"); ap.add_argument("--only", default=""); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--dir", default="~/Leap/data/dfl2"); ap.add_argument("--window", type=int, default=0)
ap.add_argument("--every", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--ntk", type=float, default=0)   # NTK-aware RoPE: base x ntk^(d/(d-2)), training-free context extension
ap.add_argument("--weights", default="")         # fine-tuned weights (dfl2c_train.py) over the incoai ones
ap.add_argument("--bs", type=int, default=8)     # L5: block 16 (anchor + 15 drafts per pass)
A = ap.parse_args()
if A.cpu:
    mx.set_default_device(mx.cpu)

class Target(nn.Module):
    """only the 125B's input embedding and LM head (4-bit affine, group 64), which draft.bind() needs"""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)

def load_target():
    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    t = Target()
    t.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                   + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    mx.eval(t.parameters())
    return t

def replay(draft, f, ids, off, gen, bs=8):
    """f: (n, 12800) features of absolute positions off..off+n-1; ids: tokens there. Returns accepted list, coverage rows."""
    n = len(ids); mask_id = int(draft.config.mask_token_id)
    cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed = start                                   # absolute position of the next feature row to feed
    a = gen                                       # absolute position of the anchor (the first generated token)
    acc, cov = [], []
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        block = mx.array([[int(ids[a - off])] + [mask_id] * (bs - 1)])
        path, cand, _ = draft.propose(block, ctx, cache, 0.0, logits_start=1)
        mx.eval(path, cand)
        fed = a
        d = np.array(path[0]).tolist(); cnd = np.array(cand[0])
        true = ids[a - off + 1:a - off + bs]
        k = 0
        while k < len(true) and k < len(d) and d[k] == true[k]:
            k += 1
        acc.append((k, len(true), off + n - 1 - a))
        cov.append([int(true[j] in cnd[j]) for j in range(len(true))])
        if (trim := cache[0].offset - a) > 0:     # as the official loop (0 in steady state)
            D._trim_recent_cache(cache, trim)
        a = a + k + 1
    return acc, cov

def tau_of(acc, fixed=True):
    """tokens per block from (accepted k, drafts checked m, tokens left after the anchor r). Fixed (Sat 10-24): k drafts
    + the target's bonus / correction token whenever a token follows them, as the official loop (8 for a fully accepted
    block). Old (L2-L3 reports): a fully accepted block counted as 7, i.e. low by the share of fully accepted blocks."""
    if fixed:
        return sum(min(k + 1, r) for k, m, r in acc) / max(1, len(acc))
    return sum(min(k + 1, m) if k == m else k + 1 for k, m, r in acc) / max(1, len(acc))

def taus(acc):
    return f"{tau_of(acc):.2f} (old {tau_of(acc, False):.2f})"

def mtp_replay(out, mtp, K=7):
    acc, a, T = [], 0, len(out)
    while a < T - 1:
        true = out[a + 1:a + 1 + K]; d = mtp[a + 1, :K] if a + 1 < T else []
        k = 0
        while k < len(true) and k < len(d) and d[k] == true[k]:
            k += 1
        acc.append((k, len(true), T - 1 - a)); a = a + k + 1
    return acc

draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(load_target())
if A.weights:
    draft.load_weights(os.path.expanduser(A.weights), strict=False)
    print(f"[dfl2e] fine-tuned weights {A.weights}", flush=True)
if A.window:                                     # inference-time window override (config + every attention layer)
    draft.config.sliding_window = A.window
    for L in draft.layers:
        L.self_attn.sliding_window = A.window
if A.ntk:
    d = draft.config.head_dim
    base = draft.config.rope_theta * A.ntk ** (d / (d - 2))
    draft.rope = D._build_rope(d, base, draft.config.max_position_embeddings, None)
    print(f"[dfl2e] NTK-aware RoPE x{A.ntk:g}: base {draft.config.rope_theta:g} -> {base:.0f}", flush=True)
print(f"[dfl2e] window {draft.config.sliding_window}, dir {A.dir}", flush=True)

def every_anchor(draft, f, ids, off, gen, bs=8):
    """DFlash2 run length at EVERY anchor (the cache holds only committed features, so drafts do not depend on the block
    history) plus its first draft: for the hybrid bound."""
    n = len(ids); mask_id = int(draft.config.mask_token_id)
    cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, runs, first = start, [], []
    for a in range(gen, off + n - 1):
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        path, _, _ = draft.propose(mx.array([[int(ids[a - off])] + [mask_id] * (bs - 1)]), ctx, cache, 0.0, logits_start=1)
        d = np.array(path[0]).tolist(); fed = a
        true = ids[a - off + 1:a - off + bs]; k = 0
        while k < len(true) and d[k] == true[k]:
            k += 1
        runs.append((k, len(true), off + n - 1 - a)); first.append(d[0])
    return runs, first

def policy_tau(rd, rm, fd, fm, mode, fixed=True):
    a, acc, T = 0, [], len(rd)
    while a < T:
        kd, m, r = rd[a]; km = rm[a][0]
        k = max(kd, km) if mode == "oracle" else (kd if fd[a] == fm[a] else km)
        acc.append((k, m, r)); a = a + k + 1
    return tau_of(acc, fixed)
print(f"[dfl2e] drafter + target embed/head loaded, device {mx.default_device()}", flush=True)

if A.smoke:                                       # mechanics only: random features, a fake 40-token continuation
    rng = np.random.default_rng(0)
    f = (rng.standard_normal((300, 12800)) * 0.5).astype(np.float16)
    ids = rng.integers(1000, 50000, 300).astype(np.int32)
    t0 = time.time(); acc, cov = replay(draft, f, ids, 0, 260)
    print(f"[dfl2e] smoke: {len(acc)} blocks in {time.time() - t0:.1f}s, accepted {[k for k, *_ in acc][:10]}", flush=True)
    sys.exit(0)

res = {}
for path in sorted(glob.glob(os.path.join(os.path.expanduser(A.dir), "*.npz"))):
    key = os.path.basename(path)[:-4]
    if A.only and not key.startswith(A.only):
        continue
    z = np.load(path)
    f = z["f"].reshape(z["f"].shape[0], -1)
    acc, cov = replay(draft, f, z["ids"], int(z["off"]), int(z["gen"]), A.bs)
    cls = key.split("_")[0] + ("_" + key.split("_")[1] if key.startswith("tf_") else "")
    cls = "agent" if key.startswith("agent") else key.split("_")[0] + "_" + key.split("_")[1]
    r = res.setdefault(cls, {"acc": [], "cov": [], "mtp": []})
    r["acc"] += acc; r["cov"] += cov
    if cls == "agent":
        m = np.load(os.path.expanduser(f"~/Chat1000/data/agent125/{key[6:]}.npz"))
        r["mtp"] += mtp_replay(m["out"].tolist(), m["mtp"])
        msg = f"[dfl2e] {key}: DFlash2 tau{A.bs} {taus(acc)} | MTP K=7 {taus(mtp_replay(m['out'].tolist(), m['mtp']))}"
        if A.every:
            rd, fd = every_anchor(draft, f, z["ids"], int(z["off"]), int(z["gen"]))
            out = m["out"].tolist(); T = len(out); rm, fm = [], []
            for a in range(T - 1):
                true = out[a + 1:a + 8]; d = m["mtp"][a + 1, :7]; k = 0
                while k < len(true) and d[k] == true[k]:
                    k += 1
                rm.append((k, len(true), T - 1 - a)); fm.append(int(d[0]))
            assert len(rd) == len(rm), (len(rd), len(rm))
            r.setdefault("pol", []).append((rd, rm, fd, fm))
            msg += f" | cross-check {policy_tau(rd, rm, fd, fm, 'x'):.2f} | oracle {policy_tau(rd, rm, fd, fm, 'oracle'):.2f}"
        print(msg, flush=True)

lines = []
for cls, r in sorted(res.items()):
    acc = r["acc"]
    full = [k for k, m, _ in acc if m == A.bs - 1]
    pj = [np.mean([k >= j for k in full]) for j in range(1, A.bs)] if full else []
    covj = np.array([c for c in r["cov"] if len(c) == A.bs - 1]).mean(0) if r["cov"] else []
    line = (f"{cls:10s} blocks {len(acc):5d} | DFlash2 tau{A.bs} {taus(acc)} | P(first j right) " +
            " ".join(f"{p:.2f}" for p in pj) + " | top-16 coverage per depth " + " ".join(f"{c:.2f}" for c in covj))
    if r["mtp"]:
        line += (f" | MTP K=7 tau {taus(r['mtp'])} (x{tau_of(acc) / tau_of(r['mtp']):.2f}; "
                 f"old x{tau_of(acc, False) / tau_of(r['mtp'], False):.2f})")
    if r.get("pol"):
        tx = sum(len(p[0]) for p in r["pol"])
        for fx in (True, False):
            bx = sum(len(p[0]) / policy_tau(*p, "x", fx) for p in r["pol"])
            bo = sum(len(p[0]) / policy_tau(*p, "oracle", fx) for p in r["pol"])
            line += f" | {'' if fx else 'old '}cross-check {tx / bx:.2f} | {'' if fx else 'old '}oracle {tx / bo:.2f}"
    lines.append(line); print(line, flush=True)
open(os.path.expanduser(f"~/Leap/data/dfl2_result{A.tag}.txt"), "w").write("\n".join(lines) + "\n")
