"""R-61N CHAT-NT1 (MTP Head, Sun 11-01; NewIdeas' QUEUE row, from the orchestrator's night brief): near-tie (NON-exact) acceptance
on CHAT, never measured (NT-1 ran 24 HumanEval samples at block 32). A copy of NewIdeas' scripts/nt_accept.py replay (L1-20,
cand21's selector at t 0.0, teacher-forced) + nt_analyze.py's rules, run on D-02a's captures (data/d02a: the drafter's taps f,
the 125B's own top-32 tk / lp per position, row i predicting token i + 1): tf_ifeval 30, tf_multi 24, agent 12, mgsm 30, and
tf_gsm 60 as the math reference. Added: the drafter's top-3 at each first miss (the selector's three best candidates there,
the prefix being all hits), for the top-3 oracle (any of them inside the accept set). Block 16 (the 45 GB loop's checks) by
default, NT_BS=32 for NT-1's.
Rules (nt_analyze.py): forgive a first miss when the drafter's token has p_d >= rho * p_top1 (rho 0.5 / 0.25 / 0.1); LB = +1
token per forgiven block, MOD = the run continues with the exact replay's conditional survival and later misses forgiven at the
rule's first-miss rate (an estimate, not a bound). Quality is not measured here.
PASS (QUEUE): chat tokens per pass +30 % or better at rho 0.25 (MOD) -> a set-trained drafter row + Kamen's quality test;
KILL: under +10 % at rho 0.1. Inputs read-only; MLX on the CPU (NT_DEV=gpu to change).
Output: ~/TrainLab/data/r61n/chatnt_b{BS}.jsonl + .txt"""
import os, sys, glob, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
if os.environ.get("NT_DEV", "cpu") == "cpu":
    mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_third_party"))
import dflash_model_mlx as D

class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
BS = int(os.environ.get("NT_BS", "16"))
OUT = os.path.expanduser("~/TrainLab/data/r61n"); os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(OUT, f"chatnt_b{BS}.jsonl")
DOMS = os.environ.get("NT_DOMS", "tf_ifeval,tf_multi,agent,mgsm,tf_gsm").split(",")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

if os.environ.get("NT_ANALYZE_ONLY") != "1":
    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    tgt = Target()
    tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                     + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    del w
    draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    draft.bind(tgt)
    draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
    mx.eval(draft.parameters(), tgt.parameters())
    MASK = int(draft.config.mask_token_id)

    def select_conf(block, hidden, logits):
        """cand21.py's selector at temperature 0 (nt_accept.py's), + the three best candidates at every step"""
        sel = draft.candidate_selector
        cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
        unary = mx.take_along_axis(logits, cand, axis=-1)
        hp = sel.hidden_projection(hidden)
        pred, path, conf, top3 = block[:, 0], [], [], []
        for p in range(hp.shape[1]):
            sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None] * sel.successor_codebook(cand[:, p]), axis=-1)
            s = mx.argmax(sc, axis=-1)
            conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s[:, None], axis=-1)[:, 0])
            o3 = mx.argsort(-sc, axis=-1)[:, :3]
            top3.append(mx.take_along_axis(cand[:, p], o3, axis=-1))
            pred = mx.take_along_axis(cand[:, p], s[:, None], axis=-1)[:, 0]
            path.append(pred)
        return mx.stack(path, axis=1), mx.stack(conf, axis=1), mx.stack(top3, axis=1)

    done = set()
    if os.path.exists(SRC):
        done = {json.loads(l)["key"] for l in open(SRC)}
    t0 = time.time()
    for dom in DOMS:
        for path in sorted(glob.glob(os.path.expanduser(f"~/TrainLab/data/d02a/{dom}_*.npz"))):
            key = os.path.basename(path)[:-4]
            if key in done:
                continue
            z = np.load(path)
            f = z["f"].reshape(z["f"].shape[0], -1); ids = z["ids"]; off, gen = int(z["off"]), int(z["gen"]); n = len(ids)
            tk, lp = z["tk"], z["lp"].astype(np.float64)
            cache = draft.make_cache()
            start = max(off, gen - (draft.config.sliding_window - 1))
            for c in cache:
                c.offset = start
            fed, a, recs = start, gen, []
            while a < off + n - 1:
                ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
                blk = mx.array([[int(ids[a - off])] + [MASK] * (BS - 1)])
                h = draft.hidden_states(blk, ctx, cache, 1)
                logits = draft.compute_logits(h)
                fed = a
                true = np.asarray(ids[a - off + 1:a - off + BS])
                p, c, t3 = select_conf(blk, h, logits); mx.eval(p, c, t3)
                d = np.array(p[0])[:len(true)]
                hit = d == true
                k = int(np.argmin(hit)) if not hit.all() else len(true)
                rem = off + n - 1 - a
                acc = min(k + 1, rem)
                r = {"dom": dom, "key": key, "a": int(a), "k": k, "acc": int(acc), "nb": int(len(true)), "rem": int(rem),
                     "hit": hit.astype(int).tolist(), "p1": [round(float(np.exp(lp[a - off + j, 0])), 5) for j in range(len(true))]}
                if k < len(true):
                    i = a - off + k
                    wh = np.where(tk[i] == d[k])[0]
                    d3 = [int(x) for x in np.array(t3[0, k])]
                    r3 = [int(np.where(tk[i] == x)[0][0]) if (tk[i] == x).any() else -1 for x in d3]
                    r.update(t1=int(tk[i, 0]), dt=int(d[k]), rank=int(wh[0]) if len(wh) else -1,
                             pd=float(np.exp(lp[i, wh[0]])) if len(wh) else 0.0, p32=float(np.exp(lp[i, -1])),
                             d3=d3, r3=r3, p3=[float(np.exp(lp[i, j])) if j >= 0 else 0.0 for j in r3])
                recs.append(r)
                if (trim := cache[0].offset - a) > 0:
                    D._trim_recent_cache(cache, trim)
                a = a + acc
            with open(SRC, "a") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"{key}: {len(recs)} blocks, {sum(r['acc'] for r in recs) / max(1, len(recs)):.2f} per pass, {time.time() - t0:.0f} s", flush=True)

# ---------------------------------------------------------------- analysis (nt_analyze.py's rules, per domain)
ALL = [json.loads(l) for l in open(SRC)]
GROUPS = [("chat (ifeval + multi)", ("tf_ifeval", "tf_multi")), ("ifeval", ("tf_ifeval",)), ("multi", ("tf_multi",)),
          ("agent", ("agent",)), ("mgsm", ("mgsm",)), ("gsm (reference)", ("tf_gsm",))]
RHOS = (0.5, 0.25, 0.1)
BINS = [(0.99, 1.01, "p >= 0.99"), (0.9, 0.99, "0.9-0.99"), (0.5, 0.9, "0.5-0.9"), (0.0, 0.5, "< 0.5")]
rng = np.random.default_rng(0)
say(f"R-61N CHAT-NT1, block {BS} (L1-20, cand21's selector at t 0.0, teacher-forced on D-02a's captures; {len(ALL)} blocks)")
verdict = {}
for name, doms in GROUPS:
    recs = [r for r in ALL if r["dom"] in doms]
    if not recs:
        continue
    nb_max = BS - 1
    tok_exact = np.mean([r["acc"] for r in recs])
    miss = [r for r in recs if r["k"] < r["nb"]]
    say(f"\n== {name}: {len(recs)} blocks from {len({r['key'] for r in recs})} answers; exact tokens per pass {tok_exact:.2f}; "
        f"first misses {len(miss)}")
    for lo, hi, nm in BINS:
        sub = [r for r in miss if lo <= r["p1"][r["k"]] < hi]
        say(f"    first misses at the 125B's top-1 p {nm:10s}: {len(sub):4d} ({len(sub) / max(1, len(miss)):.2f}); drafter's token in "
            f"its top-32 {sum(r['rank'] >= 0 for r in sub) / max(1, len(sub)):.2f}, median p_d "
            f"{np.median([r['pd'] for r in sub]) if sub else 0:.3f}")
    reach = np.zeros(nb_max); hits = np.zeros(nb_max)
    for r in recs:
        for j in range(min(r["nb"], r["k"] + 1)):
            reach[j] += 1; hits[j] += r["hit"][j]
    surv = hits / np.maximum(1, reach)
    say("    conditional survival P(hit at depth j | hits before), j = 1..: " + " ".join(f"{s:.2f}" for s in surv))
    def score(rule, label):
        forg = [r for r in miss if rule(r)]
        f = len(forg) / max(1, len(miss))
        lb = np.mean([r["acc"] + (1 if r["k"] < r["nb"] and rule(r) and r["k"] + 2 <= r["rem"] else 0) for r in recs])
        tot, RUNS = 0.0, 200
        for _ in range(RUNS):
            for r in recs:
                if r["k"] >= r["nb"] or not rule(r):
                    tot += r["acc"]; continue
                j = r["k"] + 1
                while j < r["nb"]:
                    if rng.random() < surv[j] or rng.random() < f:
                        j += 1
                    else:
                        break
                tot += min(j + 1, r["rem"])
        mod = tot / (RUNS * len(recs))
        say(f"    {label:30s}: forgives {f:.2f} of first misses ({len(forg)}); tokens per pass LB {lb:.2f} ({lb / tok_exact - 1:+.0%}), "
            f"MOD {mod:.2f} ({mod / tok_exact - 1:+.0%})")
        return mod / tok_exact - 1
    g = {}
    for rho in RHOS:
        g[rho] = score(lambda r, rho=rho: r["pd"] >= rho * r["p1"][r["k"]], f"rho {rho} (drafter's token)")
    for rho in RHOS:
        score(lambda r, rho=rho: any(p >= rho * r["p1"][r["k"]] for p in r.get("p3", [])), f"rho {rho}, drafter's top-3 oracle")
    score(lambda r: r["t1"] in r.get("d3", []), "exact top-3 oracle (125B's #1 in it)")
    verdict[name] = g
c = verdict.get("chat (ifeval + multi)")
if c:
    v = "PASS" if c[0.25] >= 0.30 else "KILL" if c[0.1] < 0.10 else "BETWEEN"
    say(f"\nR-61N verdict on chat (PASS: MOD +30 % at rho 0.25; KILL: MOD under +10 % at rho 0.1): {v} "
> (a quote from the project owner, kept private)
open(os.path.join(OUT, f"chatnt_b{BS}.txt"), "w").write("\n".join(lines) + "\n")
