"""TrainLab: on-policy agentic fine-tune of the 35B's MTP head (NewIdeas 22:35: one turn; if the held-out probe stays
flat, the MTP-drafter line closes). Kamen's workload (edit / bash / write / text turns of a coding agent) is where
the stock head's chained acceptance is already high (tau16 7.7-10.3); the generic off-policy fine-tune did not move.
  gen    ~N Claude Code turns from the main account's transcripts (SpecDecode's extract.py + render.py, read-only;
         SpecDecode's 29 samples excluded, so Chat1000's 9 held-out turns never leak) -> the 35B's greedy
         continuation (exact, 256 tokens) -> one teacher-forced pass over prompt + continuation -> features of the
         last W prompt tokens + the continuation (post-norm h fp16, top-64) -> data/mtp_onpolicy/<i>.npz
  train  mtp35's training-time test on those windows, loss only on chain starts inside the continuation (the
         positions the drafter serves), held-out probe every 50 steps; then mtp35's eval (per class)
usage: mtp_onpolicy.py gen [--n 120] [--time-limit S]   |   mtp_onpolicy.py train --run NAME [--lr 3e-6] ...
"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import train35 as T
import mtp35 as M
import decode35 as DC

OUT = os.path.join(T.TL, "data/mtp_onpolicy")
W = 768
GEN = 256


def samples(n):
    import extract as X
    from render import render
    held = [json.loads(l) for l in open(os.path.expanduser("~/SpecDecode/data/samples.jsonl"))]
    held_targets = {json.dumps(s["target"], sort_keys=True)[:400] for s in held}
    random.seed(1)
    cands = {k: [] for k in ("edit", "write", "bash", "text")}
    for path in glob.glob(f"{X.PROJ}/*/*.jsonl"):
        if "SpecDecode" in path:
            continue
        msgs = X.load_session(path)
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or i < 4:
                continue
            ctx = X.context_before(msgs, i)
            kind = X.classify(m, ctx)
            if kind and json.dumps(m, sort_keys=True)[:400] not in held_targets:
                cands[kind].append({"kind": kind, "project": path.split("/")[-2], "messages": ctx, "target": m})
    want = {"edit": int(n * 0.4), "bash": int(n * 0.25), "write": int(n * 0.15), "text": n - int(n * 0.4) - int(n * 0.25) - int(n * 0.15)}
    out = []
    for kind, k in want.items():
        pool = cands[kind]; random.shuffle(pool)
        seen = {}
        for c in pool:
            if seen.get(c["project"], 0) < 6:
                out.append(c); seen[c["project"]] = seen.get(c["project"], 0) + 1
            if sum(1 for o in out if o["kind"] == kind) >= k:
                break
    return out, render


def cmd_gen(a):
    t0 = time.time()
    limit = a.time_limit or float(os.environ.get("TL_LIMIT", "1e9"))
    os.makedirs(OUT, exist_ok=True)
    model, tok = T.setup()
    sm, render = samples(a.n)
    items = []
    for j, s in enumerate(sm):
        try:
            ids, _ = render(s, tok)
        except ValueError:
            continue
        if len(ids) > 9000:                              # keep the system prompt + tools, and the recent context
            ids = ids[:600] + ids[-8400:]
        items.append({"key": f"{s['kind']}{j:03d}", "kind": s["kind"], "prompt_ids": ids, "max_tokens": GEN})
    items = [it for it in items if not os.path.exists(os.path.join(OUT, it["key"] + ".npz"))]
    items.sort(key=lambda it: len(it["prompt_ids"]))
    print(f"gen: {len(items)} prompts to do (median {np.median([len(i['prompt_ids']) for i in items]):.0f} tokens)", flush=True)
    flow = T.Flow("exact")
    lm = model.language_model
    done = 0
    for b in range(0, len(items), a.batch):
        if time.time() - t0 > limit - 120:
            print("gen: time limit", flush=True); break
        batch = items[b:b + a.batch]
        res = DC.decode_batch(model, flow, batch, print)
        for it in batch:
            cont, fin = res[it["key"]]
            full = it["prompt_ids"] + cont
            ids = mx.array(full)[None]
            h, top, tlp = M.target_features(model, ids)
            s0 = max(0, len(it["prompt_ids"]) - W)
            np.savez(os.path.join(OUT, it["key"] + ".npz"), ids=np.array(full[s0:], np.int32),
                     h=np.array(h[0, s0:].astype(mx.float16)), top=np.array(top[0, s0:], np.int32),
                     tlp=np.array(tlp[0, s0:].astype(mx.float16)), c0=np.int32(len(it["prompt_ids"]) - s0),
                     kind=it["kind"])
            done += 1
        print(f"gen: {done} done, {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def kd_masked(head, lm_head, feats, top, tlp, start):
    """mtp35.kd_depths restricted to chain starts >= start."""
    tot, per = 0.0, []
    Tn = top.shape[1]
    for j, f in enumerate(feats, start=1):
        n = Tn - j - 1
        a0 = max(0, start)
        if n <= a0:
            continue
        ttop, ttlp = top[:, j + a0:j + n], tlp[:, j + a0:j + n]
        inv = ttop < M.V
        tl = mx.where(inv, ttlp, -mx.inf)
        lg = M.draft_logits(lm_head, f[:, a0:n]).astype(mx.float32)
        lq = mx.take_along_axis(lg, mx.where(inv, ttop, 0), axis=-1) - mx.logsumexp(lg, axis=-1, keepdims=True)
        lq = mx.where(inv, lq, -mx.inf)
        p, q = mx.exp(tl), mx.exp(lq)
        pt = mx.maximum(1 - p.sum(-1), 1e-12); qt = mx.maximum(1 - q.sum(-1), 1e-12)
        kl = (mx.where(inv, p * (tl - lq), 0.0)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))
        per.append(kl.mean()); tot = tot + per[-1]
    return tot / max(len(per), 1), per


def cmd_train(a):
    t0 = time.time()
    limit = a.time_limit or float(os.environ.get("TL_LIMIT", "1e9"))
    model, tok = T.setup()
    lm = model.language_model
    embed = lm.model.embed_tokens
    ck = os.path.join(T.TL, "ckpt", a.run + ".mtp")
    head = M.load_head(model, a.run if os.path.exists(ck + ".json") else None)
    state = json.load(open(ck + ".json")) if os.path.exists(ck + ".json") else {"tokens": 0, "step": 0}
    files = sorted(glob.glob(os.path.join(OUT, "*.npz")))
    print(f"train: {len(files)} on-policy windows", flush=True)
    data = []
    for f in files:
        z = np.load(f)
        data.append((mx.array(z["ids"])[None], mx.array(z["h"])[None], mx.array(z["top"])[None],
                     mx.array(z["tlp"].astype(np.float32))[None], int(z["c0"])))
    probe = []
    for name, p_, ans in M.heldout_sets(tok)[30:39:2]:          # held-out agentic turns (route4), first 1024 tokens of answer context
        full = (p_ + ans)[-1024:]
        ids_p = mx.array(full)[None]
        hp, tp_, tl_ = M.target_features(model, mx.array(p_ + ans)[None])
        n0 = len(p_ + ans) - len(full)
        probe.append((ids_p, hp[:, n0:], tp_[:, n0:], tl_[:, n0:], max(0, len(full) - len(ans) - 1)))

    def probe_kl():
        per = np.zeros(a.depth)
        for ids_p, hp, tp_, tl_, st in probe:
            _, pr = kd_masked(head, lm.lm_head, M.unrolled(head, embed, hp, ids_p, a.depth), tp_, tl_, st)
            per[:len(pr)] += np.array([float(x) for x in pr])
        return per / len(probe)

    pk = probe_kl()
    print(f"probe step {state['step']} held-out agentic KL per depth " + " ".join(f"{x:.3f}" for x in pk) + f" (mean {pk.mean():.3f})", flush=True)
    opt = optim.Adam(learning_rate=lambda s: a.lr * mx.minimum(1.0, (s + 1) / 20.0))
    lossf = nn.value_and_grad(head, lambda hd, ids, h, top, tlp, st: kd_masked(hd, lm.lm_head, M.unrolled(hd, embed, h, ids, a.depth), top, tlp, st))
    rng = np.random.default_rng(state["step"])
    hist = []
    while time.time() - t0 < limit - 60:
        ids, h, top, tlp, c0 = data[int(rng.integers(len(data)))]
        (loss, per), g = lossf(head, ids, h, top, tlp, c0 - 1)
        g, gn = optim.clip_grad_norm(g, 1.0)
        mx.eval(loss, gn)
        if not np.isfinite(float(loss)):
            state["step"] += 1; continue
        opt.update(head, g); mx.eval(head.trainable_parameters(), opt.state)
        state["step"] += 1; state["tokens"] += int(ids.shape[1]) - c0
        hist.append([float(x) for x in per])
        if state["step"] % 25 == 0:
            m = np.mean([x for x in hist if len(x) == a.depth], 0) if hist else []
            hist = []
            print(f"step {state['step']} cont-tokens {state['tokens']} KL per depth " + " ".join(f"{x:.3f}" for x in m)
                  + f" |g| {float(gn):.2f} {time.time() - t0:.0f}s", flush=True)
        if state["step"] % 50 == 0:
            pk = probe_kl()
            print(f"probe step {state['step']} held-out agentic KL per depth " + " ".join(f"{x:.3f}" for x in pk) + f" (mean {pk.mean():.3f})", flush=True)
    mx.save_safetensors(ck + ".safetensors", dict(tree_flatten(head.trainable_parameters())))
    json.dump(state, open(ck + ".json", "w"))
    print(f"saved {a.run} at step {state['step']}", flush=True)


def main():
    T.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--run", default="mtp_onpol")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--depth", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--time-limit", type=float, default=0)
    a = ap.parse_args()
    {"gen": cmd_gen, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
