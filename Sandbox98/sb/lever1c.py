"""WEEK98 lever 1c in Sandbox98 (ported from MoonShots' ~/MoonShots/week98/lever1c_distill.py, read-only; owner moved to
Sandbox98 by NewIdeas 01:17 / WEEK98.md 01:2x): distil Qwen3.5-2B toward the 125B's choices, then measure its
teacher-forced top-1 agreement with the 125B's runtime top-1 (the auditor's flat level, MoonShots' audit law).

Form (MoonShots'): the frozen 2B runs layers 0-22 once per file; the residual entering layer 23 (full attention, so no
Gated-DeltaNet backward) is cached in bf16. LoRA (rank r, scale 10) on layer 23's linears, cross-entropy to the 125B's
top-32 distribution (the bank's tk / lp, renormalized) on the answer rows of the bank's 360 train files (n1112 files t30b
never saw). The final norm and the tied lm_head stay frozen.
Added here: read-outs at several training times (the slope decides "more budget" vs KILL), the unsure rows (125B margin
< 4 nats, where 87 % of the 2B's misses sit), flips against the control, and a bootstrap CI over files.

LINE (NewIdeas 01:17, written before any run): PASS if MBPP >= 89.0 and val rises too (no overfit); information line for
the 98 %-class route: HE+ >= 97.0 and MBPP >= 91.0; KILL if MBPP < +1 pt (< 86.67). Control (LoRA zero) must reproduce
lever7_agree.py's 94.42 / 85.67 / 87.29 +- 0.1.
Read-only inputs: ~/datasets/deepdraft/bank, ~/BigMoE/data/deepdraft/bank/split.json. Writes ~/Sandbox98 only.
usage: lever1c.py [--minutes 8] [--evals 1,2,4] [--rank 16] [--lr 1e-4] [--tag a]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H = os.path.expanduser
BANK = H("~/datasets/deepdraft/bank")
SPLIT = H("~/BigMoE/data/deepdraft/bank/split.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--evals", default="1,2,4", help="extra read-outs at these training minutes (the end is always read)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--smoke", type=int, default=0, help="files per set for a smoke test")
    ap.add_argument("--train-head", default="bf16", choices=["bf16", "q4"], help="lm_head copy used by the training loss "
                    "(read-outs always use the target's own 4-bit head)")
    ap.add_argument("--save-every", type=float, default=90.0, help="seconds of training between resume states")
    a = ap.parse_args()
    name = f"lever1c_{a.tag}" + ("_smoke" if a.smoke else "")
    from sb import core as C
    from sb import resume as R
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.tuner.lora import LoRALinear
    C.limit_memory(9)
    t0 = time.time()
    t = C.tm()
    L23 = t.layers[23]
    assert not L23.is_linear, "layer 23 must be a full-attention layer"
    split = json.load(open(SPLIT))
    cut = (lambda v: v[:a.smoke]) if a.smoke else (lambda v: v)

    def pre23(ids):                                       # the frozen trunk up to layer 22 (inference kernels)
        x = t.embed_tokens(mx.array(np.asarray(ids, np.int32))[None])
        fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
        for i in range(23):
            x = t.layers[i](x, mask=ssm if t.layers[i].is_linear else fa, cache=None)
        mx.eval(x)
        return x

    def load_set(names, rows_of):
        out = []
        for nm in names:
            C.wait_pause()
            z = np.load(os.path.join(BANK, nm))
            ids, tk, lp = np.asarray(z["ids"]), np.asarray(z["tk"]), np.asarray(z["lp"], np.float32)
            rr = np.array([r for r in rows_of(nm, z) if r + 1 < len(ids)])
            x = pre23(ids).astype(mx.bfloat16)
            mx.eval(x)
            out.append(dict(x=x, rows=rr, tk=tk[rr], lp=lp[rr], nm=nm))
        return out

    ans = lambda nm, z: range(int(z["gen"]) - 1, z["raw"].shape[0])
    va_rows = {}
    for nm, r in json.load(open(os.path.join(BANK, "proxy", "rows_va.json"))):
        va_rows.setdefault(nm, []).append(r)
    assert not set(va_rows) & set(split["train"]), "val and train overlap"
    sets = {"heplus": load_set(cut(split["heplus"]), ans), "mbpp": load_set(cut(split["mbpp"]), ans),
            "val": load_set(cut(list(va_rows)), lambda nm, z: va_rows[nm])}
    train = load_set(cut(split["train"]), ans)
    ntr = sum(len(it["rows"]) for it in train)
    print(f"[1c] cached layer-23 inputs: {sum(len(v) for v in sets.values())} eval + {len(train)} train files "
          f"({ntr} train answer rows), {time.time() - t0:.0f}s", flush=True)

    def head(x, rows):                                    # layer 23 -> norm -> tied lm_head, on the scored rows
        fa = create_attention_mask(x, None)
        h = L23(x, mask=fa, cache=None)
        return t.embed_tokens.as_linear(t.norm(h[0, mx.array(rows)])).astype(mx.float32)

    def hits():                                           # per set: per file (hit vector, 125B margin)
        res = {}
        for st, items in sets.items():
            res[st] = []
            for it in items:
                top = np.asarray(mx.argmax(head(it["x"], it["rows"]), axis=-1))
                res[st].append((top == it["tk"][:, 0], it["lp"][:, 0] - it["lp"][:, 1]))
        return res

    rng_b = np.random.default_rng(1115)

    def summary(hh, ctl=None):
        out = {}
        for st, per in hh.items():
            hit = np.concatenate([h for h, _ in per]); m = np.concatenate([m_ for _, m_ in per])
            r = dict(all=round(100 * hit.mean(), 2), sure=round(100 * hit[m >= 4].mean(), 2),
                     unsure=round(100 * hit[m < 4].mean(), 2), rows=int(len(hit)), unsure_rows=int((m < 4).sum()))
            if ctl is not None:
                c = np.concatenate([h for h, _ in ctl[st]])
                r["gained"] = int((hit & ~c).sum()); r["lost"] = int((~hit & c).sum())
                fh = np.array([h.sum() for h, _ in per], float); fc = np.array([h.sum() for h, _ in ctl[st]], float)
                fn = np.array([len(h) for h, _ in per], float)
                bs = []
                for _ in range(2000):
                    i = rng_b.integers(0, len(fn), len(fn))
                    bs.append((fh[i].sum() - fc[i].sum()) / fn[i].sum())
                r["diff"] = round(100 * (fh.sum() - fc.sum()) / fn.sum(), 2)
                r["ci95"] = [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]
            out[st] = r
        return out

    # LoRA on layer 23's linear layers (q / k / v / o and the MLP), everything else frozen
    t.freeze()

    def wrap(mod):
        for k, v in mod.children().items():
            if isinstance(v, (nn.Linear, nn.QuantizedLinear)):
                setattr(mod, k, LoRALinear.from_base(v, r=a.rank, dropout=0.0, scale=a.scale))
            elif isinstance(v, nn.Module):
                wrap(v)
    wrap(L23)
    npar = sum(v.size for _, v in tree_flatten(L23.trainable_parameters()))
    hc = hits()                                           # LoRA's B starts at zero: this is the untrained 2B
    ctl = summary(hc)
    print(f"[1c] control (LoRA zero, {npar} trainable params): {json.dumps(ctl)}", flush=True)

    def head_train(x, rows):                              # the same head with the bf16 dequantized lm_head copy: the
        fa = create_attention_mask(x, None)               # 4-bit path's backward to its input is ~5x slower (core.py)
        h = L23(x, mask=fa, cache=None)
        return C.lm_logits_train(t.norm(h[0, mx.array(rows)]))

    def loss_fn(x, rows, tk, lp):
        lq = head_train(x, rows) if a.train_head == "bf16" else head(x, rows)
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        p = mx.softmax(mx.array(lp), axis=-1)             # the 125B's top-32, renormalized
        return -(p * mx.take_along_axis(lq, mx.array(tk.astype(np.int32)), axis=-1)).sum(-1).mean()

    opt = optim.Adam(learning_rate=a.lr)
    vg = nn.value_and_grad(L23, loss_fn)
    rng = np.random.default_rng(1115)
    marks = sorted(float(m) for m in a.evals.split(",") if m and float(m) < a.minutes)
    reads = {}
    step, ep, losses, train_s, perm, pos = 0, 0, [], 0.0, None, 0
    done = False
    sp = os.path.join(C.ROOT, "ckpt", name + "_state")         # mid-step resume (sb/resume.py)
    R.arm_sigterm()
    meta = R.load(sp, {"l23": L23}, {"l23": opt})
    if meta:
        step, ep, train_s, pos, losses, reads, marks = (meta[k] for k in ("step", "ep", "train_s", "pos", "losses", "reads", "marks"))
        perm = np.array(meta["perm"]); rng.bit_generator.state = meta["rng"]
        print(f"[1c] resumed at step {step} ({train_s:.0f}s of training done; reads {list(reads)})", flush=True)
    last_save = train_s

    def snap():
        return dict(step=step, ep=ep, train_s=train_s, pos=pos, perm=[int(v) for v in perm], losses=losses, reads=reads,
                    marks=marks, rng=rng.bit_generator.state)
    while not done:
        if perm is None or pos >= len(perm):
            if perm is not None:
                ep += 1
            perm, pos = rng.permutation(len(train)), 0
        i = int(perm[pos]); pos += 1
        C.wait_pause()
        ts = time.time()
        it = train[i]
        l, g = vg(it["x"], it["rows"], it["tk"], it["lp"])
        opt.update(L23, g); mx.eval(L23.parameters(), opt.state)
        train_s += time.time() - ts
        losses.append(float(l)); step += 1
        if step % 100 == 0:
            print(f"[1c] step {step} ep {ep} loss {np.mean(losses[-100:]):.4f} train {train_s:.0f}s", flush=True)
        if marks and train_s >= 60 * marks[0]:
            m = marks.pop(0)
            reads[f"{m:g}min"] = dict(summary(hits(), hc), steps=step, loss100=float(np.mean(losses[-100:])))
            r = reads[f"{m:g}min"]
            print(f"[1c] read at {m:g} min (step {step}): " + " ".join(
                f"{st} {r[st]['all']:.2f} ({r[st]['diff']:+.2f}, unsure {r[st]['unsure']:.2f})" for st in sets), flush=True)
            R.save(sp, {"l23": L23}, {"l23": opt}, snap()); last_save = train_s
        if train_s >= 60 * a.minutes:
            done = True
            break
        if R.STOP["f"] or train_s - last_save >= a.save_every:
            wrote = R.save(sp, {"l23": L23}, {"l23": opt}, snap()); last_save = train_s
            if R.STOP["f"]:
                print(f"[1c] SIGTERM: state saved at step {step} ({wrote / 1e6:.1f} MB), exiting to resume later", flush=True)
                sys.exit(75)
    R.finish(sp)
    after = summary(hits(), hc)
    print(f"[1c] after {a.minutes:g} min ({step} steps, {ep} epochs): {json.dumps(after)}", flush=True)
    mb = after["mbpp"]["all"]; cm = ctl["mbpp"]["all"]
    verdict = ("PASS" if mb >= 89.0 and after["val"]["all"] > ctl["val"]["all"] else
               "KILL" if mb < cm + 1.0 else "BETWEEN")
    res = dict(verdict=verdict, control=ctl, after=after, reads=reads, steps=step, epochs=round(ep + pos / len(train), 2), rank=a.rank, scale=a.scale,
               lr=a.lr, minutes=a.minutes, train_head=a.train_head, train_answer_rows=ntr, trainable_params=npar,
               loss_first100=float(np.mean(losses[:100])), loss_last100=float(np.mean(losses[-100:])),
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.join(C.ROOT, "ckpt"), exist_ok=True)
    json.dump(res, open(os.path.join(C.RES, name + ".json"), "w"), indent=1)
    mx.save_safetensors(os.path.join(C.ROOT, "ckpt", name + "_lora_l23.safetensors"), dict(tree_flatten(L23.trainable_parameters())))
    print(f"[1c] DONE verdict {verdict} (MBPP {cm:.2f} -> {mb:.2f}; line PASS >= 89.0 with val up, KILL < {cm + 1:.2f})", flush=True)


if __name__ == "__main__":
    main()
