"""WEEK98 lever 1c screen (MoonShots, Sun 15 Nov 2026; owner by NewIdeas 01:17): distil Qwen3.5-2B toward the 125B's
choices, then measure its teacher-forced agreement with the 125B's runtime top-1 (the auditor's flat level).

Cheapest safe form: the frozen 2B runs layers 0-22 once per file (inference kernels); its residual entering layer 23 (a
full-attention layer, so no Gated-DeltaNet backward is needed) is cached. LoRA (rank 16) is trained on layer 23's linear
layers only, with the cross-entropy to the 125B's top-32 distribution (the bank's tk / lp, renormalized) on the answer rows
of the bank's 360 train files. The final norm and the tied lm_head stay frozen.

Reported, in this one harness, before (control = LoRA off; must reproduce lever7_agree.py's 94.42 / 85.67 / 87.29 +- 0.1)
and after training: agreement on HE+ 24, MBPP 100, val 40, all rows and sure rows (125B margin >= 4 nats).
LINE (NewIdeas 01:17, written first): PASS if MBPP >= 89.0 and val rises too (no overfit) -> the join with the
distilled auditor, a longer run on n1112; information line for the 98 %-class route: HE+ >= 97.0 and MBPP >= 91.0; for
reference, the 125B head's depth-0 accuracy with the true state is HE+ 97.9, MBPP 94.0. KILL if MBPP < +1 pt.
Read-only inputs: the bank, the proxy's rows_va.json, Sandbox98's sb/core.py (the model loader). Writes ~/MoonShots/week98/.
usage: ~/BigMoE/.venv/bin/python -I lever1c_distill.py [--minutes 8] [--rank 16] [--lr 1e-4]"""
import argparse, json, os, sys, time
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98"))
BANK = H("~/datasets/deepdraft/bank")
SPLIT = H("~/BigMoE/data/deepdraft/bank/split.json")
OUT = H("~/MoonShots/week98")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    from sb import core as C
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
            x = pre23(ids)
            out.append(dict(x=x.astype(mx.bfloat16), rows=rr, tk=tk[rr], lp=lp[rr], nm=nm))
        return out

    ans = lambda nm, z: range(int(z["gen"]) - 1, z["raw"].shape[0])
    va_rows = {}
    for nm, r in json.load(open(os.path.join(BANK, "proxy", "rows_va.json"))):
        va_rows.setdefault(nm, []).append(r)
    sets = {"heplus": load_set(split["heplus"], ans), "mbpp": load_set(split["mbpp"], ans),
            "val": load_set(list(va_rows), lambda nm, z: va_rows[nm])}
    train = load_set(split["train"], ans)
    print(f"cached layer-23 inputs: {sum(len(v) for v in sets.values())} eval + {len(train)} train files, {time.time() - t0:.0f}s", flush=True)

    def head(x, rows):                                    # layer 23 -> norm -> tied lm_head, on the scored rows
        fa = create_attention_mask(x, None)
        h = L23(x, mask=fa, cache=None)
        return t.embed_tokens.as_linear(t.norm(h[0, mx.array(rows)])).astype(mx.float32)

    def agree():
        res = {}
        for st, items in sets.items():
            n = s = ns = ss = 0
            for it in items:
                top = np.asarray(mx.argmax(head(it["x"], it["rows"]), axis=-1))
                hit = top == it["tk"][:, 0]; m = it["lp"][:, 0] - it["lp"][:, 1]
                n += len(hit); s += int(hit.sum()); ns += int((m >= 4).sum()); ss += int((hit & (m >= 4)).sum())
            res[st] = dict(all=round(100 * s / n, 2), sure=round(100 * ss / max(ns, 1), 2), rows=n)
        return res

    # LoRA on layer 23's linear layers (q / k / v / o and the MLP), everything else frozen
    t.freeze()
    def wrap(mod, path=""):
        for k, v in mod.children().items():
            if isinstance(v, (nn.Linear, nn.QuantizedLinear)):
                setattr(mod, k, LoRALinear.from_base(v, r=a.rank, dropout=0.0, scale=10.0))
            elif isinstance(v, nn.Module):
                wrap(v, path + "." + k)
    wrap(L23)
    for k_, v_ in tree_flatten(L23.trainable_parameters()):
        pass
    ctl = agree()                                         # LoRA's B starts at zero: this is the untrained 2B
    print("control (LoRA zero):", ctl, flush=True)

    def loss_fn(x, rows, tk, lp):
        lq = head(x, rows)
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        p = mx.softmax(mx.array(lp), axis=-1)             # the 125B's top-32, renormalized
        return -(p * mx.take_along_axis(lq, mx.array(tk.astype(np.int32)), axis=-1)).sum(-1).mean()

    opt = optim.Adam(learning_rate=a.lr)
    vg = nn.value_and_grad(L23, loss_fn)
    rng = np.random.default_rng(1115)
    step, ep, tstart, losses = 0, 0, time.time(), []
    while (time.time() - tstart) < 60 * a.minutes:
        for i in rng.permutation(len(train)):
            it = train[i]
            l, g = vg(it["x"], it["rows"], it["tk"], it["lp"])
            opt.update(L23, g); mx.eval(L23.parameters(), opt.state)
            losses.append(float(l)); step += 1
            if step % 100 == 0:
                print(f"step {step} ep {ep} loss {np.mean(losses[-100:]):.4f} {time.time() - tstart:.0f}s", flush=True)
            if (time.time() - tstart) >= 60 * a.minutes:
                break
        ep += 1
    after = agree()
    print("after:", after, flush=True)
    res = dict(control=ctl, after=after, steps=step, epochs=ep, rank=a.rank, lr=a.lr, minutes=a.minutes,
               loss_first100=float(np.mean(losses[:100])), loss_last100=float(np.mean(losses[-100:])),
               seconds=round(time.time() - t0, 1))
    json.dump(res, open(os.path.join(OUT, "lever1c_distill.json"), "w"), indent=1)
    mx.save_safetensors(os.path.join(OUT, "lever1c_lora_l23.safetensors"), dict(tree_flatten(L23.trainable_parameters())))
    print("written", flush=True)


if __name__ == "__main__":
    main()
