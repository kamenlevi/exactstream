"""PORT reference (MTP Head, Wed 11-04 20:2x): runs the MLX trainer's own code (sticky/d09_train.py, executed up to its
'modes' section, so fwd / heads / loss_fn / make_batch are the trainer's, not copies) on MLX's CPU device, and saves what
the PyTorch port must reproduce.
  fwd   : one batch (the first --groups train groups, pmask from the trainer's PM_RNG): hidden, logits rows, loss.
  train : --steps optimizer steps on fixed items (step k uses train group k), first --layers layers only (memory),
          the trainer's own step(): micro-batches, summed grads, clip 1.0, MLX AdamW (no bias correction).
Usage: [BigMoE venv] python ref_mlx.py fwd|train OUT.npz [--A 8] [--groups 1] [--steps 5] [--layers 1] [--pmask 0.5]"""
import argparse, gc, math, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("out")
ap.add_argument("--A", type=int, default=8); ap.add_argument("--groups", type=int, default=1)
ap.add_argument("--steps", type=int, default=5); ap.add_argument("--layers", type=int, default=0)
ap.add_argument("--pmask", type=float, default=0.5); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--S", type=int, default=1)
ap.add_argument("--gpu", action="store_true")      # Thu 11-05: MLX's CPU qmm is ~4 % off; the GPU is the reference
ap.add_argument("--freeze-layers", action="store_true")   # grad: only the final norm + the selector get gradients
ap.add_argument("--data", default="~/Leap/data/l120")       # a folder with only the windows needed (memory)
ap.add_argument("--freeze-selector", action="store_true")   # train: no selector parameters (memory)
ap.add_argument("--dp-json", default="")                    # train: save the step-1 parameter change (grad_stats JSON)
ap.add_argument("--save-after1", default="")                # train: save the weights after step 1 (npz, fp32)
ap.add_argument("--params", default="")                     # evalp: these weights (npz) -> step 2's batch loss
ap.add_argument("--only", default="")                       # evalp: load only names with these prefixes (comma list)
ap.add_argument("--scales", default="1")                    # evalp: start + t * (W - start) for each t (comma list)
ap.add_argument("--dump", default="")                       # evalp: save the 1-layer forward's intermediates (PATH.t*.npz)
r = ap.parse_args()
import mlx.core as mx
if not r.gpu:
    mx.set_default_device(mx.cpu)
src = open(os.path.expanduser("~/TrainLab/sticky/d09_train.py")).read()
src = src[:src.index("# ---------------------------------------------------------------- modes")]
if r.layers:                                       # truncate before the trainer casts its layers to fp32 (memory)
    src = src.replace("draft.freeze()\n", f"draft.freeze()\ndraft.layers = draft.layers[:{r.layers}]\n", 1)
sys.argv = ["d09_train.py", "--A", str(r.A), "--S", str(r.S), "--bs", str(r.bs), "--micro", "1", "--pmask", str(r.pmask),
            "--lr", str(r.lr), "--data", r.data, "--epochs", "2", "--cache-gb", "0.25"]
ns = {"__name__": "d09_ref"}
exec(compile(src, "d09_train.py", "exec"), ns)
import numpy as np
draft, A = ns["draft"], ns["A"]
if r.layers:
    draft.layers = draft.layers[:r.layers]
    ns["TRAIN"] = None
    gc.collect()
wins = ns["load_windows"]()
tr = [g for w_ in wins if w_["split"] == "train" for g in ns["groups_of"](w_)]
print(f"[ref] {len(wins)} windows, {len(tr)} train groups (A {A.A}), layers {len(draft.layers)}", flush=True)
out = {}
if r.mode == "grad":                               # Thu 11-05: one batch's loss + full gradient -> port/grad_stats.py JSON
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import grad_stats as GS
    if r.freeze_layers:                            # the layers stay fp32 (the trainer's cast), without gradients
        for L in draft.layers:
            L.freeze()
    b = ns["make_batch"](tr[:r.groups], A.pmask)
    norm = mx.array(max(1.0, float(b[-1].sum())))
    l, g = nn.value_and_grad(draft, lambda m_, b_, n_: ns["loss_fn"](b_, n_))(draft, b, norm)
    mx.eval(l, g)
    per = {k: GS.stats(k, np.array(v.astype(mx.float32))) for k, v in tree_flatten(g)}
    peak = mx.get_peak_memory() / 2 ** 30
    GS.save(os.path.expanduser(r.out), float(l), per, dict(norm=float(norm), mlx_peak_gb=peak, device="gpu" if r.gpu else "cpu"))
    print(f"[ref] grad loss {float(l):.6f} (norm {float(norm):.0f}); {len(per)} parameters; MLX peak {peak:.2f} GB", flush=True)
    sys.exit(0)
if r.mode == "evalp":                              # Thu 11-05 swap test: step 2's batch loss with the given weights
    if r.freeze_selector:
        draft.candidate_selector.freeze()
    z = np.load(os.path.expanduser(r.params))
    keys = [k for k in z.files if not r.only or any(k.startswith(p) for p in r.only.split(","))]
    from mlx.utils import tree_flatten
    start = {k: v for k, v in tree_flatten(draft.parameters()) if k in keys}
    ns["make_batch"]([tr[0]], A.pmask)              # step 1's batch: advances the trainer's PM_RNG as in train mode
    b = ns["make_batch"]([tr[1 % len(tr)]], A.pmask)
    for t in [float(x) for x in r.scales.split(",")]:   # Thu 11-05: start + t * (W - start), t = 1 -> W
        draft.load_weights([(k, start[k] + t * (mx.array(z[k]) - start[k])) for k in keys], strict=False)
        mx.eval(draft.parameters())
        nrm = max(1.0, float(b[-1].sum()))
        loss = ns["loss_fn"](b, mx.array(nrm))
        hctx, upos, anc, tok, tkl, lpl, y, pred, w = b      # the loss_fn terms apart: soft CE vs the selector CE
        logits, cand, sc = ns["heads"](ns["fwd"](hctx, upos, anc, tok), pred)
        ce = mx.logsumexp(logits, axis=-1) - (mx.softmax(lpl, axis=-1) * mx.take_along_axis(logits, tkl, axis=-1)).sum(-1)
        inc = cand == y[..., None]
        sce = (mx.logsumexp(sc, axis=-1) - (sc * inc).sum(-1)) * inc.any(-1)
        n_in = float((inc.any(-1) * (w > 0)).sum())
        mx.eval(loss, ce, sce)
        if r.dump:                                 # Thu 11-05: the forward's intermediates (1 layer), for the port
            L, BSz = draft.layers[0], ns["BS"]
            S_, Aa = anc.shape; U_ = hctx.shape[1]; B_ = S_ * Aa
            st = {}
            h = (draft.embed_tokens(tok.reshape(B_, BSz)) * draft.embed_scale).astype(mx.float32); st["h0"] = h
            qpos = (anc.reshape(B_, 1) + mx.arange(BSz)[None]).reshape(B_, 1, BSz)
            r_, a_, i_ = upos[:, None, None, :], anc[:, :, None, None], mx.arange(BSz)[None, None, :, None]
            neg = mx.where((r_ >= 0) & (r_ <= a_ - 1) & ((a_ + i_) - r_ < ns["W"]), 0.0, -1e9).astype(mx.float32)
            hc = hctx.astype(mx.float32)
            st["n1"] = L.input_layernorm(h); st["x1"], st["k1"] = L.attention_conv.prepare(st["n1"])
            st["a1"] = ns["attn"](L.self_attn, st["x1"], hc, upos[:, None, :], qpos, neg, S_, Aa, U_)
            st["f1"] = L.attention_conv.finish(st["a1"], st["k1"]); st["h1"] = h + st["f1"]
            st["n2"] = L.post_attention_layernorm(st["h1"]); st["x2"], st["k2"] = L.mlp_conv.prepare(st["n2"])
            st["m2"] = L.mlp(st["x2"]); st["f2"] = L.mlp_conv.finish(st["m2"], st["k2"]); st["h2"] = st["h1"] + st["f2"]
            st["hid"] = draft.norm(st["h2"][:, 1:])
            st["logits4"] = logits.reshape(-1, logits.shape[-1])[:4]
            mx.eval(st)
            np.savez(os.path.expanduser(f"{r.dump}.t{t:g}.npz"), **{k_: np.array(v.astype(mx.float32)) for k_, v in st.items()})
        print(f"[ref] evalp {os.path.basename(r.params)} only={r.only or 'all'} ({len(keys)} tensors) t={t:g}: step-2 "
              f"batch loss {float(loss):.6f} = ce {float((w * ce).sum()) / nrm:.6f} + sce {float((w * sce).sum()) / nrm:.6f} "
              f"(rows with the true token in the top 16: {n_in:.0f} of {float((w > 0).sum()):.0f})", flush=True)
    sys.exit(0)
if r.mode == "fwd":
    b = ns["make_batch"](tr[:r.groups], A.pmask)
    hctx, upos, anc, tok, tkl, lpl, y, pred, w = b
    hid = ns["fwd"](hctx, upos, anc, tok)
    logits, cand, sc = ns["heads"](hid, pred)
    norm = mx.array(max(1.0, float(w.sum())))
    loss = ns["loss_fn"](b, norm)
    mx.eval(hid, logits, cand, sc, loss)
    rows = np.nonzero(np.array(w).reshape(-1) > 0)[0][:64]
    out = dict(hid=np.array(hid), logits_rows=np.array(logits.reshape(-1, logits.shape[-1]))[rows], rows=rows,
               cand=np.sort(np.array(cand), -1), sc=np.sort(np.array(sc), -1), loss=float(loss),
               w=np.array(w), tok=np.array(tok), tkl=np.array(tkl), pred=np.array(pred), y=np.array(y))
    print(f"[ref] fwd loss {float(loss):.6f} (norm {float(norm):.0f})", flush=True)
else:
    if r.freeze_selector:                          # Thu 11-05: inactive at 1 layer (zero gradient); saves ~2 GB
        draft.candidate_selector.freeze()
    import mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_map
    loss_and_grad = nn.value_and_grad(draft, lambda m, b, norm: ns["loss_fn"](b, norm))
    opt = optim.AdamW(learning_rate=r.lr, weight_decay=0.0)
    steps_total = max(1, int(math.ceil(2 * len(tr) / A.S)))
    warm = min(20, steps_total // 10 + 1)
    lr_at = lambda s: r.lr * (0.01 + 0.99 * s / warm) if s < warm else \
        r.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps_total - warm)))))
    losses, gnorms = [], []
    from mlx.utils import tree_flatten
    before = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(draft.trainable_parameters())} if r.dp_json else None
    for s in range(r.steps):
        items = [tr[(s * A.S + k) % len(tr)] for k in range(A.S)]
        parts = [ns["make_batch"](items[k:k + 1], A.pmask) for k in range(len(items))]
        norm = mx.array(max(1.0, sum(float(b[-1].sum()) for b in parts)))
        loss, acc = 0.0, None
        for b in parts:
            l, g = loss_and_grad(draft, b, norm)
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(l, acc)
            loss += float(l)
        acc, gn = optim.clip_grad_norm(acc, 1.0)
        opt.learning_rate = lr_at(s)
        opt.update(draft, acc)
        mx.eval(draft.trainable_parameters(), opt.state)
        if before is not None and s == 0:          # Thu 11-05: the step-1 parameter change, per parameter
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import grad_stats as GS
            after = dict(tree_flatten(draft.trainable_parameters()))
            GS.save(os.path.expanduser(r.dp_json), loss, {k: GS.stats(k, np.array(after[k].astype(mx.float32)) - v)
                                                          for k, v in before.items()})
            before = None
        if r.save_after1 and s == 0:               # Thu 11-05: the weights after step 1 (for the swap test)
            np.savez(os.path.expanduser(r.save_after1),
                     **{k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(draft.trainable_parameters())})
        losses.append(loss); gnorms.append(float(gn))
        print(f"[ref] step {s + 1}: loss {loss:.6f} grad-norm {float(gn):.4f} lr {lr_at(s):.3e}", flush=True)
    out = dict(losses=np.array(losses), gnorms=np.array(gnorms))
np.savez(os.path.expanduser(r.out), **out)
print(f"[ref] saved {r.out}", flush=True)
