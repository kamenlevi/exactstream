#!/usr/bin/env python3
"""DEEPDRAFT pre-proof (docs/PREREG_DEEPDRAFT_PREPROOF_2026-11-12.md): a learned per-token stand-in for the trunk state,
read by the unchanged MTP head b5.
  train: the helper z_k = z_(k-1) + f(z_(k-1), e(ids)) on the capture night 1 (Magicoder) rows, teacher-forced unrolls of
         K = 8, the per-stream normalised MSE; 2,500 steps x 32 unrolls; nested data fractions (file-name order); the
         last 5 % of files is validation. -> data/deepdraft/preproof/helper_<tag>.safetensors + _train.json
  eval:  b5's chain as TrainLab d23h_mtp31.py runs it (24 HumanEval+ windows of data/d16w; 100 MBPP answers of
         dlog/d1102mbpp; anchors from gen - 2; depth 31), with depth j >= 2 reading the stand-in z_(j-1) instead of the
         head's own output: arms plain (b5 as is), stale (z = W at the anchor), helper_<tag>. -> data/deepdraft/preproof/eval.json
usage: [BigMoE venv] dd_preproof.py train --frac 1.0 --tag f3 | dd_preproof.py eval --tags f1,f2,f3 [--sets code,mbpp]"""
import argparse, glob, json, os, struct, sys, time

H = os.path.expanduser
OUT = H("~/BigMoE/data/deepdraft/preproof")
CAP = H("~/FableGod/deepdraft/capture/n1112")
D, K, STEPS, BATCH = 31, 8, 2500, 32
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]


def embed_table(mx):
    """the 125B's token embedding, dequantized once from the bare store (frozen), bf16 [V, 2560]"""
    import numpy as np
    d = H("~/BigMoE/models/bare")
    ix = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    kw = [k for k in ix if k.endswith("embed_tokens.weight") and "mtp" not in k][0]
    def get(name):
        f = os.path.join(d, ix[name])
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        v = hdr[name]; a, b = v["data_offsets"]
        raw = np.fromfile(f, dtype=np.uint8, count=b - a, offset=8 + n + a)
        if v["dtype"] == "U32":
            return mx.array(raw.view(np.uint32).reshape(v["shape"]))
        return mx.array((raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])).astype(mx.bfloat16)
    w, s, bi = get(kw), get(kw[:-len("weight")] + "scales"), get(kw[:-len("weight")] + "biases")
    bits = 32 * w.shape[-1] // (s.shape[-1] * 64)
    return mx.dequantize(w, s, bi, 64, bits).astype(mx.bfloat16)


def make_helper(nn, mx):
    class Helper(nn.Module):
        def __init__(self, Dz=10240, E=2560, Hd=1024, F=2048):
            super().__init__()
            self.pz, self.pe = nn.Linear(Dz, Hd), nn.Linear(E, Hd)
            self.norm = nn.RMSNorm(Hd)
            self.f1, self.f2 = nn.Linear(Hd, F), nn.Linear(F, Hd)
            self.out = nn.Linear(Hd, Dz)
            self.out.weight = mx.zeros_like(self.out.weight); self.out.bias = mx.zeros_like(self.out.bias)   # starts as "stale"

        def __call__(self, z, e):                        # z [B, 10240] f32, e [B, 2560]
            zs = z.reshape(-1, 4, 2560)
            r = mx.sqrt((zs * zs).mean(-1, keepdims=True) + 1e-6)
            zn = (zs / r).reshape(-1, 10240)
            h = nn.gelu(self.norm(self.pz(zn) + self.pe(e.astype(mx.float32))))
            h = h + self.f2(nn.gelu(self.f1(h)))
            dz = self.out(h).reshape(-1, 4, 2560) * r
            return z + dz.reshape(-1, 10240)
    return Helper


def nmse(mx, a, b):
    a, b = a.reshape(-1, 4, 2560), b.reshape(-1, 4, 2560)
    na = a / mx.sqrt((a * a).mean(-1, keepdims=True) + 1e-6)
    nb = b / mx.sqrt((b * b).mean(-1, keepdims=True) + 1e-6)
    return ((na - nb) ** 2).mean()


def files_split():
    fs = sorted(glob.glob(os.path.join(CAP, "*.npz")))
    nv = max(1, len(fs) // 20)
    return fs[:-nv], fs[-nv:]


def load_rows(np, f):
    z = np.load(f)
    W = np.asarray(z["wd"]); ids = np.asarray(z["ids"], np.int64); gen = int(z["gen"])
    return W, ids, gen


def batches(np, rng, files, n_files=48, steps=40):
    """chunks of random files held in RAM; unroll starts s in the answer rows with s + K < len(W)"""
    while True:
        pick = [files[i] for i in rng.choice(len(files), min(n_files, len(files)), replace=False)]
        data = [load_rows(np, f) for f in pick]
        starts = [(i, s) for i, (W, ids, gen) in enumerate(data) for s in range(max(gen - 1, 0), len(W) - K - 1)]
        for _ in range(steps):
            sel = [starts[j] for j in rng.choice(len(starts), BATCH, replace=False)]
            z0 = np.stack([data[i][0][s] for i, s in sel]).astype(np.float32)
            T = np.stack([data[i][0][s + 1:s + K + 1] for i, s in sel]).astype(np.float32)
            E = np.stack([data[i][1][s + 1:s + K + 1] for i, s in sel])
            yield z0, E, T


def cmd_train(a):
    import numpy as np
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    os.makedirs(OUT, exist_ok=True)
    emb = embed_table(mx); mx.eval(emb)
    tr, va = files_split()
    tr = tr[:max(1, int(round(len(tr) * a.frac)))]
    model = make_helper(nn, mx)()
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)

    def loss_fn(m, z0, E, T):
        z, L = z0, 0.0
        for k in range(K):
            z = m(z, emb[E[:, k]])
            L = L + nmse(mx, z, T[:, k])
        return L / K
    step = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(1112)
    it = batches(np, rng, tr)
    t0, hist = time.time(), []
    for i in range(STEPS):
        z0, E, T = next(it)
        loss, g = step(model, mx.array(z0), mx.array(E), mx.array(T))
        opt.update(model, g); mx.eval(model.parameters(), opt.state, loss)
        if i % 250 == 0 or i == STEPS - 1:
            hist.append((i, float(loss))); print(f"[pp] {a.tag} step {i} loss {float(loss):.4f} ({time.time() - t0:.0f} s)", flush=True)
    # validation: per-depth nmse of the unroll, and the stale baseline (z = W_s)
    vit = batches(np, np.random.default_rng(7), va, n_files=len(va), steps=60)
    per, stale = np.zeros(K), np.zeros(K)
    for _ in range(60):
        z0, E, T = next(vit)
        z = mx.array(z0); Tm = mx.array(T)
        for k in range(K):
            z = model(z, emb[mx.array(E[:, k])])
            per[k] += float(nmse(mx, z, Tm[:, k])); stale[k] += float(nmse(mx, mx.array(z0), Tm[:, k]))
    per, stale = per / 60, stale / 60
    model.save_weights(os.path.join(OUT, f"helper_{a.tag}.safetensors"))
    json.dump(dict(tag=a.tag, frac=a.frac, train_files=len(tr), val_files=len(va), steps=STEPS, batch=BATCH, K=K, hist=hist,
                   val_nmse=per.tolist(), val_stale_nmse=stale.tolist(), mlx_peak_gb=mx.get_peak_memory() / 1e9),
              open(os.path.join(OUT, f"helper_{a.tag}_train.json"), "w"), indent=1)
    print(f"[pp] {a.tag}: val nmse by step {np.round(per, 4).tolist()} vs stale {np.round(stale, 4).tolist()}; "
          f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def cmd_eval(a):
    sys.path.insert(0, H("~/TrainLab/sticky"))
    import d22_ttt as T
    import numpy as np
    import mlx.core as mx, mlx.nn as nn

    class Args:
        run, frac = "b5", 0.03
    rt, lm, head, P = T.load_head(Args)
    Hl = make_helper(nn, mx)
    helpers = {}
    for tag in [t for t in a.tags.split(",") if t]:
        m = Hl(); m.load_weights(os.path.join(OUT, f"helper_{tag}.safetensors")); mx.eval(m.parameters()); helpers[tag] = m
    emb = lm.model.embed_tokens

    def chain2(W, ids, on_depth, a0, stand):
        """T.chain with depth j >= 2 reading stand(j, z_prev, n) instead of the head's own previous output"""
        S = W.shape[1]; layer = head.layers[0]; att = layer.self_attn; n1 = S - 1
        mixed, hyper, inject = layer.attn_hyper_connection(T.fuse(head, P, W[:, :n1], emb(ids[:, 1:n1 + 1])).astype(mx.bfloat16))
        q, gate, K1, V1 = T.qkv(att, P, mixed, mx.arange(n1))
        o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K1, V1, scale=att.scale, mask="causal")
        n = n1 - a0
        blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
        h = T.finish(head, P, hyper[:, a0:], inject[:, a0:], blk)
        on_depth(1, h)
        z = W[:, a0:a0 + n].astype(mx.float32)           # z_0 = the exact stream at each anchor
        ck, cv = [], []
        for j in range(2, D + 1):
            n = S - j - a0
            if n <= 4:
                break
            inp, z = stand(j, h, z, n)
            mixed, hyper, inject = layer.attn_hyper_connection(
                T.fuse(head, P, inp[:, :n], emb(ids[:, a0 + j:a0 + j + n])).astype(mx.bfloat16))
            q, gate, k, v = T.qkv(att, P, mixed, a0 + mx.arange(n) + (j - 1))
            ck.append(k); cv.append(v)
            o = T.ttt_attn(att.scale, q, K1, V1, [c[:, :, :n] for c in ck], [c[:, :, :n] for c in cv], a0)
            blk = P.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate))
            h = T.finish(head, P, hyper, inject, blk)
            on_depth(j, h)

    def run(x, a0, arm):
        ids = x["ids"]; Sn = len(ids); W = x["W"]; idm = mx.array(ids)[None]; right = []

        def on_depth(j, h):
            nn_ = min(h.shape[1], Sn - 1 - j - a0)
            am = mx.argmax(T.logits(lm, head, h[:, :nn_])[0], -1); mx.eval(am)
            right.append(np.array(am) == ids[a0 + j + 1:a0 + j + 1 + nn_])
        if arm == "plain":
            T.chain(lm, head, P, W, idm, D, on_depth, a0)
        elif arm == "stale":
            chain2(W, idm, on_depth, a0, lambda j, h, z, n: (W[:, a0:a0 + n], z))
        else:
            m = helpers[arm]

            def stand(j, h, z, n):                        # z_(j-1) = f(z_(j-2), e(ids[s + j - 1])) for the n anchors s
                zz = m(z[0, :n], emb(idm[:, a0 + j - 1:a0 + j - 1 + n])[0])
                return zz[None].astype(mx.bfloat16), zz[None]
            chain2(W, idm, on_depth, a0, stand)
        return right

    arms = ["plain", "stale"] + list(helpers)
    R = {}
    for st in a.sets.split(","):
        if st == "code":
            items = [(os.path.basename(f)[:-4], None) for f in sorted(glob.glob(os.path.join(T.TD, "d16w", "tf_code_HumanEval_*.npz")))]
        else:
            items = [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(os.path.join(T.TD, "dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]]
        acc = {arm: {b: [0, 0] for b in BANDS} for arm in arms}
        for key, f in items:
            if f is None:
                x, gen = T.window(key), T.gen_of(key)
            else:
                z = np.load(f); ids = np.asarray(z["ids"], np.int32)
                x, gen = {"W": mx.array(z["wd"]).astype(mx.bfloat16)[None], "ids": ids}, int(z["gen"])
            a0 = max(0, gen - 2)
            for arm in arms:
                right = run(x, a0, arm)
                for lo, hi in BANDS:
                    for j in range(lo, hi + 1):
                        if j < len(right):
                            acc[arm][(lo, hi)][0] += int(right[j].sum()); acc[arm][(lo, hi)][1] += len(right[j])
            mx.clear_cache()
        R[st] = {arm: {f"{lo}-{hi}": acc[arm][(lo, hi)][0] / max(acc[arm][(lo, hi)][1], 1) for lo, hi in BANDS} for arm in arms}
        R[st]["windows"] = len(items)
        print(f"[pp] {st}: " + "; ".join(f"{arm} " + " ".join(f"{b}:{100 * v:.1f}" for b, v in R[st][arm].items()) for arm in arms), flush=True)
    c = R.get("code")
    if c:
        base, top = c["plain"]["23-30"], c["plain"]["0-0"]
        R["gap"] = {arm: (c[arm]["23-30"] - base) / max(top - base, 1e-9) for arm in arms}
        best = max((t for t in helpers), key=lambda t: R["gap"][t], default=None)
        g = R["gap"].get("f3", R["gap"].get(best)) if helpers else None
        R["verdict"] = dict(rule="code depths 23-30: PASS g >= 0.40, KILL g < 0.15", arm="f3" if "f3" in helpers else best, g=g,
                            verdict=None if g is None else ("PASS" if g >= 0.40 else "KILL" if g < 0.15 else "INCONCLUSIVE"))
    R["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
    json.dump(R, open(os.path.join(OUT, "eval.json"), "w"), indent=1)
    print(json.dumps({k: R[k] for k in ("gap", "verdict", "mlx_peak_gb") if k in R}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--tag", default="f3")
    ap.add_argument("--tags", default="f1,f2,f3"); ap.add_argument("--sets", default="code,mbpp")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
