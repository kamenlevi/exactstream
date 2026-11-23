"""WEEK98 S5 INJECT, the no-download analog (NewIdeas 01:22; family (c): a small LM with the big model's prefix features
injected). Here the "big" model is the full 2B (24 layers) and the "small LM" is the 2B's OWN first 12 layers + its final
norm + tied lm_head (pretrained composition, early exit).

Corrector: frozen layers 0-10 (their output is cached per file), layer 11 (full attention) with LoRA r 16, then the
final norm and the lm_head. Arm I adds a zero-init cross-attention after layer 11 that reads the full 2B's post-norm
final states W[u] at prefix positions u <= s only (block-causal, s = the block's anchor), so no row sees a state the
target would not have verified yet. It sits after the last corrector layer, so no row can leak another row's injection.
Arm N is the same without it. Both train the same steps on the same files and offsets (my Magicoder base 380, the
2B's own greedy answers), cross-entropy to the full 2B's renormalized top-32 on answer rows.
Blocks: anchors a0 + off + 30 k (a0 = gen - 2); row p in a block predicts token p + 1 at lag p - s in 1..30 (lag 1 = the
chain's depth band 0: the MTP head reads W[s] and the token s + 1 to predict s + 2). Training draws off in 0..29;
read-outs use off 0, 10 and 20 on MBPP-50 / HE-30.
LINE (NewIdeas 01:22, written before any run): PASS if injection adds >= +3 pt (p < 0.05) to the corrector's
teacher-forced agreement with the full 2B (all lags, MBPP-50) -> family (c) is worth a build on the 125B. Reader
ceiling (information): arm I at lag 1 vs s30's 88.54. Also logged: per lag band, HE-30, and the control (LoRA zero).
usage: s5inject.py [--minutes 8] [--evals 2,4] [--ntrain 380] [--smoke N]"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

LAG = 30
BANDS = [(1, 1), (2, 3), (4, 7), (8, 11), (12, 15), (16, 23), (24, 30)]          # lags; = depth bands 0, 1-2, ..., 23-29
BNAME = ["0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-29"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--evals", default="2,4")
    ap.add_argument("--ntrain", type=int, default=380)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--exit", type=int, default=12, help="corrector = layers 0 .. exit-1 (layer exit-1 must be full attention)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--save-every", type=float, default=90.0, help="seconds of training between resume states")
    a = ap.parse_args()
    name = "s5inject" + (f"_exit{a.exit}" if a.exit != 12 else "") + ("_smoke" if a.smoke else "")
    from sb import core as C
    from sb.compare import paired
    from sb import resume as R
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.tuner.lora import LoRALinear
    C.limit_memory(10)
    t0 = time.time()
    t = C.tm()
    LX = a.exit - 1
    assert not t.layers[LX].is_linear, f"layer {LX} must be full attention"
    s = json.load(open(os.path.join(C.DATA, "split.json")))
    trf = [os.path.join(C.DATA, "mc", k + ".npz") for k in s["base"]][:a.smoke or a.ntrain]
    evf = {"mbpp": C.files("mbpp")[:a.smoke or 50], "he": C.files("he")[:a.smoke or 30]}

    def prefix(ids):                                      # frozen layers 0 .. LX-1, inference kernels, no cache
        x = t.embed_tokens(mx.array(np.asarray(ids, np.int32))[None])
        fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
        for i in range(LX):
            x = t.layers[i](x, mask=ssm if t.layers[i].is_linear else fa, cache=None)
        return x.astype(mx.bfloat16)

    def load(f):
        C.wait_pause()
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32); S = len(ids); a0 = max(0, int(z["gen"]) - 2)
        x = prefix(ids); W = C.load_W(z)
        mx.eval(x, W)
        rows = np.arange(a0 + 1, S - 1)
        return dict(x=x, W=W, rows=rows, a0=a0, S=S, tk=np.asarray(z["tk"])[rows], lp=np.asarray(z["lp"], np.float32)[rows],
                    key=os.path.basename(f)[:-4])
    train = [load(f) for f in trf]
    evs = {k: [load(f) for f in v] for k, v in evf.items()}
    print(f"[s5] cached layer-{LX} inputs + W: {len(train)} train, " + ", ".join(f"{k} {len(v)}" for k, v in evs.items()) +
          f" files, {sum(len(it['rows']) for it in train)} train rows, {time.time() - t0:.0f}s", flush=True)

    def anchors(it, off):
        """per row 0..S-1: the last W position it may read (block-causal); lag per scored row"""
        S, a0 = it["S"], it["a0"]
        p = np.arange(S)
        q = p - (a0 + off)
        anc = np.where(q <= 0, a0, a0 + off + LAG * ((q - 1) // LAG))
        anc = np.where(p <= a0, np.maximum(p - 1, 0), anc)          # prompt rows (not scored): their own past
        return anc.astype(np.int32), (it["rows"] - anc[it["rows"]])

    class Inject(nn.Module):
        def __init__(self, d=2048, nh=8, hd=128):
            super().__init__()
            self.nq, self.nk = nn.RMSNorm(d, eps=1e-6), nn.RMSNorm(d, eps=1e-6)
            self.q = nn.Linear(d, nh * hd, bias=False); self.k = nn.Linear(d, nh * hd, bias=False)
            self.v = nn.Linear(d, nh * hd, bias=False); self.o = nn.Linear(nh * hd, d, bias=False)
            self.o.weight = mx.zeros((d, nh * hd))                  # zero-init: arm I starts as arm N
            self.rope = nn.RoPE(64, traditional=False, base=1e7)
            self.nh, self.hd = nh, hd

        def __call__(self, h, W, anc):
            B, S, _ = h.shape
            sh = lambda z: z.reshape(B, S, self.nh, self.hd).transpose(0, 2, 1, 3)
            q = self.rope(sh(self.q(self.nq(h.astype(mx.float32)))))
            k = self.rope(sh(self.k(self.nk(W.astype(mx.float32)))))
            v = sh(self.v(W.astype(mx.float32)))
            m = mx.where(mx.arange(S)[None, :] <= anc[:, None], 0.0, -1e9).astype(mx.float32)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=m)
            return self.o(o.transpose(0, 2, 1, 3).reshape(B, S, -1))

    t.freeze()

    def wrap(mod):
        for k, v in mod.children().items():
            if isinstance(v, (nn.Linear, nn.QuantizedLinear)):
                setattr(mod, k, LoRALinear.from_base(v, r=a.rank, dropout=0.0, scale=10.0))
            elif isinstance(v, nn.Module):
                wrap(v)

    class Arm(nn.Module):
        def __init__(self, inject):
            super().__init__()
            self.layer = copy.deepcopy(t.layers[LX]); self.layer.freeze(); wrap(self.layer)
            if inject:
                self.inj = Inject()
            self.inject = inject

        def hidden(self, x, W, anc):
            h = self.layer(x, mask=create_attention_mask(x, None), cache=None)
            if self.inject:
                h = h + self.inj(h, W, mx.array(anc))
            return h

    arms = {"N": Arm(False), "I": Arm(True)}
    rng = np.random.default_rng(1115)
    mx.random.seed(1115)
    arms["I"].layer.update(tree_unflatten(tree_flatten(arms["N"].layer.trainable_parameters())))   # the same LoRA init
    npar = {k: sum(v.size for _, v in tree_flatten(m.trainable_parameters())) for k, m in arms.items()}

    def readout(arm):
        """per set, per file: band counts by lag, plus the run-time read (orchestrator 02:2x): 'kept' = [sum over blocks of
        the leading run of right guesses from lag 1, blocks], and 'cond' = per lag [right, reached] where reached means
        right at every lag before it (the strict conditional, accept-until-fail's own rows)"""
        per = {}
        for st, items in evs.items():
            cnt = {}
            for it in items:
                c = {b: [0, 0] for b in BNAME}
                c["kept"] = [0, 0]; c["cond"] = [[0, 0] for _ in range(LAG)]
                for off in (0, 10, 20):
                    anc, lag = anchors(it, off)
                    h = arm.hidden(it["x"], it["W"], anc)
                    top = np.asarray(mx.argmax(C.lm_logits(t.norm(h[0, mx.array(it["rows"])])), axis=-1))
                    hit = top == it["tk"][:, 0]
                    for b, (lo, hi) in zip(BNAME, BANDS):
                        sel = (lag >= lo) & (lag <= hi)
                        c[b][0] += int(hit[sel].sum()); c[b][1] += int(sel.sum())
                    ar = anc[it["rows"]]
                    for s_ in np.unique(ar):
                        blk = np.flatnonzero(ar == s_)
                        hb = hit[blk[np.argsort(lag[blk])]]
                        if lag[blk].min() != 1:
                            continue
                        k = int(np.argmin(hb)) if not hb.all() else len(hb)
                        c["kept"][0] += k; c["kept"][1] += 1
                        for L in range(min(k + 1, len(hb))):
                            c["cond"][L][0] += int(hb[L]); c["cond"][L][1] += 1
                c["all"] = [sum(c[b][0] for b in BNAME), sum(c[b][1] for b in BNAME)]
                cnt[it["key"]] = c
            per[st] = cnt
        return per

    def bands(per):
        out = {}
        for st, cnt in per.items():
            r = {b: round(100 * sum(v[b][0] for v in cnt.values()) / max(1, sum(v[b][1] for v in cnt.values())), 2)
                 for b in BNAME + ["all"]}
            r["kept_run"] = round(sum(v["kept"][0] for v in cnt.values()) / max(1, sum(v["kept"][1] for v in cnt.values())), 3)
            cr = [[sum(v["cond"][L][0] for v in cnt.values()), sum(v["cond"][L][1] for v in cnt.values())] for L in range(LAG)]
            r["cond"] = [round(x / n, 4) if n else None for x, n in cr]
            r["cond_n"] = [n for _, n in cr]
            out[st] = r
        return out

    def compare(pN, pI):
        out = {}
        for st in pN:
            out[st] = {}
            for b in ("all", "0-0", "23-29", "kept"):
                r = paired(pN[st], pI[st], b, np.random.default_rng(1114))
                sc = 1 if b == "kept" else 100                  # kept: a run length, not a percentage
                out[st][b] = {k: (round(sc * v, 3) if k in ("A", "B", "diff") else v) for k, v in r.items() if k != "ci95"}
                out[st][b]["ci95"] = [round(sc * x, 3) for x in r["ci95"]]
        return out

    ctl = readout(arms["N"])
    print(f"[s5] control (LoRA zero, no injection; params N {npar['N']}, I {npar['I']}): {json.dumps(bands(ctl))}", flush=True)
    opts = {k: optim.Adam(learning_rate=a.lr) for k in arms}

    def loss_fn(arm, x, W, anc, rows, tk, lp):
        h = arm.hidden(x, W, anc)
        lq = C.lm_logits_train(t.norm(h[0, mx.array(rows)]))
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        p = mx.softmax(mx.array(lp), axis=-1)
        return -(p * mx.take_along_axis(lq, mx.array(tk.astype(np.int32)), axis=-1)).sum(-1).mean()
    vgs = {k: nn.value_and_grad(m, lambda x, W, anc, rows, tk, lp, m=m: loss_fn(m, x, W, anc, rows, tk, lp)) for k, m in arms.items()}
    marks = sorted(float(m) for m in a.evals.split(",") if m and float(m) < a.minutes)
    reads, losses = {}, {k: [] for k in arms}
    step, ep, train_s, done, perm, pos = 0, 0, 0.0, False, None, 0
    sp = os.path.join(C.ROOT, "ckpt", name + "_state")
    R.arm_sigterm()
    meta = R.load(sp, arms, opts)
    if meta:
        step, ep, train_s, pos, losses, reads, marks = (meta[k] for k in ("step", "ep", "train_s", "pos", "losses", "reads", "marks"))
        perm = np.array(meta["perm"]); rng.bit_generator.state = meta["rng"]
        print(f"[s5] resumed at step {step} ({train_s:.0f}s of training done; reads {list(reads)})", flush=True)
    last_save, wrote = train_s, 0

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
        it = train[i]; off = int(rng.integers(0, LAG))
        anc, _ = anchors(it, off)
        for k, m in arms.items():
            l, g = vgs[k](it["x"], it["W"], anc, it["rows"], it["tk"], it["lp"])
            opts[k].update(m, g); mx.eval(m.parameters(), opts[k].state)
            losses[k].append(float(l))
        train_s += time.time() - ts; step += 1
        if step % 100 == 0:
            print(f"[s5] step {step} ep {ep} loss N {np.mean(losses['N'][-100:]):.4f} I {np.mean(losses['I'][-100:]):.4f} "
                  f"train {train_s:.0f}s", flush=True)
        if (marks and train_s >= 60 * marks[0]) or train_s >= 60 * a.minutes:
            last = train_s >= 60 * a.minutes
            m_ = a.minutes if last else marks.pop(0)
            pN, pI = readout(arms["N"]), readout(arms["I"])
            reads[f"{m_:g}min"] = dict(N=bands(pN), I=bands(pI), I_vs_N=compare(pN, pI), steps=step)
            r = reads[f"{m_:g}min"]
            print(f"[s5] read at {m_:g} min (step {step}): " + " | ".join(
                f"{st}: N {r['N'][st]['all']:.2f} I {r['I'][st]['all']:.2f} ({r['I_vs_N'][st]['all']['diff']:+.2f}, "
                f"p {r['I_vs_N'][st]['all']['p_sign']:.3f}); 23-29 N {r['N'][st]['23-29']:.2f} I {r['I'][st]['23-29']:.2f}; "
                f"lag 1 I {r['I'][st]['0-0']:.2f}; kept run N {r['N'][st]['kept_run']:.2f} I {r['I'][st]['kept_run']:.2f} "
                f"({r['I_vs_N'][st]['kept']['diff']:+.3f})" for st in evs), flush=True)
            if last:
                done = True
                break
            wrote = R.save(sp, arms, opts, snap()); last_save = train_s
        if R.STOP["f"] or train_s - last_save >= a.save_every:
            wrote = R.save(sp, arms, opts, snap()); last_save = train_s
            if R.STOP["f"]:
                print(f"[s5] SIGTERM: state saved at step {step} ({wrote / 1e6:.0f} MB), exiting to resume later", flush=True)
                sys.exit(75)
    R.finish(sp)
    fin = reads[f"{a.minutes:g}min"]
    d = fin["I_vs_N"]["mbpp"]["all"]
    verdict = "PASS" if d["diff"] >= 3.0 and d["p_sign"] < 0.05 else ("NO EFFECT" if abs(d["diff"]) < 1.0 else "BETWEEN")
    # cost: the corrector's pass over a 32-row block (layers 0 .. LX, no cache: weight-bound) + the lm_head on 32 rows
    z = np.load(evf["mbpp"][0]); ids0 = np.asarray(z["ids"], np.int32)[:32]
    def corr_pass():
        x = prefix(ids0); h = arms["I"].hidden(x, evs["mbpp"][0]["W"][:, :32], np.arange(32, dtype=np.int32) - 1 + (np.arange(32) == 0))
        return C.lm_logits(t.norm(h[0]))
    for _ in range(3):
        mx.eval(corr_pass())
    tt = []
    for _ in range(10):
        q = time.perf_counter(); mx.eval(corr_pass()); tt.append(time.perf_counter() - q)
    cost_ms = 1e3 * float(np.median(tt))
    res = dict(verdict=verdict, control=bands(ctl), reads=reads, steps=step, epochs=round(ep + pos / len(train), 2), params=npar, lr=a.lr, rank=a.rank,
               exit=a.exit, minutes=a.minutes, corrector_block_ms_32rows=cost_ms, seconds=round(time.time() - t0, 1))
    json.dump(res, open(os.path.join(C.RES, name + ".json"), "w"), indent=1)
    print(f"[s5] kept run per 30-row block (leading right guesses from lag 1), MBPP-50: N {fin['N']['mbpp']['kept_run']:.3f} -> "
          f"I {fin['I']['mbpp']['kept_run']:.3f} ({fin['I_vs_N']['mbpp']['kept']['diff']:+.3f}, CI {fin['I_vs_N']['mbpp']['kept']['ci95'][0]:+.3f}.."
          f"{fin['I_vs_N']['mbpp']['kept']['ci95'][1]:+.3f}); strict conditional lag 1-8 N {fin['N']['mbpp']['cond'][:8]} I {fin['I']['mbpp']['cond'][:8]}", flush=True)
    print(f"[s5] DONE verdict {verdict}: MBPP-50 all lags N {fin['N']['mbpp']['all']:.2f} -> I {fin['I']['mbpp']['all']:.2f} "
          f"({d['diff']:+.2f}, CI {d['ci95'][0]:+.2f}..{d['ci95'][1]:+.2f}, sign p {d['p_sign']:.3f}); corrector pass over 32 rows "
          f"{cost_ms:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
