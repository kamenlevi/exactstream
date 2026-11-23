"""WEEK98 S5b INJECT on the REAL pair (NewIdeas' family (c), no download; the S5 analog was VOID): does injecting the
125B's own final prefix states into the competent 2B raise its agreement with the 125B?
- Small LM: Qwen3.5-2B, frozen layers 0-22 (inputs to layer 23 cached per file, as lever 1c), layer 23 with LoRA r 16
  (scale 10, lr 1e-4: 1c's form), final norm, tied lm_head.
- Arm I adds a cross-attention after layer 23 that reads the 125B's final widened state 'wd' (10240-d, the trunk's
  stream after its last layer = the MTP head's input; the BANK) at prefix rows u <= s only (block-causal: s = the row's
  block anchor, so no row sees a state the 125B would not have verified). Flamingo-style gate: h + tanh(alpha) x(...),
  alpha = 0 at the start, so arm I starts as arm N exactly; the adapter trains on its own Adam (--xlr). Arm N: no
  injection (= lever 1c). The same LoRA init, files, block offsets and steps for both (alternating per step).
- Data: the bank's 360 train files (n1112, answer rows), cross-entropy to the 125B's renormalized top-32.
- Blocks: anchors a0 + off + 30 k (a0 = gen - 2); row p predicts token p + 1 at lag p - s in 1..30. Training draws off in
  0..29; read-outs use off 0, 10, 20 on HE+ 24, MBPP 100 and val 40 (all answer rows of the 40 val files).
- Read-out (orchestrator 02:2x: run-time measures first): kept run per 30-row block (leading right guesses from lag 1)
  and the strict conditional per lag, then agreement by lag band; paired I vs N.
LINE (NewIdeas 01:22, written before any run): PASS if injection adds >= +3 pt (p < 0.05) to MBPP agreement -> family (c)
is worth a build on the 125B; read kept-run beside it. Reader ceiling: the 125B head's own depth 0 with the true state,
MBPP 94.0 / HE+ 97.9.
usage: s5b.py [--minutes 8] [--evals 1,2,4] [--smoke N]"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H = os.path.expanduser
BANK = H("~/datasets/deepdraft/bank")
SPLIT = H("~/BigMoE/data/deepdraft/bank/split.json")
LAG = 30
BANDS = [(1, 1), (2, 3), (4, 7), (8, 11), (12, 15), (16, 23), (24, 30)]
BNAME = ["0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-29"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0); ap.add_argument("--evals", default="1,2,4")
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--xlr", type=float, default=3e-4, help="the injection adapter's own Adam (incl. its gate)")
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--save-every", type=float, default=90.0)
    a = ap.parse_args()
    name = "s5b" + ("_smoke" if a.smoke else "")
    from sb import core as C
    from sb import resume as R
    from sb.compare import paired
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.tuner.lora import LoRALinear
    C.limit_memory(9)
    t0 = time.time()
    t = C.tm()
    assert not t.layers[23].is_linear
    split = json.load(open(SPLIT))
    cut = (lambda v: v[:a.smoke]) if a.smoke else (lambda v: v)

    def pre23(ids):
        x = t.embed_tokens(mx.array(np.asarray(ids, np.int32))[None])
        fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
        for i in range(23):
            x = t.layers[i](x, mask=ssm if t.layers[i].is_linear else fa, cache=None)
        return x.astype(mx.bfloat16)

    def load_set(names):
        out = []
        for nm in names:
            C.wait_pause()
            z = np.load(os.path.join(BANK, nm))
            ids = np.asarray(z["ids"]); S = len(ids); a0 = max(0, int(z["gen"]) - 2)
            rows = np.arange(a0 + 1, S - 1)
            x = pre23(ids); mx.eval(x)
            out.append(dict(x=x, rows=rows, tk=np.asarray(z["tk"])[rows], lp=np.asarray(z["lp"], np.float32)[rows],
                            nm=nm, a0=a0, S=S))
        return out

    def wd(it):                                           # the 125B's states [1, S, 10240] (row S-1 padded, never read)
        w = np.load(os.path.join(BANK, it["nm"]))["wd"]
        return mx.concatenate([mx.array(w), mx.zeros((1, w.shape[1]), mx.float16)], 0)[None]

    va = sorted({nm for nm, _ in json.load(open(os.path.join(BANK, "proxy", "rows_va.json")))})
    assert not set(va) & set(split["train"])
    evs = {"heplus": load_set(cut(split["heplus"])), "mbpp": load_set(cut(split["mbpp"])), "val": load_set(cut(va))}
    train = load_set(cut(split["train"]))
    print(f"[s5b] cached layer-23 inputs: " + ", ".join(f"{k} {len(v)}" for k, v in evs.items()) +
          f", train {len(train)} files ({sum(len(i['rows']) for i in train)} rows), {time.time() - t0:.0f}s", flush=True)

    def anchors(it, off):
        S, a0 = it["S"], it["a0"]
        p = np.arange(S); q = p - (a0 + off)
        anc = np.where(q <= 0, a0, a0 + off + LAG * ((q - 1) // LAG))
        anc = np.where(p <= a0, np.maximum(p - 1, 0), anc)
        return anc.astype(np.int32), it["rows"] - anc[it["rows"]]

    class Inject(nn.Module):
        def __init__(self, dz=10240, d=2048, nh=8, hd=128):
            super().__init__()
            self.nq, self.nk = nn.RMSNorm(d, eps=1e-6), nn.RMSNorm(dz, eps=1e-6)
            self.q = nn.Linear(d, nh * hd, bias=False); self.k = nn.Linear(dz, nh * hd, bias=False)
            self.v = nn.Linear(dz, nh * hd, bias=False); self.o = nn.Linear(nh * hd, d, bias=False)
            self.alpha = mx.zeros((1,))                    # tanh gate: closed at the start (Flamingo)
            self.rope = nn.RoPE(64, traditional=False, base=1e7)
            self.nh, self.hd = nh, hd

        def __call__(self, h, Z, anc):
            B, S, _ = h.shape
            sh = lambda z: z.reshape(B, S, self.nh, self.hd).transpose(0, 2, 1, 3)
            zn = self.nk(Z.astype(mx.float32))
            q = self.rope(sh(self.q(self.nq(h.astype(mx.float32)))))
            k = self.rope(sh(self.k(zn))); v = sh(self.v(zn))
            m = mx.where(mx.arange(S)[None, :] <= anc[:, None], 0.0, -1e9).astype(mx.float32)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=m)
            return mx.tanh(self.alpha) * self.o(o.transpose(0, 2, 1, 3).reshape(B, S, -1))

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
            self.layer = copy.deepcopy(t.layers[23]); self.layer.freeze(); wrap(self.layer)
            if inject:
                self.inj = Inject()
            self.inject = inject

        def hidden(self, x, Z, anc):
            h = self.layer(x, mask=create_attention_mask(x, None), cache=None)
            if self.inject:
                h = h + self.inj(h, Z, mx.array(anc))
            return h

    mx.random.seed(1115)
    arms = {"N": Arm(False), "I": Arm(True)}
    arms["I"].layer.update(tree_unflatten(tree_flatten(arms["N"].layer.trainable_parameters())))
    npar = {k: sum(v.size for _, v in tree_flatten(m.trainable_parameters())) for k, m in arms.items()}

    def readout(arm):
        per = {}
        for st, items in evs.items():
            cnt = {}
            for it in items:
                c = {b: [0, 0] for b in BNAME}
                c["kept"] = [0, 0]; c["cond"] = [[0, 0] for _ in range(LAG)]
                Z = wd(it) if arm.inject else None
                for off in (0, 10, 20):
                    anc, lag = anchors(it, off)
                    h = arm.hidden(it["x"], Z, anc)
                    top = np.asarray(mx.argmax(C.lm_logits(t.norm(h[0, mx.array(it["rows"])])), axis=-1))
                    hit = top == it["tk"][:, 0]
                    for b, (lo, hi) in zip(BNAME, BANDS):
                        sel = (lag >= lo) & (lag <= hi)
                        c[b][0] += int(hit[sel].sum()); c[b][1] += int(sel.sum())
                    ar = anc[it["rows"]]
                    for s_ in np.unique(ar):
                        blk = np.flatnonzero(ar == s_)
                        if lag[blk].min() != 1:
                            continue
                        hb = hit[blk[np.argsort(lag[blk])]]
                        k = int(np.argmin(hb)) if not hb.all() else len(hb)
                        c["kept"][0] += k; c["kept"][1] += 1
                        for L in range(min(k + 1, len(hb))):
                            c["cond"][L][0] += int(hb[L]); c["cond"][L][1] += 1
                c["all"] = [sum(c[b][0] for b in BNAME), sum(c[b][1] for b in BNAME)]
                cnt[it["nm"]] = c
                mx.clear_cache()
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
            for b in ("kept", "all", "0-0", "23-29"):
                r = paired(pN[st], pI[st], b, np.random.default_rng(1114))
                sc = 1 if b == "kept" else 100
                out[st][b] = {k: (round(sc * v, 3) if k in ("A", "B", "diff") else v) for k, v in r.items() if k != "ci95"}
                out[st][b]["ci95"] = [round(sc * x, 3) for x in r["ci95"]]
        return out

    def loss_fn(arm, x, Z, anc, rows, tk, lp):
        h = arm.hidden(x, Z, anc)
        lq = C.lm_logits_train(t.norm(h[0, mx.array(rows)]))
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        p = mx.softmax(mx.array(lp), axis=-1)
        return -(p * mx.take_along_axis(lq, mx.array(tk.astype(np.int32)), axis=-1)).sum(-1).mean()

    ctl = readout(arms["N"])
    print(f"[s5b] control (LoRA zero, no injection = the plain 2B; params N {npar['N']}, I {npar['I']}): "
          + json.dumps({st: {b: v[b] for b in ("all", "0-0", "23-29", "kept_run")} for st, v in bands(ctl).items()}), flush=True)
    opts = {"N": optim.Adam(learning_rate=a.lr), "IL": optim.Adam(learning_rate=a.lr), "IX": optim.Adam(learning_rate=a.xlr)}
    vgs = {k: nn.value_and_grad(m, lambda x, Z, anc, rows, tk, lp, m=m: loss_fn(m, x, Z, anc, rows, tk, lp)) for k, m in arms.items()}
    marks = sorted(float(m) for m in a.evals.split(",") if m and float(m) < a.minutes)
    reads, losses = {}, {"N": [], "I": []}
    step, ep, train_s, done, perm, pos = 0, 0, 0.0, False, None, 0
    rng = np.random.default_rng(1115)
    sp = os.path.join(C.CKPT, name + "_state")
    mods = {"N": arms["N"], "IL": arms["I"].layer, "IX": arms["I"].inj}
    R.arm_sigterm()
    meta = R.load(sp, mods, opts)
    if meta:
        step, ep, train_s, pos, losses, reads, marks = (meta[k] for k in ("step", "ep", "train_s", "pos", "losses", "reads", "marks"))
        perm = np.array(meta["perm"]); rng.bit_generator.state = meta["rng"]
        print(f"[s5b] resumed at step {step} ({train_s:.0f}s of training done; reads {list(reads)})", flush=True)
    last_save = train_s

    def snap():
        return dict(step=step, ep=ep, train_s=train_s, pos=pos, perm=[int(v) for v in perm], losses=losses, reads=reads,
                    marks=marks, rng=rng.bit_generator.state)

    def read(tag):
        pN, pI = readout(arms["N"]), readout(arms["I"])
        bN, bI = bands(pN), bands(pI)
        r = dict(N=bN, I=bI, I_vs_N=compare(pN, pI), steps=step, gate=float(mx.tanh(arms["I"].inj.alpha)[0]))
        print(f"[s5b] read {tag} (step {step}, gate tanh(alpha) {r['gate']:+.4f}): " + " | ".join(
            f"{st}: kept run N {bN[st]['kept_run']:.3f} I {bI[st]['kept_run']:.3f} ({r['I_vs_N'][st]['kept']['diff']:+.3f}); "
            f"all N {bN[st]['all']:.2f} I {bI[st]['all']:.2f} ({r['I_vs_N'][st]['all']['diff']:+.2f}, p {r['I_vs_N'][st]['all']['p_sign']:.3f}); "
            f"lag 1 {bN[st]['0-0']:.2f} / {bI[st]['0-0']:.2f}; 23-29 {bN[st]['23-29']:.2f} / {bI[st]['23-29']:.2f}" for st in evs), flush=True)
        return r
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
        Z = wd(it)
        l, g = vgs["N"](it["x"], None, anc, it["rows"], it["tk"], it["lp"])
        opts["N"].update(arms["N"], g); mx.eval(arms["N"].parameters(), opts["N"].state)
        losses["N"].append(float(l))
        l, g = vgs["I"](it["x"], Z, anc, it["rows"], it["tk"], it["lp"])
        opts["IL"].update(arms["I"].layer, g["layer"]); opts["IX"].update(arms["I"].inj, g["inj"])
        mx.eval(arms["I"].parameters(), opts["IL"].state, opts["IX"].state)
        losses["I"].append(float(l))
        train_s += time.time() - ts; step += 1
        if step % 100 == 0:
            print(f"[s5b] step {step} ep {ep} loss N {np.mean(losses['N'][-100:]):.4f} I {np.mean(losses['I'][-100:]):.4f} "
                  f"gate {float(mx.tanh(arms['I'].inj.alpha)[0]):+.4f} train {train_s:.0f}s", flush=True)
        if (marks and train_s >= 60 * marks[0]) or train_s >= 60 * a.minutes:
            last = train_s >= 60 * a.minutes
            m_ = a.minutes if last else marks.pop(0)
            reads[f"{m_:g}min"] = read(f"at {m_:g} min")
            if last:
                done = True
                break
            R.save(sp, mods, opts, snap()); last_save = train_s
        if R.STOP["f"] or train_s - last_save >= a.save_every:
            wrote = R.save(sp, mods, opts, snap()); last_save = train_s
            if R.STOP["f"]:
                print(f"[s5b] SIGTERM: state saved at step {step} ({wrote / 1e6:.0f} MB), exiting to resume later", flush=True)
                sys.exit(75)
    R.finish(sp)
    fin = reads[f"{a.minutes:g}min"]
    d = fin["I_vs_N"]["mbpp"]["all"]; dk = fin["I_vs_N"]["mbpp"]["kept"]
    verdict = ("PASS" if d["diff"] >= 3.0 and d["p_sign"] < 0.05 else
               "HURTS" if d["diff"] <= -1.0 and d["p_sign"] < 0.05 else "NO EFFECT" if abs(d["diff"]) < 1.0 else "BETWEEN")
    best = max(reads.items(), key=lambda kv: kv[1]["I_vs_N"]["mbpp"]["all"]["diff"])
    res = dict(verdict=verdict, control=bands(ctl), reads=reads, best_read=best[0], steps=step, epochs=round(ep + pos / len(train), 2),
               params=npar, lr=a.lr, xlr=a.xlr, rank=a.rank, minutes=a.minutes, seconds=round(time.time() - t0, 1))
    json.dump(res, open(os.path.join(C.RES, name + ".json"), "w"), indent=1)
    print(f"[s5b] DONE verdict {verdict}: MBPP agreement N {fin['N']['mbpp']['all']:.2f} -> I {fin['I']['mbpp']['all']:.2f} "
          f"({d['diff']:+.2f}, CI {d['ci95'][0]:+.2f}..{d['ci95'][1]:+.2f}, p {d['p_sign']:.3f}); kept run {dk['A']:.3f} -> {dk['B']:.3f} "
          f"({dk['diff']:+.3f}); best read {best[0]} ({best[1]['I_vs_N']['mbpp']['all']['diff']:+.2f})", flush=True)


if __name__ == "__main__":
    main()
