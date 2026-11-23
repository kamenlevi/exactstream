"""WEEK98 PLAN4D item 2 in Sandbox98 (orchestrator 21:4x; NewIdeas' ~/NewIdeas/research/week98/PLAN4D_HEPLUS.md, 2d196fb):
lever 1c rerun as a FULL-DATA distilled auditor. Pre-registered in results/LEVERS_A.md (section L1CF) before any run.

Same form and recipe as sb/lever1c.py:
- rank-16 LoRA, scale 10, on every linear of the 2B's layer 23 (full attention, so no Gated-DeltaNet backward);
- Adam, lr 1e-4 constant;
- cross-entropy to the 125B's top-32, renormalized, on the answer rows;
- the final norm and the tied lm_head stay frozen; the loss uses the bf16 dequantized copy.
What changes is the data: ONE epoch over every captured n1112 file except the bank's val 40 (3,032 files, 805,005 answer
rows, ~8.4x 1c's 360 files) instead of 8 epochs over 360. Their layer-23 inputs don't fit in 7 GB, so each step runs the
frozen trunk (layers 0-22, inference kernels) on its own file first. One epoch makes the same trunk passes 1c's caching did.

Read-outs are 1c's, unchanged: teacher-forced top-1 agreement with the 125B's runtime top-1 (MoonShots' audit law: the
audited deep bands sit at this level) on HE+ 24 / MBPP 100 / val 40, as a diff vs the control (LoRA zero) with a bootstrap
CI over files. They run after steps 1000 and 2000 and at the end of the epoch, each saving the LoRA (sha256 in the JSON).

Read-only inputs: ~/FableGod/deepdraft/capture/n1112, ~/datasets/deepdraft/bank, ~/BigMoE/data/deepdraft/bank/split.json.
Writes ~/Sandbox98 only.
usage: lever1cf.py [--reads 1000,2000] [--tag a] [--smoke N] [--dry]"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H = os.path.expanduser
CAP = H("~/FableGod/deepdraft/capture/n1112")
BANK = H("~/datasets/deepdraft/bank")
SPLIT = H("~/BigMoE/data/deepdraft/bank/split.json")
CTL_REF = {"heplus": 94.42, "mbpp": 85.67, "val": 87.29}     # lever7_agree.py = 1c's control (logs/lever1c_a.log)
PAUSE = H("~/.local/share/quiet/PAUSE")


def file_lists():
    """-> (train files, the bank's eval sets, index rows by file). Train = every n1112 capture file but the bank's val 40."""
    sp = json.load(open(SPLIT))
    va_rows = {}
    for nm, r in json.load(open(os.path.join(BANK, "proxy", "rows_va.json"))):
        va_rows.setdefault(nm, []).append(r)
    assert set(va_rows) == set(sp["val"]) and len(va_rows) == 40, "the bank's val 40 differ between split.json and rows_va"
    idx = {}
    for line in open(os.path.join(CAP, "index.jsonl")):
        d = json.loads(line)
        idx[d["key"].replace("/", "__") + ".npz"] = d
    allf = sorted(f for f in os.listdir(CAP) if f.startswith("dd__mc_") and f.endswith(".npz"))
    assert set(allf) == set(idx), "capture files and index.jsonl differ"
    train = [f for f in allf if f not in va_rows]
    held = set(sp["heplus"]) | set(sp["mbpp"]) | set(va_rows)
    assert not held & set(train), "an eval file is in the training list"
    return train, sp, va_rows, idx


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verdict(reads, ctl):
    """the pre-registered lines (LEVERS_A.md, L1CF), on the reads in step order"""
    if any(abs(ctl[s]["all"] - CTL_REF[s]) > 0.1 for s in CTL_REF):
        return "VOID", "the control is off lever7_agree.py's 94.42 / 85.67 / 87.29 by > 0.1"
    rs = [reads[k] for k in sorted(reads, key=lambda k: reads[k]["steps"])]
    he = [r["heplus"]["diff"] for r in rs]
    vd = [r["val"]["diff"] for r in rs]
    va = [r["val"]["all"] for r in rs]
    why = []
    if all(d < 1.0 for d in he):
        why.append(f"HE+ < +1.0 at every read ({', '.join(f'{d:+.2f}' for d in he)})")
    if min(vd) < 0:
        why.append(f"val below the control at a read ({', '.join(f'{d:+.2f}' for d in vd)})")
    if max(va) - va[-1] >= 1.0:
        why.append(f"val fell {max(va) - va[-1]:.2f} pt from its best read")
    if why:
        return "KILL", "; ".join(why)
    if he[-1] >= 1.0:
        return "PASS", f"HE+ {he[-1]:+.2f} at the end of the epoch, val {vd[-1]:+.2f}"
    return "BETWEEN", f"HE+ >= +1.0 only before the end ({', '.join(f'{d:+.2f}' for d in he)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", default="1000,2000", help="read-outs after these steps (the end of the epoch is always read)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--smoke", type=int, default=0, help="train on the N longest files only, reads at N/2 and N "
                    "(memory peak and harness check; the eval sets stay full so the control check holds)")
    ap.add_argument("--save-every", type=float, default=120.0, help="seconds between resume states")
    ap.add_argument("--dry", action="store_true", help="check the file lists and exit before the model loads")
    a = ap.parse_args()
    name = f"lever1cf_{a.tag}" + ("_smoke" if a.smoke else "")
    train, sp, va_rows, idx = file_lists()
    if a.smoke:
        train = sorted(train, key=lambda f: -(idx[f]["P"] + idx[f]["n"]))[:a.smoke]
    marks = [a.smoke // 2] if a.smoke else sorted(int(m) for m in a.reads.split(",") if m)
    marks = [m for m in marks if 0 < m < len(train)]
    nrows = sum(idx[f]["n"] for f in train)
    print(f"[1cf] {len(train)} train files ({nrows} answer rows, longest {max(idx[f]['P'] + idx[f]['n'] for f in train)} "
          f"tokens); eval HE+ {len(sp['heplus'])} / MBPP {len(sp['mbpp'])} / val {len(va_rows)}; reads after steps "
          f"{marks} + the end ({len(train)})", flush=True)
    if a.dry:
        z = np.load(os.path.join(CAP, train[0]))
        assert {"ids", "tk", "lp", "gen"} <= set(z.files) and z["tk"].shape[1] == 32
        print("[1cf] DRY OK: lists checked, first file readable; no model loaded", flush=True)
        return

    from sb import core as C
    from sb import resume as R
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.tuner.lora import LoRALinear
    C.limit_memory(float(os.environ.get("SB_MEM_GB", "7")))
    R.arm_sigterm()
    t0 = time.time()
    t = C.tm()
    L23 = t.layers[23]
    assert not L23.is_linear, "layer 23 must be a full-attention layer"

    def stop_now(where):
        if R.STOP["f"]:
            print(f"[1cf] SIGTERM during {where}: exiting to resume later", flush=True)
            sys.exit(75)

    def wait_pause():                                    # PAUSE holds the loop; a SIGTERM still gets through
        while os.path.exists(PAUSE) and not R.STOP["f"]:
            time.sleep(1)

    def pre23(ids):                                       # the frozen trunk up to layer 22 (inference kernels)
        x = t.embed_tokens(mx.array(np.asarray(ids, np.int32))[None])
        fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
        for i in range(23):
            x = t.layers[i](x, mask=ssm if t.layers[i].is_linear else fa, cache=None)
        return x.astype(mx.bfloat16)

    def load_file(path, rows_of):
        z = np.load(path)
        ids, tk, lp = np.asarray(z["ids"]), np.asarray(z["tk"]), np.asarray(z["lp"], np.float32)
        rr = np.array([r for r in rows_of(z) if r + 1 < len(ids)])
        return ids, rr, tk[rr], lp[rr]

    ans = lambda z: range(int(z["gen"]) - 1, z["tk"].shape[0])
    sets = {}
    for st, names in (("heplus", sp["heplus"]), ("mbpp", sp["mbpp"]), ("val", list(va_rows))):
        sets[st] = []
        for nm in names:
            wait_pause(); stop_now("eval caching")
            ids, rr, tk, lp = load_file(os.path.join(BANK, nm), (lambda z, nm=nm: va_rows[nm]) if st == "val" else ans)
            x = pre23(ids); mx.eval(x)
            sets[st].append(dict(x=x, rows=rr, tk=tk, lp=lp, nm=nm))
    print(f"[1cf] cached layer-23 inputs of {sum(len(v) for v in sets.values())} eval files, {time.time() - t0:.0f}s",
          flush=True)

    def head(x, rows):                                    # layer 23 -> norm -> tied 4-bit lm_head, on the scored rows
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

    t.freeze()

    def wrap(mod):                                        # LoRA on layer 23's linears (q / k / v / o and the MLP)
        for k, v in mod.children().items():
            if isinstance(v, (nn.Linear, nn.QuantizedLinear)):
                setattr(mod, k, LoRALinear.from_base(v, r=a.rank, dropout=0.0, scale=a.scale))
            elif isinstance(v, nn.Module):
                wrap(v)
    wrap(L23)
    npar = sum(v.size for _, v in tree_flatten(L23.trainable_parameters()))
    tc = time.time()
    hc = hits()                                           # LoRA's B starts at zero: this is the untrained 2B
    ctl = summary(hc)
    print(f"[1cf] control (LoRA zero, {npar} trainable params, {time.time() - tc:.0f}s): {json.dumps(ctl)}", flush=True)
    stop_now("the control read")

    def loss_fn(x, rows, tk, lp):
        fa = create_attention_mask(x, None)
        h = L23(x, mask=fa, cache=None)
        lq = C.lm_logits_train(t.norm(h[0, mx.array(rows)]))
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        p = mx.softmax(mx.array(lp), axis=-1)             # the 125B's top-32, renormalized
        return -(p * mx.take_along_axis(lq, mx.array(tk.astype(np.int32)), axis=-1)).sum(-1).mean()

    opt = optim.Adam(learning_rate=a.lr)
    vg = nn.value_and_grad(L23, loss_fn)
    perm = np.random.default_rng(1115).permutation(len(train))
    N = len(train)
    step, losses, reads, wall_s, trunk_s, fit_s, read_s = 0, [], {}, 0.0, 0.0, 0.0, 0.0
    os.makedirs(C.CKPT, exist_ok=True)
    sp_ = os.path.join(C.CKPT, name + "_state")           # mid-run resume (sb/resume.py)
    meta = R.load(sp_, {"l23": L23}, {"l23": opt})
    if meta:
        step, losses, reads, wall_s, trunk_s, fit_s, read_s = (meta[k] for k in
                                                              ("step", "losses", "reads", "wall_s", "trunk_s", "fit_s", "read_s"))
        assert meta["n"] == N and meta["perm0"] == [int(v) for v in perm[:8]], "resume state from another file list"
        print(f"[1cf] resumed at step {step}/{N} (reads {list(reads)})", flush=True)
    marks = [m for m in marks if m > step]
    last_save, s0, w0 = time.time(), step, wall_s

    def snap():
        return dict(step=step, losses=losses, reads=reads, wall_s=wall_s, trunk_s=trunk_s, fit_s=fit_s, read_s=read_s,
                    n=N, perm0=[int(v) for v in perm[:8]])

    def read_out():
        nonlocal read_s
        tr_ = time.time()
        r = dict(summary(hits(), hc), steps=step, loss100=round(float(np.mean(losses[-100:])), 4))
        ck = os.path.join(C.CKPT, f"{name}_lora_l23_s{step}.safetensors")
        mx.save_safetensors(ck, dict(tree_flatten(L23.trainable_parameters())))
        r["ckpt"] = os.path.relpath(ck, C.ROOT); r["sha256"] = sha256(ck)
        read_s += time.time() - tr_
        r["seconds"] = round(time.time() - tr_, 1)
        reads[f"s{step}"] = r
        print(f"[1cf] read after step {step}/{N} ({r['seconds']:.0f}s): " + " ".join(
            f"{st} {r[st]['all']:.2f} ({r[st]['diff']:+.2f} [{r[st]['ci95'][0]:+.2f}, {r[st]['ci95'][1]:+.2f}], "
            f"unsure {r[st]['unsure']:.2f})" for st in sets) + f"; ckpt sha256 {r['sha256'][:12]}", flush=True)

    while step < N:
        wait_pause(); stop_now(f"step {step}")
        ts = time.time()
        ids, rr, tk, lp = load_file(os.path.join(CAP, train[int(perm[step])]), ans)
        x = pre23(ids); mx.eval(x)
        tm_ = time.time()
        l, g = vg(x, rr, tk, lp)
        opt.update(L23, g); mx.eval(L23.parameters(), opt.state)
        te = time.time()
        trunk_s += tm_ - ts; fit_s += te - tm_; wall_s += te - ts
        losses.append(round(float(l), 5)); step += 1
        if step % 25 == 0 or step == N:
            k = step - s0
            print(f"[1cf] step {step}/{N} loss {np.mean(losses[-100:]):.4f} trunk {trunk_s / step:.3f}s fit "
                  f"{fit_s / step:.3f}s {(wall_s - w0) / max(k, 1):.3f}s/step", flush=True)
        if (marks and step == marks[0]) or step == N:
            if marks and step == marks[0]:
                marks.pop(0)
            read_out()
            R.save(sp_, {"l23": L23}, {"l23": opt}, snap()); last_save = time.time()
        if R.STOP["f"] or time.time() - last_save >= a.save_every:
            wrote = R.save(sp_, {"l23": L23}, {"l23": opt}, snap()); last_save = time.time()
            if R.STOP["f"]:
                print(f"[1cf] SIGTERM: state saved at step {step} ({wrote / 1e6:.1f} MB), exiting to resume later", flush=True)
                sys.exit(75)
    v, why = verdict(reads, ctl)
    last = reads[f"s{N}"]
    res = dict(verdict=v, why=why, control=ctl, reads=reads, final=f"s{N}", steps=N, epochs=1, rank=a.rank, scale=a.scale,
               lr=a.lr, train_files=N, train_answer_rows=nrows, trainable_params=npar,
               loss_first100=float(np.mean(losses[:100])), loss_last100=float(np.mean(losses[-100:])),
               trunk_s_per_step=round(trunk_s / N, 4), fit_s_per_step=round(fit_s / N, 4), read_s=round(read_s, 1),
               mlx_peak_gb=round(mx.get_peak_memory() / 1e9, 2), seconds_this_process=round(time.time() - t0, 1),
               loss_by_100=[round(float(np.mean(losses[i:i + 100])), 4) for i in range(0, N, 100)])
    json.dump(res, open(os.path.join(C.RES, name + ".json"), "w"), indent=1)
    R.finish(sp_)
    print(f"[1cf] after one epoch ({N} steps): HE+ {last['heplus']['all']:.2f} ({last['heplus']['diff']:+.2f}) MBPP "
          f"{last['mbpp']['all']:.2f} ({last['mbpp']['diff']:+.2f}) val {last['val']['all']:.2f} ({last['val']['diff']:+.2f})",
          flush=True)
    print(f"[1cf] DONE verdict {v}: {why}", flush=True)


if __name__ == "__main__":
    main()
