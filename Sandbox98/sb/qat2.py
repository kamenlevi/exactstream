"""QAT follow-ups (orchestrator 07:3x): three single-arm screens against s30, one at a time.
Each trains ONE arm from the native head over s30's own recipe (base files, 96-anchor slices, seed 1114, one pass =
1,889 steps x depth 30, lr 1e-5: 20-step warm-up, cosine to 10 %, fresh Adam, clip 1.0). The harness is deterministic
(this morning's ctl arm reproduced s30 exactly), so s30 is the matched control.
  mix    mixed 4/8-bit by sensitivity. Phase A (eval only, on s30) prices each Linear alone at 4-bit g32 against all
         8-bit; the subset carrying the most loss within +6 MB stays at 8-bit (trained plain), the rest trains through
         fake 4-bit g32. Phase B (training) runs only if that subset carries >= 40 % of the summed per-module loss.
  lsq    learned group scales: every Linear trains through a 4-bit g32 grid with a learned per-group scale and offset
         (scale = s0 exp(qa), offset = b0 + s0 qc; s0 / b0 the min-max start; qa / qc at lr 1e-3 on their own Adam
         and clip), straight-through rounding; real 4-bit = the learned grid in MLX's affine format
  floor  plain QAT g32 with the cosine ending at --floor (0.3) of the peak lr instead of 0.1
Read-out in-process: s30 at 8-bit (THE BAR), this morning's plain QAT g32 head (ckpt/qat_full_g32_qat_q4, the
reference), the lever at real quantization, the lever's latent fp32 weights (info). Full vocab and the 8K + context
list; MBPP-50 / HE-30 carry the lines; the second halves (MBPP 51-100 / HE 31-60) read the bar and the lever.
LINES (results/LEVERS_A.md): PASS = within 1 % of the bar in all four cells (MBPP-50 / HE-30 x full / list).
Guards: MLX cache cleared per window; SIGTERM saves the state and exits 75 (resume); also saved every --save-every
steps and after training; removed once the result is written.
usage: qat2.py --lever mix|lsq|floor [--floor 0.3] [--smoke-long] [--check] [--save-every 500]"""
import argparse, copy, itertools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from sb import core as C
from sb import resume as R
from sb.compare import paired
from sb.qat import LISTS, QATLinear, Yield, from_qat, quant, tables, to_qat
from sb.train import split

V = 248320
B8, B4 = 1 + 4 / 64, 0.5 + 4 / 32            # bytes per weight: 8-bit g64 / 4-bit g32, fp16 scales and biases
MIX_MB = 6.0                                    # MIX: the 8-bit modules may add at most this much to the 38 MB body
REF = "qat_full_g32_qat_q4"                     # this morning's plain QAT g32 head at real 4-bit
MID_LINE = 3.688                                # stop line: -6.3 % of the matched ctl's 3.936 (MBPP-50 full, step 944)
QSUF = (".qa", ".qc")


class LSQLinear(nn.Module):
    """4-bit group-gs grid with a learned per-group scale and offset, straight-through rounding. scale = s0 exp(qa),
    offset = b0 + s0 qc, with s0 / b0 the min-max start of the NATIVE weights (underscored: not parameters, rebuilt
    the same way on a resume)"""
    def __init__(self, w, gs=32, bits=4):
        super().__init__()
        O, I = w.shape
        self.weight = w
        wg = w.reshape(O, I // gs, gs)
        mn, mxv = wg.min(-1, keepdims=True), wg.max(-1, keepdims=True)
        self._s0 = mx.maximum((mxv - mn) / (2 ** bits - 1), 1e-8)
        self._b0 = mn
        self.qa = mx.zeros_like(self._s0)
        self.qc = mx.zeros_like(self._s0)
        self.gs, self.qmax = gs, 2 ** bits - 1

    def sb(self):
        return self._s0 * mx.exp(self.qa), self._b0 + self._s0 * self.qc

    def __call__(self, x):
        O, I = self.weight.shape
        s, b = self.sb()
        v = (self.weight.reshape(O, I // self.gs, self.gs) - b) / s
        q = mx.clip(v + mx.stop_gradient(mx.round(v) - v), 0, self.qmax)
        return x @ (q * s + b).reshape(O, I).T

    def grid(self):
        O, I = self.weight.shape
        s, b = self.sb()
        v = (self.weight.reshape(O, I // self.gs, self.gs) - b) / s
        return mx.clip(mx.round(v), 0, self.qmax), s, b


def swap(D, fn):
    for _, m in list(D.named_modules()):
        for k, v in list(m.children().items()):
            r = fn(k, v)
            if r is not None:
                setattr(m, k, r)


def to_lsq(D, gs):
    swap(D, lambda k, v: LSQLinear(v.weight, gs) if isinstance(v, nn.Linear) else None)


def lsq_real(D):
    """LSQLinear -> QuantizedLinear holding the learned grid (MLX affine 4-bit: value i at word i // 8, shift 4 (i % 8))"""
    def f(k, v):
        if not isinstance(v, LSQLinear):
            return None
        q, s, b = v.grid()
        O, I = v.weight.shape
        sh = (mx.arange(8) * 4).astype(mx.uint32).reshape(1, 1, 8)
        packed = (q.astype(mx.uint32).reshape(O, I // 8, 8) << sh).sum(-1).astype(mx.uint32)
        ql = nn.QuantizedLinear(I, O, bias=False, group_size=v.gs, bits=4)
        ql.weight, ql.scales, ql.biases = packed, s.reshape(O, I // v.gs), b.reshape(O, I // v.gs)
        return ql
    swap(D, f)


def lsq_latent(D):
    def f(k, v):
        if not isinstance(v, LSQLinear):
            return None
        lin = nn.Linear(v.weight.shape[1], v.weight.shape[0], bias=False); lin.weight = v.weight
        return lin
    swap(D, f)


def mix_quant(D, m8, gs):
    nn.quantize(D, group_size=64, bits=8, class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and p.split(".")[-1] in m8)
    nn.quantize(D, group_size=gs, bits=4, class_predicate=lambda p, mod: isinstance(mod, nn.Linear))
    return D


def one4(D, nm, gs):
    """only module nm at 4-bit g32, every other Linear at 8-bit"""
    nn.quantize(D, group_size=gs, bits=4, class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and p.split(".")[-1] == nm)
    nn.quantize(D, group_size=64, bits=8, class_predicate=lambda p, mod: isinstance(mod, nn.Linear))
    return D


def sched(lr, warmup, steps, floor):
    return lambda s: lr * mx.minimum(1.0, (s + 1) / warmup) * (
        floor + (1 - floor) / 2 * (1 + mx.cos(np.pi * mx.minimum(1.0, s / max(1, steps - 1)))))


def qparams(D):
    return tree_unflatten([(k, v) for k, v in tree_flatten(D.trainable_parameters()) if k.endswith(QSUF)])


def pct(base_per, per, rng, key="kept"):
    d = paired(base_per, per, key, rng)
    A = max(1e-9, d["A"])
    return {"pct": round(100 * d["diff"] / A, 2), "ci_pct": [round(100 * x / A, 2) for x in d["ci95"]],
            "diff": round(d["diff"], 4), "ci": [round(x, 4) for x in d["ci95"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", required=True, choices=("mix", "lsq", "floor"))
    ap.add_argument("--floor", type=float, default=0.3); ap.add_argument("--gs", type=int, default=32)
    ap.add_argument("--smoke-long", dest="smoke_long", action="store_true")
    ap.add_argument("--check", action="store_true", help="20 steps of plain QAT g32 at floor 0.1: must equal this "
                                                         "morning's qat arm (step 1 1.0741, mean of 20 0.9108)")
    ap.add_argument("--save-every", dest="save_every", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        a.lever, a.floor = "floor", 0.1
    lv, smoke = a.lever, a.smoke_long
    floor = a.floor if lv == "floor" else 0.1
    tag = f"qat2_{lv}" + ("_smoke" if smoke else "") + ("_check" if a.check else "")
    C.limit_memory(7)
    t0 = time.time()
    C.load_target(); C.drop_trunk()
    tr, _ = split("base")
    sl = C.pass_slices(tr, 96, 1114)                      # s30's own one-pass order
    steps_sched = len(sl)
    spec = {"mbpp": ("mbpp", 0, 50), "he": ("he", 0, 30), "mbpp2": ("mbpp", 50, 100), "he2": ("he", 30, 60)}
    sets = {k: C.files(st)[lo:hi] for k, (st, lo, hi) in spec.items()}
    run_sl, mid, save_every = sl, steps_sched // 2, a.save_every
    if smoke:
        ln = {f: len(np.load(f)["ids"]) for f in set(s[0] for s in sl)}
        run_sl = sorted(sl, key=lambda s: -min(ln[s[0]], s[1] + s[2] + 32))[:3]
        sets = {k: sorted(fs, key=lambda f: -len(np.load(f)["ids"]))[:2] for k, fs in sets.items()}
        mid, save_every = 2, 2
    if a.check:
        run_sl, mid, save_every = sl[:20], 0, 0
    steps = len(run_sl)
    line_sets = {k: sets[k] for k in ("mbpp", "he")}
    conf_sets = {k: sets[k] for k in ("mbpp2", "he2")}
    cnt = np.load(os.path.join(C.DATA, "shortlist_counts.npy"))
    m8l = np.full(V, -1e9, np.float32); m8l[np.argsort(-cnt, kind="stable")[:8192]] = 0.0
    statics = {8192: mx.array(m8l)}
    rng = np.random.default_rng(1114)
    out = {"args": vars(a), "lever": lv}
    sizes = {}
    m8 = []
    try:
        if lv == "mix":                                   # ---- phase A: per-module 4-bit cost on s30, eval only
            pa_path = os.path.join(C.RES, tag + "_phaseA.json")
            s30 = C.load_drafter("s30")
            sizes = {p.split(".")[-1]: v.weight.size for p, v in s30.named_modules() if isinstance(v, nn.Linear)}
            if os.path.exists(pa_path):
                PA = json.load(open(pa_path))
                print(f"[qat2] phase A read from {os.path.basename(pa_path)}", flush=True)
            else:
                ta = time.time()
                bar_t = {k: tables(quant(copy.deepcopy(s30), 8, 64), fs, statics) for k, fs in line_sets.items()}
                rows = {}
                for nm in sizes:
                    Dq = one4(copy.deepcopy(s30), nm, a.gs)
                    t = {k: tables(Dq, fs, statics) for k, fs in line_sets.items()}; del Dq
                    r = {st: pct(C.summarize(bar_t[st]["full"])[1], C.summarize(t[st]["full"])[1], rng)["pct"] for st in t}
                    r["avg"] = round((r["mbpp"] + r["he"]) / 2, 3)
                    r["mb_saved"] = round(sizes[nm] * (B8 - B4) / 1e6, 2)
                    rows[nm] = r
                    print(f"[qat2] phase A {nm:9s} alone at 4-bit g{a.gs}: kept/32 vs all 8-bit MBPP {r['mbpp']:+.2f} % HE "
                          f"{r['he']:+.2f} % (avg {r['avg']:+.2f}); 8-bit costs +{r['mb_saved']:.2f} MB", flush=True)
                loss = {nm: max(0.0, -r["avg"]) for nm, r in rows.items()}
                best, best_l = (), -1.0
                for n in range(len(rows) + 1):
                    for sub in itertools.combinations(sorted(rows), n):
                        mb = sum(rows[x]["mb_saved"] for x in sub)
                        lo = sum(loss[x] for x in sub)
                        if mb <= MIX_MB + 1e-9 and lo > best_l + 1e-12:
                            best, best_l = sub, lo
                tot = sum(loss.values())
                PA = {"rows": rows, "m8": list(best), "carried": round(best_l / tot, 3) if tot > 0 else 0.0,
                      "extra_mb": round(sum(rows[x]["mb_saved"] for x in best), 2), "wall_s": round(time.time() - ta, 1)}
                json.dump(PA, open(pa_path, "w"), indent=1)
            m8 = PA["m8"]
            out["phaseA"] = PA
            print(f"[qat2] phase A: keep {m8} at 8-bit (+{PA['extra_mb']} MB) carrying {100 * PA['carried']:.0f} % of the "
                  f"summed per-module loss (rule: >= 40 %)", flush=True)
            del s30
            if PA["carried"] < 0.4 and not smoke:
                out.update(status="KILL at phase A", verdict="KILL", wall_s=round(time.time() - t0, 1))
                json.dump(out, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
                print("[qat2] KILL at phase A: the 8-bit modules that fit carry under 40 % of the loss; DONE", flush=True)
                return
            mx.clear_cache()
    except Yield:
        print("[qat2] SIGTERM during phase A: exiting to resume (phase A restarts)", flush=True)
        sys.exit(75)
    D = C.load_drafter("native"); D.freeze(); D.unfreeze()
    if not sizes:
        sizes = {p.split(".")[-1]: v.weight.size for p, v in D.named_modules() if isinstance(v, nn.Linear)}
    if lv == "mix":
        swap(D, lambda k, v: QATLinear(v.weight, a.gs) if isinstance(v, nn.Linear) and k not in m8 else None)
    elif lv == "lsq":
        to_lsq(D, a.gs)
    else:
        to_qat(D, a.gs)
    body_mb = round(sum(n * (B8 if k in m8 else B4) for k, n in sizes.items()) / 1e6, 2)
    opt = optim.Adam(learning_rate=sched(1e-5, 20, steps_sched, floor), bias_correction=True)
    optq = optim.Adam(learning_rate=sched(1e-3, 20, steps_sched, 0.1), bias_correction=True) if lv == "lsq" else None
    opts = {"d": opt, **({"q": optq} if optq is not None else {})}
    sp = os.path.join(C.CKPT, tag + "_state")
    i0, midres, log = 0, None, []
    if os.path.exists(sp + ".safetensors"):
        meta = R.load(sp, {"d": D}, {"d": opt})
        if optq is not None:
            arrs = mx.load(sp + ".safetensors")
            optq.init(qparams(D))
            optq.state = tree_unflatten([(n[4:], v) for n, v in arrs.items() if n.startswith("q.o.")])
        i0, midres, log = meta["i"], meta.get("mid"), meta["log"]
        print(f"[qat2] resumed at step {i0}", flush=True)
    R.arm_sigterm()

    def save(i_done):
        meta = {"i": i_done, "log": log}
        if midres is not None:
            meta["mid"] = midres
        return R.save(sp, {"d": D}, opts, meta)

    def real_form():
        Dq = copy.deepcopy(D)
        if lv == "lsq":
            lsq_real(Dq)
        else:
            from_qat(Dq)
            mix_quant(Dq, m8, a.gs) if lv == "mix" else quant(Dq, 4, a.gs)
        return Dq

    def latent_form():
        Dq = copy.deepcopy(D)
        lsq_latent(Dq) if lv == "lsq" else from_qat(Dq)
        return Dq

    def train_step(x):
        C.REPL["fn"] = None
        loss, kl, ex, g = C.ttt_step(D, x, 30, 0.97, None, None)
        if optq is None:
            g, _ = optim.clip_grad_norm(g, 1.0)
            opt.update(D, g)
            mx.eval(D.parameters(), opt.state)
        else:
            fl = tree_flatten(g)
            gw = tree_unflatten([(k, v) for k, v in fl if not k.endswith(QSUF)])
            gq = tree_unflatten([(k, v) for k, v in fl if k.endswith(QSUF)])
            gw, _ = optim.clip_grad_norm(gw, 1.0)
            gq, _ = optim.clip_grad_norm(gq, 1.0)
            opt.update(D, gw); optq.update(D, gq)
            mx.eval(D.parameters(), opt.state, optq.state)
        return float(loss)

    t1 = time.time()
    print(f"[qat2] lever {lv}: one arm, {steps} steps x depth 30 from native (cosine over {steps_sched} to {floor:.0%}), "
          f"from step {i0}; stop line at step {mid or '-'}; body {body_mb} MB (8-bit: {m8 or '-'})", flush=True)
    i = i0
    try:
        for i in range(i0, steps + 1):
            if mid and i == mid and midres is None:
                tm = time.time()
                Dq = real_form(); tq = tables(Dq, line_sets["mbpp"], statics, ("full",)); del Dq
                k = C.summarize(tq["full"])[0]["kept_per_32"]
                midres = {"step": i, "kept": round(k, 3), "line": MID_LINE}
                print(f"[qat2] STOP LINE at step {i}: MBPP-50 full kept/32 {k:.3f} (stop below {MID_LINE}) "
                      f"({time.time() - tm:.0f}s)", flush=True)
                if k < MID_LINE and not smoke:
                    out.update(status="STOPPED at the stop line", verdict="KILL", mid=midres, wall_s=round(time.time() - t0, 1))
                    json.dump(out, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
                    R.finish(sp)
                    print("[qat2] STOPPED at the stop line; DONE", flush=True)
                    return
            if i == steps:
                break
            C.wait_pause()
            f, b, r = run_sl[i]
            x = C.window_at(f, b, r, 30)
            log.append(train_step(x))
            del x
            mx.clear_cache()                                 # per window (orchestrator 06:3x)
            if (i + 1) % 20 == 0 or i == i0:
                print(f"[qat2] step {i + 1}/{steps} loss {np.mean(log[-20:]):.4f}; {(time.time() - t1) / (i + 1 - i0):.2f}s/step, "
                      f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
            if R.STOP["f"]:
                wrote = save(i + 1)
                print(f"[qat2] SIGTERM: state saved at step {i + 1} ({wrote / 1e9:.2f} GB), exiting to resume later", flush=True)
                sys.exit(75)
            if save_every and (i + 1) % save_every == 0 and i + 1 < steps:
                wrote = save(i + 1)
                print(f"[qat2] state saved at step {i + 1} ({wrote / 1e9:.2f} GB)", flush=True)
                if smoke:
                    R.load(sp, {"d": D}, {"d": opt})
                    if optq is not None:
                        arrs = mx.load(sp + ".safetensors")
                        optq.init(qparams(D))
                        optq.state = tree_unflatten([(n[4:], v) for n, v in arrs.items() if n.startswith("q.o.")])
                    print("[qat2] smoke: state reloaded (save + load round trip OK)", flush=True)
    except Yield:
        wrote = save(i)
        print(f"[qat2] SIGTERM during the stop-line read-out: state saved at step {i} ({wrote / 1e9:.2f} GB); exiting", flush=True)
        sys.exit(75)
    if a.check:
        print(f"[qat2] CHECK: step 1 loss {log[0]:.4f} (morning 1.0741), mean of 20 {np.mean(log):.4f} (morning 0.9108); DONE",
              flush=True)
        return
    if i0 < steps and not smoke:
        wrote = save(steps)
        print(f"[qat2] trained in {time.time() - t1:.0f}s; state saved ({wrote / 1e9:.2f} GB); reading out", flush=True)
    else:
        print(f"[qat2] trained in {time.time() - t1:.0f}s; reading out", flush=True)
    tr0 = time.time()
    ev = {}
    try:
        forms = [("bar", lambda: quant(C.load_drafter("s30"), 8, 64), True),
                 ("ref", None, False), ("lever", real_form, True), ("latent", latent_form, False)]
        for name, mk, conf in forms:
            if name == "ref":
                Dq = C.Drafter(); quant(Dq, 4, 32); Dq.load_weights(os.path.join(C.CKPT, REF + ".safetensors"))
            else:
                Dq = mk()
            ev[name] = {k: tables(Dq, fs, statics) for k, fs in (sets if conf else line_sets).items()}
            if name == "lever" and not smoke:
                lever_params = dict(tree_flatten(Dq.parameters()))
            del Dq
            mx.clear_cache()
            print(f"[qat2] read-out {name} done ({time.time() - tr0:.0f}s)", flush=True)
    except Yield:
        print("[qat2] SIGTERM during the read-out: the trained state is on disk; exiting to resume (the read-out restarts)",
              flush=True)
        sys.exit(75)
    res = {}
    for name in ("ref", "lever", "latent"):
        for st in ev[name]:
            for lk in LISTS:
                s_, per = C.summarize(ev[name][st][lk])
                sb_, pb = C.summarize(ev["bar"][st][lk])
                row = {"kept32": round(s_["kept_per_32"], 3), "bar": round(sb_["kept_per_32"], 3),
                       "0-0": round(100 * s_["0-0"], 2), "23-30": round(100 * s_["23-30"], 2),
                       "vs_bar": pct(pb, per, rng)}
                if name != "ref" and st in ev["ref"]:
                    row["vs_ref"] = pct(C.summarize(ev["ref"][st][lk])[1], per, rng)
                res[f"{name}|{st}|{lk}"] = row
                print(f"[qat2] {name:6s} {st:5s} {lk:12s} kept {row['kept32']:.3f} (bar {row['bar']:.3f}: {row['vs_bar']['pct']:+.2f} %, "
                      f"CI {row['vs_bar']['ci_pct'][0]:+.2f}..{row['vs_bar']['ci_pct'][1]:+.2f})"
                      + (f"; vs the plain QAT head {row['vs_ref']['diff']:+.3f} kept (CI {row['vs_ref']['ci'][0]:+.3f}.."
                         f"{row['vs_ref']['ci'][1]:+.3f})" if "vs_ref" in row else ""), flush=True)
    line = all(res[f"lever|{st}|{lk}"]["vs_bar"]["pct"] >= -1.0 for st in ("mbpp", "he") for lk in LISTS)
    conf = all(res[f"lever|{st}|{lk}"]["vs_bar"]["pct"] >= -2.0 for st in ("mbpp2", "he2") for lk in LISTS)
    if lv == "mix":
        rec = {st: round(1 - res[f"lever|{st}|full"]["vs_bar"]["pct"] / min(-1e-9, res[f"ref|{st}|full"]["vs_bar"]["pct"]), 3)
               for st in ("mbpp", "he")}
        kill = all(v < 0.5 for v in rec.values())
        out["recovery_of_ref_gap"] = rec
    else:
        kill = all(res[f"lever|{st}|full"]["vs_ref"]["diff"] <= 0 for st in ("mbpp", "he"))
    verdict = ("PASS" if conf else "PASS, not confirmed") if line else ("KILL" if kill else "BETWEEN")
    if verdict.startswith("PASS") and not smoke:              # keep the deliverable form only on a PASS
        mx.save_safetensors(os.path.join(C.CKPT, tag + "_q.safetensors"), lever_params)
    out.update(status="done", verdict=verdict, mid=midres, res=res, body_mb=body_mb, m8=m8, steps=steps,
               loss_last100=round(float(np.mean(log[-100:])), 4) if log else None, wall_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
    R.finish(sp)
    print(f"[qat2] VERDICT {lv} {verdict}; body {body_mb} MB; read-outs in {time.time() - tr0:.0f}s", flush=True)
    print(f"[qat2] DONE in {out['wall_s']:.0f}s -> results/{tag}.json", flush=True)


if __name__ == "__main__":
    main()
