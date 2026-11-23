"""CHAIN SIZE, the last blocker: can quantization-aware training (QAT) make a 4-bit chain head hold its kept/32?
(Post-training 4-bit cost -6.4 % MBPP / -10.6 % HE kept; 8-bit is free; the head part is solved by the prompt-aware list,
results/listctx.json.) Two arms in one process, on the same slices in the same order:
  ctl   plain fp32 training
  qat   every Linear of the head trains through fake 4-bit weights (group --gs, the MLX affine quantizer, straight-through
        gradient: w + stop_gradient(deq(q(w)) - w))
Night screen (default; results/qat.json): from s30o, 200 steps x depth 30 on pool2 (s30o's Adam continued, cosine
re-anneal, lr 1e-5), group 64.
FULL BUDGET (--full, orchestrator 06:3x; results/qat_full_g<gs>.json): from the NATIVE head over s30's own recipe (base
files, 96-anchor slices, seed 1114, one pass = 1,889 steps x depth 30, lr 1e-5 with a 20-step warm-up and a cosine to
10 %, fresh Adam, clip 1.0). Stop line at the half: ctl 8-bit vs qat at real 4-bit, MBPP-50 full vocab.
Read-out in-process: ctl fp32, ctl 8-bit (THE BAR: the 8-bit body at matched steps), ctl 4-bit post-training (group 64
and --gs), qat fp32 weights, qat at REAL 4-bit (--gs); each with the full vocab and the 8K + context list; kept/32 first,
paired against the bar with the SAME list. --full adds the held-out second halves (MBPP 51-100, HE 31-60) and the pooled
sets as confirmation.
LINE (results/LEVERS_A.md): PASS if qat at real 4-bit keeps kept/32 within 1 % of ctl 8-bit on MBPP-50 and HE-30, full
vocab and 8K + context.
Guards: the MLX cache is cleared per window; SIGTERM saves the run's state (both arms + Adam) and exits 75 for a resume
(also saved every --save-every steps and after training; removed once the result is written).
usage: qat.py [--steps 200] [--smoke]  |  qat.py --full --gs 32 [--smoke-long] [--save-every 500]"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from sb import core as C
from sb import resume as R
from sb.compare import paired
from sb.listctx import window_tables
from sb.train import split
from sb.train_multi import Arm

V = 248320
LISTS = ("full", "S8K+context")


class Yield(Exception):
    """SIGTERM arrived during a read-out"""


class QATLinear(nn.Module):
    """a Linear whose forward uses 4-bit fake-quantized weights (straight-through); same parameter name 'weight'"""
    def __init__(self, w, gs=64, bits=4):
        super().__init__()
        self.weight = w
        self.gs, self.bits = gs, bits

    def __call__(self, x):
        w = self.weight
        wq = mx.dequantize(*mx.quantize(w, group_size=self.gs, bits=self.bits), group_size=self.gs, bits=self.bits)
        return x @ (w + mx.stop_gradient(wq - w)).T


def to_qat(D, gs):
    for path, m in list(D.named_modules()):
        for k, v in list(m.children().items()):
            if isinstance(v, nn.Linear):
                setattr(m, k, QATLinear(v.weight, gs))


def from_qat(D):
    for path, m in list(D.named_modules()):
        for k, v in list(m.children().items()):
            if isinstance(v, QATLinear):
                lin = nn.Linear(v.weight.shape[1], v.weight.shape[0], bias=False); lin.weight = v.weight
                setattr(m, k, lin)


def quant(D, bits, gs):
    nn.quantize(D, group_size=gs, bits=bits, class_predicate=lambda p, mod: isinstance(mod, nn.Linear))
    return D


def linear_weights(D):
    return sum(v.weight.size for _, v in D.named_modules() if isinstance(v, (nn.Linear, QATLinear)))


def tables(D, fs, statics, lists=LISTS):
    tabs = {k: [] for k in lists}
    for f in fs:
        C.wait_pause()
        if R.STOP["f"]:
            raise Yield
        rts, gen, a0, _ = window_tables(D, f, statics)
        for k in lists:
            tabs[k].append((os.path.basename(f)[:-4], rts[k], gen, a0))
        del rts
        mx.clear_cache()
    return tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true", help="from the native head over s30's own recipe (1,889 steps)")
    ap.add_argument("--smoke-long", dest="smoke_long", action="store_true",
                    help="full-length smoke: the 3 longest training windows, the 2 longest files of each read-out set, "
                         "the stop line, a state save + reload and every read-out form")
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--mid", type=int, default=-1, help="stop-line step (default: the half of a --full run; 0 = none)")
    ap.add_argument("--stop-pct", dest="stop_pct", type=float, default=-6.3)
    ap.add_argument("--save-every", dest="save_every", type=int, default=0)
    a = ap.parse_args()
    smoke = a.smoke or a.smoke_long
    tag = (f"qat_full_g{a.gs}" if a.full else "qat") + ("_smoke" if smoke else "")
    C.limit_memory(7)
    t0 = time.time()
    C.load_target(); C.drop_trunk()
    if a.full:
        tr, _ = split("base")
        sl = C.pass_slices(tr, 96, 1114)                  # s30's own one-pass order (train.py defaults, files base)
        ta = argparse.Namespace(start="native", resume_opt="", const=False, lr=1e-5, warmup=20, steps=len(sl), depth=30,
                                wdecay=0.97)
        spec = {"mbpp": ("mbpp", 0, 50), "he": ("he", 0, 30), "mbpp2": ("mbpp", 50, 100), "he2": ("he", 30, 60)}
        fam = {"mbpp_all": ("mbpp", "mbpp2"), "he_all": ("he", "he2")}
    else:
        tr, _ = split("pool2")
        sl = C.pass_slices(tr, 96, 1115)
        ta = argparse.Namespace(start="s30o", resume_opt="s30o", const=False, lr=1e-5, warmup=20,
                                steps=min(a.steps, len(sl)), depth=30, wdecay=0.97)
        spec = {"mbpp": ("mbpp", 0, 50), "he": ("he", 0, 30)}
        fam = {}
    sets = {k: C.files(st)[lo:hi] for k, (st, lo, hi) in spec.items()}
    run_sl = sl[:ta.steps]
    mid = (ta.steps // 2 if a.full else 0) if a.mid < 0 else a.mid
    save_every = a.save_every
    if a.smoke_long:            # the peak comes from the longest windows: train on the 3 longest, read the 2 longest files
        ln = {f: len(np.load(f)["ids"]) for f in set(s[0] for s in sl)}
        run_sl = sorted(sl, key=lambda s: -min(ln[s[0]], s[1] + s[2] + 32))[:3]
        sets = {k: sorted(fs, key=lambda f: -len(np.load(f)["ids"]))[:2] for k, fs in sets.items()}
        mid, save_every = 2, 2
    elif a.smoke:
        run_sl = sl[:3]; sets = {k: fs[:2] for k, fs in sets.items()}; mid = 0
    steps = len(run_sl)
    arms = {"ctl": Arm("ctl", "-", ta), "qat": Arm("qat", "-", ta)}
    nlin = linear_weights(arms["ctl"].D)
    to_qat(arms["qat"].D, a.gs)
    cnt = np.load(os.path.join(C.DATA, "shortlist_counts.npy"))
    m8 = np.full(V, -1e9, np.float32); m8[np.argsort(-cnt, kind="stable")[:8192]] = 0.0
    statics = {8192: mx.array(m8)}
    sp = os.path.join(C.CKPT, tag + "_state")
    mods = lambda: {k: v.D for k, v in arms.items()}
    opts = lambda: {k: v.opt for k, v in arms.items()}
    i0, midres = 0, None
    if os.path.exists(sp + ".safetensors"):
        meta = R.load(sp, mods(), opts())
        for k, v in arms.items():
            v.log = meta["logs"][k]
        i0, midres = meta["i"], meta.get("mid")
        print(f"[qat] resumed at step {i0}" + (f" (stop line read before: {midres['pct']:+.2f} %)" if midres else ""), flush=True)
    R.arm_sigterm()

    def save(i_done):
        meta = {"i": i_done, "logs": {k: v.log for k, v in arms.items()}}
        if midres is not None:
            meta["mid"] = midres
        return R.save(sp, mods(), opts(), meta)

    def stop_line():
        Dc = quant(copy.deepcopy(arms["ctl"].D), 8, 64)
        tc = tables(Dc, sets["mbpp"], statics, ("full",)); del Dc
        Dq = copy.deepcopy(arms["qat"].D); from_qat(Dq); quant(Dq, 4, a.gs)
        tq = tables(Dq, sets["mbpp"], statics, ("full",)); del Dq
        sc, pc = C.summarize(tc["full"]); sq, pq = C.summarize(tq["full"])
        d = paired(pc, pq, "kept", np.random.default_rng(1114))
        A = max(1e-9, d["A"])
        return {"ctl_q8": round(sc["kept_per_32"], 3), "qat_q4": round(sq["kept_per_32"], 3),
                "pct": round(100 * d["diff"] / A, 2), "ci_pct": [round(100 * x / A, 2) for x in d["ci95"]]}

    t1 = time.time()
    print(f"[qat] arms ctl + qat (fake 4-bit g{a.gs}, STE): {steps} steps x depth 30 from {ta.start} "
          f"({'Adam continued' if ta.resume_opt else 'fresh Adam'}; cosine over {ta.steps}), from step {i0}; "
          f"stop line at step {mid or '-'}; {nlin / 1e6:.2f}M Linear weights", flush=True)
    i = i0
    try:
        for i in range(i0, steps + 1):
            if mid and i == mid and midres is None:      # the stop line at matched steps (also right after a resume here)
                tm = time.time()
                midres = stop_line(); midres["step"] = i
                print(f"[qat] STOP LINE at step {i}: MBPP full kept/32 ctl 8-bit {midres['ctl_q8']:.3f} vs qat 4-bit "
                      f"g{a.gs} {midres['qat_q4']:.3f} = {midres['pct']:+.2f} % (CI {midres['ci_pct'][0]:+.2f}.."
                      f"{midres['ci_pct'][1]:+.2f}); stop below {a.stop_pct:+.1f} % ({time.time() - tm:.0f}s)", flush=True)
                if midres["pct"] < a.stop_pct and not smoke:
                    json.dump({"args": vars(a), "status": "STOPPED at the stop line", "mid": midres, "steps_done": i},
                              open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
                    R.finish(sp)
                    print("[qat] STOPPED at the stop line: QAT is no better than post-training 4-bit; DONE", flush=True)
                    return
            if i == steps:
                break
            C.wait_pause()
            f, b, r = run_sl[i]
            x = C.window_at(f, b, r, 30)
            for m in arms.values():
                m.step(x, ta)
            del x
            mx.clear_cache()                                 # per window (orchestrator 06:3x)
            if (i + 1) % 20 == 0 or i == i0:
                print(f"[qat] step {i + 1}/{steps} " + " ".join(f"{k} {np.mean(m.log['loss'][-20:]):.4f}" for k, m in arms.items())
                      + f"; {(time.time() - t1) / (i + 1 - i0):.2f}s/step, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
            if R.STOP["f"]:
                wrote = save(i + 1)
                print(f"[qat] SIGTERM: state saved at step {i + 1} ({wrote / 1e9:.2f} GB), exiting to resume later", flush=True)
                sys.exit(75)
            if save_every and (i + 1) % save_every == 0 and i + 1 < steps:
                wrote = save(i + 1)
                print(f"[qat] state saved at step {i + 1} ({wrote / 1e9:.2f} GB)", flush=True)
                if smoke:
                    R.load(sp, mods(), opts())
                    print("[qat] smoke: state reloaded (save + load round trip OK)", flush=True)
    except Yield:
        wrote = save(i)
        print(f"[qat] SIGTERM during the stop-line read-out: state saved at step {i} ({wrote / 1e9:.2f} GB); exiting", flush=True)
        sys.exit(75)
    if i0 < steps and not smoke:
        wrote = save(steps)                                  # a yield during the read-out resumes here, not from step 0
        print(f"[qat] trained in {time.time() - t1:.0f}s; state saved ({wrote / 1e9:.2f} GB); reading out", flush=True)
    else:
        print(f"[qat] trained in {time.time() - t1:.0f}s; reading out", flush=True)
    tl = {k: {"last100": round(float(np.mean(m.log["loss"][-100:])), 4)} for k, m in arms.items() if m.log["loss"]}
    ev = {}
    tr0 = time.time()
    g = f"q4g{a.gs}"
    try:
        Dc = arms["ctl"].D
        ev["ctl|fp32"] = {k: tables(Dc, fs, statics) for k, fs in sets.items()}
        forms = [("ctl|q8", 8, 64), ("ctl|q4g64", 4, 64)] + ([(f"ctl|{g}", 4, a.gs)] if a.gs != 64 else [])
        for name, bits, gsz in forms:
            Dq = quant(copy.deepcopy(Dc), bits, gsz)
            ev[name] = {k: tables(Dq, fs, statics) for k, fs in sets.items()}; del Dq
        Dz = arms["qat"].D; from_qat(Dz)
        ev["qat|fp32"] = {k: tables(Dz, fs, statics) for k, fs in sets.items()}
        quant(Dz, 4, a.gs)
        ev[f"qat|{g}"] = {k: tables(Dz, fs, statics) for k, fs in sets.items()}
    except Yield:
        print("[qat] SIGTERM during the read-out: the trained state is on disk; exiting to resume (the read-out restarts)",
              flush=True)
        sys.exit(75)
    if a.full and not smoke:                             # the deliverable form only: the qat head at real 4-bit (~38 MB)
        mx.save_safetensors(os.path.join(C.CKPT, tag + "_qat_q4.safetensors"), dict(tree_flatten(Dz.parameters())))
    print(f"[qat] read-outs in {time.time() - tr0:.0f}s", flush=True)

    def summ(e, st, lk):
        if st in fam:
            return C.summarize(e[fam[st][0]][lk] + e[fam[st][1]][lk])
        return C.summarize(e[st][lk])

    rng = np.random.default_rng(1114)
    res = {}
    for key, e in ev.items():
        for st in list(sets) + list(fam):
            _, pc = summ(ev["ctl|fp32"], st, "full")
            for lk in LISTS:
                s_, per = summ(e, st, lk)
                _, p8 = summ(ev["ctl|q8"], st, lk)               # the 8-bit body with the SAME list, matched steps
                row = {"0-0": round(100 * s_["0-0"], 2), "23-30": round(100 * s_["23-30"], 2),
                       "kept32": round(s_["kept_per_32"], 3)}
                for b in ("kept", "0-0", "23-30"):
                    sc = 1 if b == "kept" else 100
                    d = paired(pc, per, b, rng)
                    row["d" + b] = round(sc * d["diff"], 3); row["ci" + b] = [round(sc * x, 3) for x in d["ci95"]]
                    d8 = paired(p8, per, b, rng)
                    row["q8d" + b] = round(sc * d8["diff"], 3); row["q8ci" + b] = [round(sc * x, 3) for x in d8["ci95"]]
                    if b == "kept":
                        A = max(1e-9, d8["A"])
                        row["kept_vs_q8_pct"] = round(100 * d8["diff"] / A, 2)
                        row["kept_vs_q8_ci_pct"] = [round(100 * x / A, 2) for x in d8["ci95"]]
                res[f"{key}|{st}|{lk}"] = row
                print(f"[qat] {key:10s} {st:8s} {lk:12s} kept {row['kept32']:.3f} (vs 8-bit same list "
                      f"{row['kept_vs_q8_pct']:+.2f} %, CI {row['kept_vs_q8_ci_pct'][0]:+.2f}..{row['kept_vs_q8_ci_pct'][1]:+.2f}; "
                      f"vs fp32-full {row['dkept']:+.3f}) 0-0 {row['0-0']:.2f} 23-30 {row['23-30']:.2f}", flush=True)
    q, p = f"qat|{g}", f"ctl|{g}"
    line = all(res[f"{q}|{st}|{lk}"]["kept_vs_q8_pct"] >= -1.0 for st in ("mbpp", "he") for lk in LISTS)
    rec = {}
    for st in ("mbpp", "he"):
        pg = res[f"{p}|{st}|full"]["kept_vs_q8_pct"]
        rec[st] = round(1 - res[f"{q}|{st}|full"]["kept_vs_q8_pct"] / pg, 3) if pg < 0 else 1.0
    conf = all(res[f"{q}|{st}|{lk}"]["kept_vs_q8_pct"] >= -2.0 for st in ("mbpp2", "he2") for lk in LISTS) if a.full else True
    verdict = ("PASS" if conf else "PASS, not confirmed") if line else ("KILL" if all(v < 0.5 for v in rec.values()) else "BETWEEN")
    body = {"fp32": round(nlin * 4 / 1e6, 1), "q8g64": round(nlin * (1 + 4 / 64) / 1e6, 1),
            "q4g64": round(nlin * (0.5 + 4 / 64) / 1e6, 1), f"q4g{a.gs}": round(nlin * (0.5 + 4 / a.gs) / 1e6, 1)}
    out = {"args": vars(a), "status": "done", "verdict": verdict, "recovery_of_ptq_loss": rec, "mid": midres, "res": res,
           "steps": steps, "loss": tl, "body_mb": body, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
    R.finish(sp)
    print(f"[qat] VERDICT {verdict}; recovery of the post-training {g} loss {json.dumps(rec)}; body MB {json.dumps(body)}",
          flush=True)
    print(f"[qat] DONE in {out['wall_s']:.0f}s -> results/{tag}.json", flush=True)


if __name__ == "__main__":
    main()
