"""WEEK98 CARRY400 confirmation (orchestrator 02:2x, for MTP Head while sub2 is blocked): MTP Head's lever 'allrows'
(~/TrainLab/sb98/levers/allrows.py, unmodified, loaded by absolute path as MTP Head ran it) at s30's full recipe.
Two arms in ONE process on the same slices, order and schedule, so the control is exact:
  n1  the plain head = s30's recipe (sb/train.py run s30: native head, fresh bias-corrected Adam, base files = 1,889
      slices of 96 rows in seed-1114 order, rollout depth 30, lr 1e-5 with a 20-step warm-up, cosine to 1e-6 at the
      last step, grad clip 1.0). train_multi's Arm reproduces train.py's step exactly; n1 should land on s30's MBPP-50
      23-30 55.47 within ~1 pt (control check).
  ar  the same + allrows (no accept-until-fail truncation in the training KL).
Read-outs in-process (no checkpoints written): at step 945 (MBPP-50, the stop line) and at the end (MBPP-50 + HE-30):
bands, kept/32 and the strict conditional per depth; paired ar vs n1. A SIGTERM (the guard's yield) saves both arms +
Adam once (sb/resume.py) and the rerun resumes.
LINES (pre-registered in results/LEVERS_A.md before the run): MTP Head's: CONFIRMED if MBPP-50 23-30 >= +2.0 pt
(p < 0.05) at 1,889 steps. Stop line at 945: stop unless ar >= +2.0 at 23-30. Run-time value (information): kept/32 and
the conditional at depths 1-8, where accept-until-fail decoding earns its tokens.
usage: carryconf.py [--steps 1889] [--stop-at 945] [--smoke]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb import resume as R
from sb.compare import paired
from sb.evalx import run_eval
from sb.train import split
from sb.train_multi import Arm

ALLROWS = os.path.expanduser("~/TrainLab/sb98/levers/allrows")


def readout(arms, sets):
    out = {}
    for k, m in arms.items():
        for st, fs in sets.items():
            out[f"{k}|{st}"] = run_eval(m.D, fs, "chain")
            mx.clear_cache()
    return out


def compare(ev, sets, rng):
    res = {}
    for st in sets:
        (ra, pa), (rb, pb) = ev[f"n1|{st}"], ev[f"ar|{st}"]
        row = {"n1": {b: round(100 * ra[b], 2) for b in ("0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-30")},
               "ar": {b: round(100 * rb[b], 2) for b in ("0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-30")},
               "kept": [round(ra["kept_per_32"], 3), round(rb["kept_per_32"], 3)],
               "cond_n1": [None if c is None else round(c, 3) for c in ra["cond"][:31]],
               "cond_ar": [None if c is None else round(c, 3) for c in rb["cond"][:31]]}
        for b in ("23-30", "0-0", "kept"):
            d = paired(pa, pb, b, rng)
            sc = 1 if b == "kept" else 100
            row[f"d{b}"] = {"diff": round(sc * d["diff"], 3), "ci95": [round(sc * x, 3) for x in d["ci95"]],
                            "wins": d["wins"], "losses": d["losses"], "p_sign": round(d["p_sign"], 4)}
        res[st] = row
    return res


def line(tag, cmp):
    return " | ".join(f"{st}: 23-30 n1 {r['n1']['23-30']:.2f} ar {r['ar']['23-30']:.2f} ({r['d23-30']['diff']:+.2f}, p {r['d23-30']['p_sign']:.3f}); "
                      f"0-0 {r['d0-0']['diff']:+.2f}; kept {r['kept'][0]:.3f} -> {r['kept'][1]:.3f} ({r['dkept']['diff']:+.3f}, "
                      f"CI {r['dkept']['ci95'][0]:+.3f}..{r['dkept']['ci95'][1]:+.3f})" for st, r in cmp.items()) + f" [{tag}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1889); ap.add_argument("--stop-at", type=int, default=945)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--mem", type=float, default=6.0)
    a = ap.parse_args()
    nm, nh = (2, 2) if a.smoke else (50, 30)
    if a.smoke:
        a.steps, a.stop_at = 4, 2
    tag = "carryconf" + ("_smoke" if a.smoke else "")
    C.limit_memory(a.mem)
    t0 = time.time()
    C.load_target(); C.drop_trunk()
    tr, _ = split("base")
    sl = C.pass_slices(tr, 96, 1114)                      # train.py run s30: epochs 1, seed 1114, rows 96
    steps = min(a.steps, len(sl))
    ta = argparse.Namespace(start="native", resume_opt="", const=False, lr=1e-5, warmup=20, steps=steps, depth=30,
                            wdecay=0.97)
    arms = {"n1": Arm("n1", "-", ta), "ar": Arm("ar", ALLROWS, ta)}
    sets_m = {"mbpp": C.files("mbpp")[:nm]}
    sets_all = {"mbpp": C.files("mbpp")[:nm], "he": C.files("he")[:nh]}
    sp = os.path.join(C.CKPT, tag + "_state")
    out = {"args": vars(a), "steps": steps, "reads": {}}
    i0 = 0
    if os.path.exists(sp + ".safetensors"):
        meta = R.load(sp, {k: v.D for k, v in arms.items()}, {k: v.opt for k, v in arms.items()})
        for k, v in arms.items():
            v.log = meta["logs"][k]
        out, i0 = meta["out"], meta["i"]
        print(f"[cc] resumed at step {i0}", flush=True)
    R.arm_sigterm()
    rng = np.random.default_rng(1114)
    print(f"[cc] arms n1 + ar (allrows, {ALLROWS}.py): {steps} steps x depth 30 on base from native (fresh Adam), "
          f"from step {i0}; ready in {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    for i in range(i0, steps):
        C.wait_pause()
        f, b, r = sl[i]
        x = C.window_at(f, b, r, 30)
        for m in arms.values():
            m.step(x, ta)
        if (i + 1) % 50 == 0 or i == i0:
            print(f"[cc] step {i + 1}/{steps} " + " ".join(f"{k} {np.mean(m.log['loss'][-50:]):.4f}" for k, m in arms.items())
                  + f"; {(time.time() - t1) / (i + 1 - i0):.2f}s/step, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        if i + 1 == a.stop_at and "stop" not in out["reads"]:
            cmp = compare(readout(arms, sets_m), sets_m, rng)
            out["reads"]["stop"] = {"step": i + 1, **cmp}
            print("[cc] stop-line read " + line(f"step {i + 1}", cmp), flush=True)
            if cmp["mbpp"]["d23-30"]["diff"] < 2.0 and not a.smoke:
                out["verdict"] = "NOT CONFIRMED (stopped at the 945 line: 23-30 < +2.0)"
                R.finish(sp)
                json.dump(out, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
                print(f"[cc] DONE verdict {out['verdict']}", flush=True)
                return
        if R.STOP["f"]:
            wrote = R.save(sp, {k: v.D for k, v in arms.items()}, {k: v.opt for k, v in arms.items()},
                           {"i": i + 1, "out": out, "logs": {k: v.log for k, v in arms.items()}})
            print(f"[cc] SIGTERM: state saved at step {i + 1} ({wrote / 1e9:.2f} GB), exiting to resume later", flush=True)
            sys.exit(75)
    R.finish(sp)
    print(f"[cc] trained {steps} steps in {time.time() - t1:.0f}s; final read-out", flush=True)
    cmp = compare(readout(arms, sets_all), sets_all, rng)
    out["reads"]["final"] = {"step": steps, **cmp}
    d = cmp["mbpp"]["d23-30"]
    out["verdict"] = ("CONFIRMED" if d["diff"] >= 2.0 and d["p_sign"] < 0.05 else
                      "NOT CONFIRMED" if d["diff"] < 2.0 else "BETWEEN")
    out["control_check_vs_s30_55.47"] = cmp["mbpp"]["n1"]["23-30"]
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(C.RES, tag + ".json"), "w"), indent=1)
    print("[cc] final " + line(f"step {steps}", cmp), flush=True)
    print(f"[cc] DONE verdict {out['verdict']} (n1 23-30 {cmp['mbpp']['n1']['23-30']:.2f} vs s30 55.47)", flush=True)


if __name__ == "__main__":
    main()
