#!/usr/bin/env python3
"""TAPSQ (Sat 14 Nov 2026; Kamen's 15-min rule): does the MTP head get deeper when it also reads the multi-layer taps?
A quick test, not a run (docs/PREREG_DEEPDRAFT_TAPSQ_2026-11-14.md has the lines and the actions).

- h, already in every capture file: DFlash2's own fused context input, hidden_norm(fc(taps)), taps = the mean of the 4 HC
  streams after layers 5/15/25/35/45, 2560 wide (bf16 bits). The head reads only wd (the top pre-mixer stream) today.
- Warm start t30b. One fixed order (seed 1114) of 192-anchor slices of the 795 n1112 files t30b never saw (pinned in
  data/deepdraft/tapsq/split.json); every arm reads the same slices in the same order, so each step's loss is on a slice
  new to that arm (prequential), paired by step across arms.
- Arms: C = t30b continues (head lr 1e-5 constant after a 10-step warm-up);
        T = C plus xb_tap, a zero-init Linear(2560 -> 2560) of h added to the fused input of every target row, the way the
            embedding term enters (each HC stream gets the same term); not on drafted rows (EAGLE-3's place). Own Adam,
            lr 1e-5 (the head's; amendment 1: 1e-4 was pre-registered, the smoke showed one step at 1e-4 lifting the
            loss 0.35 -> 3.78), each group's gradient clipped to norm 1 on its own;
        R = T fed h from a random other row of the same window (nonzero rows only): a new trainable path at 10x lr
            without the row's information (the orchestrator's control, run only if T passes).
- Logged per step: the loss (accept-until-fail KL over depths 1-30, as b5 / t30b train), the depth-1 KL and the mean KL of
  depths 24-30 (the eval's bands 0 and 23-29).
- eval: MBPP 100 (d1102mbpp, which has h) depth bands per run, as dd_ttt.py's eval, plus per-window counts for a paired
  comparison. (The 24 HumanEval+ windows, d16w, have no h.)
usage: dd_taps.py smoke | train --arm C|T|R [--steps 80] | eval --runs t30b,tq_C,tq_T
"""
import argparse, glob, json, os, signal, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dd_ttt as DD                                                  # noqa: E402  (T = verbatim d22_ttt, M = mtp125)
import numpy as np                                                   # noqa: E402
import mlx.core as mx                                                # noqa: E402
import mlx.nn as nn                                                  # noqa: E402
import mlx.optimizers as optim                                       # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten                   # noqa: E402

T, M = DD.T, DD.M
CAP, CKD, BANDS = DD.CAP, DD.CKD, DD.BANDS
OUT = os.path.expanduser("~/BigMoE/data/deepdraft/tapsq")
SPLIT = os.path.join(OUT, "split.json")
PINNED = os.path.expanduser("~/BigMoE/data/deepdraft/ttt30b/split.json")
START, DEPTH, ROWS, WDECAY, SEED, HW = "t30b", 30, 192, 0.97, 1114, 2560
TAPS = {"H": None, "armed": False}

_fuse = T.fuse


def fuse_taps(head, P, Win, emb):
    """T.fuse, plus xb_tap(h) on the chain's first call (the target rows W); later calls (drafted rows) are unchanged"""
    out = _fuse(head, P, Win, emb)
    if TAPS["armed"]:
        TAPS["armed"] = False
        if "xb_tap" in P and TAPS["H"] is not None:
            B, n, Wd = out.shape
            add = P.xb_tap(TAPS["H"][:, :n].astype(mx.float32))
            out = (out.reshape(B, n, head._hc, -1) + add[:, :, None, :]).reshape(B, n, Wd)
    return out


T.fuse = fuse_taps                                                   # T.chain looks fuse up at call time


def set_tap(P, on):
    if "xb_tap" in P:
        del P["xb_tap"]
    if on:
        d = P.fc_hidden.weight.shape[0]
        P.xb_tap = nn.Linear(HW, d, bias=False)
        P.xb_tap.weight = mx.zeros((d, HW), mx.float32)


def split_files():
    """the n1112 files t30b never saw (neither its train nor its held-out list), pinned at the first call"""
    if not os.path.exists(SPLIT):
        s = json.load(open(PINNED))
        seen = set(s["train"]) | set(s["held_out"])
        new = sorted(os.path.basename(f) for f in glob.glob(os.path.join(CAP, "*.npz")) if os.path.basename(f) not in seen)
        os.makedirs(OUT, exist_ok=True)
        json.dump({"night": "n1112", "rule": "the n1112 files on disk at the pin that t30b's split (ttt30b/split.json) does not "
                   "list", "pinned": time.strftime("%Y-%m-%d %H:%M"), "train": new}, open(SPLIT, "w"), indent=0)
    return [os.path.join(CAP, n) for n in json.load(open(SPLIT))["train"]]


def h_of(z, end):
    """h's bf16 bits -> an exact float32 array -> bf16 [1, end - 1, 2560]"""
    hb = np.asarray(z["h"][:end - 1]).astype(np.uint32) << 16
    return mx.array(hb.view(np.float32))[None].astype(mx.bfloat16)


def shuffle_rows(H, rng):
    """R: each nonzero row of H gets another nonzero row's h (one permutation of the nonzero rows)"""
    a = np.array(H[0].astype(mx.float32))
    nz = np.flatnonzero(np.abs(a).sum(-1) > 0)
    b = a.copy()
    b[nz] = a[rng.permutation(nz)]
    return mx.array(b)[None].astype(mx.bfloat16)


def win_at(f, b, r):
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32)
    end = min(len(ids), b + r + DEPTH + 2)
    return {"W": mx.array(z["wd"][:end - 1]).astype(mx.bfloat16)[None], "ids": ids[:end],
            "tk": np.asarray(z["tk"][:end - 1], np.int32), "lp": np.asarray(z["lp"][:end - 1], np.float32),
            "asst": np.asarray(z["asst"], np.float32)[:end], "a0": b, "H": h_of(z, end)}


def loss_parts(lm, head, P, x):
    """b5's training loss (T.ttt_loss with accept-until-fail), the taps armed when P has xb_tap -> (loss, per-depth KL)"""
    TAPS["H"], TAPS["armed"] = x.get("H"), "xb_tap" in P
    loss, per = T.ttt_loss(lm, head, P, x, DEPTH, WDECAY, True)
    TAPS["armed"] = False
    return loss, mx.stack(per)


def load_start(P, tap):
    set_tap(P, False)
    P.load_weights(os.path.join(CKD, f"dd_{START}.safetensors"))
    set_tap(P, tap)


def groups(g):
    return {k: v for k, v in g.items() if not k.startswith("xb_")}, {k: v for k, v in g.items() if k.startswith("xb_")}


def cmd_smoke(a):
    t0 = time.time()
    rt, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    tr = split_files()
    sl = DD.pass_slices(tr, ROWS, SEED)
    x = win_at(*sl[0])
    h = np.array(x["H"][0].astype(mx.float32))
    nz = int((np.abs(h).sum(-1) > 0).sum())
    print(f"[tq] {len(tr)} files t30b never saw, {len(sl)} slices; slice 0: {os.path.basename(sl[0][0])} rows {sl[0][1]}+{sl[0][2]}; "
          f"h rows {h.shape[0]} ({nz} nonzero), finite {bool(np.isfinite(h).all())}, rms {float(np.sqrt((h[h != 0] ** 2).mean())):.3f}", flush=True)
    load_start(P, False)
    l0 = float(T.ttt_loss(lm, head, P, x, DEPTH, WDECAY, True)[0])
    set_tap(P, True)
    l1, per = loss_parts(lm, head, P, x)
    print(f"[tq] smoke: t30b's loss {l0:.7f}; with the zero-init tap {float(l1):.7f}: {'EQUAL' if float(l1) == l0 else 'DIFFERENT'}"
          f"; per-depth KL {per.shape[0]} depths (1: {float(per[0]):.4f}, 24-30: {float(per[23:30].mean()):.4f})", flush=True)
    P.freeze(); P.unfreeze()
    lossf = nn.value_and_grad(P, lambda: loss_parts(lm, head, P, x))
    (lv, _), g = lossf()
    gb, gx = groups(g)
    gn = float(mx.sqrt(sum((v ** 2).sum() for _, v in tree_flatten(gx))))
    print(f"[tq] smoke: loss {float(lv):.7f}; the tap's gradient norm {gn:.3e} ({'> 0' if gn > 0 else 'ZERO'})", flush=True)
    tlr = a.lr * a.tap_lr_mult
    optx = optim.Adam(learning_rate=tlr)
    gx, _ = optim.clip_grad_norm(gx, 1.0)
    optx.update(P, gx); mx.eval(P.parameters(), optx.state)
    l2 = float(loss_parts(lm, head, P, x)[0])
    wn = float(mx.abs(P.xb_tap.weight).max())
    print(f"[tq] smoke: after one tap-only step at {tlr:g} (full lr, no warm-up): loss {l2:.7f} ({'CHANGED' if l2 != l0 else 'UNCHANGED'}), tap |w| max {wn:.2e}", flush=True)
    xr = dict(x); xr["H"] = shuffle_rows(x["H"], np.random.default_rng(SEED))
    lr_ = float(loss_parts(lm, head, P, xr)[0])
    print(f"[tq] smoke: R's shuffled h -> loss {lr_:.7f} ({'DIFFERENT from T' if lr_ != l2 else 'SAME as T'})", flush=True)
    print(f"[tq] smoke done in {time.time() - t0:.0f}s; peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def cmd_train(a):
    t0 = time.time()
    run = f"{a.tag}_{a.arm}"
    rt, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    load_start(P, a.arm in ("T", "R"))
    P.freeze(); P.unfreeze()
    tr = split_files()
    sl = DD.pass_slices(tr, ROWS, SEED)[:a.steps]
    warm = lambda s: mx.minimum(1.0, (s + 1) / 10.0)
    if a.warm_opt:   # amendment 2: the head continues on t30b's own Adam moments and step, at t30b's final lr (no restart)
        opt = optim.Adam(learning_rate=a.lr)
        opt.init(groups(P.trainable_parameters())[0])
        opt.state = tree_unflatten(list(mx.load(os.path.join(CKD, f"dd_{START}_opt.safetensors")).items()))
        opt.learning_rate = a.lr
    else:            # the void run (tq_): fresh Adam (no bias correction) at a.lr after a 10-step warm-up
        opt = optim.Adam(learning_rate=lambda s: a.lr * warm(s))
    tap_lr = a.tap_lr if a.tap_lr is not None else a.lr * a.tap_lr_mult
    optx = optim.Adam(learning_rate=lambda s: tap_lr * warm(s), bias_correction=a.warm_opt) if "xb_tap" in P else None
    rng = np.random.default_rng(SEED)
    cur = {}
    lossf = nn.value_and_grad(P, lambda: loss_parts(lm, head, P, cur["x"]))
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    st = {"run": run, "arm": a.arm, "start": START, "steps": a.steps, "seed": SEED, "lr": a.lr, "warm_opt": a.warm_opt,
          "tap_lr": tap_lr if optx is not None else None, "loss": [], "d1": [], "d24_30": [], "sec": []}
    ntr = sum(v.size for _, v in tree_flatten(P.trainable_parameters()))
    print(f"[tq] train {run}: {len(tr)} unseen files, {len(sl)} slices (seed {SEED}), depth {DEPTH}, rows {ROWS}, lr {a.lr}"
          + (f" ({'t30b' + chr(39) + 's Adam state' if a.warm_opt else 'fresh Adam'})")
          + (f", tap lr {tap_lr} (fresh Adam, bias correction {a.warm_opt})" if optx is not None else "") + f"; {ntr / 1e6:.1f}M trainable; loaded in {time.time() - t0:.0f}s", flush=True)
    for i, (f_, b_, r_) in enumerate(sl):
        while os.path.exists(T.PAUSE) and not stop["now"]:
            time.sleep(1)
        if stop["now"]:
            print(f"[tq] stopped at step {i}: not saved (a quick arm reruns from the start)", flush=True); sys.exit(3)
        x = win_at(f_, b_, r_)
        if a.arm == "R":
            x["H"] = shuffle_rows(x["H"], rng)
        cur["x"] = x
        t1 = time.time()
        (loss, per), g = lossf()
        gb, gx = groups(g)
        gb, _ = optim.clip_grad_norm(gb, 1.0)
        opt.update(P, gb)
        if optx is not None:
            gx, _ = optim.clip_grad_norm(gx, 1.0)
            optx.update(P, gx)
            mx.eval(P.parameters(), opt.state, optx.state, loss, per)
        else:
            mx.eval(P.parameters(), opt.state, loss, per)
        pv = np.array(per)
        st["loss"].append(float(loss)); st["d1"].append(float(pv[0]))
        st["d24_30"].append(float(pv[23:30].mean()) if len(pv) >= 30 else None); st["sec"].append(time.time() - t1)
        pk = mx.get_peak_memory() / 1e9
        if (i + 1) % 10 == 0 or i < 3:
            print(f"[tq] {run} step {i + 1}/{len(sl)} loss {float(loss):.4f} (mean of last 10 {np.mean(st['loss'][-10:]):.4f}); "
                  f"{time.time() - t1:.1f}s/step, peak {pk:.1f} GB", flush=True)
        if a.peak_max and pk > a.peak_max:
            print(f"[tq] MLX peak {pk:.1f} GB > {a.peak_max}: stopped", flush=True); sys.exit(4)
    mx.save_safetensors(os.path.join(CKD, f"dd_{run}.safetensors"), dict(tree_flatten(P.parameters())))
    st["peak_gb"], st["wall_s"] = mx.get_peak_memory() / 1e9, time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(st, open(os.path.join(OUT, f"{run}.json"), "w"))
    print(f"[tq] DONE {run}: {len(sl)} steps in {time.time() - t0:.0f}s; peak {st['peak_gb']:.1f} GB", flush=True)


def cmd_eval(a):
    rt_, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    items = sorted(glob.glob(os.path.join(T.TD, "dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]
    R = {}
    for run in a.runs.split(","):
        wts = mx.load(os.path.join(CKD, f"dd_{run}.safetensors"))
        set_tap(P, "xb_tap.weight" in wts)
        P.load_weights(list(wts.items())); mx.eval(P.parameters())
        shuf = run.endswith("_R")
        rng = np.random.default_rng(SEED + 1)
        band = {b: [0, 0] for b in BANDS}; per_win = []; kept = []
        for f in items:
            while os.path.exists(T.PAUSE):
                time.sleep(1)
            z = np.load(f)
            x = {"W": mx.array(z["wd"]).astype(mx.bfloat16)[None], "ids": np.asarray(z["ids"], np.int32)}
            H = h_of(z, len(x["ids"]))
            TAPS["H"] = shuffle_rows(H, rng) if shuf else H
            gen = int(z["gen"]); a0 = max(0, gen - 2)
            TAPS["armed"] = "xb_tap" in P
            hits = T.chain_hits(lm, head, P, x, 31, a0)
            TAPS["armed"] = False
            n = len(x["ids"]) - 1 - a0
            rt = np.full((n, 31), -1, np.int8)
            for j, (right, _am) in enumerate(hits):
                rt[:len(right), j] = np.asarray(right).astype(np.int8)
            w = {}
            for lo, hi in BANDS:
                c = rt[:, lo:hi + 1]; r_, v_ = int((c == 1).sum()), int((c >= 0).sum())
                band[(lo, hi)][0] += r_; band[(lo, hi)][1] += v_; w[f"{lo}-{hi}"] = [r_, v_]
            per_win.append(w)
            kept += DD.kept_walk(rt, max(0, gen - 1 - a0))
            mx.clear_cache()
        R[run] = {f"{lo}-{hi}": band[(lo, hi)][0] / max(band[(lo, hi)][1], 1) for lo, hi in BANDS}
        R[run]["kept_per_32"] = float(np.mean(kept)) if kept else None
        R[run]["windows"] = len(items); R[run]["per_window"] = per_win
        print(f"[tq] {run} mbpp: " + " ".join(f"{b}:{100 * v:.1f}" for b, v in R[run].items() if "-" in b)
              + f"; kept/32 {R[run]['kept_per_32']:.2f}", flush=True)
    R["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, a.out_name), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "train", "eval"])
    ap.add_argument("--arm", choices=["C", "T", "R"]); ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--tap-lr-mult", dest="tap_lr_mult", type=float, default=1.0)
    ap.add_argument("--tap-lr", dest="tap_lr", type=float, default=1e-5); ap.add_argument("--tag", default="tq2")
    ap.add_argument("--fresh-opt", dest="warm_opt", action="store_false", help="the void run's fresh Adam (tq_)")
    ap.add_argument("--frac", type=float, default=0.03); ap.add_argument("--peak-max", dest="peak_max", type=float, default=22.0)
    ap.add_argument("--runs", default="t30b,tq2_C,tq2_T"); ap.add_argument("--out-name", dest="out_name", default="eval2.json")
    a = ap.parse_args()
    {"smoke": cmd_smoke, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
