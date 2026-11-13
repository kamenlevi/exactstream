#!/usr/bin/env python3
"""DEEPDRAFT step 3, first small test (docs/PREREG_DEEPDRAFT_TTT30_2026-11-12.md): the MTP head b5 trained further with
D-22's training-time-test chain to depth 30, on DEEPDRAFT capture night 1 (Magicoder Python), starting from b5. Only the
head trains; the 125B is untouched.
scripts/dd_d22_ttt.py is a verbatim copy of TrainLab sticky/d22_ttt.py (read-only source); its chain and ttt_loss (KL to the
125B's top-32 per depth, Spec-AUF accept-until-fail as b5) are used unchanged. Changes against its cmd_train:
- data: the capture files, which hold wd / ids / tk / lp / asst in one npz; the last 5 % of files (name order) is never
  trained on;
- anchors: a random slice of up to --rows anchors in the answer per step (memory);
- depth --depth (30), --wdecay (0.97), cosine lr from --lr (1e-5), --steps (600);
- start: b5's weights;
- checkpoints in ~/datasets/deepdraft/ckpt/ (outside every repo); a --peak-max GB guard stops and saves.
eval: the plain chain as TrainLab d23h_mtp31.py runs it (anchors gen - 2, depth 31) on the 24 held-out HumanEval+ windows
(data/d16w) and the 100 MBPP answers (dlog/d1102mbpp), per run. Agreement by depth band counts valid cells only. Kept per
32-row check: from each answer's first anchor, kept = the leading right depths + 1, then advance by kept (t2_laglaw's real
chain). -> data/deepdraft/ttt30/eval.json
usage: [BigMoE venv] dd_ttt.py train --run t30a [--steps 600] | dd_ttt.py eval --runs b5,t30a
TTT30B (docs/PREREG_DEEPDRAFT_TTT30B_2026-11-12.md): train --run t30b --manifest data/deepdraft/ttt30b/split.json
  --pass-seed 1113 --save-at 600,1500,3000 (one pass over the pinned training files; Adam state saved with the weights,
  so a SIGTERM stop resumes where it was); eval --runs b5,t30b_s600,t30b_s1500,t30b_s3000,t30b --out data/deepdraft/ttt30b
TTT30C (docs/PREREG_DEEPDRAFT_TTT30C_2026-11-13.md): t30b's train arguments plus one change per arm:
  --run t30f --feat               the carried state: a feature loss at every depth (lambda fixed from b5's first 20 slices)
  --run t30x2 --xb 10240 / --run t30x4 --xb 30720   a zero-init dense SwiGLU branch beside the MoE (branch lr x --xb-lr-mult)
  smoke: [--feat | --xb W] --smoke (the PREREG's checks on one slice + 3 real steps on a throwaway run name; no copies)
  eval finds a run's branch in its checkpoint (xb_down.weight)."""
import argparse, glob, json, os, signal, sys, time

sys.path.insert(0, os.path.expanduser("~/TrainLab/sticky"))          # mtp125 and its imports, read-only
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dd_d22_ttt as T                                               # noqa: E402  (verbatim d22_ttt.py)
import numpy as np                                                   # noqa: E402
import mlx.core as mx                                                # noqa: E402
import mlx.nn as nn                                                  # noqa: E402
import mlx.optimizers as optim                                       # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten                   # noqa: E402

M = T.M
CAP = os.path.expanduser("~/FableGod/deepdraft/capture/n1112")
CKD = os.path.expanduser("~/datasets/deepdraft/ckpt")
OUT = os.path.expanduser("~/BigMoE/data/deepdraft/ttt30")
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
_first = T.first_asst
T.first_asst = lambda x: x["a0"] if "a0" in x else _first(x)         # a step's anchor-slice start
_moe_train = M.moe_train


def moe_train_xb(moe, P, x):
    """TTT30C: the MoE + shared expert unchanged, plus the dense SwiGLU branch when P has one (t30x2 / t30x4)"""
    y = _moe_train(moe, P, x)
    if "xb_down" in P:
        xf = x.astype(mx.float32)
        y = y + P.xb_down(nn.silu(P.xb_gate(xf)) * P.xb_up(xf))
    return y


M.moe_train = moe_train_xb                                           # d22's finish() calls M.moe_train at run time


def set_branch(P, width, seed=1113):
    """TTT30C: give P a dense SwiGLU branch `width` wide beside the MoE (0: none); the down projection starts at zero, so
    the head is unchanged until it trains"""
    for k in ("xb_gate", "xb_up", "xb_down"):
        if k in P:
            del P[k]
    if width:
        mx.random.seed(seed)
        d = P.fc_hidden.weight.shape[0]
        P.xb_gate, P.xb_up = nn.Linear(d, width, bias=False), nn.Linear(d, width, bias=False)
        P.xb_down = nn.Linear(width, d, bias=False)
        P.xb_down.weight = mx.zeros((d, width), mx.float32)


def split_tree(t):
    """(the head's own parameters, the branch's) of a top-level parameter / gradient dict"""
    return ({k: v for k, v in t.items() if not k.startswith("xb_")}, {k: v for k, v in t.items() if k.startswith("xb_")})


def ttt_loss_x(lm, head, P, x, D, wdecay, lam=0.0, parts=False):
    """T.ttt_loss(..., auf=True) (b5's loss, the same ops) plus TTT30C's feature term when lam > 0 or parts: at every
    depth the relative squared error between the head's next input as read, pre_fc_norm_hidden(h_j), and the exact
    path's, pre_fc_norm_hidden(W[s + j]), on the KL's rows (assistant targets x accept-until-fail) and weights.
    -> (loss, kl term, feature term)"""
    W, ids = x["W"], x["ids"]
    S, a0 = len(ids), T.first_asst(x)
    tk, lp, asst = mx.array(x["tk"]), mx.array(x["lp"]), mx.array(x["asst"])
    acc = {"tot": 0.0, "f": 0.0, "w": 0.0}
    alive, ida = [None], mx.array(ids)
    nrm, feat_on = head.pre_fc_norm_hidden, lam > 0 or parts

    def on_depth(j, h):
        n = min(h.shape[1], S - 1 - j - a0, tk.shape[0] - j - a0)
        if n <= 0:
            return
        m = asst[a0 + j + 1:a0 + j + 1 + n]
        w = wdecay ** (j - 1)
        if alive[0] is not None:
            m = m * alive[0][:n]
        l, hit = T.depth_kl_auf(lm, head)(h[:, :n], tk[a0 + j:a0 + j + n], lp[a0 + j:a0 + j + n], m,
                                          ida[a0 + j + 1:a0 + j + 1 + n])
        if j == 1:
            acc["m1"] = mx.maximum(m.sum(), 1.0)
        w = w * mx.stop_gradient(m.sum() / acc["m1"])
        alive[0] = hit if alive[0] is None else alive[0][:n] * hit
        acc["tot"] = acc["tot"] + w * l; acc["w"] = acc["w"] + w
        if feat_on:                                  # h_j row i is depth j + 1's stand-in for W[a0 + i + j]
            pred = nrm(h[:, :n]).astype(mx.float32)
            tgt = nrm(W[:, a0 + j:a0 + j + n]).astype(mx.float32)
            rel = ((pred - tgt) ** 2).sum(-1) / mx.maximum((tgt ** 2).sum(-1), 1e-6)
            acc["f"] = acc["f"] + w * (rel[0] * m).sum() / mx.maximum(m.sum(), 1.0)
    T.chain(lm, head, P, W, mx.array(ids)[None], D, on_depth, a0)
    kl = acc["tot"] / mx.maximum(acc["w"], 1e-6)
    fe = acc["f"] / mx.maximum(acc["w"], 1e-6)
    return kl + lam * fe, kl, fe


def cap_files(manifest=None):
    """--manifest (TTT30B): the split pinned in a JSON file {train: [names], held_out: [names]}, so files the capture adds
    later change nothing; else the rule: name order, the last 5 % held out"""
    if manifest:
        m = json.load(open(manifest))
        return [os.path.join(CAP, n) for n in m["train"]], [os.path.join(CAP, n) for n in m["held_out"]]
    fs = sorted(glob.glob(os.path.join(CAP, "*.npz")))
    nv = max(1, len(fs) // 20)
    return fs[:-nv], fs[-nv:]


def pass_slices(tr, R, seed):
    """--pass-seed (TTT30B): one pass = every training file's anchors (2 before the first answer token .. len - 3) cut into
    ceil(n / R) near-equal consecutive slices, all slices shuffled once with the seed; step i takes slice i"""
    out = []
    for f in tr:
        z = np.load(f)
        n_ids = len(z["ids"]); a = np.flatnonzero(np.asarray(z["asst"]) > 0)
        first = max(0, int(a[0]) - 2) if len(a) else 0
        n = (n_ids - 3) - first + 1
        k = -(-n // R)
        bs = [first + (i * n) // k for i in range(k + 1)]
        out += [(f, bs[i], bs[i + 1] - bs[i]) for i in range(k)]
    order = np.random.default_rng(seed).permutation(len(out))
    return [out[i] for i in order]


def cap_window_at(f, b, r, D):
    """cap_window with the slice fixed: anchors from b, r rows, the crop as cap_window's (b + r + D + 2)"""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32)
    end = min(len(ids), b + r + D + 2)
    return {"W": mx.array(z["wd"][:end - 1]).astype(mx.bfloat16)[None], "ids": ids[:end],
            "tk": np.asarray(z["tk"][:end - 1], np.int32), "lp": np.asarray(z["lp"][:end - 1], np.float32),
            "asst": np.asarray(z["asst"], np.float32)[:end], "a0": b}


def cap_window(f, rng, R, D):
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32)
    asst = np.asarray(z["asst"], np.float32)
    a = np.flatnonzero(asst > 0)
    first = max(0, int(a[0]) - 2) if len(a) else 0
    last = len(ids) - 3
    b = first if last - first <= R else int(rng.integers(first, last - R + 1))
    end = min(len(ids), b + R + D + 2)
    return {"W": mx.array(z["wd"][:end - 1]).astype(mx.bfloat16)[None], "ids": ids[:end],
            "tk": np.asarray(z["tk"][:end - 1], np.int32), "lp": np.asarray(z["lp"][:end - 1], np.float32),
            "asst": asst[:end], "a0": b}


def cmd_train(a):
    t0 = time.time()
    os.makedirs(CKD, exist_ok=True)
    if a.smoke:                                          # TTT30C smoke: a throwaway run name, no copies, 3 steps
        a.run, a.save_at = "smk_" + a.run, ""
        for f in glob.glob(os.path.join(CKD, f"dd_{a.run}.*")) + glob.glob(os.path.join(CKD, f"dd_{a.run}_opt.*")):
            os.remove(f)
    rt, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    ckw, ckj = os.path.join(CKD, f"dd_{a.run}.safetensors"), os.path.join(CKD, f"dd_{a.run}.json")
    cko = os.path.join(CKD, f"dd_{a.run}_opt.safetensors")
    tr, va = cap_files(a.manifest)
    sl = pass_slices(tr, a.rows, a.pass_seed) if a.pass_seed is not None else None
    state = {"step": 0, "loss": [], "args": vars(a), "status": "running"}
    if os.path.exists(ckj):
        state = json.load(open(ckj))
        set_branch(P, state["args"].get("xb") or 0)     # TTT30C: the branch exists before its weights load
        P.load_weights(ckw)
        print(f"[dd] resume {a.run} at step {state['step']}", flush=True)
    else:
        P.load_weights(os.path.join(T.CK, "d22_b5.safetensors"))
        print("[dd] start from b5 (TrainLab ckpt d22_b5, read-only)", flush=True)
        if a.smoke:                                      # the PREREG's checks, on the first slice of the pass
            x0 = cap_window_at(*sl[0], a.depth)
            l_ref = float(T.ttt_loss(lm, head, P, x0, a.depth, a.wdecay, True)[0])
            l_x, kl_x, fe_x = ttt_loss_x(lm, head, P, x0, a.depth, a.wdecay, 0.0, parts=True)
            ok = abs(l_ref - float(l_x)) <= 1e-6 * max(1.0, abs(l_ref))
            print(f"[dd] smoke: b5's loss {l_ref:.7f}; ttt_loss_x at lambda 0 {float(l_x):.7f} (kl {float(kl_x):.7f}, feature "
                  f"term {float(fe_x):.5f}): {'EQUAL' if ok else 'DIFFERENT'}", flush=True)
            if a.xb:
                set_branch(P, a.xb)
                l_b = float(T.ttt_loss(lm, head, P, x0, a.depth, a.wdecay, True)[0])
                print(f"[dd] smoke: with the zero-init {a.xb}-wide branch {l_b:.7f} vs b5 {l_ref:.7f}: "
                      f"{'EQUAL' if l_b == l_ref else 'DIFFERENT'}", flush=True)
        set_branch(P, a.xb)
    P.freeze(); P.unfreeze()
    steps = 3 if a.smoke else (a.steps or (len(sl) if sl else 600))
    save_at = {int(s) for s in a.save_at.split(",") if s} if a.save_at else set()
    sched = lambda s: a.lr * mx.minimum(1.0, (s + 1) / 20.0) * (0.1 + 0.45 * (1 + mx.cos(np.pi * mx.minimum(1.0, s / max(1, steps - 1)))))
    opt = optim.Adam(learning_rate=sched)
    optx = optim.Adam(learning_rate=lambda s: a.xb_lr_mult * sched(s)) if "xb_down" in P else None   # TTT30C: the branch's lr
    if state["step"]:
        opt.state["step"] = mx.array(state["step"], mx.uint64)
        if optx is not None:
            optx.state["step"] = mx.array(state["step"], mx.uint64)
        if os.path.exists(cko):                          # TTT30B: Adam's moments survive a stop (a yield to a quiet lock)
            st = tree_unflatten(list(mx.load(cko).items()))
            if optx is not None:                         # TTT30C: two groups, saved as b.* / x.*
                pb, px = split_tree(P.trainable_parameters())
                opt.init(pb); optx.init(px); opt.state, optx.state = st["b"], st["x"]
            else:
                opt.init(P.trainable_parameters()); opt.state = st
            print(f"[dd] optimizer state restored ({cko})", flush=True)
    cur, lam = {}, {"v": 0.0}
    if a.feat:                                           # TTT30C t30f: lambda fixed once, before the first update
        if "lam" in state:
            lam["v"] = state["lam"]
        else:
            ks, fs = [], []
            for i in range(20):
                _, kl_, fe_ = ttt_loss_x(lm, head, P, cap_window_at(*sl[i], a.depth), a.depth, a.wdecay, 0.0, parts=True)
                mx.eval(kl_, fe_); ks.append(float(kl_)); fs.append(float(fe_))
            lam["v"] = 0.5 * float(np.mean(ks)) / max(float(np.mean(fs)), 1e-9)
            state["lam"], state["lam_cal"] = lam["v"], {"kl": float(np.mean(ks)), "feat": float(np.mean(fs))}
            print(f"[dd] lambda {lam['v']:.5f} = 0.5 x kl {np.mean(ks):.5f} / feature term {np.mean(fs):.5f} (b5, slices 0-19)", flush=True)
        state.setdefault("feat", [])
        lossf = nn.value_and_grad(P, lambda: ttt_loss_x(lm, head, P, cur["x"], a.depth, a.wdecay, lam["v"]))
    else:
        lossf = nn.value_and_grad(P, lambda: T.ttt_loss(lm, head, P, cur["x"], a.depth, a.wdecay, True)[0])
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    rng = np.random.default_rng(1112 + state["step"])
    ntr = sum(v.size for _, v in tree_flatten(P.trainable_parameters()))
    print(f"[dd] train {a.run}: {len(tr)} files (val {len(va)} held out), {steps} steps, depth {a.depth}, rows {a.rows}, "
          f"lr {a.lr}, wdecay {a.wdecay}, AUF; {ntr / 1e6:.1f}M trainable; loaded in {time.time() - t0:.0f}s"
          + (f"; one pass = {len(sl)} slices ({sum(s[2] for s in sl)} anchors), seed {a.pass_seed}" if sl else "")
          + (f"; copies at {sorted(save_at)}" if save_at else "")
          + (f"; feature loss, lambda {lam['v']:.5f}" if a.feat else "")
          + (f"; branch {P.xb_down.weight.shape[1]} wide, lr x{a.xb_lr_mult}" if optx is not None else ""), flush=True)

    def save(status):
        mx.save_safetensors(ckw + ".tmp.safetensors", dict(tree_flatten(P.parameters())))
        os.replace(ckw + ".tmp.safetensors", ckw)
        if a.pass_seed is not None:
            ost = {"b": opt.state, "x": optx.state} if optx is not None else opt.state
            mx.save_safetensors(cko + ".tmp.safetensors", dict(tree_flatten(ost)))
            os.replace(cko + ".tmp.safetensors", cko)
        state["status"] = status
        json.dump(state, open(ckj + ".tmp", "w")); os.replace(ckj + ".tmp", ckj)
    last = time.time()
    while state["step"] < steps:
        while os.path.exists(T.PAUSE) and not stop["now"]:   # TTT30B: a stop (yield) also ends a PAUSE wait
            time.sleep(1)
        if stop["now"]:
            save("stopped"); print(f"[dd] stopped at step {state['step']}: saved", flush=True); sys.exit(3)
        if sl:
            f_, b_, r_ = sl[state["step"] % len(sl)]
            cur["x"] = cap_window_at(f_, b_, r_, a.depth)
        else:
            cur["x"] = cap_window(tr[int(rng.integers(len(tr)))], rng, a.rows, a.depth)
        t1 = time.time()
        loss, g = lossf()
        fe = None
        if a.feat:                                       # (total, kl, feature term)
            loss, _kl, fe = loss
        g, _ = optim.clip_grad_norm(g, 1.0)
        if optx is not None:                             # TTT30C: the branch on its own Adam (10x lr)
            gb, gx = split_tree(g)
            opt.update(P, gb); optx.update(P, gx)
            mx.eval(P.parameters(), opt.state, optx.state, loss)
        else:
            opt.update(P, g)
            mx.eval(P.parameters(), opt.state, loss)
        state["step"] += 1; state["loss"].append(float(loss))
        if fe is not None:
            state["feat"].append(float(fe))
        pk = mx.get_peak_memory() / 1e9
        if state["step"] % 20 == 0 or state["step"] <= 3:
            print(f"[dd] step {state['step']}/{steps} loss {float(loss):.4f} (mean of last 20 {np.mean(state['loss'][-20:]):.4f}"
                  + (f"; feature term {np.mean(state['feat'][-20:]):.4f}" if fe is not None else "")
                  + f"); {time.time() - t1:.1f}s/step, peak {pk:.1f} GB", flush=True)
        if state["step"] in save_at:                     # TTT30B: the curve's copies (weights only)
            mx.save_safetensors(os.path.join(CKD, f"dd_{a.run}_s{state['step']}.safetensors"), dict(tree_flatten(P.parameters())))
            print(f"[dd] copy at step {state['step']}: dd_{a.run}_s{state['step']}", flush=True)
        if a.peak_max and pk > a.peak_max:
            save("stopped: peak"); print(f"[dd] MLX peak {pk:.1f} GB > {a.peak_max}: stopped and saved", flush=True); sys.exit(4)
        if stop["now"]:
            save("stopped"); print(f"[dd] stopped at step {state['step']}: saved", flush=True); sys.exit(3)
        if time.time() - last > 300:
            save("running"); last = time.time()
    save("done")
    print(f"[dd] DONE {a.run}: {state['step']} steps in {time.time() - t0:.0f}s; peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    if a.smoke:                                          # a resume check of the saved state, then the throwaway files go
        st = tree_unflatten(list(mx.load(cko).items())) if os.path.exists(cko) else {}
        print(f"[dd] smoke: saved optimizer groups {sorted(st) if optx is not None else 'one'}; files removed", flush=True)
        for f in (ckw, ckj, cko):
            if os.path.exists(f):
                os.remove(f)


def kept_walk(rt, r0):
    sub = rt[r0:]; out = []; p = 0
    while p < sub.shape[0]:
        k = 0
        while k < 31 and sub[p, k] == 1:
            k += 1
        out.append(k + 1); p += k + 1
    return out


def cmd_eval(a):
    out = a.out or OUT
    os.makedirs(out, exist_ok=True)
    rt_, lm, head = M.load(a.frac)
    P = M.Trainable(head)
    R = {}
    for run in a.runs.split(","):
        w = os.path.join(T.CK, "d22_b5.safetensors") if run == "b5" else os.path.join(CKD, f"dd_{run}.safetensors")
        wts = mx.load(w)                                 # TTT30C: a run with a branch carries xb_*; b5 / t30b / t30f do not
        set_branch(P, int(wts["xb_down.weight"].shape[1]) if "xb_down.weight" in wts else 0)
        P.load_weights(list(wts.items())); mx.eval(P.parameters())
        R[run] = {}
        for st in a.sets.split(","):
            if st == "code":
                items = [(os.path.basename(f)[:-4], None) for f in sorted(glob.glob(os.path.join(T.TD, "d16w", "tf_code_HumanEval_*.npz")))]
            else:
                items = [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(os.path.join(T.TD, "dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]]
            band = {b: [0, 0] for b in BANDS}; kept = []
            for key, f in items:
                while os.path.exists(T.PAUSE):
                    time.sleep(1)
                if f is None:
                    x, gen = T.window(key), T.gen_of(key)
                else:
                    z = np.load(f); x = {"W": mx.array(z["wd"]).astype(mx.bfloat16)[None], "ids": np.asarray(z["ids"], np.int32)}
                    gen = int(z["gen"])
                a0 = max(0, gen - 2)
                hits = T.chain_hits(lm, head, P, x, 31, a0)
                n = len(x["ids"]) - 1 - a0
                rt = np.full((n, 31), -1, np.int8)
                for j, (right, _am) in enumerate(hits):
                    rt[:len(right), j] = np.asarray(right).astype(np.int8)
                for lo, hi in BANDS:
                    c = rt[:, lo:hi + 1]; band[(lo, hi)][0] += int((c == 1).sum()); band[(lo, hi)][1] += int((c >= 0).sum())
                kept += kept_walk(rt, max(0, gen - 1 - a0))
                mx.clear_cache()
            R[run][st] = {f"{lo}-{hi}": band[(lo, hi)][0] / max(band[(lo, hi)][1], 1) for lo, hi in BANDS}
            R[run][st]["kept_per_32"] = float(np.mean(kept)) if kept else None
            R[run][st]["windows"] = len(items)
            print(f"[dd] {run} {st}: " + " ".join(f"{b}:{100 * v:.1f}" for b, v in R[run][st].items() if "-" in b)
                  + f"; kept/32 {R[run][st]['kept_per_32']:.2f}", flush=True)
    if "b5" in R and len(R) > 1 and "code" in R["b5"]:
        b, cand = R["b5"]["code"], [r for r in R if r != "b5"][-1]
        c = R[cand]["code"]
        v = ("CONTINUE" if c["23-30"] >= 0.90 and b["0-0"] - c["0-0"] <= 0.01 else "KILL" if c["23-30"] < 0.87 else "INCONCLUSIVE")
        R["verdict"] = dict(run=cand, rule="code 23-30 >= 90 % and depth 0 within 1 pt of b5: CONTINUE; < 87 %: KILL",
                            b5_23_30=b["23-30"], run_23_30=c["23-30"], b5_d0=b["0-0"], run_d0=c["0-0"], verdict=v)
    R["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
    json.dump(R, open(os.path.join(out, "eval.json"), "w"), indent=1)
    print(json.dumps(R.get("verdict")), flush=True)


def main():
    M.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--run", default="t30a"); ap.add_argument("--runs", default="b5,t30a")
    ap.add_argument("--frac", type=float, default=0.03); ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--rows", type=int, default=192); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wdecay", type=float, default=0.97); ap.add_argument("--steps", type=int, default=0)   # 0: 600, or one pass
    ap.add_argument("--peak-max", dest="peak_max", type=float, default=18.0); ap.add_argument("--sets", default="code,mbpp")
    ap.add_argument("--manifest"); ap.add_argument("--pass-seed", dest="pass_seed", type=int)   # TTT30B
    ap.add_argument("--save-at", dest="save_at", default=""); ap.add_argument("--out")
    ap.add_argument("--feat", action="store_true"); ap.add_argument("--xb", type=int, default=0)          # TTT30C
    ap.add_argument("--xb-lr-mult", dest="xb_lr_mult", type=float, default=10.0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
