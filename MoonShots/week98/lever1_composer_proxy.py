"""WEEK98 lever 1 screen (MoonShots, Sun 15 Nov 2026 01:xx): can a small pretrained model of the same family predict the
125B's deep state? Qwen3.5-2B (same tokenizer as the 125B: 248,077 ids, checked) runs over the BANK's proxy files; a ridge
maps its hidden state at a layer -> the 125B's wd; scored exactly like BigMoE's bank_proxy.py (held-out R2, the
RMS-normalized direction error, the predicted MBPP 23-30 from calibration.json, interpolated on the direction error).

LINE (written before running, LEVERS_A_MOONSHOTS.md lever 1): PASS if the best layer's predicted MBPP 23-30 >= 85 ->
REPLACE eval with t30b (BigMoE); BETWEEN 78-85 -> an MLP map / the 4B first; KILL if < 78.

Read-only inputs: ~/datasets/deepdraft/bank/{<file>.npz, proxy/wd_{tr,va}.npy, proxy/raw_{tr,va}.npy, proxy/rows_*.json},
~/BigMoE/data/deepdraft/bank/calibration.json; Sandbox98's sb/core.py (target_states, imported, not changed).
Writes only ~/MoonShots/week98/. Sanity first: the tap-45 ridge must reproduce the proxy's calibration point
(R2 0.8285, direction error 0.3398, predicted 91.96).
usage: ~/BigMoE/.venv/bin/python -I lever1_composer_proxy.py [--files N] [--layers 5,11,17,post]"""
import argparse, json, os, sys, time
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98"))
BANK = H("~/datasets/deepdraft/bank")
PC = os.path.join(BANK, "proxy")
CAL = H("~/BigMoE/data/deepdraft/bank/calibration.json")
OUT = H("~/MoonShots/week98")


def score(P, Y):                                   # bank_proxy.py's score(), verbatim logic
    Y = np.asarray(Y, np.float32); ym = Y.mean(0)
    r2 = 1 - float(((P - Y) ** 2).sum()) / float(((Y - ym) ** 2).sum())
    pn = P / np.sqrt((P ** 2).mean(-1, keepdims=True) + 1e-12); yn = Y / np.sqrt((Y ** 2).mean(-1, keepdims=True) + 1e-12)
    return r2, float(np.sqrt(((pn - yn) ** 2).sum() / (yn ** 2).sum()))


def predict_acc(ne, family="ridge"):               # bank_proxy.py's predict_acc(), same rule
    pts = sorted([p for p in json.load(open(CAL))["points"] if p.get("family", "ridge") in (family, "oracle") and "ne" in p],
                 key=lambda p: p["ne"])
    xs = np.array([p["ne"] for p in pts]); ys = np.minimum.accumulate(np.array([p["mbpp_23_30"] for p in pts]))
    return float(np.interp(ne, xs, ys))


def ridge(Xtr, Ytr, Xva, Yva):                     # bank_proxy.py cmd_ridge: lambda in {1e-3, 1e-2, 1e-1} x mean diag
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc = Xtr - xm
    XtX = Xc.T @ Xc; XtY = Xc.T @ (Ytr - ym); dg = float(np.mean(np.diag(XtX)))
    best = None
    for lf in (1e-3, 1e-2, 1e-1):
        W = np.linalg.solve(XtX + lf * dg * np.eye(Xtr.shape[1], dtype=np.float32), XtY).astype(np.float32)
        r2, ne = score((Xva - xm) @ W + ym, Yva)
        if best is None or r2 > best[0]:
            best = (r2, ne, lf)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=0, help="limit files per part (0 = all), for a smoke run")
    ap.add_argument("--layers", default="5,11,17,post")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    Ytr = np.load(os.path.join(PC, "wd_tr.npy"), mmap_mode="r"); Yva = np.load(os.path.join(PC, "wd_va.npy"), mmap_mode="r")
    rows = {p: json.load(open(os.path.join(PC, f"rows_{p}.json"))) for p in ("tr", "va")}
    res = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # sanity: the proxy's own tap-45 point
    raw_tr = np.load(os.path.join(PC, "raw_tr.npy"), mmap_mode="r"); raw_va = np.load(os.path.join(PC, "raw_va.npy"), mmap_mode="r")
    r2, ne, lf = ridge(np.asarray(raw_tr[:, 4 * 2560:5 * 2560], np.float32), np.asarray(Ytr, np.float32),
                       np.asarray(raw_va[:, 4 * 2560:5 * 2560], np.float32), Yva)
    res["sanity_tap45"] = dict(r2=r2, ne=ne, lam=lf, pred=predict_acc(ne))
    print(f"[sanity] tap45 ridge: R2 {r2:.4f} ne {ne:.4f} pred {predict_acc(ne):.2f} (calibration point: 0.8285 / 0.3398 / 91.96)"
          f"  {time.time() - t0:.0f}s", flush=True)
    del raw_tr, raw_va

    # the composer: Qwen3.5-2B over each file once (no cache, full sequence), states at the proxy's rows
    from sb import core as C
    C.limit_memory(6)
    import mlx.core as mx
    names = [x.strip() for x in a.layers.split(",")]
    taps = tuple(int(x) for x in names if x != "post")
    feats = {}
    for p in ("tr", "va"):
        by_file = {}
        for i, (nm, r) in enumerate(rows[p]):
            by_file.setdefault(nm, []).append((i, r))
        files = list(by_file)[: a.files] if a.files else list(by_file)
        keep = np.concatenate([[i for i, _ in by_file[nm]] for nm in files]).astype(np.int64)
        F = {nm_: np.zeros((len(rows[p]), 2048), np.float16) for nm_ in names}
        for k, nm in enumerate(files):
            C.wait_pause()
            z = np.load(os.path.join(BANK, nm))
            ids = np.asarray(z["ids"])
            _, post, tp, _ = C.target_states(ids, taps=taps, logits=False)
            idx = np.array([i for i, _ in by_file[nm]]); rr = np.array([r for _, r in by_file[nm]])
            for nm_ in names:
                src = post if nm_ == "post" else tp[int(nm_)]
                F[nm_][idx] = np.asarray(src[mx.array(rr)].astype(mx.float32)).astype(np.float16)
            if k % 50 == 0:
                print(f"[{p}] {k}/{len(files)} files, {time.time() - t0:.0f}s", flush=True)
        feats[p] = (F, keep)
    res["n_rows"] = {p: int(len(feats[p][1])) for p in feats}

    Ytr_k = np.asarray(Ytr[feats["tr"][1]], np.float32); Yva_k = np.asarray(Yva[feats["va"][1]], np.float32)
    sets = [[n] for n in names] + [["11", "post"], names]
    res["maps"] = {}
    for s in sets:
        Xtr = np.concatenate([feats["tr"][0][n][feats["tr"][1]].astype(np.float32) for n in s], 1)
        Xva = np.concatenate([feats["va"][0][n][feats["va"][1]].astype(np.float32) for n in s], 1)
        r2, ne, lf = ridge(Xtr, Ytr_k, Xva, Yva_k)
        tag = "2B:" + "+".join(s)
        res["maps"][tag] = dict(r2=r2, ne=ne, lam=lf, pred=predict_acc(ne))
        print(f"[map] {tag}: R2 {r2:.4f} ne {ne:.4f} pred MBPP 23-30 {predict_acc(ne):.2f}  {time.time() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "lever1_composer_proxy.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("written", os.path.join(OUT, "lever1_composer_proxy.json"))


if __name__ == "__main__":
    main()
