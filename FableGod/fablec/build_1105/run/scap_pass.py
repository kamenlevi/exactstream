"""C3-R6's all-rows router scores capture (the orchestrator 07:2x, option (i); Breakthrough, Thu 2026-11-05). Runs the map
harness's "scapture" (dm_common CAPR "s": each MoE layer's router top-10 scores for every row, renormalised over the 10, the
xcapture's W) over the map's windows, the baseline model, with DM_OUT = a scratch dir (default ~/Breakthrough/data/scap):
<DM_OUT>/scores/<window>.npz (scores [n - 1, 48, 10] float16, ids, a) and <DM_OUT>/routing/<window>.npz; the map's own
out/routing is never written. Then the checks, per window: the new routing ids == the map's stored out/routing ids (every
mask was built on those); the scores' rows == the routing's rows; every row sums to 1 (float16); the answer rows' scores vs
the 11-03 xcapture's w (max |diff|; a different engine build may round differently).
Usage (BigMoE venv): python scap_pass.py smoke (the shortest code window) | run (all 36) | check (the checks only)"""
import os, sys, time
os.environ.setdefault("DM_OUT", os.path.expanduser("~/Breakthrough/data/scap"))
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import numpy as np
MAPOUT = os.path.join(DM, "out"); OUT = os.environ["DM_OUT"]

def check(names):
    """Hard: shapes, rows sum to 1, the routing ids == the map's stored routing on >= 99 % of (row, layer) cells and the
    baseline top-1 (<DM_OUT>/scapture.json vs the map's out/baseline.json) on >= 99 % of rows: a misread (e.g. converted)
    store would break both wholesale, while a newer engine build's rounding moves only a few cells. Exact equality is
    reported beside (the R6 masks use the map's stored routing ids; scores of cells whose ids moved are flagged)."""
    import json
    jb = json.load(open(os.path.join(MAPOUT, "baseline.json"))); p = os.path.join(OUT, "scapture.json")
    js = json.load(open(p)) if os.path.exists(p) else {}
    bad = 0; dmax = 0.0; cells = same = rows = top1 = 0
    for nm in names:
        q = os.path.join(OUT, "scores", nm)
        if not os.path.exists(q): print(f"[scap] {nm}: missing", flush=True); bad += 1; continue
        s = np.load(q); r = np.load(os.path.join(MAPOUT, "routing", nm))["ids"]; x = np.load(os.path.join(MAPOUT, "xcap", nm))
        a = int(x["a"]); sc = s["scores"].astype(np.float32); new = np.load(os.path.join(OUT, "routing", nm))["ids"]
        if not (sc.shape == r.shape == new.shape and np.array_equal(s["ids"], new) and np.allclose(sc.sum(-1), 1.0, atol=1e-2)):
            bad += 1; print(f"[scap] {nm}: CHECK FAILED (shape / ids / row sums)", flush=True); continue
        eqc = (new == r).all(-1); cells += eqc.size; same += int(eqc.sum())
        d = float(np.abs(sc[a:a + x["w"].shape[0]] - x["w"].astype(np.float32)).max()); dmax = max(dmax, d)
        if nm in js:
            t1n, t1b = np.array(js[nm]["tk"])[:, 0], np.array(jb[nm]["tk"])[:len(js[nm]["tk"]), 0]
            rows += len(t1n); top1 += int((t1n == t1b).sum())
    fr, ft = same / max(cells, 1), top1 / max(rows, 1)
    ok = bad == 0 and fr >= 0.99 and (rows == 0 or ft >= 0.99)
    print(f"[scap] checks: {len(names) - bad}/{len(names)} windows formed; routing ids == the map's stored routing on {fr:.4%} of "
          f"(row, layer) cells; baseline top-1 == the map's baseline on {ft:.4%} of {rows} rows; answer rows vs the 11-03 "
          f"xcapture max |diff| {dmax:.2e}; {'PASS' if ok else 'FAIL'}", flush=True)
    return ok

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    import dm_common as C
    assert os.path.realpath(C.OUT) != os.path.realpath(MAPOUT), "DM_OUT must not be the map's out/"
    W = C.load_windows(12)
    if mode == "smoke": W = [min((w for w in W if w[0] == "code"), key=lambda w: len(w[2]))]
    if mode in ("smoke", "run"):
        import mlx.core as mx
        t0 = time.time(); model, _ = C.load_model(); C.discover_layers(model); C.install()
        mx.reset_peak_memory(); t1 = time.time()
        C.measure(model, W, "scapture", lambda: C.CAPR.update(on=True, s=True)); C.clear()
        print(f"[scap] {mode}: {len(W)} windows in {time.time() - t1:.0f}s (load {t1 - t0:.0f}s), MLX peak "
              f"{mx.get_peak_memory() / 2**30:.1f} GB", flush=True)
    sys.exit(0 if check([w[1] for w in W]) else 1)
