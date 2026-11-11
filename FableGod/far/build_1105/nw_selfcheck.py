"""NEVER-WAIT's three self-checks on the REAL model (NEVERWAIT_TEST.md 2.5; Fable FAR, Thu 5 Nov 2026). GPU, ~2 min:
one map process, the model loaded once, the two shortest code windows, five arms through the map's own measure() with
DM_OUT = a scratch folder (default ~/Breakthrough/data/nw_selfcheck; the map's out/ is never written):
  twin   qom_0_live_ball_c208        the reference
  twin2  the twin again              run-to-run noise of the harness itself (the yardstick for checks i and ii)
  nwz    qom_0_live_ball_c208_nwz    check (i): the twin's masks through the never-wait path, late flags off
  nw0    qom_0_live_ball_c208_nw0    check (ii): lateness 0 (each late term written straight after its own layer's write)
  nw1    qom_0_live_ball_c208_nw1    check (iii) on the decisive arm's path (the tail assert, the late-term count)
Lines:
  (i)   PASS = nwz's top-8 ids and log-probs equal the twin's exactly. If twin2 itself is not exactly the twin, the
        harness is not run-to-run exact on this build and (i) passes when nwz is no farther from the twin than twin2 is.
  (ii)  PASS = nw0's top-1 differs from the twin's on at most max(2, 0.4 %) of the rows and the mean |d log p1| is
        <= 0.02 (bf16: the split add re-rounds ~10 of a row's 96 residual writes). FAIL = more than 3 % of the rows or
        a mean above 0.2 (a gross fault: a term doubled, dropped or misplaced wholesale). Between the two: REPORT: the
        arms may run; read nw1 beside nw0 and twin2. Only (i) and (iii) are exact gates; (ii) has little power on two
        windows, and the lateness-0 path is checked to 5e-16 against the twin on CPU (test_nw.py).
  (iii) PASS = no WARNING on any window, every late flag of the masks used exactly once (res["nw"]["terms"]), late
        writes > 0 for nw0 and nw1 and 0 for nwz.
The verdict (PASS / REPORT / FAIL) is the last line and is written to <DM_OUT>/VERDICT. Exit 0 = PASS or REPORT (run
the arms), 1 = FAIL (do not run them; under gslot.sh a non-zero exit is retried, up to 3 runs); NWSC_STRICT=1: REPORT
exits 2.
Usage (BigMoE venv, as Breakthrough, under a share lock, e.g. through scripts/dm_s7/gslot.sh):
  BIGMOE_EXT_BUILD=build_nat python nw_selfcheck.py            # run + read
  python nw_selfcheck.py read                                  # the read-out alone (CPU), from the scratch JSONs"""
import json, os, sys, time
os.environ.setdefault("DM_OUT", os.path.expanduser("~/Breakthrough/data/nw_selfcheck"))
os.environ["DM_SETS"] = "code"
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import numpy as np
MAPOUT = os.path.join(DM, "out"); OUT = os.path.expanduser(os.environ["DM_OUT"])
TWIN = "qom_0_live_ball_c208"
ARMS = [("twin", TWIN), ("twin2", TWIN), ("nwz", TWIN + "_nwz"), ("nw0", TWIN + "_nw0"), ("nw1", TWIN + "_nw1")]

def read():
    R = {}
    for tag, _ in ARMS:
        p = os.path.join(OUT, "sc_" + tag + ".json")
        if not os.path.exists(p): print(f"[nwsc] {tag}: not run ({p})", flush=True); return 1
        R[tag] = json.load(open(p))
    W = list(R["twin"]); rows = sum(len(R["twin"][w]["tk"]) for w in W)
    def cmp_(a, b):
        """(exact, top-1 flips, max |d lp| over the top-8 where the ids agree, mean |d log p1|)."""
        ex = True; fl = 0; mx_ = 0.0; s = 0.0
        for w in W:
            ta, tb = np.array(R[a][w]["tk"]), np.array(R[b][w]["tk"]); la, lb = np.array(R[a][w]["lp"]), np.array(R[b][w]["lp"])
            ex &= np.array_equal(ta, tb) and np.array_equal(la, lb); fl += int((ta[:, 0] != tb[:, 0]).sum())
            same = ta == tb; mx_ = max(mx_, float(np.abs(la - lb)[same].max()) if same.any() else 0.0); s += float(np.abs(la[:, 0] - lb[:, 0]).sum())
        return ex, fl, mx_, s / max(rows, 1)
    e2, f2, m2, a2 = cmp_("twin2", "twin"); ez, fz, mz, az = cmp_("nwz", "twin"); _, f0, m0, a0 = cmp_("nw0", "twin"); _, f1, m1, a1 = cmp_("nw1", "twin")
    print(f"[nwsc] {len(W)} code windows, {rows} answer rows; the twin against itself (twin2): {'exactly equal' if e2 else f'{f2} top-1 flips, max |d lp| {m2:.2e}'}", flush=True)
    ok_i = ez or ((not e2) and fz <= f2 and mz <= max(m2, 1e-6) * 1.5)
    print(f"[nwsc] (i)   nwz vs the twin: {'exactly equal (top-8 ids and log-probs)' if ez else f'{fz} top-1 flips, max |d lp| {mz:.2e}'}: "
          f"{'PASS' if ok_i else 'FAIL'}" + ("" if ez or e2 else " (judged against twin2: the harness is not run-to-run exact here)"), flush=True)
    lim = max(2, int(0.004 * rows))
    v_ii = "PASS" if (f0 <= lim and a0 <= 0.02) else "FAIL" if (f0 > 0.03 * rows or a0 > 0.2) else "REPORT"
    print(f"[nwsc] (ii)  nw0 vs the twin: {f0} top-1 flips of {rows} rows ({100 * f0 / max(rows, 1):.2f} %), mean |d log p1| {a0:.4f}, max |d lp| {m0:.3f}: {v_ii} "
          f"(PASS <= {lim} flips and mean <= 0.02; FAIL > {int(0.03 * rows)} flips or mean > 0.2; between: REPORT); for scale, nw1 vs the twin: {f1} flips, mean |d log p1| {a1:.4f}", flush=True)
    bad = []; note = []
    for tag in ("nwz", "nw0", "nw1"):
        for w in W:
            r = R[tag][w]; nw = r.get("nw")
            if "warn" in r: bad.append(f"{tag} {w}: {r['warn'][:120]}")
            if nw is None: bad.append(f"{tag} {w}: no never-wait record (the patch is not on this path)"); continue
            mdir = TWIN if tag == "nwz" else TWIN + "_" + tag
            path = os.path.join(MAPOUT, "far", mdir, w)
            lm = np.load(path)["late"] if (tag != "nwz" and os.path.exists(path)) else None
            want = 0 if tag == "nwz" else (int(lm.sum()) if lm is not None else None)
            if want is not None and nw["terms"] != want: bad.append(f"{tag} {w}: {nw['terms']} late terms used, the mask holds {want}")
            if (tag == "nwz") != (nw["writes"] == 0): bad.append(f"{tag} {w}: {nw['writes']} late writes")
        note.append(f"{tag} " + " / ".join(f"{R[tag][w]['nw']['terms']} terms, {R[tag][w]['nw']['writes']} writes" for w in W if "nw" in R[tag][w]))
    for tag in ("twin", "twin2"):
        bad += [f"{tag} {w}: {R[tag][w]['warn'][:120]}" for w in W if "warn" in R[tag][w]]
    ok_iii = not bad
    print(f"[nwsc] (iii) rows and late flags: {'PASS' if ok_iii else 'FAIL'} ({'; '.join(note)})" + ("" if ok_iii else "\n[nwsc]       " + "\n[nwsc]       ".join(bad)), flush=True)
    v = "FAIL" if (not ok_i or v_ii == "FAIL" or not ok_iii) else "REPORT" if v_ii == "REPORT" else "PASS"
    print(f"[nwsc] SELF-CHECK {v}: " + {"PASS": "run the arms", "FAIL": "do not run the arms",
                                      "REPORT": "(i) and (iii) pass, (ii) is between its lines: run the arms, read nw1 beside nw0"}[v], flush=True)
    open(os.path.join(OUT, "VERDICT"), "w").write(v + "\n")
    return 1 if v == "FAIL" else 2 if (v == "REPORT" and os.environ.get("NWSC_STRICT") == "1") else 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "read": sys.exit(read())
    import dm_common as C
    from dm_ablations import ablation
    assert os.path.realpath(C.OUT) != os.path.realpath(MAPOUT), "DM_OUT must not be the map's out/"
    assert hasattr(C, "NW"), "the never-wait patch is not installed in " + C.__file__
    setups = [(tag, ablation(name)) for tag, name in ARMS]          # every name resolves before the model loads
    W = sorted(C.load_windows(12), key=lambda w: len(w[2]))[:int(os.environ.get("NWSC_WINDOWS", "2"))]
    assert W and all(w[0] == "code" for w in W), [w[0] for w in W]
    import mlx.core as mx
    t0 = time.time(); model, _ = C.load_model(); C.discover_layers(model); C.install()
    mx.reset_peak_memory(); t1 = time.time()
    for tag, setup in setups:
        C.measure(model, W, "sc_" + tag, setup); C.clear()
    print(f"[nwsc] {len(setups)} arms x {len(W)} windows in {time.time() - t1:.0f}s (load {t1 - t0:.0f}s), MLX peak {mx.get_peak_memory() / 2**30:.1f} GB", flush=True)
    sys.exit(read())
