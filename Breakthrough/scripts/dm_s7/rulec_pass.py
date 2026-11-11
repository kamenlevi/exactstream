"""Rule C fresh data (Fable FULL's Round 2, ~/FableGod/full/FULL.md "Fresh data and its cost"; the orchestrator 21:4x;
Breakthrough, Wed 2026-11-04 21:5x): teacher-forced top-8 passes with the decision map's harness in its BASELINE configuration
(no perturbation, as the map's out/baseline.json that M-39 replayed) over
  (i)  rulec_he12: the 12 HPD-0 HumanEval+ windows that are not in the map (~/TrainLab/data/d02b_walk_hpd0_code.npz minus
       the map's 12 code windows), and
  (ii) rulec_mbpp: the MBPP HPD-0 walk's windows (~/TrainLab/data/d02b_walk_hpd0_mbpp.npz).
Windows are built as dm_common.load_windows builds the map's: (domain, name, ids, off, gen); measure() keeps the answer
positions (a = gen + 1 - off on), per window {dom, a, off, tk [rows][8], lp [rows][8]}.
Outputs: $DM_OUT (default ~/Breakthrough/data/rulec)/rulec_he12.json and rulec_mbpp.json, for MoonShots' CPU replay.
Usage (BigMoE venv): python rulec_pass.py check   (CPU only: window counts, positions, the overlap check; no model)
                     python rulec_pass.py run     (GPU, one model load; a set whose JSON exists is skipped: resume)"""
import os, sys, json, time
os.environ.setdefault("DM_OUT", os.path.expanduser("~/Breakthrough/data/rulec"))
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
MAP_BASE = os.path.join(DM, "out", "baseline.json")

def windows(f, dom, skip=()):
    z = np.load(os.path.join(TD, f)); W = []
    for s, nm in enumerate(z["names"]):
        nm = str(nm)
        if nm in skip: continue
        W.append((dom, nm, z[f"{s}_ids"].astype(np.int32), int(z[f"{s}_off"]), int(z[f"{s}_gen"])))
    return W

def sets():
    base = json.load(open(MAP_BASE))
    he = windows("d02b_walk_hpd0_code.npz", "code", skip=set(base))
    mb = windows("d02b_walk_hpd0_mbpp.npz", "mbpp")
    return base, [("rulec_he12", he), ("rulec_mbpp", mb)]

def pos(W):   # measured rows per window: tk has len(ids) - 1 rows (row t predicts ids[t + 1]); the answer from a on
    return sum(len(ids) - 1 - (gen + 1 - off) for _, _, ids, off, gen in W)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    base, S = sets()
    print(f"[rulec] DM_OUT {os.environ['DM_OUT']}", flush=True)
    for name, W in S:
        print(f"[rulec] {name}: {len(W)} windows, {sum(len(w[2]) for w in W)} ids, {pos(W)} measured positions; "
              f"first {[w[1] for w in W[:3]]}", flush=True)
    if mode == "check":
        z = np.load(os.path.join(TD, "d02b_walk_code.npz")); m = {str(n): z[f"{s}_ids"] for s, n in enumerate(z["names"])}
        hp = np.load(os.path.join(TD, "d02b_walk_hpd0_code.npz"))
        both = [(str(n), s) for s, n in enumerate(hp["names"]) if str(n) in base and str(n) in m]
        same = sum(np.array_equal(m[n], hp[f"{s}_ids"]) for n, s in both)
        print(f"[rulec] overlap check: the map's code windows in the HPD-0 walk {len(both)}; same ids as the map's set {same}/{len(both)}; "
              f"map windows' measured positions {sum(len(base[n]['tk']) for n, _ in both)}", flush=True)
        sys.exit(0)
    import mlx.core as mx
    import dm_common as C
    assert os.path.realpath(C.OUT) != os.path.realpath(os.path.join(DM, "out")), "DM_OUT must not be the map's out/"
    t0 = time.time(); model, _ = C.load_model(); C.discover_layers(model); C.install()
    print(f"[rulec] model loaded in {time.time() - t0:.0f}s", flush=True)
    for name, W in S:
        if os.path.exists(os.path.join(C.OUT, name + ".json")): print(f"[rulec] {name}: exists, skipped", flush=True); continue
        mx.reset_peak_memory(); t1 = time.time()
        res = C.measure(model, W, name, lambda: None)
        n = sum(len(v["tk"]) for v in res.values() if isinstance(v, dict) and "tk" in v)
        print(f"[rulec] {name} done: {len(res)} windows, {n} positions, {time.time() - t1:.0f}s, MLX peak "
              f"{mx.get_peak_memory() / 2**30:.1f} GB", flush=True)
    C.clear()
