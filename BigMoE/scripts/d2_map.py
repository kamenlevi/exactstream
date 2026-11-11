#!/usr/bin/env python3
"""D2's map run (Wed 11 Nov; docs/PREREG_D2_2026-11-11.md). It uses Breakthrough's decision map as it stands: dm_common's
loader, windows and measure(), and the D2 ablations in ~/FableGod/decision_map/dm_d2.py (the owner's OK, 18:4x). The
only addition is the agent set (d02b_walk_d02a_agent.npz), which the map's SETS lacks. All windows of each set are used,
in the files' order.
Order (the verdict first):
  1. gate: the map's validity check (baseline.json vs D-02a's stored top-1; exit 3 if bad).
  2. bm_d2: D2 = d2_trunk+d0_resid, on code, IFEval, multi and agent.
  3. bm_res: the control, d0_resid with the store's trunk, on the same windows.
  4. bm_ops: d0_ops, the ops path's noise floor, on code, IFEval and multi.
  5. bm_d2t: d2_trunk alone, the trunk's own error, on code, IFEval and multi.
  6. bm_d3 / bm_d4: the 3- and 4-bit trunk + d0_resid (the curve), on code, IFEval and multi.
  7. bm_d2_rep: D2 again on the first IFEval window (the determinism check against bm_d2).
  8. bm_base_agent: the stock exact pass on the agent windows, the agent set's gate: its top-1 vs D-02a's stored
     top-1 must be >= 99.9 % (baseline.json has no agent rows; Breakthrough 18:5x).
Outputs: ~/FableGod/decision_map/out/bm_*.json (measure()'s per-window top-8 ids and log-probs on the answer rows).
usage: cd ~/FableGod/decision_map && ~/BigMoE/.venv/bin/python ~/BigMoE/scripts/d2_map.py [--only bm_d2,bm_res]"""
import os, sys, time, traceback

DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
os.chdir(DM)
import dm_common as C  # noqa: E402
import dm_ablations as A  # noqa: E402

only = None
if "--only" in sys.argv:
    only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
if not any(s[0] == "agent" for s in C.SETS):
    C.SETS.append(("agent", "d02b_walk_d02a_agent.npz"))
model, tok = C.load_model()
C.discover_layers(model)
C.install()
W = C.load_windows(10 ** 6)                              # every window of every set, in file order
by = {}
for w in W:
    by.setdefault(w[0], []).append(w)
print("[d2] windows: " + ", ".join(f"{d} {len(v)}" for d, v in by.items()), flush=True)
chat_code = by["code"] + by["ifeval"] + by["multi"]
allw = chat_code + by["agent"]
ARMS = [("bm_d2", "d2_trunk+d0_resid", allw), ("bm_res", "d0_resid", allw), ("bm_ops", "d0_ops", chat_code),
        ("bm_d2t", "d2_trunk", chat_code), ("bm_d3", "d3_trunk+d0_resid", chat_code), ("bm_d4", "d4_trunk+d0_resid", chat_code),
        ("bm_d2_rep", "d2_trunk+d0_resid", by["ifeval"][:1]),
        ("bm_base_agent", "none", by["agent"])]           # Breakthrough 18:5x: the agent windows' wiring check (an exact
                                                      # pass vs D-02a's stored top-1, >= 99.9 %), run with --only
ag, ok = C.gate()
print(f"[d2] gate: agreement {ag}, ok {ok}", flush=True)
if not ok:
    sys.exit(3)
for label, spec, ws in ARMS:
    if only and label not in only:
        continue
    t0 = time.time()
    try:
        C.measure(model, ws, label, (lambda: None) if spec == "none" else A.ablation(spec))
    except Exception:
        C.clear(); C.REC["on"] = False
        print(f"[d2] {label} ({spec}) FAILED after {time.time() - t0:.0f}s:\n{traceback.format_exc()}", flush=True)
        continue
    print(f"[d2] {label} ({spec}) done in {time.time() - t0:.0f}s, {len(ws)} windows", flush=True)
print("[d2] DONE", flush=True)
