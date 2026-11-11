"""GPU smoke of the INSTALLED screen 6 / 7 hook (Breakthrough, Wed 2026-11-04 15:0x; the orchestrator 14:3x: "a tiny smoke
for the hook: keep it under 2 min, share lock"). One code window (the shortest), one map process, outputs to DM_OUT (a
scratch dir, never the map's out/):
  (1) C regression: qom_0_live_bq_c147_t3 on the window == the stored out/qom_0_live_bq_c147_t3.json (tk and lp bit for
      bit), so the patch leaves the existing t3mask path untouched end to end;
  (2) s7_trl with DM_TRL_FALLBACK=1 (by day the store is partial: layer 18 part-encoded, the rest fall back to MSB-3,
      counted), s7_ref and s7_cal_all run; their seconds per window and MLX peaks price tonight's slot.
Usage (BigMoE venv): DM_OUT=<scratch> python smoke_s7.py [arm ...]"""
import os, sys, time, json, resource
os.environ.setdefault("DM_TRL_FALLBACK", "1")
DMD = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DMD)
import numpy as np
import mlx.core as mx
import dm_common as C
from dm_ablations import ABLATIONS
REAL = os.path.join(DMD, "out")
assert os.path.realpath(C.OUT) != os.path.realpath(REAL), "set DM_OUT to a scratch dir"
arms = sys.argv[1:] or ["qom_0_live_bq_c147_t3", "s7_trl", "s7_ref", "s7_cal_all"]
t_all = time.time()
model, _ = C.load_model(); C.discover_layers(model); C.install()
code = [w for w in C.load_windows(12) if w[0] == "code"]
W = [min(code, key=lambda w: len(w[2]))]
nm = W[0][1]; print(f"[smoke] window {nm}: {len(W[0][2])} ids, {len(W[0][2]) - 1 - (W[0][4] + 1 - W[0][3])} measured rows", flush=True)
out = {}
for a in arms:
    if a == "s7_cal_all" and time.time() - t_all > 75:
        print(f"[smoke] {a} skipped: {time.time() - t_all:.0f}s used (the 2-min cap)", flush=True); continue
    mx.reset_peak_memory(); t0 = time.time()
    res = C.measure(model, W, a, ABLATIONS[a]); dt = time.time() - t0
    out[a] = res[nm]
    print(f"[smoke] {a}: {dt:.1f}s for the window, MLX peak {mx.get_peak_memory() / 2**30:.1f} GB, lsrc {res[nm].get('lsrc')}", flush=True)
    if a == "qom_0_live_bq_c147_t3":
        ref = json.load(open(os.path.join(REAL, a + ".json")))[nm]
        print(f"[smoke] C REGRESSION {nm}: tk equal {ref['tk'] == res[nm]['tk']}, lp bit-equal {ref['lp'] == res[nm]['lp']} "
              f"({len(res[nm]['tk'])} rows)", flush=True)
if "s7_trl" in out and "s7_ref" in out:
    a, b = np.array(out["s7_trl"]["tk"])[:, 0], np.array(out["s7_ref"]["tk"])[:, 0]
    d = np.abs(np.array(out["s7_trl"]["lp"])[:, 0] - np.array(out["s7_ref"]["lp"])[:, 0])
    print(f"[smoke] s7_trl vs s7_ref: top-1 flips {int((a != b).sum())}/{len(a)}, mean |dlp1| {d.mean():.4f} (differ, as they should)", flush=True)
print(f"[smoke] total {time.time() - t_all:.0f}s; max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB", flush=True)
