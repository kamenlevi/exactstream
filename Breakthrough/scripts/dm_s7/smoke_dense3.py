"""GPU smoke of the INSTALLED DENSE-3 arm on the real model (Breakthrough, Wed 2026-11-04 17:0x; NewIdeas 16:5x). One code
window (the shortest), one map process, outputs to DM_OUT (a scratch dir, never the map's out/), share lock, < 2 min:
  (1) C regression: qom_0_live_bq_c147_t3 on the window == the stored out/ JSON (tk and lp bit for bit): the install
      left the harness unchanged;
  (2) screen 6 (qom_0_live_ball_c147_t3) -> R6a;
  (3) qom_0_live_ball_c147_t3+dense3 (ablation()'s "+" stack) -> RD: the "[dm] dense3" line (modules, MB), seconds,
      MLX peak; NOT INVALID = RD's top-8 ids and log-probs differ from R6a's on at least one row (NewIdeas' QUEUE 16:5x
      rule on the full run), plus top-1 flips and mean |dlp1| vs R6a and each one's flips vs the baseline;
  (4) screen 6 again (measure() runs clear() first, so TEARDOWN puts the weights back) -> R6b == R6a bit for bit.
Usage (BigMoE venv): DM_OUT=<scratch> python smoke_dense3.py"""
import os, sys, time, json, hashlib, resource
DMD = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DMD)
import numpy as np
import mlx.core as mx
import dm_common as C
from dm_ablations import ablation
REAL = os.path.join(DMD, "out"); ARM, SIX, CC = "qom_0_live_ball_c147_t3+dense3", "qom_0_live_ball_c147_t3", "qom_0_live_bq_c147_t3"
assert os.path.realpath(C.OUT) != os.path.realpath(REAL), "set DM_OUT to a scratch dir"
print(f"[smoke] env BIGMOE_PKD1={os.environ.get('BIGMOE_PKD1')} BIGMOE_SCOUT_BUNDLE={os.environ.get('BIGMOE_SCOUT_BUNDLE')}; "
      f"Leap dense3.py sha256 {hashlib.sha256(open(os.path.expanduser('~/Leap/scripts/dense3.py'), 'rb').read()).hexdigest()[:16]}", flush=True)
t_all = time.time()
model, _ = C.load_model(); C.discover_layers(model); C.install()
code = [w for w in C.load_windows(12) if w[0] == "code"]
W = [min(code, key=lambda w: len(w[2]))]; nm = W[0][1]
base = json.load(open(os.path.join(REAL, "baseline.json")))[nm]
print(f"[smoke] window {nm}: {len(W[0][2])} ids; model loaded in {time.time() - t_all:.0f}s", flush=True)
out = {}
for key, a in (("C", CC), ("6a", SIX), ("D", ARM), ("6b", SIX)):
    mx.reset_peak_memory(); t0 = time.time()
    res = C.measure(model, W, a, ablation(a)); dt = time.time() - t0
    out[key] = res[nm]
    fl = np.mean(np.array(res[nm]["tk"])[:, 0] != np.array(base["tk"])[:len(res[nm]["tk"]), 0]) * 100
    print(f"[smoke] {key} {a}: {dt:.1f}s, MLX peak {mx.get_peak_memory() / 2**30:.1f} GB, flips vs baseline {fl:.2f} % "
          f"({len(res[nm]['tk'])} rows)", flush=True)
    if key == "C":
        ref = json.load(open(os.path.join(REAL, a + ".json")))[nm]
        print(f"[smoke] C REGRESSION: tk equal {ref['tk'] == res[nm]['tk']}, lp bit-equal {ref['lp'] == res[nm]['lp']}", flush=True)
C.clear()
a6, b6, d = out["6a"], out["6b"], out["D"]
same = sum(1 for i in range(len(a6["tk"])) if a6["tk"][i] == d["tk"][i] and a6["lp"][i] == d["lp"][i])
t1a, t1d = np.array(a6["tk"])[:, 0], np.array(d["tk"])[:, 0]
dl = np.abs(np.array(a6["lp"])[:, 0] - np.array(d["lp"])[:, 0])
print(f"[smoke] RESTORE: screen 6 after the dense3 run == before: tk {a6['tk'] == b6['tk']}, lp bit-equal {a6['lp'] == b6['lp']}", flush=True)
print(f"[smoke] dense3 vs screen 6: rows with identical top-8 ids + log-probs {same}/{len(a6['tk'])} "
      f"({'INVALID' if same == len(a6['tk']) else 'not INVALID: the rewrite reached the forward'}); top-1 flips "
      f"{int((t1a != t1d).sum())}, mean |dlp1| {dl.mean():.4f}", flush=True)
print(f"[smoke] total {time.time() - t_all:.0f}s; max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB", flush=True)
