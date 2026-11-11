"""PORT steps check (MTP Head, Wed 11-04 22:0x): compare a train_port.py log (--log-json) with L1-20's logged per-step
losses (~/Leap/data/l120_ft.state.json, the MLX trainer on the GPU), or (Thu 11-05) with another run's {"loss": [...]}
JSON given as the 2nd argument (e.g. port/ref_mlx_steps.py with MLX_ENABLE_TF32=0). PASS = every compared step within
2e-3 relative (set before the run: step 1 already agrees to 1e-4; later steps add GPU/CPU accumulation-order drift
through Adam).
Usage: python compare_l120.py PORT_LOG.json [REF.json]"""
import json, os, sys
port = json.load(open(os.path.expanduser(sys.argv[1])))["loss"]
if len(sys.argv) > 2:
    ref, label = json.load(open(os.path.expanduser(sys.argv[2])))["loss"], os.path.basename(sys.argv[2])
else:
    ref, label = json.load(open(os.path.expanduser("~/Leap/data/l120_ft.state.json")))["log"]["loss"], "L1-20"
worst = 0.0
for s, (a, b) in enumerate(zip(port, ref)):
    rel = abs(a - b) / abs(b)
    worst = max(worst, rel)
    print(f"step {s + 1}: port {a:.6f}  {label} {b:.6f}  rel {rel:.2e}")
print(f"STEPS CHECK vs {label} ({min(len(port), len(ref))} steps): max rel {worst:.2e} -> "
      f"{'PASS' if worst < 2e-3 else 'FAIL'} (line 2e-3)")
