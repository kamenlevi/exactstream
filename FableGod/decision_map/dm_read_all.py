"""Run every map read-out once (Breakthrough, Mon 2026-11-02 23:3x): dm_cert_read.py, dm_day9_read.py, dm_q1u_read.py.
Usage (BigMoE venv): python dm_read_all.py"""
import os, subprocess, sys
D = os.path.dirname(os.path.abspath(__file__))
for s in ("dm_cert_read.py", "dm_day9_read.py", "dm_q1u_read.py", "dm_t3all_read.py", "dm_oea_read.py"):
    r = subprocess.run([sys.executable, os.path.join(D, s)], cwd=D, capture_output=True, text=True)
    print(f"[read-all] {s} rc {r.returncode}", flush=True)
