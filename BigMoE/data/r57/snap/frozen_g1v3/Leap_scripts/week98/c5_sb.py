"""c5_sb.py (Leap, Sun 15 Nov 2026 01:4x): C5 TIME32 with the engine's SB on every arm. The exact-45 loop submits a
2-16-row check's graph every SBN = 4 layers (r57_screen_rel1.py ref_on: Q._prefill_midloop_eval_layers -> SBN for
2 <= rows <= 16), so Python's build overlaps the GPU; above 16 rows it submits nothing until the end. Here every arm
(A16, B2x16, C32) gets SBN for 2-32 rows: scheduling only (MTPLX: the same graph and kernels, every output bit unchanged),
so the comparison is the engine's GPU-overlapped one. Runs c5_time32.py unchanged otherwise.
usage: [venv] c5_sb.py [smoke]   (C5_SBN, default 4)"""
import os
import runpy
import sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
SBN = int(os.environ.get("C5_SBN", "4"))
import mtplx.models.qwen4_exp as Q
_orig = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: _orig(rows) or (SBN if 2 <= int(rows) <= 32 else 0)
print(f"c5_sb: the engine's SB on every arm, a submit every {SBN} layers for 2-32-row forwards", flush=True)
sys.argv = ["c5_time32.py"] + sys.argv[1:]
runpy.run_path(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/week98/c5_time32.py"), run_name="__main__")
