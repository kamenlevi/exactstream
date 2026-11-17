"""kt_launch.py (BigMoE, Tue 17 Nov 2026): run an r57 screen in THIS process as `python <screen> <args>` would (sys.argv,
sys.path[0] = the screen's folder, run as __main__) and print one line at exit, "V3KT {json}": Leap's ref2_switches_v3
state (ref2.CFG bof / hc_b, bofi loaded / ON, bofi.CNT, ref2.COUNTS hc_b). Nothing in the run changes.
usage: python scripts/kt_launch.py <screen.py> <args...>"""
import atexit
import json
import os
import runpy
import sys


def _report():
    r, b = sys.modules.get("ref2"), sys.modules.get("bofi")
    cfg = getattr(r, "CFG", {}) if r else {}
    cnt = getattr(r, "COUNTS", {}) if r else {}
    print("V3KT " + json.dumps({"ref2": getattr(r, "__file__", None), "bof_cfg": cfg.get("bof"), "hc_b_cfg": cfg.get("hc_b"),
                                "bofi": getattr(b, "__file__", None), "bofi_on": bool(b.ON[0]) if b else None,
                                "bofi_cnt": dict(b.CNT) if b else None, "hc_b_count": cnt.get("hc_b")}), flush=True)


atexit.register(_report)
screen = os.path.abspath(sys.argv[1])
sys.argv = [sys.argv[1]] + sys.argv[2:]
sys.path[0] = os.path.dirname(screen)
runpy.run_path(screen, run_name="__main__")
