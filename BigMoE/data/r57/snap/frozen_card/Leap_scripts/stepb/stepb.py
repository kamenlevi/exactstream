"""STEP 0 PART B (Leap, Fri 2026-11-06 07:4x; the orchestrator 07:2x: Kamen made the dense-kernel rebuild the top
priority). zero's FLAT-2 harness (copied from ~/FableGod/zero/build_1105, unchanged but for the probe) on BigMoE's
r57_screen.py at its HEAD, with one more insertion: `import stepb_k; stepb_k.patch()` right after the harness's
`import evalsuite.env`, before anything imports mlx. The probe (flat2_probe.py here) adds kernel-level synced forwards
(STEPB_KREP at the live check's 2 rows and at 16 rows), and the driver ends with stepb_k.calibrate() (round trip,
dependent-dispatch cost, read floor). Read-out: stepb_read.py on out/<tag>_flat2.json.
Usage (inside a quiet lock): [BigMoE venv] <the arm's env> python stepb.py C   (run_stepb.sh sets it all)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import flat2_driver
import flat2_patch

if os.environ.get("STEPB_QOS") == "1":                   # S2c (Sat 01:1x): the decode thread at QOS user-interactive (as
    import ctypes                                        # BigMoE's loader_main), so the scheduler keeps it on a P-core
    _qrc = ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(0x21, 0)
    print(f"[stepb] main thread QOS user-interactive: rc {_qrc}", flush=True)

assert os.environ.get("R57_PRESET") == "45", "step B runs on the 45 GB preset: use run_stepb.sh (or set the arm's env)"
F2_CFG = flat2_driver.config()
os.makedirs(F2_CFG["out"], exist_ok=True)
os.environ.setdefault("R57_TAG", F2_CFG["tag"])
os.environ.setdefault("R57_KEYS", F2_CFG["key"])
F2_GEN, F2_INFO = flat2_patch.build(open(F2_CFG["screen"]).read())
_anc = [m.start() for m in re.finditer(r"(?m)^import evalsuite\.env\b.*$", F2_GEN)]
assert len(_anc) == 1, f"step B: 'import evalsuite.env' found {len(_anc)} times at column 0 in the harness"
_eol = F2_GEN.index("\n", _anc[0])
F2_GEN = F2_GEN[:_eol + 1] + "import stepb_k as _SBK; _SBK.patch()        # Leap step B: wrap mx before any mlx import\n" \
    + "_SBK.preload('d7b_nax', os.environ.get('STEPB_D7BNAX'))   # Leap: a candidate drafter module, not installed\n" \
    + F2_GEN[_eol + 1:]
_r2 = [m.start() for m in re.finditer(r"(?m)^import ref2\b.*$", F2_GEN)]
assert len(_r2) >= 1, "step B: no 'import ref2' at column 0 in the harness"
_eol2 = F2_GEN.index("\n", _r2[0])
F2_GEN = F2_GEN[:_eol2 + 1] + ("if os.environ.get('STEPB_ATT_SINGLE') == '1':   # Leap step B: R-62's single-call SDPA (FAST)\n"
                               "    ref2.CFG['att_single'] = True\n"
                               "if os.environ.get('STEPB_KEEP_REF2C') == '1':   # Leap S2c 1e: the loop's per-cycle ref2.remove()\n"
                               "    def _soft_remove(_r=ref2):                  # switches REF2 off but keeps its per-weight caches\n"
                               "        _r._ON[0] = False                       # (_DN / _SN / _HC8: identity-validated, so the same bits)\n"
                               "        _r._CARRY[0] = None\n"
                               "    ref2.remove = _soft_remove\n"
                               "    print('[stepb] STEPB_KEEP_REF2C: ref2.remove() keeps _DN / _SN / _HC8', flush=True)\n"
                               "if os.environ.get('STEPB_WIRED') == '1':   # Leap PLAN 220: v1' + wired (R57_WIRED's mx.set_wired_limit)\n"
                               "    import mlx.core as _mxw\n"
                               "    print('[stepb] STEPB_WIRED: set_wired_limit, was', _mxw.set_wired_limit(int(_mxw.device_info()['max_recommended_working_set_size'])), flush=True)\n") + F2_GEN[_eol2 + 1:]
_sb = "(4 if SB_LO <= int(rows) <= 16 else 0)"            # r57's SB cadence (MTPLX's mid-loop async_eval every 4 layers)
if F2_GEN.count(_sb) == 1:
    F2_GEN = F2_GEN.replace(_sb, "(STEPB_SBN[0] if SB_LO <= int(rows) <= 16 else 0)")
else:                                                    # r57_screen_fast.py (PLAN 220): its cadence is R57_SBN's already
    assert F2_GEN.count("(SBN if SB_LO <= int(rows) <= 16 else 0)") == 1, "step B: no SB cadence expression (old or R57_SBN form)"
_e2 = F2_GEN.index("\n", F2_GEN.index("_SBK.preload("))
F2_GEN = F2_GEN[:_e2 + 1] + "STEPB_SBN = [int(os.environ.get('STEPB_SB_N', '4'))]   # Leap step B: the SB cadence, toggleable\n" + F2_GEN[_e2 + 1:]
F2_INFO["stepb"] = "stepb_k.patch() after import evalsuite.env; STEPB_ATT_SINGLE after import ref2; SB cadence STEPB_SBN"
F2_GENPATH = os.path.join(F2_CFG["out"], f"{F2_CFG['tag']}_generated.py")
open(F2_GENPATH, "w").write(F2_GEN)
F2ON, F2H, F2C = [False], None, []
sys.argv = [F2_GENPATH, sys.argv[1] if len(sys.argv) > 1 else "208", str(F2_CFG["cap"])]
exec(compile(F2_GEN, F2_GENPATH, "exec"), globals())
sys.exit(flat2_driver.main(globals(), F2_CFG, F2_INFO))
