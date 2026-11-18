"""ADAPTK-SAT: ADAPTK-LOSSY re-priced at Saturday's measured costs (NewIdeas, Wed 18 Nov). Imports adaptk_lossy unchanged, swaps
COSTS, writes adaptk_sat.txt / .json beside this file (the original adaptk_lossy outputs are untouched).
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_sat.py Models (ms): check16 = 42.7 (slot #3 no-tier upper bound 45.73 minus REF2 ~3) or 38.0;
check32 = check16 x 1.458 (E14 ratio, as adaptk_lossy's v4 models); overhead 3; draft per level L: Q8H32 SF 0.94
(MoonShots ra_sf.py), PK 0.80 realistic / 0.76 best (MoonShots' 11-18 census of the live Q8H32 pop: 178.8 MB per pop at the ~260 GB/s cold-GEMV cap
is >= 0.69 ms, so the earlier 0.66 was below this format's floor), c2 SF 0.59 (draft cost only; c2's own kept loss not modelled).
'h' models charge L per tree level (the replay's height); 'p' models charge L per pop (15 / 31), folded into T."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adaptk_lossy as A
R = 1.4372 * 1.04 / 1.025
def m(c16, L, pops):
    c32 = c16 * R
    if pops:
        return (c16 + 3 + 15 * L, c32 + 3 + 31 * L, 0.0)
    return (c16 + 3, c32 + 3, L)
A.COSTS = {"v4pass": A.COSTS["v4pass"],
           "sat_q8_h": m(42.7, 0.94, False), "sat_q8_p": m(42.7, 0.94, True),
           "sat_q8lo_h": m(38.0, 0.94, False),
           "sat_pk80_h": m(42.7, 0.80, False), "sat_pk76_h": m(42.7, 0.76, False), "sat_c2_h": m(42.7, 0.59, False), "sat_c2_p": m(42.7, 0.59, True)}
import tempfile; _T = tempfile.mkdtemp(); A.HERE = _T
A.main()
import shutil; HERE = os.path.dirname(os.path.abspath(__file__))
shutil.move(os.path.join(_T, "adaptk_lossy.txt"), os.path.join(HERE, "adaptk_sat.txt")); shutil.move(os.path.join(_T, "adaptk_lossy.json"), os.path.join(HERE, "adaptk_sat.json"))
