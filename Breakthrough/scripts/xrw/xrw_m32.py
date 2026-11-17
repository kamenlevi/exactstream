"""xrw_m32.py (Breakthrough, Tue 17 Nov 2026 10:1x; EXPERT REWRITE, Leap's item e): Leap's ooo1c_m32_check.py scenarios
(layers 20 / 44 x COH on / off x rand, rand_2d, pop1, pop2, pop2_2d, rand_late: random routing, an expert on all 32 rows,
two experts on > 16 rows, 2-D nolsb masks, late release that the kernel must poll for) run UNCHANGED through
ref2._ooo_sel with the candidate switched on by the knob (xknob: install, MODE, ON). Leap's file is read and exec'd with
two edits only: its OUT points into ~/Breakthrough/data/xrw, and the knob is installed right after its imports.
Its lines per scenario: the 32-row call == two 16-row calls bit for bit, and the base posts cover every used expert;
a hang would stop it (the late scenarios poll the release region).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/xrw_m32.py <w4a8|tgl16>   (share lock ~3 GB)
  -> data/xrw/m32_<cand>.txt"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/Leap/scripts/week98/ooo1c_m32_check.py")
OUT_LINE = 'OUT = os.path.expanduser("~/Leap/data/week98/check25/ooo1c_m32_check.txt")'
HOOK_AT = "import ooo1c as OC\n"


def main():
    cand = sys.argv[1]
    assert cand in ("w4a8", "tgl16"), cand
    src = open(SRC).read()
    assert src.count(OUT_LINE) == 1 and src.count(HOOK_AT) == 1, "Leap's m32 check changed: re-read it before running"
    out = os.path.join(HERE, "..", "..", "data", "xrw", f"m32_{cand}.txt")
    src = src.replace(OUT_LINE, f"OUT = {os.path.abspath(out)!r}")
    src = src.replace(HOOK_AT, HOOK_AT + f"sys.path.insert(0, {HERE!r})\nimport xknob as XK\nXK.install()\n"
                      f"XK.MODE[0] = {cand!r}\nXK.ON[0] = True\n"
                      f"print('KNOB: ooo1cq.tier_glu_nax6n_1cq -> {cand} (xknob ON)', flush=True)\n")
    src = src.replace('say(f"OOO1C M32: {n - bad} of {n} scenarios PASS")',
                      'say(f"OOO1C M32 on the candidate {XK.MODE[0]} (knob calls {XK.COUNTS}): {n - bad} of {n} scenarios PASS")')
    g = {"__name__": "__main__", "__file__": SRC}
    try:
        exec(compile(src, SRC + " [xrw_m32: OUT + knob]", "exec"), g)
    except SystemExit as e:
        print(f"xrw_m32 {cand}: exit {e.code}", flush=True)
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
