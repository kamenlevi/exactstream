"""R-41D (docs/PREREG_r41d_2026-10-29.md): add the arms b12llfx2 / b12llff2 to evalsuite/cli.py and evalsuite/model.py.
Run once, after runner 27 (R-41C part B) has ended and Breakthrough has swapped bt_b12: it refuses while any evalsuite
stage or runner runs, and unless the live ~/Breakthrough/scripts/bt_b12.py hashes to the sha given (sha256[:10]).
Usage: .venv/bin/python tools/r41d_arms.py SHA"""
import hashlib, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM = "2700:60,8600:60,16500:56,32800:47"          # b12llff2's B12_FOLD_CM (amendment 1: Breakthrough's FFX points)


def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{os.path.relpath(path, ROOT)}: {n} matches for {old[:70]!r}: nothing written")
        s = s.replace(old, new)
    return s


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sha = sys.argv[1]
    live = hashlib.sha256(open(os.path.expanduser("~/Breakthrough/scripts/bt_b12.py"), "rb").read()).hexdigest()[:10]
    if live != sha:
        raise SystemExit(f"bt_b12.py is {live}, not {sha}: nothing written")
    ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    busy = [l for l in ps.splitlines() if "-m evalsuite.stages" in l or "day_runner27.sh" in l or "day_runner28.sh" in l]
    if busy:
        raise SystemExit(f"a stage or runner runs (never edit its code meanwhile): {busy[0][:120]}: nothing written")
    re_old = 'B12_RE = re.compile(r"(b12c192|'
    re_new = 'B12_RE = re.compile(r"(b12llfx2|b12llff2|b12c192|'
    cli = os.path.join(ROOT, "evalsuite", "cli.py")
    c = edit(cli, [
        ('"b12llfx": 14, "b12llff": 14}', '"b12llfx": 14, "b12llff": 14,\n          "b12llfx2": 14, "b12llff2": 14}'),
        ('                          B12_FOLD_AW="1", B12_FOLD_CM="1", B12_PFR_GRID="balanced")         # Kamen\'s call) = old fold-on\n',
         '                          B12_FOLD_AW="1", B12_FOLD_CM="1", B12_PFR_GRID="balanced")         # Kamen\'s call) = old fold-on\n'
         '# R-41D (docs/PREREG_r41d_2026-10-29.md): the fold arms + R-45\'s fix (B12_FRS_NOCYCLE frees each replaced FR-Spec\n'
         '# draft head at once) on Breakthrough\'s swapped bt_b12 (model.BT_B12_SHA); b12llff2 = FFX: C_m capped at 60\n'
         'B12_ENV["b12llfx2"] = dict(B12_ENV["b12llfx"], B12_FRS_NOCYCLE="1")\n'
         f'B12_ENV["b12llff2"] = dict(B12_ENV["b12llff"], B12_FOLD_CM="{CM}", B12_FRS_NOCYCLE="1")\n'),
        ('"B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV",',
         '"B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV", "B12_FRS_NOCYCLE",'),
        ('           "b12llfoff": True, "b12llfon": True, "b12llox": True, "b12llfx": True, "b12llff": True}\nB12_RE',
         '           "b12llfoff": True, "b12llfon": True, "b12llox": True, "b12llfx": True, "b12llff": True,\n'
         '           "b12llfx2": True, "b12llff2": True}\nB12_RE'),
        (re_old, re_new)])
    model = os.path.join(ROOT, "evalsuite", "model.py")
    m = edit(model, [
        ('       "b12llox": 0.10, "b12llfx": 0.10, "b12llff": 0.10}   # the fold confirmation (R-41): old shipped / fold-exact / fold-fast\n',
         '       "b12llox": 0.10, "b12llfx": 0.10, "b12llff": 0.10,   # the fold confirmation (R-41): old shipped / fold-exact / fold-fast\n'
         '       "b12llfx2": 0.10, "b12llff2": 0.10}        # R-41D: fold-exact / fold-fast (FFX) + R-45\'s fix (B12_FRS_NOCYCLE)\n'),
        ('"b12llfx": ("v3m", "1", "1"), "b12llff": ("v3m", "1", "1")}',
         '"b12llfx": ("v3m", "1", "1"), "b12llff": ("v3m", "1", "1"),\n'
         '               "b12llfx2": ("v3m", "1", "1"), "b12llff2": ("v3m", "1", "1")}'),
        ('                 "b12llff": ("balanced", "1", "1")}',
         '                 "b12llff": ("balanced", "1", "1"),\n'
         '                 "b12llfx2": ("keep", None, None),      # R-41D (docs/PREREG_r41d_2026-10-29.md, amendment 1)\n'
         f'                 "b12llff2": ("balanced", "1", "{CM}")}}'),
        ('"b12llox": "c3f957cba6", "b12llfx": "c3f957cba6", "b12llff": "c3f957cba6"}   # the code each arm\'s rows',
         f'"b12llox": "c3f957cba6", "b12llfx": "c3f957cba6", "b12llff": "c3f957cba6",   # the code each arm\'s rows\n'
         f'              "b12llfx2": "{sha}", "b12llff2": "{sha}"}}   # R-41D: Breakthrough\'s swap with R-45\'s fix'),
        ('           "b12llfoff": True, "b12llfon": True, "b12llox": True, "b12llfx": True, "b12llff": True}\nB12_RE',
         '           "b12llfoff": True, "b12llfon": True, "b12llox": True, "b12llfx": True, "b12llff": True,\n'
         '           "b12llfx2": True, "b12llff2": True}\nFRS_NOCYCLE = ("b12llfx2", "b12llff2")   # R-41D: R-45\'s fix on, asserted at load\nB12_RE'),
        (re_old, re_new),
        ('                    f"{v[\'name\']}: (B12_PFR_GRID, B12_FOLD_AW, B12_FOLD_CM) = {got}, want {FOLD_SWITCHES[v[\'base\']]}"\n',
         '                    f"{v[\'name\']}: (B12_PFR_GRID, B12_FOLD_AW, B12_FOLD_CM) = {got}, want {FOLD_SWITCHES[v[\'base\']]}"\n'
         '            if v["base"] in FRS_NOCYCLE:\n'
         '                assert os.environ.get("B12_FRS_NOCYCLE") == "1", \\\n'
         '                    f"{v[\'name\']}: B12_FRS_NOCYCLE={os.environ.get(\'B12_FRS_NOCYCLE\')!r}, want \'1\' (R-45\'s fix)"\n')])
    open(cli, "w").write(c)
    open(model, "w").write(m)
    print(f"cli.py and model.py: b12llfx2 / b12llff2 added (bt_b12 {sha})")


if __name__ == "__main__":
    main()
