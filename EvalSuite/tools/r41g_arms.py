"""b12llff3 (docs/PREREG_b12llff3_2026-10-31.md): add the arm to evalsuite/cli.py and evalsuite/model.py.
b12llff3 = b12llff2's env (fold-fast, FFX C_m points, R-45's fix) on Breakthrough's live bt_b12 with its default engine
v3o (set explicitly: cli.SHIP_AR would otherwise force v3f) and B12_PFR_MIN=512 (prompts over 512 tokens release the
pools while they are read). Run once, while no evalsuite stage or runner runs, and only if the live
~/Breakthrough/scripts/bt_b12.py hashes to the sha given (sha256[:10]). Usage: .venv/bin/python tools/r41g_arms.py SHA"""
import hashlib, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM = "2700:60,8600:60,16500:56,32800:47"          # b12llff2's B12_FOLD_CM (Breakthrough's FFX points)


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
    busy = [l for l in ps.splitlines() if "-m evalsuite.stages" in l or "tools/day_runner3" in l]
    if busy:
        raise SystemExit(f"a stage or runner runs (never edit its code meanwhile): {busy[0][:120]}: nothing written")
    re_old = 'B12_RE = re.compile(r"(b12llfx2|b12llff2|'
    re_new = 'B12_RE = re.compile(r"(b12llff3|b12llfx2|b12llff2|'
    cli = os.path.join(ROOT, "evalsuite", "cli.py")
    c = edit(cli, [
        ('          "b12llfx2": 14, "b12llff2": 14}', '          "b12llfx2": 14, "b12llff2": 14, "b12llff3": 14}'),
        (f'B12_ENV["b12llff2"] = dict(B12_ENV["b12llff"], B12_FOLD_CM="{CM}", B12_FRS_NOCYCLE="1")\n',
         f'B12_ENV["b12llff2"] = dict(B12_ENV["b12llff"], B12_FOLD_CM="{CM}", B12_FRS_NOCYCLE="1")\n'
         '# b12llff3 (docs/PREREG_b12llff3_2026-10-31.md, decision 5): b12llff2 on the live bt_b12 (v3o, M-01 by default) +\n'
         '# B12_PFR_MIN 512 (pools released while prompts over 512 tokens are read)\n'
         'B12_ENV["b12llff3"] = dict(B12_ENV["b12llff2"], B12_ENGINE="v3o", B12_PFR_MIN="512")\n'),
        ('"B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV", "B12_FRS_NOCYCLE",',
         '"B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV", "B12_FRS_NOCYCLE", "B12_PFR_MIN",'),
        ('           "b12llfx2": True, "b12llff2": True}\nB12_RE', '           "b12llfx2": True, "b12llff2": True, "b12llff3": True}\nB12_RE'),
        (re_old, re_new)])
    model = os.path.join(ROOT, "evalsuite", "model.py")
    m = edit(model, [
        ('       "b12llfx2": 0.10, "b12llff2": 0.10}        # R-41D: fold-exact / fold-fast (FFX) + R-45\'s fix (B12_FRS_NOCYCLE)\n',
         '       "b12llfx2": 0.10, "b12llff2": 0.10,        # R-41D: fold-exact / fold-fast (FFX) + R-45\'s fix (B12_FRS_NOCYCLE)\n'
         '       "b12llff3": 0.10}                          # decision 5: b12llff2 on v3o + B12_PFR_MIN 512\n'),
        ('               "b12llfx2": ("v3m", "1", "1"), "b12llff2": ("v3m", "1", "1")}',
         '               "b12llfx2": ("v3m", "1", "1"), "b12llff2": ("v3m", "1", "1"), "b12llff3": ("v3o", "1", "1")}'),
        (f'                 "b12llff2": ("balanced", "1", "{CM}")}}',
         f'                 "b12llff2": ("balanced", "1", "{CM}"),\n'
         f'                 "b12llff3": ("balanced", "1", "{CM}")}}'),
        ('              "b12llfx2": "ce6d4c63d2", "b12llff2": "ce6d4c63d2"}   # R-41D: Breakthrough\'s swap with R-45\'s fix',
         '              "b12llfx2": "ce6d4c63d2", "b12llff2": "ce6d4c63d2",   # R-41D: Breakthrough\'s swap with R-45\'s fix\n'
         f'              "b12llff3": "{sha}"}}   # decision 5: the live file with v3o + B12_PFR_MIN'),
        ('           "b12llfx2": True, "b12llff2": True}\nFRS_NOCYCLE = ("b12llfx2", "b12llff2")   # R-41D: R-45\'s fix on, asserted at load',
         '           "b12llfx2": True, "b12llff2": True, "b12llff3": True}\n'
         'FRS_NOCYCLE = ("b12llfx2", "b12llff2", "b12llff3")   # R-41D: R-45\'s fix on, asserted at load\n'
         'PFR_MIN = {"b12llff3": "512"}             # decision 5: B12_PFR_MIN, asserted at load'),
        (re_old, re_new),
        ('                    f"{v[\'name\']}: B12_FRS_NOCYCLE={os.environ.get(\'B12_FRS_NOCYCLE\')!r}, want \'1\' (R-45\'s fix)"\n',
         '                    f"{v[\'name\']}: B12_FRS_NOCYCLE={os.environ.get(\'B12_FRS_NOCYCLE\')!r}, want \'1\' (R-45\'s fix)"\n'
         '            assert os.environ.get("B12_PFR_MIN") == PFR_MIN.get(v["base"]), \\\n'
         '                f"{v[\'name\']}: B12_PFR_MIN={os.environ.get(\'B12_PFR_MIN\')!r}, want {PFR_MIN.get(v[\'base\'])!r}"\n')])
    open(cli, "w").write(c)
    open(model, "w").write(m)
    print(f"cli.py and model.py: b12llff3 added (bt_b12 {sha})")


if __name__ == "__main__":
    main()
