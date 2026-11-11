"""Freeze Leap's REF2 at a commit into snap/leap-<sha>/, the code a REF2 variant's rows are made on (model.ref2_import).

  .venv/bin/python tools/snap_leap.py <leap-commit> "<what it is, why frozen>" [--with mod,mod]

Files come from the commit (git show), not from Leap's working tree: scripts/ref2.py and every scripts/ module it
imports at module level, followed through their own module-level imports; and third_party/ms_gdn/ (MoonShots' exact GDN
files, which ref2's CFG gdn_exact imports by absolute path; ref2_import imports the snapshot's copy first). Lazy imports
inside functions are listed in SOURCE, not copied: those of ~/Leap's live tree must stay unused (tier_nax4 off).
--with copies the named lazy modules too, again followed through their module-level imports, for a path that uses them
(the records3 conversion's checks: tierglu_3cn, which ref2's layout router imports for converted layers).
Then BigMoE's Python imports the snapshot through model.ref2_import, which must take ref2 and its modules from it."""
import ast, os, subprocess, sys, time

LEAP = os.path.expanduser("~/Leap")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*a):
    return subprocess.run(["git", "-C", LEAP, *a], capture_output=True, text=True, check=True).stdout


def imports(src):
    """-> (module-level imported top names, names imported inside functions)"""
    top, lazy = set(), set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            (top if node in tree.body else lazy).update(n.split(".")[0] for n in names)
    return top, lazy - top


def main():
    sha, why = git("rev-parse", "--short=7", sys.argv[1]).strip(), sys.argv[2]
    extra = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[3] == "--with" else []
    out = os.path.join(ROOT, "snap", f"leap-{sha}")
    assert not os.path.exists(out), f"{out} exists"
    have = {os.path.basename(p)[:-3] for p in git("ls-tree", "--name-only", sha, "scripts/").split() if p.endswith(".py")}
    assert set(extra) <= have, f"not in {sha}'s scripts/: {sorted(set(extra) - have)}"
    files, todo, lazy = {}, ["ref2", *extra], set()
    while todo:
        m = todo.pop()
        if m in files:
            continue
        files[m] = git("show", f"{sha}:scripts/{m}.py")
        top, lz = imports(files[m])
        todo += sorted(top & have)
        lazy |= lz & have
    gdn = [p for p in git("ls-tree", "--name-only", sha, "third_party/ms_gdn/").split()]
    os.makedirs(os.path.join(out, "ms_gdn"))
    for m, src in files.items():
        open(os.path.join(out, f"{m}.py"), "w").write(src)
    for p in gdn:
        open(os.path.join(out, "ms_gdn", os.path.basename(p)), "w").write(git("show", f"{sha}:{p}"))
    lazy -= set(files)
    open(os.path.join(out, "SOURCE"), "w").write(
        f"source ~/Leap {sha} scripts/{{{','.join(sorted(files))}}}.py + third_party/ms_gdn/ "
        f"({' '.join(sorted(os.path.basename(p) for p in gdn))}), from the commit (git show). {why}. "
        + (f"Copied with --with (lazy imports a path uses): {', '.join(extra)}. " if extra else "")
        + f"Lazy imports of ~/Leap/scripts not copied (must stay unused): {', '.join(sorted(lazy)) or 'none'}. "
        f"Copied {time.strftime('%Y-%m-%d %H:%M')} by tools/snap_leap.py.\n")
    check = (f"import sys, os; sys.path.insert(0, {ROOT!r}); import evalsuite.model as EM\n"
             f"ref2 = EM.ref2_import(dict(ref2=True, ref2_code='leap-{sha}'))\n"
             f"import importlib\nfor m in {extra!r}: assert importlib.import_module(m).__file__.startswith({out!r}), m\n"
             f"leap = sorted(n for n, m in sys.modules.items() if (getattr(m, '__file__', None) or '').startswith({LEAP!r}))\n"
             f"snap = sorted(n for n, m in sys.modules.items() if (getattr(m, '__file__', None) or '').startswith({out!r}))\n"
             f"print('from ~/Leap:', leap or 'none'); print('from the snapshot:', snap); print('CFG', ref2.CFG)\n"
             f"assert not leap and 'ref2' in snap")
    r = subprocess.run([os.path.expanduser("~/BigMoE/.venv/bin/python"), "-c", check], capture_output=True, text=True)
    print(open(os.path.join(out, "SOURCE")).read() + r.stdout + r.stderr[-2000:])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
