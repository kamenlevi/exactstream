"""Build the self-contained r57_draft bundle for BigMoE's blob-1 freeze (MoonShots, Wed 18 Nov 2026).

Flat directory week98/draft/bundle_<code>/: every .py r57_draft (incl. PK) can import at run time (static AST closure over
week98{,/draft,/bet7}, ~/BigMoE/scripts, ~/TrainLab; no module name exists in two of those dirs), each file taken with
`git show <commit>:<path>` (MoonShots at the pinned code commit, BigMoE / TrainLab at their HEADs), then patched ONLY in
paths:
  * sys.path inserts of ~/MoonShots/week98{,/draft,/bet7}, ~/BigMoE/scripts, ~/TrainLab{,/sticky} -> the bundle dir
    (os.path.dirname(os.path.abspath(__file__)));
  * sys.path inserts of ~/Leap/scripts -> os.environ["R57_LEAP_SCRIPTS"] (only screened_head.py is used from there);
  * mtp125's ~/BigMoE + ~/SpecDecode inserts dropped (only its load() / samples() import from them; not on r57's path);
  * KDIR (the stock MLX kernel headers the bitwise replicas parse) -> the running mlx package's include dir.
Data files (weights, screen rank) keep their absolute paths; MANIFEST lists them with sha256.
usage: ~/BigMoE/.venv/bin/python -I -B make_bundle.py [--code b5c7c80] [--force]"""
import argparse
import ast
import difflib
import hashlib
import os
import re
import shutil
import subprocess as sp

H = os.path.expanduser
REPOS = {"MoonShots": H("~/MoonShots"), "BigMoE": H("~/BigMoE"), "TrainLab": H("~/TrainLab"), "Leap": H("~/Leap")}
FILES = [  # (repo, path in repo)
    ("MoonShots", "week98/draft/r57_draft.py"), ("MoonShots", "week98/draft/ra_sf.py"),
    ("MoonShots", "week98/draft/pk_level.py"), ("MoonShots", "week98/draft/pk_ops.py"),
    ("MoonShots", "week98/draft/pk_skel2.py"), ("MoonShots", "week98/draft/sf_tree.py"),
    ("MoonShots", "week98/draft/sf_tail.py"), ("MoonShots", "week98/draft/fta_kernel.py"),
    ("MoonShots", "week98/draft/fta_match.py"), ("MoonShots", "week98/draft/s22_format.py"),
    ("MoonShots", "week98/p3_dd_tree.py"),
    ("MoonShots", "week98/bet7/fl_exact.py"), ("MoonShots", "week98/bet7/fl_qmatch.py"),
    ("MoonShots", "week98/bet7/fl_hc.py"), ("MoonShots", "week98/bet7/fl_attn.py"),
    ("BigMoE", "scripts/tree16_draft.py"), ("BigMoE", "scripts/dd_ttt.py"), ("BigMoE", "scripts/dd_d22_ttt.py"),
    ("BigMoE", "scripts/dd_tree.py"), ("BigMoE", "scripts/dd_topk.py"), ("BigMoE", "scripts/dd_ceiling.py"),
    ("BigMoE", "scripts/dd_manifest.py"),
    ("TrainLab", "mtp125.py"),
]
LEAP_ENV = [("Leap", "scripts/screened_head.py")]       # read through R57_LEAP_SCRIPTS (not copied)
DATA = ["~/datasets/deepdraft/q8h32/t30b_q8h32.safetensors", "~/datasets/deepdraft/ckpt/dd_t30b.safetensors",
        "~/Leap/data/frspec_rank.npy"]
KHDRS = ["quantized.h", "gemv.h", "sdpa_vector.h"]
SELF = 'os.path.dirname(os.path.abspath(__file__))'
CALL = r'(?:H|os\.path\.expanduser)\("(~/[^"]*)"\)'
TO_SELF = ("~/MoonShots/week98", "~/MoonShots/week98/draft", "~/MoonShots/week98/bet7", "~/BigMoE/scripts",
           "~/TrainLab", "~/TrainLab/sticky")
DROP = ("~/BigMoE", "~/SpecDecode")
KDIR_OLD = 'KDIR = os.path.expanduser("~/BigMoE/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels")'
KDIR_NEW = ('KDIR = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")'
            '  # [bundle] the running mlx\'s headers (was ~/BigMoE/.venv/.../kernels)')


def git(repo, *a):
    return sp.run(["git", "-C", REPOS[repo], *a], capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def patch(src, name):
    out, notes = [], []
    for ln in src.splitlines(keepends=True):
        if "sys.path." in ln and "~/" in ln:
            m = re.search(CALL, ln)
            assert m, (name, ln)
            p = m.group(1)
            ind = ln[:len(ln) - len(ln.lstrip())]
            if p == "~/Leap/scripts":
                new = ln[:m.start()] + 'os.environ["R57_LEAP_SCRIPTS"]' + ln[m.end():]
            elif p in TO_SELF:
                new = ln[:m.start()] + SELF + ln[m.end():]
            elif p in DROP:
                new = f"{ind}# [bundle] dropped (only load() / samples() import from there): {ln.strip()}\n"
            else:
                raise SystemExit(f"{name}: unmapped sys.path entry {p}")
            out.append(new)
            notes.append(f"{p} -> {'env R57_LEAP_SCRIPTS' if p == '~/Leap/scripts' else 'dropped' if p in DROP else 'bundle dir'}")
        elif ln.strip() == KDIR_OLD:
            assert re.search(r"^import mlx\.core as mx", src, re.M), name
            out.append(KDIR_NEW + "\n")
            notes.append("KDIR -> running mlx's include dir")
        else:
            out.append(ln)
    new = "".join(out)
    for ln in new.splitlines():                          # nothing outside the bundle / env left on sys.path
        if "sys.path." in ln and not ln.lstrip().startswith("#"):
            assert "~/" not in ln and "/Users/" not in ln, (name, ln)
    ast.parse(new)
    return new, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="b5c7c80")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    code = git("MoonShots", "rev-parse", a.code).decode().strip()
    heads = {r: git(r, "rev-parse", "HEAD").decode().strip() for r in ("BigMoE", "TrainLab", "Leap")}
    out = H(f"~/MoonShots/week98/draft/bundle_{a.code}")
    if os.path.exists(out):
        assert a.force, f"{out} exists (--force to rebuild)"
        shutil.rmtree(out)
    os.makedirs(out)
    rows, diffs, notes_all = [], [], []
    names = set()
    for repo, path in FILES:
        ref = code if repo == "MoonShots" else heads[repo]
        src = git(repo, "show", f"{ref}:{path}")
        live = open(os.path.join(REPOS[repo], path), "rb").read()
        name = os.path.basename(path)
        assert name not in names, name
        names.add(name)
        new, notes = patch(src.decode(), name)
        nb = new.encode()
        open(os.path.join(out, name), "wb").write(nb)
        rows.append((name, sha(nb), f"~/{repo}/{path}@{ref[:10]}", sha(src), "live==commit" if live == src else "LIVE DIFFERS",
                     "; ".join(notes) or "verbatim"))
        if notes:
            diffs += difflib.unified_diff(src.decode().splitlines(keepends=True), new.splitlines(keepends=True),
                                          f"{repo}/{path}@{ref[:10]}", f"bundle/{name}")
    lrows = []
    for repo, path in LEAP_ENV:
        b = git(repo, "show", f"{heads[repo]}:{path}")
        first = git(repo, "log", "--format=%h", "--", path).decode().split()
        lrows.append((os.path.basename(path), sha(b), f"~/{repo}/{path}@{heads[repo][:10]} (only commit touching it: "
                      f"{', '.join(first)}; same bytes at b4331e4 / 3ec85e8 / 913250f)"))
        for c in ("b4331e4", "3ec85e8", "913250f"):
            assert git(repo, "show", f"{c}:{path}") == b, (path, c)
    import mlx.core as mx
    kdir = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")
    assert os.path.samefile(kdir, H("~/BigMoE/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels"))
    krows = [(h, sha(open(os.path.join(kdir, h), "rb").read())) for h in KHDRS]
    drows = []
    for d in DATA:
        hs = hashlib.sha256()
        with open(H(d), "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                hs.update(blk)
        drows.append((d, hs.hexdigest(), os.path.getsize(H(d))))
    vers = sp.run([H("~/BigMoE/.venv/bin/python"), "-I", "-c", "import importlib.metadata as m;"
                   "print(' '.join(f'{p} {m.version(p)}' for p in ('mlx','mtplx','numpy','tokenizers')))"],
                  capture_output=True, text=True, check=True).stdout.strip()
    open(os.path.join(out, "PATCHES.diff"), "w").write("".join(diffs))
    with open(os.path.join(out, "MANIFEST"), "w") as f:
        f.write(f"# r57_draft bundle at MoonShots code commit {code[:10]} (incl. PK: R57_TREE_PK=1 read by r57_draft)\n"
                f"# built by week98/draft/make_bundle.py; flat dir; import: sys.path.insert(0, <bundle>); import r57_draft\n"
                "# env needed: R57_LEAP_SCRIPTS = a dir holding screened_head.py with the sha256 below (KeyError if unset)\n"
                "# patches: paths only (PATCHES.diff); every other byte equals the source commit\n"
                "#\n# file\tsha256\tsource@commit\tsource_sha256\tlive\tpatch\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
        f.write("#\n# via env R57_LEAP_SCRIPTS (not copied): file\tsha256\tsource\n")
        for r in lrows:
            f.write("# " + "\t".join(r) + "\n")
        f.write(f"#\n# venv (not copied): {vers}; stock MLX kernel headers read at import (KDIR = running mlx):\n")
        for h, s in krows:
            f.write(f"#   {h}\t{s}\n")
        f.write("#\n# data read at run time (absolute paths kept, not copied): path\tsha256\tbytes\n")
        for d, s, n in drows:
            f.write(f"#   {d}\t{s}\t{n}\n")
        f.write("#\n# not bundled: mtp125.load() (bigmoe.loader) and mtp125.samples() (SpecDecode extract / render): "
                "test-harness only, never called by r57_draft\n"
                "# side effect kept as in the source: FMT 'stored' imports dd_ttt -> dd_d22_ttt -> mtp125, which sets "
                "MTPLX_ / TL_ env defaults at import (install provenance logs them)\n")
    print(f"bundle {out}: {len(rows)} files, {sum(1 for r in rows if r[5] != 'verbatim')} patched")
    for r in rows:
        print("  ", r[0], r[1][:12], r[2], r[4], r[5])


if __name__ == "__main__":
    main()
