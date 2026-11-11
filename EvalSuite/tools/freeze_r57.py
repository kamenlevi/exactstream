"""Freeze the outside code BigMoE's r57_screen_fast.py loads, so every FAST process of a run reads one version
(docs/PREREG_agentfast_2026-11-06.md, Fri 09:3x: Leap's ref2.py / d7b_nax.py changed between processes twice in an hour).
  python3 tools/freeze_r57.py <FZ dir> <reference manifest> <Leap commit> <BigMoE commit>
FZ/BigMoE/{bigmoe,scripts} = `git archive` of BigMoE at the commit, + ext/<build> copied (the .so); FZ/Leap/{scripts,
third_party} = `git archive` of Leap at the commit; FZ/TrainLab/{sticky,mtp125.py} and FZ/Breakthrough/keepset copied live.
In the frozen screen and every frozen .py file, code paths (the trees above) are rewritten to the copies; data paths
(models, data/, ckpt/, FableGod, EvalSuite) stay live. Then every module of the reference manifest (an earlier process's
loaded modules + sha256) must have a frozen copy with the same sha, or differ only by the rewritten path strings.
Writes FZ/FREEZE.json (sources, commits, per-module shas, the rewrites) and prints the screen path to run.
  python3 tools/freeze_r57.py now <FZ dir> <ext build>
S4's freeze (docs/PREREG_s4_2026-11-07.md: the code current at S4's start, no reference to verify against): the same
trees copied LIVE (BigMoE/{bigmoe,scripts}, Leap/{scripts,third_party}; no __pycache__) and the same rewrites. FREEZE.json
records each repo's HEAD, `git status` of the copied paths and every frozen file's sha256 (a dirty tree's diff goes to
FZ/<repo>.diff). The run's first process manifest is then the reference for the later ones."""
import hashlib, json, os, re, shutil, subprocess, sys, tarfile, io

H = os.path.expanduser("~")
sha = lambda b: hashlib.sha256(b).hexdigest()[:12]
IGN = shutil.ignore_patterns("__pycache__", "*.pyc")


def archive(repo, commit, paths, dest):
    tar = subprocess.run(["git", "-C", repo, "archive", commit, *paths], capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(dest)


def copy_side(fz, ext_build):
    """the ext build's .so, TrainLab and Breakthrough: copied live in both modes"""
    so_dir = f"{H}/BigMoE/ext/{ext_build}"
    os.makedirs(f"{fz}/BigMoE/ext/{ext_build}")
    for f in os.listdir(so_dir):
        if f.endswith(".so"):
            shutil.copy2(f"{so_dir}/{f}", f"{fz}/BigMoE/ext/{ext_build}/{f}")
    os.makedirs(f"{fz}/TrainLab")
    shutil.copytree(f"{H}/TrainLab/sticky", f"{fz}/TrainLab/sticky", ignore=IGN)
    shutil.copy2(f"{H}/TrainLab/mtp125.py", f"{fz}/TrainLab/mtp125.py")
    os.makedirs(f"{fz}/Breakthrough")
    shutil.copytree(f"{H}/Breakthrough/keepset", f"{fz}/Breakthrough/keepset", ignore=IGN)


def rewrite(fz):
    """the code-path rewrites in the frozen screen and every .py file of the frozen trees -> (rewrites, files rewritten)"""
    # (quoted string as written in the sources) -> the frozen copy; longest first
    rw = [('"~/Leap/scripts"', f'"{fz}/Leap/scripts"'), ('"~/Leap/third_party/ms_gdn"', f'"{fz}/Leap/third_party/ms_gdn"'),
          ('"~/Leap/third_party"', f'"{fz}/Leap/third_party"'), ('"~/TrainLab/sticky"', f'"{fz}/TrainLab/sticky"'),
          ('"~/TrainLab"', f'"{fz}/TrainLab"'), ('"/Users/kamen/Breakthrough/keepset"', f'"{fz}/Breakthrough/keepset"'),
          ('"~/Breakthrough/keepset"', f'"{fz}/Breakthrough/keepset"'), ('"~/BigMoE"', f'"{fz}/BigMoE"'),
          ('"~/BigMoE/scripts/', f'"{fz}/BigMoE/scripts/')]
    targets = [f"{fz}/BigMoE/scripts/r57_screen_fast.py"]   # every .py file of the frozen trees: a frozen module the
    for top in ("BigMoE", "Leap", "TrainLab", "Breakthrough"):   # reference does not load (e.g. tierglu_3cn on the 3c base)
        for dp, _, fs in os.walk(f"{fz}/{top}"):                 # must not put a live path on sys.path either (Fri 12:2x)
            targets += [os.path.join(dp, f) for f in fs if f.endswith(".py")]
    done = {}
    for t in sorted(set(targets)):
        if not os.path.exists(t):
            continue
        s = open(t).read()
        n_ = {}
        for a, b in rw:
            c = s.count(a)
            if c:
                s = s.replace(a, b)
                n_[a] = c
        if n_:
            open(t, "w").write(s)
            done[t[len(fz) + 1:]] = n_
    return rw, done


def main(fz, refman, leap_c, bm_c):
    assert not os.path.exists(fz), f"{fz} exists: pick a new directory"
    ref = json.load(open(refman))["modules"]
    ext_build = next((os.path.basename(os.path.dirname(f)) for f, _ in ref.values() if f.endswith(".so") and "/ext/" in f), None)
    os.makedirs(fz)
    archive(f"{H}/BigMoE", bm_c, ["bigmoe", "scripts"], f"{fz}/BigMoE")
    archive(f"{H}/Leap", leap_c, ["scripts", "third_party"], f"{fz}/Leap")
    copy_side(fz, ext_build)
    rw, done = rewrite(fz)
    live_of = {f"{fz}/BigMoE/": f"{H}/BigMoE/", f"{fz}/Leap/": f"{H}/Leap/", f"{fz}/TrainLab/": f"{H}/TrainLab/",
               f"{fz}/Breakthrough/": f"{H}/Breakthrough/"}
    frozen_of = {v: k for k, v in live_of.items()}
    # verify: every reference module under the frozen trees has a frozen copy equal to it, or equal after the rewrites
    rows, bad = {}, []
    for name, (f, s) in sorted(ref.items()):
        lp = next((lp for lp in frozen_of if f.startswith(lp)), None)
        if lp is None:
            rows[name] = dict(path=f, sha=s, frozen=None)            # EvalSuite / MoonShots / venv: stays live (pinned)
            continue
        fp = frozen_of[lp] + f[len(lp):]
        if not os.path.exists(fp):
            bad.append(f"{name}: no frozen copy {fp}")
            continue
        b = open(fp, "rb").read()
        if sha(b) == s:
            rows[name] = dict(path=fp, sha=s, live=f, rewritten=False)
            continue
        txt = b.decode()                                     # equal to the reference after undoing the rewrites?
        for a, r in rw:
            txt = txt.replace(r, a)
        ok = sha(txt.encode()) == s
        rows[name] = dict(path=fp, sha=sha(b), ref_sha=s, live=f, rewritten=True, equal_modulo_paths=ok)
        if not ok:
            bad.append(f"{name}: frozen {sha(b)} != reference {s} even modulo the path rewrites")
    out = dict(fz=fz, refman=refman, leap=leap_c, bigmoe=bm_c, ext_build=ext_build, rewrites=done, modules=rows,
               screen=f"{fz}/BigMoE/scripts/r57_screen_fast.py",
               screen_sha=sha(open(f"{fz}/BigMoE/scripts/r57_screen_fast.py", "rb").read()))
    json.dump(out, open(f"{fz}/FREEZE.json", "w"), indent=1, sort_keys=True)
    print(f"frozen {sum(1 for r in rows.values() if r.get('frozen', 1) is not None)} modules in {fz}; live (pinned): "
          f"{sorted(n for n, r in rows.items() if r.get('frozen', 1) is None)}")
    print(f"rewritten files: {done}")
    if bad:
        print("FREEZE FAILED:", bad)
        sys.exit(1)
    print(f"screen: {out['screen']} ({out['screen_sha']})")


def main_now(fz, ext_build):
    assert not os.path.exists(fz), f"{fz} exists: pick a new directory"
    assert os.path.isdir(f"{H}/BigMoE/ext/{ext_build}"), f"no ext build {ext_build}"
    os.makedirs(fz)
    src = {"BigMoE": ["bigmoe", "scripts"], "Leap": ["scripts", "third_party"], "TrainLab": ["sticky", "mtp125.py"],
           "Breakthrough": ["keepset"]}
    for repo in ("BigMoE", "Leap"):
        for p in src[repo]:
            shutil.copytree(f"{H}/{repo}/{p}", f"{fz}/{repo}/{p}", ignore=IGN)
    copy_side(fz, ext_build)
    prov = {}
    for repo, paths in src.items():                          # what the live trees were at the freeze
        git = lambda *a: subprocess.run(["git", "-C", f"{H}/{repo}", *a], capture_output=True).stdout
        diff = git("diff", "HEAD", "--binary", "--", *paths)
        if diff:
            open(f"{fz}/{repo}.diff", "wb").write(diff)
        prov[repo] = dict(head=git("rev-parse", "--short", "HEAD").decode().strip(), paths=paths, diff=bool(diff),
                          status=git("status", "--porcelain", "--", *paths).decode().splitlines())
    rw, done = rewrite(fz)
    files = {}
    for dp, _, fs in os.walk(fz):
        for f in fs:
            p = os.path.join(dp, f)
            files[p[len(fz) + 1:]] = sha(open(p, "rb").read())
    screen = f"{fz}/BigMoE/scripts/r57_screen_fast.py"
    out = dict(fz=fz, mode="now", ext_build=ext_build, sources=prov, rewrites=done, files=files, screen=screen,
               screen_sha=files["BigMoE/scripts/r57_screen_fast.py"])
    json.dump(out, open(f"{fz}/FREEZE.json", "w"), indent=1, sort_keys=True)
    print(f"froze {len(files)} files in {fz} (live trees, no reference); " + "; ".join(
        f"{r} {v['head']}" + (f" DIRTY ({len(v['status'])} paths)" if v["status"] else "") for r, v in prov.items()))
    print(f"rewritten files: {len(done)}")
    print(f"screen: {screen} ({out['screen_sha']})")


if __name__ == "__main__":
    main_now(*sys.argv[2:4]) if sys.argv[1] == "now" else main(*sys.argv[1:5])
