#!/usr/bin/env python3
"""make_release.py <snap> [--commit C] [--out DIR] [--no-commit]

Builds the engine package (RELEASE.md section 4) as a fresh git repository, from:
  - BigMoE's frozen snapshot <snap> (deep_freeze.sh: data/r57/snap/<snap> + frozen_<snap>) and bigmoe/, ext/, two
    by-path modules and the pool-size table, taken READ-ONLY with `git archive` of the commit that holds the snapshot;
  - EvalSuite's harness package at its HEAD (git archive) + the item files and reference rows the decode loop reads
    (untracked there, copied and md5-recorded);
  - ~/Leap/release/src (run.py, install.sh, README, configs, tools) and two small data files.
Then: paths rewritten to @ROOT@ (install.sh writes the clone's absolute path), comment-safe scrub (the public snapshot's
rules), leak check, py_compile of every .py, MANIFEST.json (every file, sha256, source). No engine code is edited: only
path strings, comments and docstrings change (the scrub reports each one).
The module list is the union of every R57_MODMAN manifest BigMoE's runs wrote (data/r57/*_modules.json), which covers
every config ever run, plus what MODMAN does not see (bigmoe/, evalsuite/, scripts/copygate.py, scripts/relax_core.py)."""
import argparse, hashlib, io, json, os, re, shutil, subprocess, sys, tarfile, tokenize

HOME = os.path.expanduser("~")
BM, ES, LEAP = f"{HOME}/BigMoE", f"{HOME}/EvalSuite", f"{HOME}/Leap"
SRC = f"{LEAP}/release/src"

ap = argparse.ArgumentParser()
ap.add_argument("snap")
ap.add_argument("--commit", help="BigMoE commit holding the snapshot (default: the last one that touched it)")
ap.add_argument("--out", default=f"{LEAP}/release/out/pkg")
ap.add_argument("--no-commit", action="store_true")
ap.add_argument("--refs", help="JSON {config: tokens.json path} for verify/ (default: the PF-D-era reference files)")
ap.add_argument("--es-commit", default="HEAD", help="EvalSuite commit for harness/evalsuite (default HEAD); a docs-only "
                "rebuild pins the commit the clean-clone run tested, so the harness stays byte-identical")
ap.add_argument("--extra-snap", action="append", default=[], help="another snapshot's decode loop at the same BigMoE commit "
                "(e.g. w24, the 24 GB configs' SNAPSHOT=w24): engine/loop/r57_screen_<name>.py; its other modules must equal "
                "the main snapshot's")
a = ap.parse_args()
S = a.snap
XS = a.extra_snap


def sh(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


C = a.commit or sh("git", "-C", BM, "log", "-1", "--format=%H", "--", f"data/r57/snap/frozen_{S}")
assert C, f"no BigMoE commit holds data/r57/snap/frozen_{S}"
C = sh("git", "-C", BM, "rev-parse", C)
EC = sh("git", "-C", ES, "rev-parse", a.es_commit)
OUT = os.path.abspath(a.out)
STAGE = OUT + ".stage"
for d in (OUT, STAGE):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs(STAGE)
WARN = []

# ---- 1. read-only extraction (git archive of committed trees) ----
BM_PATHS = ["bigmoe", "ext/CMakeLists.txt", "ext/bigmoe_ext.cpp", "data/seed45/laysz45_256.json", "scripts/copygate.py",
            "scripts/relax_core.py", "scripts/repack_records.py", "scripts/test_records3.py",
            f"data/r57/snap/{S}", f"data/r57/snap/frozen_{S}"] + [f"data/r57/snap/{x}" for x in XS]
os.makedirs(f"{STAGE}/bm")
tar = subprocess.run(["git", "-C", BM, "archive", C, *BM_PATHS], check=True, capture_output=True).stdout
tarfile.open(fileobj=io.BytesIO(tar)).extractall(f"{STAGE}/bm", filter="data")
os.makedirs(f"{STAGE}/es")
tar = subprocess.run(["git", "-C", ES, "archive", EC, "evalsuite"], check=True, capture_output=True).stdout
tarfile.open(fileobj=io.BytesIO(tar)).extractall(f"{STAGE}/es", filter="data")
for p in ("bigmoe", "ext/bigmoe_ext.cpp", "scripts/copygate.py", "scripts/relax_core.py"):   # live == committed?
    if sh("git", "-C", BM, "diff", "--stat", C, "--", p):
        WARN.append(f"BigMoE {p}: the working tree differs from {C[:8]} (the package takes the commit)")
if sh("git", "-C", ES, "status", "--short", "evalsuite"):
    WARN.append("EvalSuite evalsuite/: uncommitted changes (the package takes HEAD)")

B = f"{STAGE}/bm/data/r57/snap"
FZ = f"{B}/frozen_{S}"
assert os.path.isfile(f"{B}/{S}/r57_screen_{S}.py"), f"no r57_screen_{S}.py in the snapshot"

# ---- 2. the module closure: union of every R57_MODMAN manifest ----
mods = set()
for f in sorted(os.listdir(f"{BM}/data/r57")):
    if f.endswith("_modules.json"):
        try:
            mods |= {m for m, _ in json.load(open(f"{BM}/data/r57/{f}")) if m != "__main__"}
        except Exception:
            pass
ROOTS = [("Leap_third_party/ms_gdn", "kernels/ms_gdn"), ("Leap_third_party", "third_party"), ("Leap_scripts", "kernels"),
         ("TrainLab_sticky", "drafters/sticky"), ("TrainLab_root", "drafters"), ("Breakthrough_keepset", "keepset")]
os.makedirs(OUT)
placed, missing = {}, []
for m in sorted(mods):
    hit = None
    for sub, dst in ROOTS:
        p = f"{FZ}/{sub}/{m}.py"
        if os.path.isfile(p):
            hit = (p, f"{dst}/{m}.py"); break
    if hit is None and os.path.isfile(f"{B}/{S}/{m}.py"):
        hit = (f"{B}/{S}/{m}.py", f"engine/loop/{m}.py")
    if hit is None:
        missing.append(m); continue
    placed[hit[1]] = (hit[0], f"BigMoE {C[:8]} data/r57/snap/" + os.path.relpath(hit[0], B))
for f in sorted(os.listdir(f"{B}/{S}")):                      # the decode loop and its siblings
    if f.endswith(".py"):
        placed[f"engine/loop/{f}"] = (f"{B}/{S}/{f}", f"BigMoE {C[:8]} data/r57/snap/{S}/{f}")
for x in XS:                                                   # extra decode loops (--extra-snap): the loop only; the package
    assert os.path.isfile(f"{B}/{x}/r57_screen_{x}.py"), f"no r57_screen_{x}.py in snapshot {x}"   # has one set of siblings
    for f in sorted(os.listdir(f"{B}/{x}")):
        if f.endswith(".py") and f != f"r57_screen_{x}.py":
            assert os.path.isfile(f"{B}/{S}/{f}") and open(f"{B}/{x}/{f}", "rb").read() == open(f"{B}/{S}/{f}", "rb").read(), \
                f"--extra-snap {x}: {f} differs from snapshot {S}'s (the package ships one copy)"
    placed[f"engine/loop/r57_screen_{x}.py"] = (f"{B}/{x}/r57_screen_{x}.py", f"BigMoE {C[:8]} data/r57/snap/{x}/r57_screen_{x}.py")
for p in BM_PATHS[:8]:
    src = f"{STAGE}/bm/{p}"
    if os.path.isdir(src):
        for r, _, fs in os.walk(src):
            for f in fs:
                if "__pycache__" in r or f.endswith(".pyc"):
                    continue
                rel = os.path.relpath(os.path.join(r, f), f"{STAGE}/bm")
                placed[f"engine/{rel}"] = (os.path.join(r, f), f"BigMoE {C[:8]} {rel}")
    else:
        placed[f"engine/{p}"] = (src, f"BigMoE {C[:8]} {p}")
for r, _, fs in os.walk(f"{STAGE}/es/evalsuite"):
    for f in fs:
        if "__pycache__" in r or f.endswith(".pyc"):
            continue
        rel = os.path.relpath(os.path.join(r, f), f"{STAGE}/es")
        placed[f"harness/{rel}"] = (os.path.join(r, f), f"EvalSuite {EC[:8]} {rel}")
for rel in ("data/items/dec_ifeval.jsonl", "data/items/dec_math.jsonl", "data/items/dec_multi.jsonl",
            "data/items/dec_humaneval.jsonl", "data/items/dec_mbpp.jsonl", "results/ref2/decode.jsonl"):
    placed[f"harness/{rel}"] = (f"{ES}/{rel}", f"EvalSuite working tree {rel} (untracked there) md5 "
                                + hashlib.md5(open(f"{ES}/{rel}", "rb").read()).hexdigest())
placed["data/frspec_rank.npy"] = (f"{LEAP}/data/frspec_rank.npy", "Leap data/frspec_rank.npy (vocabulary rank for the "
                                  "drafter's screened head) md5 " + hashlib.md5(open(f"{LEAP}/data/frspec_rank.npy", "rb").read()).hexdigest())
NB = f"{BM}/.nb215/nanobind"                                  # nanobind 2.15.0 as BigMoE builds the extension with it
for r, _, fs in os.walk(NB):
    for f in fs:
        if "__pycache__" in r or f.endswith(".pyc"):
            continue
        rel = os.path.relpath(os.path.join(r, f), NB)
        placed[f"third_party/nanobind/{rel}"] = (os.path.join(r, f), f"nanobind 2.15.0 (BSD-3-Clause) {rel}")
for r, _, fs in os.walk(SRC):                                  # run.py, install.sh, README, configs, tools
    for f in fs:
        if "__pycache__" in r or f.endswith(".pyc") or f.startswith("."):
            continue
        rel = os.path.relpath(os.path.join(r, f), SRC)
        placed[rel] = (os.path.join(r, f), f"Leap release/src/{rel}")
EXTRA = {"convert/make_records3.py": ("git", f"{HOME}/MoonShots", "b660ce9", "patches/planes_engine/make_records3.py"),
         "convert/convert_records_p11.py": ("git", LEAP, "HEAD", "scripts/convert_records_p11.py")}
REFS = json.load(open(a.refs)) if a.refs else {
    "exact-45_ar": f"{BM}/data/n45/ar_ref_tokens.json", "exact-32_ar": f"{BM}/data/n45/ar_ref_tokens.json",
    "chat-45_pdz0": f"{BM}/data/r57/pdz0_256_tokens.json", "chat-32_zs32": f"{BM}/data/r57/zs32_160_tokens.json"}
for k, v in REFS.items():
    EXTRA[f"verify/{k}.json"] = ("file", v)
os.makedirs(f"{STAGE}/extra", exist_ok=True)
for dst, src in EXTRA.items():
    sp = f"{STAGE}/extra/" + dst.replace("/", "__")
    if src[0] == "git":
        data = subprocess.run(["git", "-C", src[1], "show", f"{src[2]}:{src[3]}"], check=True, capture_output=True).stdout
        rev = sh("git", "-C", src[1], "rev-parse", "--short", src[2])
        label = f"{os.path.basename(src[1])} {rev} {src[3]}"
    else:
        data = open(src[1], "rb").read()
        label = f"{os.path.relpath(src[1], HOME).split('/')[0]} working tree {os.path.relpath(src[1], HOME).split('/', 1)[1]} md5 " + hashlib.md5(data).hexdigest()
    open(sp, "wb").write(data)
    placed[dst] = (sp, label)
open(f"{STAGE}/SNAPSHOT", "w").write(S + "\n")                  # run.py reads the decode loop's name from it
placed["engine/loop/SNAPSHOT"] = (f"{STAGE}/SNAPSHOT", f"make_release.py ({S})")
for dst, (src, _) in placed.items():
    os.makedirs(os.path.dirname(f"{OUT}/{dst}"), exist_ok=True)
    shutil.copy2(src, f"{OUT}/{dst}")

# ---- 3. path rewrite (literal, longest first) ----
F = f"{HOME}/BigMoE/data/r57/snap/frozen_{S}"
RW = [(f"{F}/{sub}", f"@ROOT@/{dst}") for sub, dst in ROOTS] + [(f"{HOME}/BigMoE/data/r57/snap/{S}", "@ROOT@/engine/loop")]
for P in (HOME, "~"):
    RW += [(f"{P}/BigMoE/data/r57/snap/{x}", "@ROOT@/engine/loop") for x in XS]
    RW += [(f"{P}/BigMoE/data/r57/snap/{S}", "@ROOT@/engine/loop"),
           (f"{P}/Leap/third_party/ms_gdn", "@ROOT@/kernels/ms_gdn"), (f"{P}/Leap/third_party", "@ROOT@/third_party"),
           (f"{P}/Leap/scripts", "@ROOT@/kernels"), (f"{P}/Leap/data/l120_ft.safetensors", "@ROOT@/weights/l120_ft.safetensors"),
           (f"{P}/Leap/data/frspec_rank.npy", "@ROOT@/data/frspec_rank.npy"),
           (f"{P}/TrainLab/ckpt", "@ROOT@/weights"), (f"{P}/TrainLab/sticky", "@ROOT@/drafters/sticky"),
           (f"{P}/Breakthrough/keepset", "@ROOT@/keepset"), (f"{P}/EvalSuite", "@ROOT@/harness"),
           (f"{P}/BigMoE", "@ROOT@/engine"),
           # bigmoe/tier.py (_tier_glu, every FAST arm with 3-bit sure rows) and bigmoe/loader.py (BIGMOE_GDN_EXACT) put
           # MoonShots' planes_engine on sys.path for tierglu / gdn_exact_patch; the frozen copies in kernels/ms_gdn are
           # byte-identical (md5 checked 11-09 07:0x: tierglu dc81729b, gdn_exact 3b835480, gdn_exact_patch 936c2635).
           # The module manifests never list them: R57_MODMAN records only Leap / TrainLab / Breakthrough / snapshot files
           # (the 11-09 07:00 smoke: lossy / chat 'No module named tierglu' with HOME empty).
           (f"{P}/MoonShots/patches/planes_engine", "@ROOT@/kernels/ms_gdn")]
TL = re.compile(r"(~|" + re.escape(HOME) + r")/TrainLab(?=[\"'/])")      # ~/TrainLab as a whole word
code = [p for p in placed if p.endswith((".py", ".sh")) and not p.startswith("third_party/nanobind/")]
nrw = 0
for p in code:
    t0 = t = open(f"{OUT}/{p}").read()
    for x, y in RW:
        t = t.replace(x, y)
    t = TL.sub("@ROOT@/drafters", t)
    if t != t0:
        nrw += 1
        open(f"{OUT}/{p}", "w").write(t)

# ---- 4. comment-safe scrub (build_public.sh's rules; quotes only inside comments / docstrings) ----
# The private names and phrases the scrub replaces and the leak check refuses are read from release/private_patterns.json,
# which is gitignored: written out here they would block the public snapshot, whose own leak check refuses them (11-09).
PRIV = json.load(open(f"{LEAP}/release/private_patterns.json", encoding="utf-8"))
SUBS = [tuple(x) for x in PRIV["subs"]]
QUOTE = re.compile(PRIV["quote"])
RED = "(a quote from the project owner, kept private)"
nsub = nq = 0
unsafe = []
for p in placed:
    if p.startswith("third_party/nanobind/") or not p.endswith((".py", ".sh", ".md", ".txt", ".cpp", ".json", ".jsonl", ".env")):
        continue
    fp = f"{OUT}/{p}"
    t0 = t = open(fp, encoding="utf-8").read()
    for x, y in SUBS:
        t = re.sub(x, y, t)
    nsub += t != t0
    if not QUOTE.search(t):
        if t != t0:
            open(fp, "w", encoding="utf-8").write(t)
        continue
    L = t.split("\n")
    if p.endswith(".py"):
        toks = list(tokenize.generate_tokens(io.StringIO(t).readline))
        for tk in toks:
            if tk.type == tokenize.COMMENT and QUOTE.search(tk.string):
                r, c = tk.start
                L[r - 1] = L[r - 1][:c] + "# " + RED; nq += 1
            elif tk.type == tokenize.STRING and tk.start[0] != tk.end[0]:      # multi-line strings (docstrings)
                for r in range(tk.start[0], tk.end[0] + 1):
                    m = QUOTE.search(L[r - 1])
                    if not m:
                        continue
                    if tk.start[0] < r < tk.end[0]:
                        L[r - 1] = RED; nq += 1
                    elif r == tk.start[0] and m.start() >= tk.start[1] + 3:
                        L[r - 1] = L[r - 1][:m.start()] + RED; nq += 1
                    elif r == tk.end[0] and L[r - 1].rstrip().endswith(('"""', "'''")) and m.end() <= len(L[r - 1].rstrip()) - 3:
                        L[r - 1] = RED + L[r - 1].rstrip()[-3:]; nq += 1
                    else:
                        unsafe.append(f"{p}:{r}")
        for i, line in enumerate(L):
            if QUOTE.search(line) and f"{p}:{i + 1}" not in unsafe:
                unsafe.append(f"{p}:{i + 1}")
    elif p.endswith((".sh", ".cpp")):
        for i, line in enumerate(L):
            m = QUOTE.search(line)
            if m:
                k = line.find("#" if p.endswith(".sh") else "//")
                if 0 <= k < m.start():
                    L[i] = line[:k] + ("# " if p.endswith(".sh") else "// ") + RED; nq += 1
                else:
                    unsafe.append(f"{p}:{i + 1}")
    elif p.endswith((".md", ".txt")):
        for i, line in enumerate(L):
            if QUOTE.search(line):
                L[i] = "> " + RED; nq += 1
    else:
        unsafe += [f"{p}:{i + 1}" for i, line in enumerate(L) if QUOTE.search(line)]
    open(fp, "w", encoding="utf-8").write("\n".join(L))

# ---- 4b. comment rewording (11-09, the owner's rule via the orchestrator: no person's or product's name in public files) --
# Exact (path, old, new) replacements from release/private_patterns.json "comment_subs" (gitignored: the old wording is
# private). Each old text must occur exactly once in the placed copy; a .py file's AST with docstrings removed must stay
# equal, so only comments and docstrings change (the source repos are not touched).
import ast


def _ast_nodoc(src):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        b = getattr(n, "body", None)
        if isinstance(b, list) and b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                and isinstance(b[0].value.value, str):
            n.body = b[1:] or [ast.Pass()]
    return ast.dump(tree)


ncs = 0
# Not private (the old text is step 4's own redaction line): the 24 GB loop's row-cap comment, readable again (Tue 10 Nov).
EXTRA_CS = [(f"engine/loop/r57_screen_{x}.py", f"# {RED}\n# rows (drafts cut to N - 1",
             "# R57_ROWCAP=N (W24, Mon 9 Nov: the 24 GB line's row cap): EVERY prompt's check verifies at most N\n"
             "# rows (drafts cut to N - 1") for x in XS if x == "w24"]
for rel, old, new in [tuple(x) for x in PRIV.get("comment_subs", [])] + EXTRA_CS:
    fp = f"{OUT}/{rel}"
    t0 = open(fp, encoding="utf-8").read()
    assert t0.count(old) == 1, f"comment_subs: {rel}: the old text occurs {t0.count(old)} times"
    t = t0.replace(old, new)
    if rel.endswith(".py"):
        assert _ast_nodoc(t0) == _ast_nodoc(t), f"comment_subs: {rel}: the change reaches code, not only comments / docstrings"
    open(fp, "w", encoding="utf-8").write(t); ncs += 1
print(f"  comment rewording: {ncs} lines (comments / docstrings only, AST without docstrings unchanged)")

# ---- 5. checks: leak patterns, live paths, compile ----
LEAK = re.compile(PRIV["leak"] + r"|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
                  r"claude\.ai/code/session|" + re.escape(HOME), re.I)
leaks = []
for p in placed:
    fp = f"{OUT}/{p}"
    try:
        t = open(fp, encoding="utf-8").read()
    except (UnicodeDecodeError, IsADirectoryError):
        continue
    for i, line in enumerate(t.split("\n")):
        if LEAK.search(line):
            leaks.append(f"{p}:{i + 1}: {line.strip()[:120]}")
live = []
LIVE = re.compile(r"(~|" + re.escape(HOME) + r")/(BigMoE|Leap/(scripts|third_party)|EvalSuite|TrainLab|Breakthrough/keepset|"
                  r"MoonShots/patches/planes_engine)\b")
for p in code:
    for i, line in enumerate(open(f"{OUT}/{p}").read().split("\n")):
        if LIVE.search(line):
            live.append(f"{p}:{i + 1}")
# every other ~/ path the code still names (provenance records, bench outputs, optional self-tests, switches EvalSuite
# refuses): listed in the manifest, not rewritten; with HOME empty the clean-clone run shows whether a config reaches one
HOMEP = re.compile(r"[\"'(]~/(?!\.local/share/quiet/)([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)")
home_left = []
for p in code:
    for i, line in enumerate(open(f"{OUT}/{p}").read().split("\n")):
        for m in HOMEP.finditer(line):
            home_left.append(f"{p}:{i + 1}: ~/{m.group(1)}")
bad = []
for p in placed:
    if p.endswith(".py") and not p.startswith("third_party/nanobind/"):
        r = subprocess.run([sys.executable, "-c", "import ast,sys; ast.parse(open(sys.argv[1]).read())", f"{OUT}/{p}"],
                           capture_output=True, text=True)
        if r.returncode:
            bad.append(f"{p}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}")

# ---- 6. manifest + git ----
man = dict(snapshot=S, bigmoe_commit=C, evalsuite_commit=EC, leap_commit=sh("git", "-C", LEAP, "rev-parse", "HEAD"),
           modules_union=sorted(mods), modules_missing=missing, path_rewrites=nrw, scrub_subs=nsub, scrub_quotes=nq,
           warnings=WARN, home_paths_left=home_left, files={p: dict(sha256=sha(f"{OUT}/{p}"), source=s) for p, (_, s) in sorted(placed.items())})
json.dump(man, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
shutil.rmtree(STAGE, ignore_errors=True)
ok = not leaks and not bad and not unsafe and not live          # a live research path blocks the commit too (11-09)
print(f"PACKAGE {S}: {len(placed)} files, {sum(os.path.getsize(f'{OUT}/{p}') for p in placed) / 1e6:.1f} MB; BigMoE {C[:8]}, "
      f"EvalSuite {EC[:8]}; {len(mods)} modules in the manifest union, {len(missing)} not in this snapshot {missing}; "
      f"{nrw} files path-rewritten, {nsub} scrubbed, {nq} quotes redacted")
for w in WARN:
    print("  WARN", w)
print(f"  leaks {len(leaks)}" + "".join(f"\n    {x}" for x in leaks[:20]))
print(f"  quote lines left unscrubbed (unsafe to edit) {len(unsafe)}" + "".join(f"\n    {x}" for x in unsafe[:20]))
print(f"  other ~/ paths named in code (MANIFEST home_paths_left) {len(home_left)}")
print(f"  live research paths left in code (fail loudly under HOME=empty if executed) {len(live)}"
      + "".join(f"\n    {x}" for x in live[:12]))
print(f"  syntax errors {len(bad)}" + "".join(f"\n    {x}" for x in bad[:20]))
if not ok:
    print("NOT COMMITTED: fix the leaks / syntax errors / unsafe quote lines / live research paths first"); sys.exit(2)
if not a.no_commit:
    with open(f"{OUT}/.gitignore", "w") as f:
        f.write("__pycache__/\n*.pyc\n.venv/\nengine/ext/build*/\nengine/models/\nweights/\nengine/data/r57/\nout/\n.root\n")
    sh("git", "-C", OUT, "init", "-q", "-b", "main")
    sh("git", "-C", OUT, "add", "-A")
    sh("git", "-C", OUT, "-c", "user.name=kamenlevi", "-c", "user.email=kamenlevi@users.noreply.github.com", "commit", "-q",
       "-m", f"Engine package from snapshot {S} (BigMoE {C[:8]}, EvalSuite {EC[:8]})")
    print("COMMITTED", sh("git", "-C", OUT, "rev-parse", "--short", "HEAD"), OUT)
