"""check_fasttax.py (Fable B, build_1105): CPU proof that fasttax_r57.patch leaves every arm without R57_FASTTAX untouched,
and that with it only the two intended ref2 switches move. No MLX, no GPU, no model: text, the AST and a stub ref2.
  T1  BigMoE's scripts/r57_screen.py: working tree == HEAD, and `git apply --check` of the patch passes there (read-only).
  T2  the patch applies to a scratch copy of HEAD's file and the result compiles (T3 / T4 test that result, so the check
      stays valid when BigMoE's HEAD moves, as long as the patch still applies).
  T3  AST: the patched module minus the top-level `if os.environ.get("R57_FASTTAX"):` blocks == the unpatched module, node
      for node; exactly two such blocks; the name R57_FASTTAX appears nowhere else. So with the variable unset or empty
      the patched program runs the unpatched statements, one for one: EXACT arms cannot change.
  T4  the two blocks executed against a stub (ref2's real CFG literal, read from ~/Leap/scripts/ref2.py and evaluated under
      each arm's environment): arms without R57_FASTTAX leave CFG identical and print nothing; FAST arms with it flip
      att_single (and gdn_inproj only without R57_DNR); EXACT arms with it, or a FAST arm that still carries R57_REFTOK,
      stop on the assert.
  T5  Leap's ref2 at its HEAD has what the patch relies on (att_single / gdn_inproj False by default, _sdpa_one, _InProjPad),
      and _InProjPad is still not DENSE-N-RAM-aware (the reason inproj is skipped under R57_DNR=1).
  T6  the all-in arm string in fasttax_aba.sh == BigMoE's own (scripts/fri_night.sh).
Run: ~/BigMoE/.venv/bin/python -I check_fasttax.py   (any python >= 3.9; exit 0 = all PASS)"""
import ast, os, re, shutil, subprocess, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
BIG = os.path.expanduser(os.environ.get("FTA_BIGMOE", "~/BigMoE"))
LEAP = os.path.expanduser(os.environ.get("FTA_LEAP", "~/Leap"))
PATCH = os.path.join(HERE, "fasttax_r57.patch")
PATCHED = os.path.join(HERE, "patched", "r57_screen.py")
REL = "scripts/r57_screen.py"
ok_all = True


def report(name, ok, note=""):
    global ok_all
    ok_all &= bool(ok)
    print("%-4s %s  %s" % (name, "PASS" if ok else "FAIL", note))


def git(repo, *a, **kw):
    return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, **kw)


# ---------------------------------------------------------------- T1
head = git(BIG, "rev-parse", "--short", "HEAD").stdout.strip()
orig = git(BIG, "show", "HEAD:" + REL).stdout
wt = open(os.path.join(BIG, REL)).read()
already = "R57_FASTTAX" in wt
chk = git(BIG, "apply", "--check", PATCH)
rev = git(BIG, "apply", "--check", "--reverse", PATCH)
report("T1", (wt == orig and chk.returncode == 0) or (already and rev.returncode == 0),
       "BigMoE HEAD %s; working tree %s HEAD; git apply --check %s%s" % (
           head, "==" if wt == orig else "!=", "OK" if chk.returncode == 0 else "refused",
           " (the patch is already applied: --reverse --check OK)" if already and rev.returncode == 0 else ""))

# ---------------------------------------------------------------- T2
tmp = tempfile.mkdtemp(prefix="fasttax_")
try:
    os.makedirs(os.path.join(tmp, "scripts"))
    base = orig if "R57_FASTTAX" not in orig else open(os.path.join(HERE, "orig", "r57_screen.py")).read()
    open(os.path.join(tmp, REL), "w").write(base)
    p = subprocess.run(["patch", "-p1", "-s", "-d", tmp, "-i", PATCH], capture_output=True, text=True)
    got = open(os.path.join(tmp, REL)).read()
    stored = open(PATCHED).read() if os.path.exists(PATCHED) else ""
    want = got                                        # everything below tests HEAD + the patch, not a stored copy
    try:
        compile(got, REL, "exec")
        comp = True
    except SyntaxError as e:
        comp = False
    report("T2", p.returncode == 0 and comp and got != base,
           "patch -p1 on HEAD's file rc %d; the result compiles %s; the stored patched/r57_screen.py is %s" % (
               p.returncode, comp, "the same file" if got == stored else "for another HEAD (regenerate it; the tests below use HEAD + the patch)"))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- T3
GATE = ast.dump(ast.parse('os.environ.get("R57_FASTTAX")', mode="eval").body)
t_orig, t_new = ast.parse(base), ast.parse(want)
gated = [n for n in t_new.body if isinstance(n, ast.If) and ast.dump(n.test) == GATE and not n.orelse]
rest = ast.Module(body=[n for n in t_new.body if n not in gated], type_ignores=[])
same = ast.dump(rest) == ast.dump(ast.Module(body=t_orig.body, type_ignores=[]))
inside = sum(ast.dump(n).count("R57_FASTTAX") for n in gated)
everywhere = ast.dump(t_new).count("R57_FASTTAX")
COMP = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def module_stores(node):
    """names a statement binds at module level (comprehension targets are local to their comprehension)"""
    out = set()

    def walk(n, in_comp):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and not in_comp:
            out.add(n.id)
        for c in ast.iter_child_nodes(n):
            walk(c, in_comp or isinstance(n, COMP))
    walk(node, False)
    return out


names_new = set().union(*[module_stores(g) for g in gated]) if gated else set()
report("T3", same and len(gated) == 2 and inside == everywhere and "R57_FASTTAX" not in base,
       "%d gated blocks; the rest of the AST %s the unpatched file's (%d top-level nodes); R57_FASTTAX outside the blocks: %d; "
       "module-level names the blocks bind: %s" % (len(gated), "==" if same else "!=", len(t_orig.body), everywhere - inside, sorted(names_new)))
# those names must be throwaway temporaries: wherever the unpatched code reads one at module level, the same top-level
# statement has bound it first (so a value left by the gated blocks can never be observed), and no function reads it
bad_reads = []
for top in rest.body:
    for name in names_new:
        occ = sorted(((n.lineno, n.col_offset, isinstance(n.ctx, ast.Store)) for n in ast.walk(top)
                      if isinstance(n, ast.Name) and n.id == name), key=lambda t: (t[0], t[1]))
        if occ and not (isinstance(top, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Assign)) and occ[0][2]
                        and not isinstance(top, (ast.FunctionDef, ast.ClassDef))):
            bad_reads.append((name, occ[0][0]))
        if isinstance(top, (ast.FunctionDef, ast.ClassDef)) and occ:
            bad_reads.append((name, occ[0][0]))
report("T3b", not bad_reads,
       "the unpatched code never reads %s before binding it again in the same top-level statement%s" % (
           " / ".join(sorted(names_new)), "" if not bad_reads else "; READ FIRST at %s" % bad_reads[:6]))

# ---------------------------------------------------------------- T4
ref2_src = open(os.path.join(LEAP, "scripts", "ref2.py")).read()
t_ref = ast.parse(ref2_src)
cfg_node = next(n for n in t_ref.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CFG")
counts_node = next(n for n in t_ref.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "COUNTS")


def ref2_stub(env):
    ns = {"os": types.SimpleNamespace(environ=env), "int": int, "dict": dict}
    cfg = eval(compile(ast.Expression(cfg_node.value), "ref2.CFG", "eval"), ns)
    cnt = eval(compile(ast.Expression(counts_node.value), "ref2.COUNTS", "eval"), ns)
    return types.SimpleNamespace(CFG=cfg, COUNTS=cnt, counts=lambda: dict(cnt))


def run_blocks(env):
    """execute the two gated blocks exactly as they stand in the patched file, under env; -> (CFG before, after, output)"""
    r2 = ref2_stub(env)
    before, out = dict(r2.CFG), []
    ns = {"os": types.SimpleNamespace(environ=env), "ref2": r2, "FAST": env.get("R57_FAST", ""),
          "print": lambda *a, **k: out.append(" ".join(map(str, a))), "say": lambda s: out.append(s)}
    for g in gated:
        exec(compile(ast.Module(body=[g], type_ignores=[]), "fasttax block", "exec"), ns)
    return before, dict(r2.CFG), out


REF = os.path.expanduser("~/BigMoE/data/r57/s45t_B_tokens.json:gate+t")
EXACT = {"R57_PRESET": "45", "BIGMOE_EXT_BUILD": "build", "R57_REFTOK": REF}
QA = dict(EXACT, BIGMOE_EXT_BUILD="build_qom", R57_REFTOK="", R57_FAST="QOM", R57_D17="1", R57_D7B="code", R57_HC8="1",
          REF2_HC8_FAST="1", R57_DNR="1", R57_HCB="1", REF2_SMALL_NATIVE="1", R57_D7BNAX="1", REF2_OOO_1C="1")
QA_NODNR = {k: v for k, v in QA.items() if k != "R57_DNR"}
ARMS_OFF = {
    "exact default (R57_PRESET=45)": EXACT,
    "exact + K-73 / K-83": dict(EXACT, K73_EXACT="1", K83_EXACT="1"),
    "exact bundle (DNR + hc_b + GDNB + OOO-1C)": dict(EXACT, R57_DNR="1", R57_HCB="1", R57_GDNB="1", REF2_OOO_1C="1"),
    "exact NAX6 (R57_NAX=1)": dict(EXACT, R57_NAX="1", R57_REFTOK=""),
    "FAST U3C len": dict(EXACT, BIGMOE_EXT_BUILD="build_qom", R57_FAST="U3C", R57_U3C_LEN="1", R57_REFTOK=""),
    "FAST all-in (qall)": QA,
    "FAST all-in, R57_FASTTAX empty": dict(QA, R57_FASTTAX=""),
    "FAST all-in + split 0.9": dict(QA, R57_QOM_SPLIT="0.9"),
}
bad = []
for name, env in ARMS_OFF.items():
    b, a, out = run_blocks(dict(env))
    if b != a or out:
        bad.append(name)
report("T4a", not bad, "%d arms without R57_FASTTAX: ref2.CFG identical and nothing printed%s" % (
    len(ARMS_OFF), "" if not bad else "; DIFFERENT: %s" % bad))

diff = lambda b, a: {k: a[k] for k in a if a[k] != b.get(k)}
ON = [("FAST all-in + R57_FASTTAX=1 (DNR on)", dict(QA, R57_FASTTAX="1"), {"att_single": True}),
      ("FAST all-in + R57_FASTTAX=att", dict(QA, R57_FASTTAX="att"), {"att_single": True}),
      ("FAST all-in without DNR + R57_FASTTAX=1", dict(QA_NODNR, R57_FASTTAX="1"), {"att_single": True, "gdn_inproj": True}),
      ("FAST all-in without DNR + R57_FASTTAX=att+inproj", dict(QA_NODNR, R57_FASTTAX="att+inproj"), {"att_single": True, "gdn_inproj": True}),
      ("FAST all-in without DNR + R57_FASTTAX=inproj", dict(QA_NODNR, R57_FASTTAX="inproj"), {"gdn_inproj": True}),
      ("FAST U3C len + R57_FASTTAX=1", dict(ARMS_OFF["FAST U3C len"], R57_FASTTAX="1"), {"att_single": True, "gdn_inproj": True})]
bad = []
for name, env, want_d in ON:
    b, a, out = run_blocks(dict(env))
    d = diff(b, a)
    if d != want_d or not any("FAST-TAX" in o for o in out):
        bad.append((name, d))
    print("       %-52s -> %s | %s" % (name, d, out[0][:110] if out else ""))
report("T4b", not bad, "%d FAST arms with R57_FASTTAX: only the intended ref2 switches move%s" % (len(ON), "" if not bad else "; WRONG: %s" % bad))

MUST_STOP = [("exact default + R57_FASTTAX=1", dict(EXACT, R57_FASTTAX="1")),
             ("exact default, no REFTOK, + R57_FASTTAX=1", dict(EXACT, R57_REFTOK="", R57_FASTTAX="1")),
             ("FAST all-in that still carries R57_REFTOK + R57_FASTTAX=1", dict(QA, R57_REFTOK=REF, R57_FASTTAX="1")),
             ("FAST all-in + R57_FASTTAX=inproj under R57_DNR=1", dict(QA, R57_FASTTAX="inproj")),
             ("FAST all-in + R57_FASTTAX=fatt (unknown piece)", dict(QA, R57_FASTTAX="fatt"))]
bad = []
for name, env in MUST_STOP:
    try:
        b, a, out = run_blocks(dict(env))
        bad.append((name, diff(b, a)))
    except AssertionError as e:
        print("       %-58s -> AssertionError: %s" % (name, str(e)[:90]))
report("T4c", not bad, "%d misuse cases stop on an assert before any switch moves%s" % (len(MUST_STOP), "" if not bad else "; NOT STOPPED: %s" % bad))

# ---------------------------------------------------------------- T5
lhead = git(LEAP, "rev-parse", "--short", "HEAD").stdout.strip()
cfg0 = ref2_stub({}).CFG
cls = next((n for n in t_ref.body if isinstance(n, ast.ClassDef) and n.name == "_InProjPad"), None)
cls_src = ast.get_source_segment(ref2_src, cls) if cls else ""
fn = {n.name for n in t_ref.body if isinstance(n, ast.FunctionDef)}
dnr_aware = "_dn_mlx" in cls_src or "_DNR" in cls_src
report("T5", cfg0.get("att_single") is False and cfg0.get("gdn_inproj") is False and "_sdpa_one" in fn and cls is not None,
       "Leap HEAD %s: CFG att_single %s, gdn_inproj %s by default; _sdpa_one %s; _InProjPad %s, DENSE-N-RAM-aware: %s%s" % (
           lhead, cfg0.get("att_single"), cfg0.get("gdn_inproj"), "_sdpa_one" in fn, cls is not None, dnr_aware,
           " (the inproj guard under R57_DNR=1 can be lifted)" if dnr_aware else " (so inproj is skipped under R57_DNR=1)"))

# ---------------------------------------------------------------- T6
def arm(path, var="QA"):
    s, v = open(path).read(), ""
    for m in re.finditer(r'^\s*%s="([^"]*)"' % var, s, re.M):
        v = m.group(1).replace("$%s," % var, v + ",") if m.group(1).startswith("$%s," % var) else m.group(1)
    return v
mine = arm(os.path.join(HERE, "fasttax_aba.sh")) if os.path.exists(os.path.join(HERE, "fasttax_aba.sh")) else None
theirs = arm(os.path.join(BIG, "scripts", "fri_night.sh"))
env_of = lambda s: dict(kv.split("=", 1) for kv in s.split(",") if kv)
report("T6", mine is not None and env_of(mine) == env_of(theirs) and env_of(theirs) == {k: v for k, v in QA.items() if k not in ("R57_PRESET",)},
       "fasttax_aba.sh's all-in arm %s BigMoE's fri_night.sh QA (%d switches) and the arm T4 tested" % (
           "==" if mine is not None and env_of(mine) == env_of(theirs) else "!=", len(env_of(theirs))))
print("\n%s" % ("ALL PASS" if ok_all else "SOME FAILED"))
sys.exit(0 if ok_all else 1)
