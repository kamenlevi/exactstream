"""FLAT-2's patcher (zero session, Thu 5 Nov 2026): BigMoE's scripts/r57_screen.py at its current HEAD -> the harness
flat2.py runs. Nothing outside this folder is edited: the source is read, cut before its main section (everything up
to and including run() stays: the env preset, the load, every lever's install, the live loop), and six one-line
insertions are made in run():
  F2C.append(c0) / F2C.append(now)   each cycle's start and the loop's end: the true cycle period, whatever the loop
                                     does between a commit and the next draft (always on: a list append)
  F2H.after_prefill(locals())        once per run(), before the decode clock (the pool prewarm)
  F2H.stash / before_commit / after_commit   behind `if F2ON[0]:`, on in the stage and probe passes only
Each insertion point is found by the exact line it sits next to and must be found once. The loop's rules the probe
copies are compared with what they were copied from, on the syntax tree (comments and layout are free):
  the cut of a draft block and its gate masks   every copied statement must still be there, unchanged
  the live check's FG.update and its reset      every copied argument must still be there, unchanged
A HEAD that changed a copied rule stops here with the line, before any GPU second is spent: fix flat2_probe.py
(masks_for / fwd) and the expected text below together. Statements or arguments the HEAD ADDED are reported as
warnings and the run goes on: the probe leaves a new gate key the way the loop's own reset leaves it (as the harness's
warm-split re-runs do), and the report's "warm repeats bitwise == live" line shows whether that was right.
CLI: flat2_patch.py [screen.py [generated.py]] -> prints the report; exit 1 on a mismatch. CPU only."""
import ast
import hashlib
import os
import sys

RUN = 'def run(ids, arm, name=None, drafter="dfl", cpy=False, maxd=0, fset=None, tcut=None):'
SPLIT = "REFTOK, REFARM = {}, None"                      # the main section starts here (after run()'s return)
A_PREFILL = "    out, cyc, ck_t, kept, sm, wl, cut, drf = [tok], [], [], [], [], [], False, []"
A_C0 = "        c0 = time.perf_counter()"
A_CUT = '            if arm in ("gate+t", "gate+t+pf") and not wsp:'
A_CUT_END = "            if (trim := dcache[0].offset - a) > 0:"
A_FG = "        FG.update(on=fk_ is not None or sq_ is not None, keep=mx.array(fk_) if fk_ is not None else None,"
A_SCOPE = "            with ref2.verify_scope(model):"
A_FWD = "                lg = model(mx.array([[tok] + dr]), cache=cache)"
A_RESET = "            FG.update(on=False,"
A_COMMIT = "        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)"
A_ASYNC = "        mx.async_eval(states(cache))"
A_DEC = "    dec = time.perf_counter() - td"
MARK = "   # FLAT-2 (flat2_patch.py)"
INS = dict(prefill="    if F2H is not None: F2H.after_prefill(locals())", c0="        F2C.append(c0)",
           stash="            if F2ON[0]: F2H.stash(locals())", before="        if F2ON[0]: F2H.before_commit(locals())",
           after="        if F2ON[0]: F2H.after_commit(locals())", end="    F2C.append(time.perf_counter())")

# the cut of a draft block and its gate masks, as flat2_probe.masks_for copies it (HEAD 6adad93, lines 1009-1027)
G_CUT = '''
if arm in ("gate+t", "gate+t+pf") and not wsp:
    dr = dr[:int((np.cumprod(np.array(cfl)) >= tcut).sum())]
if maxd:
    dr = dr[:maxd]
if fk_ is not None:
    fk_ = fk_[:len(dr) + 1].copy(); fk_[len(dr)] = True
    fst["gated"] += int((~fk_.all(axis=1)).sum())
if sq_ is not None:
    sq_ = sq_[:len(dr) + 1].copy()
    if D17 and fz.get("qom"):
        if len(dr) == R_ - 1:
            sq_[len(dr)] = c16 is not None and c16 >= fz["thr"]
    else:
        sq_[len(dr)] = False
    if QSPLIT is not None:
        qcf_ = np.array((list(cg_) + [c16 if c16 is not None else 1.0])[:len(dr) + 1], dtype=np.float32)
    fst["gated"] += int(sq_.sum())
    if not sq_.any() and not U3FORCE:
        sq_ = None
'''
# the live check's gate, as flat2_probe's fwd sets and resets it (lines 1083-1094)
G_FG = dict(on="fk_ is not None or sq_ is not None", keep="mx.array(fk_) if fk_ is not None else None",
            sure="mx.array(sq_) if sq_ is not None else None", sure_np="sq_", t3='bool(fz and fz.get("t3"))',
            cold='bool(fz and fz.get("cold")) and sq_ is not None', qom='bool(fz and fz.get("qom")) and sq_ is not None',
            qcf="mx.array(qcf_) if qcf_ is not None else None")
G_OFF = dict(on="False", keep="None", sure="None", sure_np="None", t3="False", cold="False", qom="False", qcf="None")
# single lines the driver leans on
G_LINES = ("    tcut = tcut or TRUNC", "    while len(out) < CAP and tok not in EOS:", "        s0, b0_ = stats(), disk_read_bytes()",
           "        snap = CS.snapshot_untrimmable_cache_lazy(cache)", "            cfl = np.array(cf[0]).tolist()",
           "        drf.append(c1 - c0)", "        ck_t.append(c2 - c1); kept.append(k + 1); wl.append(c2 - c0)")
RUN_LOCALS = {"kept", "dr", "fz", "cache", "snap", "tok", "lg", "k", "s0", "fk_", "sq_", "qcf_", "mtp", "cg_", "c16", "R_"}
MODULE_NAMES = {"mx", "np", "model", "CS", "ref2", "FG", "TAP", "taps", "tap_reset", "states", "stats", "Q", "lay", "SW_", "BG",
                "BS", "D17", "QSPLIT", "run", "ED", "rule", "SETOF", "DRAFT4", "DRAFT16", "draft", "ARMS_", "CAP", "CHATSW",
                "say", "lines", "el", "HARD", "FAST", "C", "TIER", "footprint_now_gb", "footprint_peak_gb", "st", "time", "os",
                "sys", "json", "EM", "VTIER", "TRUNC", "t0"}
RUN_KEYS = {"pc", "bad", "cut", "out", "n", "tps", "tpsw", "kept", "checks", "fast", "pf", "drf", "ck", "miss"}
PC_KEYS = {"rows", "kept", "ck", "drf", "miss", "io", "svc", "uses", "stl", "bld", "wt"}


class PatchError(Exception):
    pass


def _canon(expr):
    return ast.unparse(ast.parse(expr, mode="eval"))


def _stmt(lines, i, what):
    """the one statement that starts at source line i (it may run over several lines) -> (node, the line after it)"""
    ind = len(lines[i]) - len(lines[i].lstrip(" "))
    for j in range(i + 1, min(i + 16, len(lines)) + 1):
        try:
            body = ast.parse("\n".join(x[ind:] for x in lines[i:j])).body
        except SyntaxError:
            continue
        if len(body) == 1:
            return body[0], j
    raise PatchError(f"{what}: the statement at source line {i + 1} does not parse")


def _fg_kw(node, what, line):
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "FG.update"):
        raise PatchError(f"{what}: source line {line} is not an FG.update(...) call")
    return {k.arg: ast.unparse(k.value) for k in node.value.keywords}


def _fg_guard(kw, want, what, line, warn):
    for k, v in want.items():
        if kw.get(k) != _canon(v):
            raise PatchError(f"{what}: the harness changed at source line ~{line}\n  expected {k}={_canon(v)}\n  found    "
                             f"{k}={kw.get(k)}\n  -> re-derive flat2_probe.py's fwd (its FG.update) and G_FG / G_OFF in flat2_patch.py")
    extra = sorted(set(kw) - set(want) - {"chk"})
    if extra:
        warn.append(f"{what} (source line {line}) has keys the probe does not copy: " + ", ".join(f"{k}={kw[k]}" for k in extra)
                    + " (probe forwards leave them as the loop's reset does)")


def build(src):
    """-> (the generated harness source: definitions + run() with the insertions, a report dict with its warnings)"""
    lines, warn = src.split("\n"), []

    def find(prefix, lo=0, hi=None, what=""):
        idx = [i for i in range(lo, len(lines) if hi is None else hi) if lines[i].startswith(prefix)]
        if len(idx) != 1:
            raise PatchError(f"{what or prefix.strip()[:50]}: expected exactly one line starting with {prefix!r} in source "
                             f"lines {lo + 1}-{len(lines) if hi is None else hi}, found {len(idx)} (at {[i + 1 for i in idx][:5]})")
        return idx[0]

    i_run, i_split = find(RUN, what="run()'s signature"), find(SPLIT, what="the main section's first line")
    if not i_run < i_split:
        raise PatchError("run() should come before the main section")
    a_pre = find(A_PREFILL, i_run, i_split, "the loop's state line (after_prefill)")
    a_c0 = find(A_C0, i_run, i_split, "the cycle's start (c0)")
    a_cut = find(A_CUT, i_run, i_split, "the draft cut (stash)")
    a_end = find(A_CUT_END, a_cut, i_split, "the end of the draft cut")
    a_fg = find(A_FG, i_run, i_split, "the live check's gate")
    a_com = find(A_COMMIT, i_run, i_split, "the commit (before_commit)")
    a_asy = find(A_ASYNC, i_run, i_split, "the commit replay's async_eval (after_commit)")
    a_dec = find(A_DEC, i_run, i_split, "the loop's end (dec)")
    if not (a_pre < a_c0 < a_cut < a_end < a_fg < a_com and a_asy == a_com + 1 and a_asy < a_dec):
        raise PatchError(f"anchor order changed: prefill {a_pre + 1}, c0 {a_c0 + 1}, cut {a_cut + 1}-{a_end + 1}, gate {a_fg + 1}, "
                         f"commit {a_com + 1}, async {a_asy + 1} (must follow the commit directly), end {a_dec + 1}")
    # ---- the cut of a draft block: every copied statement still there, in order, unchanged
    ind = len(A_CUT) - len(A_CUT.lstrip(" "))
    try:
        got = [ast.unparse(n) for n in ast.parse("\n".join(x[ind:] for x in lines[a_cut:a_end])).body]
    except SyntaxError as e:
        raise PatchError(f"the draft cut (source lines {a_cut + 1}-{a_end}) does not parse on its own: {e}")
    want, j = [ast.unparse(n) for n in ast.parse(G_CUT).body], 0
    for w in want:
        while j < len(got) and got[j] != w:
            j += 1
        if j == len(got):
            raise PatchError(f"the draft block's cut and gate masks changed (source lines {a_cut + 1}-{a_end}); the copied "
                             f"statement no longer there:\n  {w.splitlines()[0]} ...\n  -> re-derive flat2_probe.masks_for and "
                             f"G_CUT in flat2_patch.py")
        j += 1
    for g in got:
        if g not in want:
            warn.append(f"the draft cut (source lines {a_cut + 1}-{a_end}) has a statement the probe does not copy: "
                        f"{g.splitlines()[0][:90]}")
    # ---- the live check's gate, forward and reset
    node, a_fg_end = _stmt(lines, a_fg, "the live check's gate")
    _fg_guard(_fg_kw(node, "the live check's gate", a_fg + 1), G_FG, "the live check's FG.update", a_fg + 1, warn)
    a_sc = find(A_SCOPE, a_fg_end, a_com, "the live check's verify scope")
    if not lines[a_sc + 1].startswith(A_FWD):
        raise PatchError(f"the live check's forward changed at source line {a_sc + 2}: expected {A_FWD.strip()!r}")
    a_rs = find(A_RESET, a_sc, a_com, "the live check's gate reset")
    node, _ = _stmt(lines, a_rs, "the live check's gate reset")
    _fg_guard(_fg_kw(node, "the live check's gate reset", a_rs + 1), G_OFF, "the live check's FG reset", a_rs + 1, warn)
    for g in G_LINES:
        find(g, i_run, i_split, "a line the driver leans on")
    out = lines[:i_split]
    for i, text in sorted(((a_dec, INS["end"]), (a_asy + 1, INS["after"]), (a_com, INS["before"]), (a_cut, INS["stash"]),
                           (a_c0 + 1, INS["c0"]), (a_pre, INS["prefill"])), reverse=True):
        out.insert(i, text + MARK)
    gen = "\n".join(out) + "\n"
    tree = ast.parse(gen)                                # syntax; then the names the hooks and the driver read
    names = check_names(tree)
    return gen, dict(md5=hashlib.md5(src.encode()).hexdigest(), source_lines=len(lines), kept_lines=i_split, warnings=warn,
                     anchors=dict(run=i_run + 1, main=i_split + 1, after_prefill=a_pre + 1, c0=a_c0 + 1, stash=a_cut + 1,
                                  gate=a_fg + 1, before_commit=a_com + 1, after_commit=a_asy + 2, end=a_dec + 1), **names)


def _stores(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in n.names)
    return out


def _module_names(body):
    out = set()
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                out.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            for part in ("body", "orelse", "finalbody"):
                out.update(_module_names(getattr(n, part, [])))
            for h in getattr(n, "handlers", []):
                out.update(_module_names(h.body))
            if isinstance(n, ast.For):
                out.update(x.id for x in ast.walk(n.target) if isinstance(x, ast.Name))
            if isinstance(n, ast.With):
                out.update(x.id for i in n.items if i.optional_vars is not None for x in ast.walk(i.optional_vars)
                           if isinstance(x, ast.Name))
    return out


def check_names(tree):
    """the names flat2_probe.py reads from run()'s locals and from the harness's globals, and the keys of run()'s
    result, must exist in the generated source (a rename upstream fails here, on the CPU)"""
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if fn is None:
        raise PatchError("no run() in the generated source")
    miss = RUN_LOCALS - _stores(fn)
    if miss:
        raise PatchError(f"run() no longer binds the locals the hooks read: {sorted(miss)}")
    mod = _module_names(tree.body)
    miss = MODULE_NAMES - mod
    if miss:
        raise PatchError(f"the harness no longer binds the module names FLAT-2 uses: {sorted(miss)}")
    ret = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
           and getattr(n.value.func, "id", "") == "dict"]
    if not ret:
        raise PatchError("run() no longer returns dict(...)")
    kw = {k.arg: k.value for k in ret[-1].keywords}
    if RUN_KEYS - set(kw):
        raise PatchError(f"run()'s result lost keys: {sorted(RUN_KEYS - set(kw))}")
    pc = kw["pc"]
    pck = {k.arg for k in pc.keywords} if isinstance(pc, ast.Call) else set()
    if PC_KEYS - pck:
        raise PatchError(f"run()'s per-check log lost keys: {sorted(PC_KEYS - pck)}")
    return dict(run_args=[a.arg for a in fn.args.args], run_keys=sorted(kw), pc_keys=sorted(pck),
                module_names=len(mod), optional_missing=sorted({"U3FORCE", "QK", "RXC", "FASTARMS"} - mod))


if __name__ == "__main__":
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/BigMoE/scripts/r57_screen.py")
    try:
        gen, info = build(open(path).read())
    except PatchError as e:
        print(f"FLAT-2 patch: MISMATCH on {path}\n{e}")
        sys.exit(1)
    print(f"FLAT-2 patch: OK on {path} (md5 {info['md5']}, {info['source_lines']} lines; definitions kept: lines 1-"
          f"{info['kept_lines']}); insertions at source lines {info['anchors']}; run({', '.join(info['run_args'])}); per-check "
          f"keys {info['pc_keys']}")
    for w in info["warnings"]:
        print(f"  WARNING: {w}")
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(gen)
        print(f"  generated source -> {sys.argv[2]} ({gen.count(chr(10))} lines)")
