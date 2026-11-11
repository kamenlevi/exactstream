"""test_inert.py (GOAL session, Thu 2026-11-05): R57_INERT's CPU self-test. No GPU, no model; python -I, < 0.5 GB.
1. the module: parse_arm, rule_ok == relax_core's, inert_k == the exact rule when no row is open (random blocks), the
   tagger's cases on synthetic fenced answers (comments, the line-end guard, tool comments, the fence's first two lines,
   '#' in a string, a C fence, an unlabeled fence; docstrings, their closing quotes, doctest lines, a triple-quoted DATA
   string, one inside brackets, after a backslash, with an f prefix; the guarded arm's comment blocks), pre_block and
   copyrun_extend on BigMoE's own CopyGate.
2. the patch: apply_inert.py's edits apply once each to ~/BigMoE/scripts/r57_screen_rlx.py (read-only; nothing written),
   the result compiles, every inserted block opens behind INERT / iscan / CRUN, and the replaced expressions equal the old
   ones for R57_ROWS 16 and 32 with the flags unset.
3. the walk streams: (a) the engine-path replay (precheck_inert.py) lands on the lexer replay's CD16 numbers
   (x1.17 fresh HumanEval+, x1.63 MBPP, +- 0.03); (b) the safety property against Python's own parser: on every answer of
   the 24 HumanEval+ and 100 MBPP walks, every token the tagger opens to wholesale acceptance lies inside a COMMENT token
   (tokenize) or a string statement (ast) of the answer's code.
usage: python -I test_inert.py [--no-walk]"""
import ast
import hashlib
import importlib.util
import io
import os
import random
import re
import sys
import tokenize

H = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inert_span as INM                                             # noqa: E402
import apply_inert as AI                                             # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RXC = load("relax_core_bigmoe", f"{H}/BigMoE/scripts/relax_core.py")
fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        fails.append(what)


def mk(text):
    s = INM.InertScan(RXC)
    s.feed(text)
    return s


def modes(s, d, a, arm):
    return [c[1] for c in INM.classify(RXC, s, d, a, arm)]


def t_module():
    print("1. the module")
    CD, D, G, C = (INM.parse_arm(x) for x in ("CD16", "D16", "G16", "C16"))
    check(INM.parse_arm("") is None and INM.parse_arm("0") is None and CD["rows"] == 16 and INM.parse_arm("CD32")["rows"] == 32
          and CD["doctest"] and not INM.parse_arm("CD16", "0")["doctest"], "parse_arm: unset / 0 = None, rows, the doctest switch")
    try:
        INM.parse_arm("XX"); bad = False
    except ValueError:
        bad = True
    check(bad, "parse_arm rejects an unknown arm")
    check(all(INM.rule_ok(rr, dl, rk) == RXC.rule_ok(rr, dl, rk) for rr in (INM.RULE_C, INM.RULE_TOP8, ("R1", 0.5))
              for dl in (-30.0, -3.0, -2.31, -2.3, -1.0, 0.0) for rk in range(10)), "rule_ok equals relax_core.rule_ok")
    rng = random.Random(11)
    ok = True
    for _ in range(2000):
        n = rng.randint(1, 31)
        am = [rng.randint(0, 4) for _ in range(n)]
        dr = [a if rng.random() < 0.8 else rng.randint(0, 4) for a in am]
        kx = 0
        while kx < n and dr[kx] == am[kx]:
            kx += 1
        k, rl = INM.inert_k(dr, am, [-9.0] * n, [50] * n, [(None, None, "")] * n)
        ok &= (k == kx and rl == [])
    check(ok, "flag-off identity: inert_k with no open row == the exact rule (2,000 random blocks)")
    k, rl = INM.inert_k([1, 2, 9, 4, 8], [1, 2, 3, 4, 5], [0, 0, -9, 0, -9], [0, 0, 50, 0, 50],
                        [(None, None, ""), (None, None, ""), ("comment", "whole", ""), (None, None, ""), ("docstring", "whole", "")])
    check((k, rl) == (5, [(2, "comment", "whole"), (4, "docstring", "whole")]), "inert_k keeps whole rows whatever the argmax and goes on")
    check(INM.inert_k([1, 9, 3], [1, 2, 3], [0, -1.0, 0], [0, 1, 0], [(None, None, "")] * 3) == (1, []),
          "inert_k stops at a mismatch on a row that is not open")
    c3 = [(None, None, ""), ("comment", INM.RULE_C, ""), (None, None, "")]
    check(INM.inert_k([1, 9, 3], [1, 2, 3], [0, -1.0, 0], [0, 1, 0], c3) == (3, [(1, "comment", "rule")])
          and INM.inert_k([1, 9, 3], [1, 2, 3], [0, -2.5, 0], [0, 1, 0], c3) == (1, []),
          "inert_k: a rule row is kept only when its rule accepts the draft (rule C: p >= 0.1 p1)")
    # ---- comments
    s = mk("Here it is:\n```python\ndef f(x):\n    y = 1\n    # compute the")
    check(modes(s, [" sum", "\n", " total"], [" total", "\n", "\n"], CD) == ["whole", None, None],
          "comment: a row inside the comment is whole; the draft that ends the line and the row after it are exact")
    check(modes(s, [" sum"], ["\n"], CD) == [None], "comment: the line-end guard (the 125B ends the line -> exact)")
    check(s.s.ctext.endswith("compute the") and s.in_comment(), "classify feeds a copy (the run's tagger is unchanged)")
    check(modes(s, [" sum"], [" total"], D) == [INM.RULE_C] and modes(s, [" sum"], [" total"], C) == ["whole"],
          "arms: D puts comments on rule C, C16 keeps them whole")
    check(modes(mk("```python\nimport os\nx = 1  # type"), [":"], [" int"], CD) == [None],
          "comment: one that is or could become a tool comment ('# type:') is never open")
    check(modes(mk("```python\n# header"), [" line"], [" text"], CD) == [None], "comment: the fence's first two lines are never open")
    check(modes(mk("```python\na = 1\nb = 2\ns = 'x # y"), [" z"], [" w"], CD) == [None], "comment: a '#' inside a string is not one")
    check(modes(mk("```c\nint a;\nint b;\n#include <stdio"), [".h"], [".h"], CD) == [None]
          and modes(mk("```\na = 1\nb = 2\n# note"), [" x"], [" y"], CD) == [None]
          and modes(mk("```py\na = 1\nb = 2\n# note"), [" x"], [" y"], CD) == ["whole"],
          "only a fence that says python / py is open (a C fence's '#include' and an unlabeled fence are not)")
    # ---- docstrings
    s7 = mk('```python\ndef f(x):\n    """Return the')
    cl = INM.classify(RXC, s7, [" sum", ' of "x" and', '."""', "\n"], [" total", " of", '."""', "\n"], CD)
    check([c[1] for c in cl] == ["whole", None, None, None] and [c[0] for c in cl] == ["docstring"] * 3 + [None],
          "docstring: a row inside is whole; a draft holding its quote character and the closing quotes are exact")
    cl = INM.classify(RXC, s7, [' "x"', " next"], [" a", " b"], CD)
    check([c[0] for c in cl] == ["docstring", None], "docstring: not open while a run of its quote character is pending")
    check(modes(s7, [" sum"], ['."""'], CD) == [None], "docstring: the 125B closes it -> exact")
    check(modes(s7, [" sum"], [" total"], C) == [None], "arm C16 leaves docstrings exact")
    s8 = mk('```python\ndef f(x):\n    """Add one.\n\n    >>> f(1')
    check(modes(s8, [")"], [")"], CD) == [None] and modes(s8, [")"], [")"], INM.parse_arm("CD16", "0")) == ["whole"]
          and modes(mk('```python\ndef f(x):\n    """Add one.'), ["\n    >>>"], ["\n"], CD) == [None],
          "docstring: doctest lines stay exact, before and after the draft (R57_INERT_DOCTEST=0 lifts it)")
    check(modes(mk('```python\ndef f(x):\n    msg = """Hello'), [" a"], [" b"], CD) == [None]
          and modes(mk('```python\nprint(\n    """Hello'), [" a"], [" b"], CD) == [None]
          and modes(mk('```python\nx = \\\n    """Hello'), [" a"], [" b"], CD) == [None]
          and modes(mk('```python\ndef f(x):\n    return """Hello'), [" a"], [" b"], CD) == [None]
          and modes(mk('```python\ndef f(x):\n    f"""Hello'), [" a"], [" b"], CD) == [None],
          "a triple-quoted string that is data (assigned, returned, inside brackets, after a backslash, f-string) is never open")
    check(modes(mk('```python\ndef f(x):\n    r"""Hello'), [" a"], [" b"], CD) == ["whole"]
          and modes(mk('```python\n"""Module doc'), [" a"], [" b"], CD) == ["whole"]
          and modes(mk('```python\nclass A:\n    def g(self):\n        """Doc'), [" a"], [" b"], CD) == ["whole"],
          "a docstring that is a statement of its own is open (raw prefix, module level, a method's)")
    # ---- the guarded arm
    check(modes(mk("```python\nx = 1\ny = 2\n# one\n# two\n# three is"), [" long"], [" short"], G) == [INM.RULE_TOP8]
          and modes(mk("```python\nx = 1\ny = 2\n# one\n# two is"), [" long"], [" short"], G) == ["whole"]
          and modes(mk("```python\nx = 1\ny = 2\nz = 3  # trailing"), [" a"], [" b"], G) == ["whole"],
          "guarded arm: lines 1-2 of a comment block and trailing comments whole, line 3 on the 125B's top-8")
    # ---- pre_block
    s = mk("```python\ndef f(x):\n    y = 1\n    # note")
    n, c2, w = INM.pre_block(RXC, s, [1, 2, 3, 4, 5], [0.2] * 5, [" a", " b", "\n", "    z", " = 2"], CD)
    check((n, c2, w) == (5, [1.0, 1.0, 0.2, 0.2, 0.2], [True, True, False, False, False]),
          "pre_block: rows in a span are sure in the cut; a 16-row arm leaves the block's length alone")
    sd = mk('```python\ndef f(x):\n    """Return the')
    dtx = [" w"] * 20 + ['"""'] + [" x"] * 10
    n16, _, _ = INM.pre_block(RXC, sd, list(range(31)), [0.5] * 31, dtx, CD)
    n32, _, w32 = INM.pre_block(RXC, sd, list(range(31)), [0.5] * 31, dtx, INM.parse_arm("CD32"))
    n32u, _, _ = INM.pre_block(RXC, sd, list(range(31)), [0.5] * 31, dtx, INM.parse_arm("CD32"), [False] * 20 + [True] * 5 + [False] * 6)
    check((n16, n32, n32u) == (31, 20, 25) and w32[:21] == [True] * 20 + [False],
          "pre_block at 32 rows: drafts past the 15th are kept only while they sit in a span (or are used copies)")
    # ---- copyrun_extend on BigMoE's CopyGate
    try:
        CGM = load("copygate_bigmoe", f"{H}/BigMoE/scripts/copygate.py")
        prompt = list(range(1000, 1200))
        cg = CGM.CopyGate(prompt + prompt[20:36], 200, "prompt")
        d0 = [7] * 31
        dr, cu, ci = cg.propose(d0, len(cg.ids) - 1)
        e = INM.copyrun_extend(cg, d0, dr, cu, ci, [1.0] * 31, 64)
        ok = (dr == prompt[36:67] and all(cu) and e[1] == prompt[36:99] and all(e[2]) and len(e[0]) == len(e[1]) == len(e[3]) == len(e[4]) == 63
              and e[0][:31] == d0 and e[0][31:] == e[1][31:])
        cg.learn(e[0], e[3], e[1] + [1], 63)
        ok &= cg.duel == [31, 0]
        same = INM.copyrun_extend(cg, d0, dr, cu, ci, [1.0] * 31, 32)
        short = INM.copyrun_extend(cg, d0, dr, [True] * 20 + [False] + [True] * 10, ci, [1.0] * 31, 64)
        ok &= same[1] == dr and short[1] == dr
        check(ok, "copyrun_extend (BigMoE's CopyGate): a 31-copy block goes on to 63 drafts, its first 31 unchanged, no duel "
              "label past the drafter's picks; 32 rows or a broken run: unchanged")
    except FileNotFoundError as ex:
        check(False, f"copyrun_extend: {ex}")


def t_patch():
    print("2. the patch")
    path = f"{H}/BigMoE/scripts/r57_screen_rlx.py"
    src = open(path).read()
    print(f"     r57_screen_rlx.py sha1 {hashlib.sha1(src.encode()).hexdigest()[:12]}")
    try:
        new = AI.apply(src); applied = True
    except SystemExit as e:
        print("     ", e); applied = False
    check(applied, f"apply_inert.py: all {len(AI.EDITS)} anchors found once")
    if not applied:
        return
    try:
        compile(new, path, "exec"); comp = True
    except SyntaxError as e:
        print("     ", e); comp = False
    check(comp, "the patched copy compiles")
    guarded = ("INERT, CRUN = None", "assert not (INERT is not None", "elif CRUN == 64", "if iscan is not None",
               "inert=dict(ist, arm=INERT")
    heads = []
    for name, _, how, text in AI.EDITS:
        if how == "replace":
            continue
        code = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        heads.append(code[0])
    check(all(any(h.startswith(g) for g in guarded) for h in heads), "every inserted block opens behind INERT / iscan / CRUN: "
          + " | ".join(h[:34] for h in heads))
    run_state = [e for e in AI.EDITS if e[0] == "run state"][0][3].split("\n")
    check(all(ln.strip().startswith(("assert not (INERT", "iscan =", "ist =", "ilog =", "if iscan", "iscan.feed")) or not ln.strip()
              for ln in run_state), "run state with the flag unset: a no-op assert, iscan / ilog None, an unused counter")
    # the replaced expressions, flags unset, against the old ones
    os.environ.pop("R57_COPYRUN", None)
    INERT = None                                         # noqa: F841 (read by eval below)
    ok = all((R in (16, 32)) == (R in (16, 32) or (R == 64 and os.environ.get("R57_COPYRUN") == "64")) for R in (8, 16, 32, 64))
    ok &= all(R_ - 1 == min(R_, 32) - 1 for R_ in (16, 32))
    ok &= all((R if cg is not None else 16) == (R if (cg is not None or (INERT is not None and INERT["rows"] > 16)) else 16)
              for R in (16, 32) for cg in (None, 1))
    rng = random.Random(3)
    for _ in range(2000):
        R_ = rng.choice((16, 32))
        cu_ = [rng.random() < rng.choice((0.2, 0.8, 0.95)) for _ in range(R_ - 1)]
        ok &= (R_ == 32 and sum(cu_) < 24) == (R_ >= 32 and sum(cu_[:31]) < 24 and not (INERT is not None and INERT["rows"] > 16))
    check(ok, "flags unset: the four replaced expressions equal the old ones for R57_ROWS 16 and 32")
    check(new.count("R57_INERT") >= 3 and 'inert=dict(ist, arm=INERT["name"]) if INERT is not None else None' in new
          and '"lh", "inert")}' in new, "the result and the runs record carry inert (None when unset)")


def py_inert_spans(code):
    """char spans of the code's comments (tokenize) and string statements (ast); None when the code does not parse"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    lines = code.split("\n")
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)
    spans = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.COMMENT:
                spans.append((starts[t.start[0] - 1] + t.start[1], starts[t.end[0] - 1] + t.end[1]))
    except (tokenize.TokenError, IndentationError):
        return None
    def off(lineno, col):                               # ast columns are UTF-8 byte offsets
        return starts[lineno - 1] + len(lines[lineno - 1].encode()[:col].decode(errors="ignore"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            spans.append((off(node.value.lineno, node.value.col_offset), off(node.value.end_lineno, node.value.end_col_offset)))
    return spans


def t_walk():
    print("3. the walk streams")
    import precheck_inert as P
    fresh = set(__import__("json").load(open(P.TOP8["he12"])).keys())
    for label, S, want in (("fresh HumanEval+ 12", P.samples("code", P.TOP8["he12"], (fresh, True)), 1.17),
                           ("MBPP 100", P.samples("mbpp", P.TOP8["mbpp"]), 1.63)):
        base, cd = P.replay(S, None), P.replay(S, INM.parse_arm("CD16"))
        r = cd["tps"] / base["tps"]
        check(abs(r - want) <= 0.03, f"{label}: CD16 through the engine path x{r:.3f} (kept {base['kept']:.2f} -> {cd['kept']:.2f}, "
              f"length {cd['E'] / cd['T']:.3f}) vs the lexer replay's x{want:.2f} +- 0.03")
    for label, dom in (("HumanEval+ 24", "code"), ("MBPP 100", "mbpp")):
        S = P.samples(dom)
        for arm, tag in ((INM.parse_arm("CD16"), "doctest guard on"), (INM.parse_arm("CD16", "0"), "guard off")):
            n_open = n_out = n_parsed = n_ans = cov = tot = 0
            worst = []
            for x in S:
                ids, gen = x["ids"], x["gen"]
                pcs = [P.dec(t) for t in ids[gen:]]
                text = "".join(pcs)
                m = re.search(r"```(?:python|py|python3)\n", text)
                if not m:
                    continue
                n_ans += 1
                c0 = m.end()
                c1 = text.find("\n```", c0)
                c1 = len(text) if c1 < 0 else c1 + 1
                spans = py_inert_spans(text[c0:c1])
                if spans is None:
                    continue
                n_parsed += 1
                inert = bytearray(c1 - c0)
                for a_, b_ in spans:
                    for q in range(max(a_, 0), min(b_, c1 - c0)):
                        inert[q] = 1
                sc = INM.InertScan(P.RXC); sc.feed(pcs[0])
                cls = INM.classify(P.RXC, sc, pcs[1:], pcs[1:], arm)
                o = len(pcs[0]); covd = bytearray(c1 - c0)
                for j, (kind, mode, _) in enumerate(cls):
                    L = len(pcs[1 + j])
                    if mode == "whole":
                        n_open += 1
                        bad = [q for q in range(o, o + L) if not (c0 <= q < c1 and inert[q - c0])]
                        if bad:
                            n_out += 1
                            if len(worst) < 4:
                                worst.append((x["name"], kind, repr(text[max(o - 30, 0):o]), repr(pcs[1 + j])))
                        for q in range(o, o + L):
                            if c0 <= q < c1:
                                covd[q - c0] = 1
                    o += L
                cov += sum(1 for q in range(c1 - c0) if inert[q] and covd[q]); tot += sum(inert)
            check(n_out == 0 and n_parsed >= 0.9 * n_ans, f"{label} ({tag}): {n_open} tokens open to wholesale acceptance, {n_out} outside "
                  f"Python's own comments / string statements ({n_parsed} of {n_ans} answers parse); the open tokens cover "
                  f"{100 * cov / max(tot, 1):.0f} % of that text")
            for w in worst:
                print("       outside:", *w)


if __name__ == "__main__":
    t_module()
    t_patch()
    if "--no-walk" not in sys.argv:
        t_walk()
    print(f"{'ALL OK' if not fails else f'{len(fails)} FAILED'}")
    sys.exit(1 if fails else 0)
