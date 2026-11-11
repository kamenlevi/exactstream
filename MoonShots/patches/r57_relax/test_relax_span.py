"""test_relax_span.py (MoonShots, Wed 2026-11-04): R57_RELAX_SPAN's tests (the orchestrator 22:5x: flag off == exact, and
the walk streams reproduce the replay's accept decisions token for token). CPU only; run outside others' quiet locks.
1. the module: parse_rule, rule_ok, span_k == the exact rule with no span rows (random blocks), span_k's relaxed accepts and
   stop, span_rows' four conditions on a synthetic fenced answer (comment rows, a draft that starts a line, the line-end
   guard, a tool comment, the fence's first two lines); the engine's vendored relax_core equals EvalSuite's in every
   function rule C uses.
2. the patch: apply_span.py's edits apply once to ~/BigMoE/scripts/r57_screen_rlx.py (read-only; nothing written), the
   result compiles, and every inserted statement is behind SPAN / sscan (unset = exact).
3. the walk streams: the engine path (relax_span.span_rows / span_k on the vendored relax_core, the tagger fed the answer
   from the first generated token the way r57 feeds it, every token decoded alone by BigMoE's tokenizer, a copy fed the
   block's drafts) against the replay (scripts/span_relax_c.py: M-41's 'window' feed and the engine's 'answer' feed) on
   every tested row of the three top-8 sources (the map's 12 HumanEval+ windows, Breakthrough's fresh 12 and MBPP 100):
   per-row accept decisions must match the 'answer' replay token for token; differences from M-41's 'window' feed are
   listed and the kept ratios of both printed.
usage: [BigMoE venv] python test_relax_span.py [--no-walk]"""
import ast, hashlib, importlib.util, math, os, random, sys

H = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import relax_span as RSP                                             # noqa: E402
import apply_span as AS                                              # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RXC = load("relax_core_bigmoe", f"{H}/BigMoE/scripts/relax_core.py")   # the engine's copy (r57 loads this one)
RXE = load("relax_core_evalsuite", f"{H}/EvalSuite/evalsuite/relax_core.py")
fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        fails.append(what)


def t_module():
    print("1. the module")
    check(RSP.parse_rule("") == ("R2", 3, 0.1) and RSP.parse_rule("R2:2:0.25") == ("R2", 2, 0.25)
          and RSP.parse_rule("R1:0.5") == ("R1", 0.5), "parse_rule: default R2:3:0.1, R2 / R1 forms")
    try:
        RSP.parse_rule("R3:1"); bad = False
    except ValueError:
        bad = True
    check(bad, "parse_rule rejects an unknown rule")
    r = ("R2", 3, 0.1)
    check(RSP.rule_ok(r, -2.0, 2) and not RSP.rule_ok(r, -2.0, 3) and not RSP.rule_ok(r, -2.31, 0)
          and RSP.rule_ok(r, math.log(0.1), 0), "rule_ok R2 top-3 at 0.1 p1 (rank < 3, dlm >= log 0.1)")
    check(all(RSP.rule_ok(rr, dl, rk) == RXC.rule_ok(rr, dl, rk) for rr in (r, ("R2", 2, 0.25), ("R1", 0.5))
              for dl in (-3.0, -2.31, -2.3, -1.0, -0.4, 0.0) for rk in range(5)), "rule_ok equals relax_core.rule_ok")
    rng = random.Random(7)
    ok = True
    for _ in range(2000):
        n = rng.randint(1, 15)
        am = [rng.randint(0, 4) for _ in range(n)]
        dr = [a if rng.random() < 0.8 else rng.randint(0, 4) for a in am]
        dlm = [rng.uniform(-4, 0) for _ in range(n)]; drk = [rng.randint(0, 6) for _ in range(n)]
        kx = 0
        while kx < n and dr[kx] == am[kx]:
            kx += 1
        k, rl = RSP.span_k(dr, am, dlm, drk, set())
        ok &= (k == kx and rl == [])
    check(ok, "flag-off identity: span_k with no span rows == the exact rule (2,000 random blocks)")
    k, rl = RSP.span_k([1, 2, 9, 4, 8], [1, 2, 3, 4, 5], [0, 0, -1.0, 0, -0.1], [0, 0, 1, 0, 1], {2, 4})
    check((k, rl) == (5, [2, 4]), "span_k keeps a near draft on a span row and goes on")
    k, rl = RSP.span_k([1, 9, 3], [1, 2, 3], [0, -1.0, 0], [0, 1, 0], set())
    check((k, rl) == (1, []), "span_k stops at a mismatch outside the span rows")
    k, rl = RSP.span_k([1, 9, 3], [1, 2, 3], [0, -2.5, 0], [0, 1, 0], {1})
    check((k, rl) == (1, []), "span_k stops when the span row's draft fails the rule (p < 0.1 p1)")
    # span_rows on a synthetic answer: a ``` fence, two code lines, then a comment on the fence's third line
    sc = RXC.CodeScan()
    sc.feed("Here it is:\n```python\ndef f(x):\n    y = 1\n    # compute the")
    rows, pre = RSP.span_rows(RXC, sc, [" sum", "\n", " total"], [" total", "\n", "\n"])
    check(pre[0][0] and rows == {0}, "span_rows: a comment row relaxes; a draft that starts a line and a row whose argmax "
          "ends the line (the guard) do not")
    check(sc.ctext.endswith("compute the") and sc.in_comment(), "span_rows feeds a copy (the run's tagger is unchanged)")
    sc2 = RXC.CodeScan(); sc2.feed("```python\nimport os\nx = 1  # type")
    rows2, _ = RSP.span_rows(RXC, sc2, [":", " the"], [" int", " the"])
    check(rows2 == set(), "span_rows: a comment that is or could become a tool comment ('# type:') never relaxes")
    sc3 = RXC.CodeScan(); sc3.feed("```python\n# header")
    rows3, _ = RSP.span_rows(RXC, sc3, [" line"], [" text"])
    check(rows3 == set(), "span_rows: a comment on the fence's first two lines never relaxes")
    sc4 = RXC.CodeScan(); sc4.feed("```python\na = 1\nb = 2\ns = 'x # y")
    rows4, _ = RSP.span_rows(RXC, sc4, [" z"], [" w"])
    check(rows4 == set(), "span_rows: a '#' inside a string is not a comment")
    def defs(path):
        src = open(path).read()
        return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    de, dv = defs(f"{H}/EvalSuite/evalsuite/relax_core.py"), defs(f"{H}/BigMoE/scripts/relax_core.py")
    same = all(de.get(f) is not None and de.get(f) == dv.get(f)
               for f in ("CodeScan", "nontool", "comment_rows", "comment_draft_ok", "rule_ok")) and RXC.TOOL_PREFIXES == RXE.TOOL_PREFIXES
    check(same, "the engine's vendored relax_core equals EvalSuite's in CodeScan, nontool, comment_rows, comment_draft_ok, rule_ok")


def t_patch():
    print("2. the patch")
    path = f"{H}/BigMoE/scripts/r57_screen_rlx.py"
    src = open(path).read()
    print(f"     r57_screen_rlx.py sha1 {hashlib.sha1(src.encode()).hexdigest()[:12]}")
    try:
        new = AS.apply(src); applied = True
    except SystemExit as e:
        print("     ", e); applied = False
    check(applied, f"apply_span.py: all {len(AS.EDITS)} anchors found once")
    if not applied:
        return
    try:
        compile(new, path, "exec"); comp = True
    except SyntaxError as e:
        print("     ", e); comp = False
    check(comp, "the patched copy compiles")
    heads = []
    for name, _, _, text in AS.EDITS:
        code = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        heads.append(code[0])
    guarded = ("SPAN = os.environ", "assert not (SPAN", "if SPAN and dr:", "if sscan is not None:", "spn=sst if SPAN else None,")
    check(all(any(h.startswith(g) for g in guarded) for h in heads), "every inserted block opens behind SPAN / sscan: "
          + " | ".join(h[:40] for h in heads))
    run_state = AS.EDITS[1][3].split("\n")
    check(all(ln.strip().startswith(("assert not (SPAN", "sscan =", "sst =", "slog =", "if sscan", "sscan.feed")) or not ln.strip()
              for ln in run_state), "run state with SPAN unset: a no-op assert, sscan / slog None, an unused counter")
    check(new.count("R57_RELAX_SPAN") >= 3 and "spn=sst if SPAN else None" in new, "the result carries spn (None when unset)")


def t_walk():
    print("3. the walk streams")
    from tokenizers import Tokenizer
    TKZ = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
    sys.path.insert(0, f"{H}/MoonShots/scripts")
    import span_relax_c as SRC
    srcs = [("code", f"{H}/FableGod/decision_map/out/baseline.json", "map 12"),
            ("code", f"{H}/Breakthrough/data/rulec/rulec_he12.json", "fresh HumanEval+ 12"),
            ("mbpp", f"{H}/Breakthrough/data/rulec/rulec_mbpp.json", "fresh MBPP 100")]
    for walk, srcp, label in srcs:
        if not os.path.exists(srcp):
            check(False, f"{label}: {srcp} missing"); continue
        rep = {}
        for feed in ("window", "answer"):
            POS, _ = SRC.build(walk, srcp, feed)
            rep[feed] = {(q["win"], q["c"], q["j"]): (SRC.acc(q, "C"), q) for q in POS}
        D = SRC.load_rows(srcp)
        W = SRC.npz(f"{SRC.TD}/d02b_walk_hpd0_{walk}.npz")
        eng = {}
        for i, nm in enumerate(W["names"]):
            key = nm[:-4] if nm.endswith(".npz") else nm
            if key not in D:
                continue
            row0, rows = D[key]
            ids = list(W[f"{i}_ids"][0]); gen = int(W[f"{i}_gen"][0][0])
            checks, (nc, _) = W[f"{i}_checks"]
            dtok = W[f"{i}_dtok"][0]; per = len(dtok) // max(nc, 1)
            sc = RXC.CodeScan()
            sc.feed(TKZ.decode([ids[gen]]))                              # r57: the first generated token
            fed = gen
            for c in range(nc):
                k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
                if A > fed:                                              # r57: each check's committed tokens, decoded alone
                    sc.feed("".join(TKZ.decode([int(x)]) for x in ids[fed + 1:A + 1])); fed = A
                dr = [int(x) for x in dtok[c * per:c * per + m]]
                dr = dr[:max(0, min(len(dr), len(ids) - 1 - A))]
                if not dr:
                    continue
                am, dlm, drk = [], [], []
                for j, d in enumerate(dr):
                    if 0 <= A + j - row0 < len(rows):
                        tk, lp = rows[A + j - row0]
                        am.append(tk[0])
                        dlm.append(lp[tk.index(d)] - lp[0] if d in tk else -math.inf)
                        drk.append(sum(1 for v in lp if v > lp[tk.index(d)]) if d in tk else 99)
                    else:                                                # no top-8 row (a window's first row): the
                        am.append(int(ids[A + j + 1])); dlm.append(-math.inf); drk.append(99)   # walk's own token; not compared
                dtx = [TKZ.decode([x]) for x in dr]; atx = [TKZ.decode([x]) for x in am]
                srows, _ = RSP.span_rows(RXC, sc, dtx, atx)
                for j in range(min(k + 1, len(dr))):
                    eng[(key, c, j)] = dr[j] == am[j] or (j in srows and RSP.rule_ok(RSP.DEFAULT_RULE, dlm[j], drk[j]))
        for feed in ("answer", "window"):
            R = rep[feed]
            common = [x for x in R if x in eng]
            mism = [x for x in common if R[x][0] != eng[x]]
            if feed == "answer":
                check(len(common) == len(R) and not mism, f"{label}: the engine path reproduces the replay (answer feed) on "
                      f"{len(common)} / {len(R)} tested rows, {len(mism)} differ")
            else:
                print(f"     {label}: vs M-41's window feed: {len(mism)} of {len(common)} rows differ")
            for x in mism[:6]:
                q = R[x][1]
                print(f"       {feed} {x}: replay {R[x][0]} engine {eng[x]}; draft {q['dtext']!r} argmax {q['atext']!r}, "
                      f"in_comment {q['inc']}, c13 {q['c13']}, guard {q['guard']}, rank {q['rank']}")
            P_ = [R[x][1] for x in common]
            a0 = [SRC.acc(q, "exact") for q in P_]
            e16 = SRC.kfp(SRC.prof(P_, a0), 16)
            print(f"     {label} ({feed} feed): kept ratio replay x{SRC.kfp(SRC.prof(P_, [R[x][0] for x in common]), 16) / e16:.3f}, "
                  f"engine path x{SRC.kfp(SRC.prof(P_, [eng[x] for x in common]), 16) / e16:.3f}")


if __name__ == "__main__":
    t_module()
    t_patch()
    if "--no-walk" not in sys.argv:
        t_walk()
    print(f"{'ALL OK' if not fails else f'{len(fails)} FAILED'}")
    sys.exit(1 if fails else 0)
