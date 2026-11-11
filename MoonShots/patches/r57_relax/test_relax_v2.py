"""test_relax_v2.py (MoonShots, Fri 2026-11-06; the orchestrator 01:4x): the host-light v2 of both relaxed rules.
Done means (the orchestrator): v2's accept decisions equal v1's on recorded checks (same kept), and host us per check, v1
vs v2, measured on CPU; plus the combined arm's replay (rule C + R1 0.5) if the two gains stack.
1. the patch: apply_v2.py's edits apply once to ~/BigMoE/scripts/r57_screen_rlx.py (read-only; nothing written), the
   result compiles, R57_RELAX_V2=0 keeps v1's code paths verbatim, and the module's may_touch is exact on synthetic text.
2. the recorded checks (the HPD-0 walk, the three top-8 sources of M-41): the engine's per-check host work emulated twice,
   v1 (every token decoded by BigMoE's tokenizer, the tagger fed every block) and v2 (cached texts, the tagger only where
   relax_span.may_touch says a comment is possible); span rows, kept per check (rule C; rule C + R1 0.5) must be equal on
   every check, and host us per check is timed for both (perf_counter_ns, one core). The GPU sync v2 removes cannot be
   timed here: it is counted (one fewer per check).
3. the replay's kept ratios (span_relax_c.build, the engine's answer feed): exact, R1 0.5, rule C, rule C + R1 0.5.
usage: [BigMoE venv] python test_relax_v2.py -> stdout + ~/MoonShots/data/test_relax_v2.txt"""
import importlib.util, math, os, sys, time

H = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, f"{H}/MoonShots/scripts")
import relax_span as RSP                                             # noqa: E402
import apply_v2 as AV                                                # noqa: E402
lines, fails = [], []


def say(s):
    print(s, flush=True); lines.append(s)


def check(cond, what):
    say(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails.append(what)


spec = importlib.util.spec_from_file_location("relax_core_bigmoe", f"{H}/BigMoE/scripts/relax_core.py")
RXC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RXC)


def t_patch():
    say("1. the patch")
    src = open(f"{H}/BigMoE/scripts/r57_screen_rlx.py").read()
    try:
        new = AV.apply(src); ok = True
    except SystemExit as e:
        say(f"     {e}"); ok = False
    check(ok, f"apply_v2.py: all {len(AV.EDITS)} anchors found once")
    if not ok:
        return
    try:
        compile(new, "r57_screen_rlx.py", "exec"); comp = True
    except SyntaxError as e:
        say(f"     {e}"); comp = False
    check(comp, "the patched copy compiles")
    keep_v1 = ["sd_, sr_ = lds[:, 0] - l0s.max(axis=-1), (l0s > lds).sum(axis=-1)", "mx.eval(sd_, sr_)",
               "dtx = [_TKZ.decode([int(x)]) for x in dr]", "srows, spre = RSP.span_rows(RXC, sscan, dtx, atx)",
               "dl_, rk_ = ld[:, 0] - l0.max(axis=-1), (l0 > ld).sum(axis=-1)"]
    check(all(k in new for k in keep_v1), "R57_RELAX_V2=0 keeps v1's statements (both rules' own sync and decodes)")
    check(new.count("mx.eval(am, tp, *((dlv_, rkv_) if rx2 else ()))") == 1,
          "v2: dlm / rank join the argmax readback's single mx.eval")
    for txt, want in (("x = 1  # a", True), ("x = 1", False), ("s = 'a#b'", True)):
        sc = RXC.CodeScan(); sc.feed("```python\na = 1\nb = 2\n")
        check(RSP.may_touch(sc, [txt]) == want, f"may_touch({txt!r}) = {want} (a '#' anywhere in a draft counts, even in a string)")
    sc = RXC.CodeScan(); sc.feed("```python\na = 1\nb = 2\nc = 3  # note")
    check(RSP.may_touch(sc, [" more"]), "may_touch is true inside an open comment")


def t_checks():
    say("2. recorded checks: v1 vs v2 host work (span tagger), decisions and us per check")
    from tokenizers import Tokenizer
    TKZ = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
    import span_relax_c as SRC
    cache = {}

    def tdec(x):
        t = cache.get(x)
        if t is None:
            t = cache[x] = TKZ.decode([x])
        return t

    srcs = [("code", f"{H}/FableGod/decision_map/out/baseline.json", "map 12"),
            ("code", f"{H}/Breakthrough/data/rulec/rulec_he12.json", "fresh HumanEval+ 12"),
            ("mbpp", f"{H}/Breakthrough/data/rulec/rulec_mbpp.json", "fresh MBPP 100")]
    for walk, srcp, label in srcs:
        D = SRC.load_rows(srcp)
        W = SRC.npz(f"{SRC.TD}/d02b_walk_hpd0_{walk}.npz")
        res = {}
        for ver in ("v1", "v2"):
            ns, n, rows_on, ks, kcs, srs = 0, 0, 0, [], [], []
            for i, nm in enumerate(W["names"]):
                key = nm[:-4] if nm.endswith(".npz") else nm
                if key not in D:
                    continue
                row0, rows = D[key]
                ids = [int(x) for x in W[f"{i}_ids"][0]]; gen = int(W[f"{i}_gen"][0][0])
                checks, (nc, _) = W[f"{i}_checks"]
                dtok = W[f"{i}_dtok"][0]; per = len(dtok) // max(nc, 1)
                dec = (lambda x: TKZ.decode([x])) if ver == "v1" else tdec
                sc = RXC.CodeScan(); sc.feed(dec(ids[gen])); fed = gen
                for c in range(nc):
                    k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
                    dr = [int(x) for x in dtok[c * per:c * per + m]][:max(0, min(m, len(ids) - 1 - A))]
                    if not dr:
                        continue
                    am, dlm, drk = [], [], []
                    for j, d in enumerate(dr):
                        if 0 <= A + j - row0 < len(rows):
                            tk, lp = rows[A + j - row0]
                            am.append(tk[0]); dlm.append(lp[tk.index(d)] - lp[0] if d in tk else -math.inf)
                            drk.append(sum(1 for v in lp if v > lp[tk.index(d)]) if d in tk else 99)
                        else:
                            am.append(ids[A + j + 1]); dlm.append(-math.inf); drk.append(99)
                    t0 = time.perf_counter_ns()                      # the engine's host work for this check (span tagger)
                    if A > fed:
                        sc.feed("".join(dec(x) for x in ids[fed + 1:A + 1])); fed = A
                    dtx = [dec(x) for x in dr]
                    if ver == "v1" or RSP.may_touch(sc, dtx):
                        atx = [dec(x) for x in am]
                        sr, _ = RSP.span_rows(RXC, sc, dtx, atx)
                        rows_on += 1
                    else:
                        sr = set()
                    kk, _ = RSP.span_k(dr, am, dlm, drk, sr)
                    kc, _ = RSP.span_k(dr, am, dlm, drk, sr, fast_ok=lambda j: dlm[j] >= -0.5)
                    ns += time.perf_counter_ns() - t0; n += 1
                    ks.append(kk); kcs.append(kc); srs.append(sorted(sr))
            res[ver] = dict(us=ns / 1e3 / max(n, 1), n=n, on=rows_on, ks=ks, kcs=kcs, srs=srs)
        a, b = res["v1"], res["v2"]
        check(a["srs"] == b["srs"] and a["ks"] == b["ks"] and a["kcs"] == b["kcs"],
              f"{label}: v2's span rows and kept equal v1's on all {a['n']} checks (rule C and rule C + R1 0.5)")
        say(f"     {label}: host us per check v1 {a['us']:.1f} -> v2 {b['us']:.1f} (x{b['us'] / a['us']:.2f}); the tagger ran on "
            f"{b['on']} of {b['n']} blocks in v2 ({100 * b['on'] / b['n']:.0f} %); GPU syncs per check: v1 2 (argmax + dlm / rank), v2 1")


def t_replay():
    say("3. replay kept (span_relax_c, answer feed; the profile ratio to exact): R1 0.5, rule C, rule C + R1 0.5")
    import span_relax_c as SRC

    def ok(q, rule):
        if q["right"]:
            return True
        if q["lpd"] is None:
            return False
        r1 = q["lpd"] >= q["lp0"] - 0.5
        c = q["c13"] and not q["guard"] and q["rank"] < 3 and q["lpd"] >= q["lp0"] + math.log(0.1)
        return {"r1": r1, "c": c, "c+r1": c or r1}[rule]

    for walk, srcp, label in (("code", f"{H}/FableGod/decision_map/out/baseline.json", "map 12"),
                              ("code", f"{H}/Breakthrough/data/rulec/rulec_he12.json", "fresh HumanEval+ 12"),
                              ("mbpp", f"{H}/Breakthrough/data/rulec/rulec_mbpp.json", "fresh MBPP 100")):
        P, _ = SRC.build(walk, srcp, "answer")
        e16 = SRC.kfp(SRC.prof(P, [SRC.acc(q, "exact") for q in P]), 16)
        r = {rule: SRC.kfp(SRC.prof(P, [ok(q, rule) for q in P]), 16) / e16 for rule in ("r1", "c", "c+r1")}
        say(f"     {label}: R1 0.5 x{r['r1']:.3f}, rule C x{r['c']:.3f}, rule C + R1 0.5 x{r['c+r1']:.3f} "
            f"(stacking: {100 * (r['c+r1'] - 1) / max(1e-9, (r['r1'] - 1) + (r['c'] - 1)):.0f} % of the sum of the two gains)")


if __name__ == "__main__":
    t_patch()
    t_checks()
    t_replay()
    say(f"{'ALL OK' if not fails else f'{len(fails)} FAILED'}")
    open(f"{H}/MoonShots/data/test_relax_v2.txt", "a").write("\n".join(lines) + "\n")
    sys.exit(1 if fails else 0)
