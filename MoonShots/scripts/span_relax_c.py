"""SPANRELAX-C (MoonShots, Wed 2026-11-04; the orchestrator 21:4x: Fable FULL's round 2 R2.1, ~/FableGod/full/FULL.md
"## Round 2", lines pre-registered there): rule C = relax only on rows the engine's causal tagger puts inside a '#'
comment, with the line end left to the 125B.
A row j may relax iff (relax_core's terms, read-only import of ~/EvalSuite/evalsuite/relax_core.py):
  1. CodeScan fed the answer text up to and excluding row j's draft: in_comment() (inside a ``` fence, in a '#' comment,
     the fence's third line or later, no quote run open);
  2. comment_draft_ok(draft_j);  3. nontool(ctext + draft_j) and nontool(ctext + argmax_j);
  4. the length guard: '\\n' not in argmax_j;
  5. then R2 top-3 at 0.1 p1: the draft's rank < 3 and log p(d) >= log p1 + log 0.1.
Every other row is exact. Positions = the HPD-0 walk's tested positions (relax_curves' convention, executed read-only:
right = the draft is the 125B's argmax) joined with a top-8 source: baseline.json (the 12 map windows, M-39's) or the
fresh arrays (argv[2]: a baseline-like .json {name: {a, tk, lp}} or a folder of them). kept = relax_curves'
kept_from_profile at 16 rows; ratio to the exact profile; engine kept = 10.54 x ratio.
Feeding (--feed): 'window' (M-41's runs, 22:24 / 22:37: the tagger fed the whole window from its first token, prompt
included; a window without any ``` starts in code) or 'answer' (the engine's and FULL's: fed from the first generated token
on, no override; added Wed 22:5x for the R57_RELAX_SPAN tests).
Checks per relaxed accept: the accepted draft's text, re-scanned by Mission 25's character scan with the draft in place,
lies in a comment (or is blank) = 'tagger and walk tokenizer agree'.
Lines (FULL R2.1): fresh HumanEval+ 12: PASS kept(16) >= x1.10 with 0 relaxed accepts outside in_comment rows and 0
code-token deviations; BETWEEN x1.05-1.10; KILL < x1.05, or any code-token deviation, or > 2 % of relaxed accepts where the
walk's tokenizer and CodeScan disagree. MBPP: PASS >= x1.03, KILL < x1.01. Prediction: fresh HumanEval+ x1.13, MBPP x1.04.
usage: [BigMoE venv] python span_relax_c.py <walk: code | mbpp> [top-8 source] [label] [--feed window|answer]
       -> stdout + data/span_relax_c.txt; importable (build / acc / prof) by patches/r57_relax/test_relax_span.py"""
import importlib.util, json, math, os, sys, time

H = os.path.expanduser("~")
TD = f"{H}/TrainLab/data"
RC, CL = f"{H}/FableGod/relax_curves.py", f"{H}/FableGod/comment_loophole.py"
BASE16 = 10.54
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


_src = open(RC).read()
env = {"__name__": "relax_curves_prefix"}
exec(compile(_src[:_src.index("\nDOMS = [")], RC, "exec"), env)      # npz(), dist_tables() only
_k0 = _src.index("def kept_from_profile"); _k1 = _src.index("\nfor dom in DOMS:\n    P = [q for q in POS")
exec(compile(_src[_k0:_k1], RC, "exec"), env)                         # kept_from_profile
npz, kfp = env["npz"], env["kept_from_profile"]
_src = open(CL).read()
envc = {"__name__": "comment_loophole_prefix"}
exec(compile(_src[:_src.index("\nSETS = [")], CL, "exec"), envc)
regions, dec = envc["regions"], envc["dec"]
_spec = importlib.util.spec_from_file_location("relax_core_ro", f"{H}/EvalSuite/evalsuite/relax_core.py")
RCORE = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RCORE)


def load_rows(p):
    """name -> (row0, [(tk, lp)]) from a baseline-like json or a folder of them"""
    files = [p] if os.path.isfile(p) else [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".json")]
    D = {}
    for f in files:
        for nm, b in json.load(open(f)).items():
            if isinstance(b, dict) and "tk" in b and "lp" in b:
                D[nm[:-4] if nm.endswith(".npz") else nm] = (b["a"], list(zip(b["tk"], b["lp"])))
    return D


def build(walk, srcp, feed="window"):
    """the tested positions with rule C's inputs; feed: 'window' (M-41) or 'answer' (the engine's)"""
    D = load_rows(srcp)
    W = npz(f"{TD}/d02b_walk_hpd0_{walk}.npz")
    POS, nwin = [], 0
    for i, nm in enumerate(W["names"]):
        key = nm[:-4] if nm.endswith(".npz") else nm
        if key not in D:
            continue
        nwin += 1
        row0, rows = D[key]
        ids = list(W[f"{i}_ids"][0])
        gen = int(W[f"{i}_gen"][0][0])
        checks, (nc, _) = W[f"{i}_checks"]
        dcf, dtok = W[f"{i}_dcf"][0], W[f"{i}_dtok"][0]
        per = len(dtok) // max(nc, 1)
        pieces = [dec(t) for t in ids]
        sc = RCORE.CodeScan()
        t0_ = 0 if feed == "window" else gen
        if feed == "window" and "```" not in "".join(pieces):
            sc.code = True
        pre = [(False, "")] * len(ids)
        for t in range(t0_, len(ids)):
            sc.feed(pieces[t])
            pre[t] = (sc.in_comment(), sc.ctext)
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            for j in range(min(k + 1, m)):
                p = A + j; r = p - row0
                if p + 1 >= len(ids) or not (0 <= r < len(rows)):
                    continue
                tk, lp = rows[r]
                d = dtok[c * per + j]
                dtext, atext = dec(d), dec(tk[0])
                inc, ct = pre[p]
                q = dict(win=key, c=c, j=j, p=p, depth=j + 1, right=d == tk[0], wright=j < k, lp0=lp[0],
                         lpd=lp[tk.index(d)] if d in tk else None,
                         rank=sum(1 for v in lp if v > lp[tk.index(d)]) if d in tk else 99, inc=bool(inc), dtext=dtext,
                         atext=atext, c13=bool(inc and RCORE.comment_draft_ok(dtext) and RCORE.nontool(ct + dtext)
                                               and RCORE.nontool(ct + atext)),
                         guard="\n" in atext, ids=ids, d=d)
                POS.append(q)
    return POS, nwin


def acc(q, rule):
    if q["right"]:
        return True
    if q["lpd"] is None or rule == "exact":
        return False
    ok13 = q["c13"] and (rule == "C_noguard" or not q["guard"])
    return ok13 and q["rank"] < 3 and q["lpd"] >= q["lp0"] + math.log(0.1)


def prof(POS, a):
    out = {}
    for dd in range(1, 16):
        S = [x for x, q in zip(a, POS) if q["depth"] == dd]
        out[dd] = sum(S) / len(S) if S else float("nan")
    return out


def main():
    t0 = time.time()
    args = [a for a in sys.argv[1:] if not a.startswith("--feed")]
    feed = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--feed=")), "window")
    walk = args[0] if args else "code"
    srcp = args[1] if len(args) > 1 else f"{H}/FableGod/decision_map/out/baseline.json"
    label = args[2] if len(args) > 2 else os.path.basename(srcp)
    POS, nwin = build(walk, srcp, feed)
    say(f"== SPANRELAX-C rule C ({time.strftime('%a %Y-%m-%d %H:%M')}): walk {walk}, top-8 source {label}, feed {feed}: {nwin} "
        f"windows, {len(POS)} positions; in_comment rows {sum(q['inc'] for q in POS)}, conditions 1-3 {sum(q['c13'] for q in POS)}, "
        f"blocked by the line-end guard {sum(q['c13'] and q['guard'] for q in POS)}; alignment: the 125B's top-1 == the draft agrees "
        f"with the walk's own exact acceptance on {100 * sum(q['right'] == q['wright'] for q in POS) / max(len(POS), 1):.2f} % of positions")
    a0 = [acc(q, "exact") for q in POS]
    e16 = kfp(prof(POS, a0), 16)
    say(f"  exact profile kept(16) {e16:.2f}" + (" [RELAX_350 7.78 on the 12 map windows]" if label == "baseline.json" and walk == "code" else ""))
    R = {}
    for rule in ("C", "C_noguard"):
        a = [acc(q, rule) for q in POS]
        rel = [q for q, x, x0 in zip(POS, a, a0) if x and not x0]
        out_rows = sum(1 for q in rel if not q["inc"])
        dis = 0
        for q in rel:                                                # re-scan with the draft in place: is its text in a comment?
            if not q["dtext"].strip():
                continue
            reg = regions(q["ids"][:q["p"] + 1] + [q["d"]])
            dis += reg[-1] != "comment"
        k16 = kfp(prof(POS, a), 16)
        R[rule] = dict(ratio=k16 / e16, rel=len(rel), out=out_rows, dis=dis)
        say(f"  rule {rule:9s}: kept(16) ratio x{k16 / e16:.3f} = engine {BASE16 * k16 / e16:.2f}; relaxed accepts {len(rel)} "
            f"({100 * len(rel) / len(POS):.2f} % of positions), outside in_comment rows {out_rows}, code-token deviations {out_rows}; "
            f"draft text outside a comment on re-scan {dis} ({100 * dis / max(len(rel), 1):.1f} %)")
    r = R["C"]
    if walk == "code":
        call = ("KILL" if r["ratio"] < 1.05 or r["out"] > 0 or 100 * r["dis"] / max(r["rel"], 1) > 2 else
                "PASS" if r["ratio"] >= 1.10 else "BETWEEN")
        say(f"Lines (FULL R2.1, HumanEval+): x{r['ratio']:.3f} (PASS >= 1.10, KILL < 1.05), {r['out']} accepts outside in_comment "
            f"rows, re-scan disagreement {100 * r['dis'] / max(r['rel'], 1):.1f} % (KILL > 2) -> {call} [prediction fresh x1.13]")
    else:
        call = "KILL" if r["ratio"] < 1.01 or r["out"] > 0 else ("PASS" if r["ratio"] >= 1.03 else "BETWEEN")
        say(f"Lines (FULL R2.1, MBPP): x{r['ratio']:.3f} (PASS >= 1.03, KILL < 1.01) -> {call} [prediction x1.04]")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/span_relax_c.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
