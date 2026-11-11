"""nocom_report.py (Fable FULL, LOSSY350 T-B1): the comment-free prompt arm against qall, from the Friday run's files.
CPU, no model. usage (from BigMoE's root, as fri_night.sh calls it): python -I nocom_report.py <C> [out.txt] [--dir data/r57]
Reads {dir}/f{qall,qallb,qallp,qallpb}_{C}_{runs,checks,tokens}.json (A = qall, P = qall + R57_NOCOM=1; two passes each,
pair means per prompt as fri_night.sh's own report) and, when present, data/fri/nocom_tests.json (nocom_tests.py).
Per prompt and arm: generated tokens, comment tokens (relax_core.CodeScan over the decoded answer: a token with any char
inside a '#' comment of a code block), the answer's verbatim copy run of the prompt (>= 8 tokens), kept per check over all
checks and over the checks that start OUTSIDE the usable copy run (the run minus its first 8 tokens), steady tok/s (tpsw),
seconds per task (n / tps), and the checks that keep only comment tokens.
Lines (pre-registered in ~/FableGod/full/LOSSY350_FULL.md section 5, for the benchmark's CAP 256 window):
  PASS    comment tokens <= 2 % of P's tokens AND kept per check outside the copy run >= x1.15 AND tok/s >= +8 %
          AND no task that passes in A fails in P (tasks that ended before the cap in both arms; nocom_tests.py).
  KILL    kept outside the copy run < x1.05, OR tok/s < +3 %, OR >= 2 new failures.
  BETWEEN anything else; with comment tokens > 2 % the sentence was ignored and the mask form decides."""
import importlib.util, json, math, os, statistics as S, sys

H = os.path.expanduser("~")
EOS = {248044, 248046}
TRIG = 8


def _load_core():
    for p in (f"{H}/BigMoE/scripts/relax_core.py", f"{H}/EvalSuite/evalsuite/relax_core.py"):
        if os.path.exists(p):
            sp = importlib.util.spec_from_file_location("relax_core_nocom", p)
            m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
            return m
    raise SystemExit("nocom_report: relax_core.py not found")


def flags(RC, dec, ids):
    """per token: 1 = a comment token (any char inside a comment), 2 = the newline that ends a comment, 0 = other"""
    sc = RC.CodeScan(); out = []
    for t in ids:
        s = dec(t); before = bool(sc.code and sc.com); anyc = False
        for ch in s:
            sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
        out.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
    return out


def copy_run(prompt, ans):
    """the answer's first verbatim copy of the prompt of >= 8 tokens: (start, end) in the answer, or None"""
    pos = {}
    for i, t in enumerate(prompt):
        pos.setdefault(t, []).append(i)
    q = 0
    while q < len(ans):
        best = 0
        for i in pos.get(ans[q], ()):
            n = 0
            while q + n < len(ans) and i + n < len(prompt) and ans[q + n] == prompt[i + n]:
                n += 1
            best = max(best, n)
        if best >= 8:
            return q, q + best
        q += 1
    return None


def geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def read_arm(d, tag, C, items, RC, dec, cap):
    """{key: per-run measures} for one pass of an arm, or {} when its files are missing"""
    try:
        runs = {r["key"]: r for r in json.load(open(os.path.join(d, f"{tag}_{C}_runs.json")))}
        toks = json.load(open(os.path.join(d, f"{tag}_{C}_tokens.json")))
        chks = {c["key"]: c for c in json.load(open(os.path.join(d, f"{tag}_{C}_checks.json")))}
    except (OSError, ValueError):
        return {}
    out = {}
    for key, r in runs.items():
        if "PAUSED" in r.get("bad", "") or key not in toks or key not in items:
            continue
        ans = next(iter(toks[key].values()))[-1][:cap]              # {arm: [tokens of each run]}: the one run of this pass
        fl = flags(RC, dec, list(items[key]["prompt_ids"]) + ans)[len(items[key]["prompt_ids"]):]
        run = copy_run(list(items[key]["prompt_ids"]), ans)
        kept = [int(k) for k in chks.get(key, {}).get("kept", [])]
        k_out, k_all, com_only, p = [], [], 0, 1                     # token 0 comes from the prefill; check i starts at p
        for k in kept:
            span = range(p, min(p + k, len(ans)))
            if len(span):
                in_copy = run is not None and run[0] + TRIG <= p < run[1]
                k_all.append(k)
                if not in_copy:
                    k_out.append(k)
                com_only += all(fl[i] == 1 for i in span)
            p += k
        out[key] = dict(n=len(ans), com=sum(1 for f in fl if f == 1), tpsw=r["tpsw"], secs=r["n"] / r["tps"], kept_all=k_all,
                        kept_out=k_out, com_only=com_only, checks=len(k_all), ended=bool(ans) and ans[-1] in EOS and len(ans) < cap,
                        copy=(run[1] - run[0]) if run else 0)
    return out


def pair(a, b, f):
    """per prompt: the mean of f over the arm's passes that have the prompt"""
    out = {}
    for k in set(a) | set(b):
        v = [f(x[k]) for x in (a, b) if k in x and f(x[k]) == f(x[k])]
        if v:
            out[k] = sum(v) / len(v)
    return out


def report(C, d="data/r57", tests="data/fri/nocom_tests.json", cap=256, items=None, dec=None):
    RC = _load_core()
    if dec is None:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
        dec = lambda t: tk.decode([int(t)])
    if items is None:
        items = {r["key"]: r for r in map(json.loads, open(f"{H}/EvalSuite/data/items/dec_humaneval.jsonl"))}
    A1, A2, P1, P2 = (read_arm(d, t, C, items, RC, dec, cap) for t in ("fqall", "fqallb", "fqallp", "fqallpb"))
    L = [f"T-B1 comment-free prompt arm (R57_NOCOM=1) vs qall, C {C}, cap {cap}: passes with data A {len(A1)} / {len(A2)} prompts, "
         f"P {len(P1)} / {len(P2)}"]
    if not (A1 or A2) or not (P1 or P2):
        L.append("Verdict T-B1: MISSING (an arm has no files)")
        return "\n".join(L)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    m = {}
    for nm, f in (("tpsw", lambda x: x["tpsw"]), ("secs", lambda x: x["secs"]), ("n", lambda x: x["n"]),
                  ("kall", lambda x: mean(x["kept_all"])), ("kout", lambda x: mean(x["kept_out"]))):
        m[nm] = (pair(A1, A2, f), pair(P1, P2, f))
    def ratio(nm, agg):
        a, p = m[nm]; ks = [k for k in p if k in a and a[k] > 0]
        return (agg([p[k] / a[k] for k in ks]), len(ks)) if ks else (float("nan"), 0)
    tot = lambda arms, f: sum(f(x) for a in arms for x in a.values())
    cA = tot((A1, A2), lambda x: x["com"]) / max(1, tot((A1, A2), lambda x: x["n"]))
    cP = tot((P1, P2), lambda x: x["com"]) / max(1, tot((P1, P2), lambda x: x["n"]))
    coA = tot((A1, A2), lambda x: x["com_only"]) / max(1, tot((A1, A2), lambda x: x["checks"]))
    coP = tot((P1, P2), lambda x: x["com_only"]) / max(1, tot((P1, P2), lambda x: x["checks"]))
    cpA = tot((A1, A2), lambda x: x["copy"]) / max(1, tot((A1, A2), lambda x: x["n"]))
    r_t, n_ = ratio("tpsw", geo); r_ko, _ = ratio("kout", geo); r_ka, _ = ratio("kall", geo)
    r_s, _ = ratio("secs", S.median); r_n, _ = ratio("n", S.median)
    L.append(f"  comment tokens: A {100 * cA:.1f} % of its tokens, P {100 * cP:.1f} % (line <= 2 %); checks that keep only comment "
             f"tokens: A {100 * coA:.1f} % of its checks, P {100 * coP:.1f} % (FULL's prediction for A inside CAP 256: 15-20 %); "
             f"the copy run is {100 * cpA:.0f} % of A's tokens")
    L.append(f"  P vs A over {n_} prompts (pair means per prompt): steady tok/s {100 * (r_t - 1):+.1f} % (geo); kept per check outside "
             f"the copy run x{r_ko:.3f}, over all checks x{r_ka:.3f} (geo); tokens per task x{r_n:.3f}, seconds per task x{r_s:.3f} "
             f"(medians)")
    newf, tnote = None, "tests not run (nocom_tests.py under EvalSuite's venv writes data/fri/nocom_tests.json)"
    if os.path.exists(tests):
        T = json.load(open(tests))
        ended = lambda arms, k: all(a[k]["ended"] for a in arms if k in a) and any(k in a for a in arms)
        ks = [k for k in set(A1) | set(A2) if ended((A1, A2), k) and ended((P1, P2), k)]
        okA = lambda k: all(T.get(t, {}).get(k) for t in ("fqall", "fqallb") if k in T.get(t, {})) and any(k in T.get(t, {}) for t in ("fqall", "fqallb"))
        failP = lambda k: any(T.get(t, {}).get(k) is False for t in ("fqallp", "fqallpb"))
        newf = [k for k in ks if okA(k) and failP(k)]
        fixed = [k for k in ks if not okA(k) and not failP(k) and any(k in T.get(t, {}) for t in ("fqallp", "fqallpb"))]
        tnote = (f"HumanEval+ tests on the {len(ks)} tasks that ended before the cap in both arms: A passes {sum(okA(k) for k in ks)}, "
                 f"new failures in P {len(newf)} {newf}, fails in A that pass in P {len(fixed)}")
    L.append("  " + tnote)
    if r_ko != r_ko or r_t != r_t:
        v = "NO VERDICT (no shared prompts)"
    elif r_ko < 1.05 or r_t < 1.03 or (newf is not None and len(newf) >= 2):
        v = "KILL"
    elif cP <= 0.02 and r_ko >= 1.15 and r_t >= 1.08 and newf is not None and not newf:
        v = "PASS"
    elif cP <= 0.02 and r_ko >= 1.15 and r_t >= 1.08 and newf is None:
        v = "PASS pending the tests"
    elif cP > 0.02:
        v = "BETWEEN (the sentence was ignored on part of the answers: the mask form decides)"
    else:
        v = "BETWEEN"
    L.append(f"Verdict T-B1 (pre-registered): {v}; PASS = comment tokens <= 2 % AND kept outside the copy run >= x1.15 AND tok/s >= "
             f"+8 % AND no new test failure; KILL = kept < x1.05 OR tok/s < +3 % OR >= 2 new failures")
    return "\n".join(L)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    d = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--dir"), "data/r57")
    a = [x for x in a if x != d]
    out = report(a[0], d)
    print(out)
    if len(a) > 1:
        open(a[1], "w").write(out + "\n")
