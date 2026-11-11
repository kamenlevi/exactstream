"""inert_report.py (GOAL session, Thu 2026-11-05): read one INERT engine arm against its control. CPU only, python -I.
    python -I inert_report.py <control>_runs.json <arm>_runs.json [--log <R57_INERT_LOG file>] [--arm CD16]
Reads the two runs files of r57_screen (and the tokens files beside them, <tag>_tokens.json). Per prompt both arms ran:
  S  seconds per task (n / tps), arm / control          K  kept per check, arm / control
  T  tokens per task (n), arm / control                 W  the arm's share of tokens kept by INERT (whole + by a rule)
  code  whether the answer's CODE is the same in both arms once comments and string statements are removed (ast): 'same'
        means the tests cannot tell the arms apart; 'differs' means the 125B wrote other code after the drafter's text and
        the task's tests must decide; 'n/a' = one of the two answers does not parse (cut at the token cap, or not python).
Then the arm's lines (INSTALL.md): PASS / BETWEEN / KILL on S and T; the code column is the quality read-out.
The log audit: every relaxed accept's region must be 'comment' or 'docstring'."""
import ast
import importlib.util
import json
import os
import re
import statistics as st
import sys

H = os.path.expanduser("~")
LINES = {"CD16": (0.90, 0.97), "C16": (0.93, 0.98), "G16": (0.95, 0.99), "D16": (0.97, 1.00),
         "CD32": (0.87, 0.95), "G32": (0.93, 0.97), "D32": (0.93, 0.97),      # (PASS: median S <=, KILL: median S >=)
         "T16": (0.975, 1.00), "TC16": (0.98, 1.00), "U16": (0.97, 1.00)}     # the guarded arms (Fri 6 Nov; GUARD.md)
GUARD_RULE = {"T16": (3, 0.1), "TC16": (3, 0.1), "U16": (8, 0.01)}             # top-k at x p1: the log audit checks every accept
T_PASS, T_KILL = 1.05, 1.15


def code_of(text):
    m = re.search(r"```(?:python|py|python3)\n", text)
    if not m:
        return None
    c1 = text.find("\n```", m.end())
    return text[m.end():len(text) if c1 < 0 else c1 + 1]


def bare(code):
    """the code's ast with every string statement (docstrings included) removed; None when it does not parse"""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            kept = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                            and isinstance(s.value.value, str))]
            node.body = kept or [ast.Pass()]
    return ast.dump(tree)


def load_runs(path):
    runs = {}
    for r in json.load(open(path)):
        if "PAUSED" not in r.get("bad", "") and not r.get("cut"):
            runs.setdefault(r["key"], []).append(r)
    return runs


def load_tokens(runs_path):
    p = runs_path.replace("_runs.json", "_tokens.json")
    if not os.path.exists(p):
        return {}
    return {k: [o for outs in v.values() for o in outs] for k, v in json.load(open(p)).items()}


def mean(xs):
    return sum(xs) / len(xs)


def main():
    a = sys.argv[1:]
    log = a[a.index("--log") + 1] if "--log" in a else None
    arm = a[a.index("--arm") + 1] if "--arm" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--log", "--arm"))]
    ctl_p, arm_p = pos[0], pos[1]
    C, A = load_runs(ctl_p), load_runs(arm_p)
    TC, TA = load_tokens(ctl_p), load_tokens(arm_p)
    dec = None
    if TC and TA:
        from tokenizers import Tokenizer
        tkz = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
        dec = lambda ids: "".join(tkz.decode([int(t)]) for t in ids)
    keys = sorted(set(C) & set(A))
    if not keys:
        raise SystemExit("no prompt ran in both files")
    ist = [r.get("inert") for k in keys for r in A[k]]
    arm = arm or next((i["arm"] for i in ist if i), None)
    print(f"control {ctl_p}\narm     {arm_p}  (R57_INERT={arm})\n{len(keys)} prompts in both\n")
    print(f"  {'prompt':34s} {'S':>6s} {'T':>6s} {'K':>6s} {'W %':>6s}  code")
    S, T, K, same, differs = [], [], [], 0, []
    for k in keys:
        cs, as_ = mean([r["n"] / r["tps"] for r in C[k]]), mean([r["n"] / r["tps"] for r in A[k]])
        cn, an = mean([r["n"] for r in C[k]]), mean([r["n"] for r in A[k]])
        ck, ak = mean([r["kept"] for r in C[k]]), mean([r["kept"] for r in A[k]])
        w = [100 * (i["whole"] + i["ruled"]) / max(r["n"], 1) for r in A[k] for i in [r.get("inert")] if i]
        code = "n/a"
        if dec and k in TC and k in TA:
            bc, ba = [bare(code_of(dec(o)) or "") for o in TC[k]], [bare(code_of(dec(o)) or "") for o in TA[k]]
            if all(x is not None for x in bc + ba):
                code = "same" if set(ba) <= set(bc) else "differs"
        same += code == "same"
        if code == "differs":
            differs.append(k)
        S.append(as_ / cs); T.append(an / cn); K.append(ak / ck)
        print(f"  {k:34s} {S[-1]:6.3f} {T[-1]:6.3f} {K[-1]:6.3f} {mean(w) if w else float('nan'):6.1f}  {code}")
    ms, mt = st.median(S), st.median(T)
    print(f"\n  median S {ms:.3f}   median T {mt:.3f}   median K {st.median(K):.3f}   code same on {same} of {len(keys)}"
          + (f"; differs: {', '.join(differs)} (their tests decide)" if differs else ""))
    tot = {f: sum(i[f] for i in ist if i) for f in ("rows", "kx", "k", "whole", "ruled", "com", "doc", "open")} if any(ist) else None
    if tot:
        print(f"  the arm's counters: rows {tot['rows']}, kept exact {tot['kx']} -> in all {tot['k']} (x{tot['k'] / max(tot['kx'], 1):.3f}); "
              f"kept whole {tot['whole']}, by a rule {tot['ruled']}; comment {tot['com']}, docstring {tot['doc']}; rows open {tot['open']}")
    if arm in LINES:
        p, kl = LINES[arm]
        v = ("PASS" if ms <= p and mt <= T_PASS else "KILL" if ms >= kl or mt >= T_KILL else "BETWEEN")
        print(f"  lines for {arm}: PASS median S <= {p} and T <= {T_PASS}; KILL S >= {kl} or T >= {T_KILL}  ->  speed {v}")
        print(f"  quality: PASS needs the tests' verdicts equal on >= {len(keys) - 1} of {len(keys)} "
              f"({same} are settled by 'same' code); KILL at >= 2 tasks that flip to fail")
    if log and os.path.exists(log):
        reg = {}
        for ln in open(log):
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            reg[(e.get("region"), e.get("mode"))] = reg.get((e.get("region"), e.get("mode")), 0) + 1
        bad = {k_: v_ for k_, v_ in reg.items() if k_[0] not in ("comment", "docstring")}
        print(f"  log audit: {sum(reg.values())} relaxed accepts: " + ", ".join(f"{r} / {m} {n}" for (r, m), n in sorted(reg.items(), key=str))
              + ("; ALL in comments / docstrings" if not bad else f"; OUTSIDE: {bad} -> the arm is void"))
        if arm in GUARD_RULE:                               # a guarded arm keeps nothing wholesale and nothing outside its rule
            import math
            topk, x = GUARD_RULE[arm]
            es = [json.loads(ln) for ln in open(log) if ln.strip()]
            out = [e for e in es if e.get("mode") == "whole" or e.get("rank", 99) >= topk or e.get("dlm", -99) < math.log(x) - 1e-3]
            print(f"  guard audit ({arm}: top-{topk} at {x} p1): {len(es) - len(out)} of {len(es)} accepts inside the rule"
                  + ("; ALL inside" if not out else f"; {len(out)} OUTSIDE (first: {out[0]}) -> the arm is void"))


if __name__ == "__main__":
    main()
